# Code Review Report — ConnectClaw 全量代码审查

Generated: 2026-10-16 00:25
Scope: **全量代码**（71 个 Python 文件 / ~2.1 万行），非增量 commit 审查
Method: 分 5 个子系统并行精读（provider+agent+harness / coding+safety / hashline / memory / channel+入口+配置），关键结论已由审查者本地复现；报告作者另对 8 条最高危结论做了独立复核。

---

## Overall Score: D (38/100)

| 指标 | 数量 |
|---|---|
| 审查文件 | 71 |
| 高严重问题 | 14 |
| 中严重问题 | 30+ |
| 低严重问题 | 30+ |
| 现有测试 | 394 passed（**未覆盖**下述绝大多数缺陷） |

> **评分说明**：按技能评分标准，高严重 ≥ 3 即判 D。这里判 D 并非否定工程质量——恰恰相反，架构（前缀缓存纪律、结构化注入、bwrap 隔离模型、hashline 设计）明显是有取舍记录、有深度沉淀的，测试也全绿。但**"测试全绿"与"存在一批已复现的正确性 / 安全缺陷"可以并存**：现有测试几乎都在 happy path，没有覆盖错误注入、并发、失败降级、以及"模型能输入什么"这条信任边界。因此这份报告的 D 是**缺陷密度**，不是**设计水平**。

**一句话结论**：这是一个设计成熟、实现有细悟的项目，但存在 **3 类必须优先修的问题**——(1) 文件工具的 cwd/受保护路径约束在 `hash_edit`/`read` 上**完全缺失**（宿主任意读写 = 沙箱逃逸）；(2) 记忆子系统有"每轮必崩"和"做梦被空转/复活"的正确性 bug；(3) 会话落盘/飞书授权等异步路径有丢数据与泄漏。见文末修复优先级。

---

## 一、高严重问题

### H1. `stream.py` 客户端缓存用 `dict` 调用 `popitem(last=False)` → 必抛 `TypeError`

`connectclaw/provider/stream.py:24,110-111`
```python
24: _client_cache: dict[tuple[str, str], Any] = {}          # 普通 dict
...
110:            while len(_client_cache) > _client_cache_max:   # 默认上限 8
111:                _, old_client = _client_cache.popitem(last=False)
```
`last=` 只属于 `OrderedDict`。**已独立复核**（本机 `dict.popitem(last=False)` → `TypeError: dict.popitem() takes no keyword arguments`）。该分支在第 9 个不同 `(base_url, api_key[:8], proxy)` 组合出现时触发——"死 key 清理后换 key"、"模型发现探测"都会累积不同 key。异常在 `for attempt` 之外抛出，**整个 `stream_simple` 生成器崩溃，本轮直接失败**。对照：`provider/tokenizer.py:33`、`provider/deepseek.py:21` 同款缓存都正确用了 `OrderedDict`。

**修复**：改用 `OrderedDict`（命中时 `move_to_end`），或 `_client_cache.pop(next(iter(_client_cache)))`。

---

### H2. 文件工具层缺 cwd/受保护约束 → 宿主任意读写（沙箱逃逸）**已复核**

三处独立漏洞，合起来构成完整逃逸链：

**(a) `hash_edit` 无 cwd 限制** — `connectclaw/coding/tools/hash_edit.py:351-354`
```python
def _resolve_path(self, path: str) -> str:
    if os.path.isabs(path):
        return path                       # 绝对路径原样返回
    return os.path.normpath(os.path.join(self._cwd, path))
```
它只调 `is_protected`（:153），**完全没有 cwd 收敛**。而 `coding_agent.on_before_tool` 的授权钩子只拦 `write` 和 `bash`（`coding_agent.py:880`），对 `hash_edit` **不拦截**。`hash_edit` 默认在白名单里（`config.py:128-133`），也作为子 agent 基础原语（`coding_agent.py:234`）→ **模型可 `hash_edit("/home/user/.ssh/authorized_keys")`、`~/.bashrc`、systemd user unit、cron.d 等**，绕过全部沙箱。

**(b) `read` / `hash_read` 无 cwd 限制，且不拒读受保护文件** — `connectclaw/coding/tools/read.py:108-111`、`hash_read.py:266-269`
```python
if os.path.isabs(path):
    return path                           # 任意绝对路径直读；'../' 亦可越出 cwd
```
`read` 甚至**不调 `is_protected`**；`hash_read` 调了但只是**加一行警告后照样返回内容**（`hash_read.py:256-260`）。于是向机器人说一句 `read ~/.connectclaw/config.toml`，就会把 `app_secret`、`LLM api_key`、`models.toml` 里**所有 provider 的 key** 贴回聊天——`security.py:22` 声称的"agent 不可读 config/registry 契约"**没有落地**。

**(c) 受保护判定 & 写 cwd 判定不解析符号链接** — `connectclaw/security.py:31-33`
```python
ap = os.path.normpath(os.path.abspath(os.path.expanduser(path)))   # 非 realpath
```
`ln -s ~/.connectclaw /proj/link`（`ln` 在 BashGuard 里是 SAFE）后 `hash_edit("/proj/link/config.toml")` 的 `is_protected` 为 False → 落地真实 `config.toml` → **agent 可自行扩大 `[bash] operator_open_ids` 白名单**（正是 `security.py` 文档承诺封死的逃逸口）。

**修复**：给 `hash_edit`/`read`/`hash_read` 加与 `write` 一致的 cwd 收敛 + 受保护读写拒绝；`is_protected`/cwd 判定统一改 `os.path.realpath`；`on_before_tool` 对 `hash_edit` 也接授权钩子。

---

### H3. 沙箱降级时"受保护文件只读"承诺失效 — `sandbox.py:347-350`

`BwrapSandbox` 用 `--ro-bind / /` + 对 protected 文件附加 `--ro-bind`（:297-309），但 `NamespaceSandbox` **只建 namespace、不做任何 remount**，rootfs 依旧可写：
```python
347:    def build(self, command: str) -> list[str]:
348:        inner = f"cd {self.cwd}; {command}"          # 另见 M19：cwd 未 shell 引用
349:        return ["unshare", "--mount", "--fork", "--pid", "--mount-proc",
350:                "--", "bash", "-c", inner]
```
`resolve_sandbox_cls("auto"/"bwrap"/"unshare")` 在探测失败时**自动降级到 `DirectSandbox`**（`sandbox.py:195-205`）。即无 bwrap 的机器（容器/受限内核很常见）上，bash 可写 `config.toml`，而 `security.py` 仍宣称"bash 不能改"。**修复**：降级时要么做等价只读 remount，要么拒绝启动，至少把 `environment_facts` 升格为强告警。

---

### H4. `web_fetch` 无 URL/scheme/host 校验 → SSRF（含重定向绕过）

`connectclaw/coding/tools/web_search.py:187` 直接透传 `params["url"]`；`lightpanda.fetch_once` 先走裸 `httpx.get(url, follow_redirects=True)`（`lightpanda.py:772-806`），失败再回退真浏览器。可指向 `http://127.0.0.1:<port>/`（本机 Lightpanda CDP `/json`）、`http://169.254.169.254/latest/meta-data/`、内网服务；`follow_redirects=True` 使"公网外链 302 到内网"绕过任何前缀判断。**修复**：校验 scheme ∈ {http,https}，解析**最终** URL 的 host，禁 loopback/私网/链路本地/元数据网段。

---

### H5. 会话落盘 fire-and-forget：不持 Task 引用 + 无锁 → 丢消息 / 破坏 parent 链

`connectclaw/agent/harness/agent_harness.py:454-462`
```python
asyncio.get_running_loop().create_task(
    self._session.append_message(self._serialize_message(msg))
)   # 返回值被丢弃
```
叠加 `session.py:201-211`：`entry_id`/`parent_id` 在 `await` 前同步读取 `self._current_leaf_id`，`await append_entry()` 之后才更新。同一轮多条 `message_end` 会在同一同步批次内连续 emit → 多个任务**读到同一个尚未更新的 leaf**，本应线性串联的消息变**兄弟分支**，落盘顺序还取决于 aiofiles 线程池完成顺序。此外 CPython 对 `create_task` 返回值仅持弱引用，任务可能在执行中途被 GC（消息静默丢失，异常无人 retrieve）。审查者已复现三条并发 append 得到同一父节点。

**修复**：用容器持 task 引用 + `done_callback` retrieve 异常；给 JSONL 存储加 `asyncio.Lock` 串行化 `append_*`，让 `parent_id` 与写入同临界区决定。

---

### H6. 记忆召回错误路径返回 `""` → 每轮 `TypeError` 直接崩 **已复核**

`connectclaw/memory/subsystem.py:150,166`
```python
if not self._retriever:
    return "", []            # 应为 return [], []
...
except Exception as e:
    logger.debug("Memory: recall failed: %s", e)
    return "", []            # docstring 承诺"错误不破坏本轮"，实则必崩
```
调用方 `connectclaw/coding/coding_agent.py:410,427`：
```python
mem_ops, recalled_memories = await self._memory.recall(...)
...
op_list = mem_ops + context_state.catalog_delta(...)   # "" + [list] → TypeError
```
`handle_message` 只兜 `CancelledError`，拦不住。触发条件：`[memory] enabled=false`，或检索期任意异常（如 H7/M24 的维度不匹配）。**修复**：错误/禁用路径返回 `[]`。

---

### H7. 做梦把已删除记忆**复活**并覆盖合并结果 — `consolidator.py:458-473` **已复核**

```python
        for mid in ids[1:]:
            self._store.delete(mid)              # 合并：删掉被并入项
        ...
        for mid in result.get("forget", []):
            self._store.delete(mid)              # 遗忘：删除
        for ep in episodes:
            ep.strength *= 0.7                   # ← 对整批 episode 重放写回
            self._store.update(ep)
```
`update()` 实为 `INSERT OR REPLACE`（`store.py:131`），最后这段把**刚删除的条目全量写回**，并用内存中**合并前**的旧 `content` 覆盖 `ids[0]` 已写入的 `merged_content`。**复核确认**：`forget e1` + `merge e1+e2` 后，e1 复活、合并内容丢失、e2 复活 → 整个 consolidation 空转。每次 dream（≥5 条旧 episodic）必触发。

**修复**：先衰减、后删除；或只对未进 `forget`/未被合并的 episode 写回。

---

### H8. 软遗忘被衰减地板复位 — `consolidator.py:212-216`

```python
new_strength = entry.strength * decay_factor
floor = entry.importance * 0.3
new_strength = max(new_strength, floor)          # ← 把 strength=0 抬回
```
软删把 `strength=0`（`subsystem.py:596-599`），下次 dream 第 1 步 `_apply_decay` 因地板抬回 `importance*0.3`。已复现：`importance≥0.34` 的记忆强度回到 ≥0.1（高于 `min_strength`），又被召回并逃过 cleanup——`coding/tools/memory.py` 承诺的"下次 dream 回收"不成立。**修复**：对显式软删（strength 恰为 0）跳过地板。

---

### H9. 飞书授权超时提示被二次更新覆盖为"已拒绝" **已复核**

`connectclaw/channel/feishu.py:665-679`
```python
except TimeoutError:
    auth.approved = False
    if msg_id:
        await self._update_auth_card(msg_id, command, rid,
                                     approved=False, timed_out=True)   # 第一次：⏰超时
self._auth_requests.pop(rid, None)
if msg_id:
    await self._update_auth_card(msg_id, command, rid,
                                 approved=auth.approved, timed_out=False)  # 第二次：覆盖成❌拒绝
```
超时分支后未 return，无条件再更新一次 → 飞书永远只显示"❌ 已拒绝"，且多发一次卡片更新。**修复**：超时分支后 `return auth.approved`，或用 `resolved` 标志跳过。

---

### H10. `_auth_requests` 在取消/异常时泄漏 → 僵尸授权卡

`connectclaw/channel/feishu.py:655-679` 全段**无 try/finally**。两条泄漏路径：(1) 等授权时同会话 `/stop` → `event.wait()` 抛 `CancelledError`（非 `TimeoutError`）→ `pop` 被跳过；(2) `send_card` 抛异常 → 同样跳过。条目只增不减，卡片仍是可点的活按钮。**修复**：`try/finally: self._auth_requests.pop(rid, None)`。

---

### H11. 无 SIGTERM 处理 → 守护进程停机跳过优雅清理 **已复核**

`connectclaw/main.py:354-359` 只注册 `SIGINT`（`grep SIGTERM` 仅命中 `bg_tasks.py`）。`systemd`/`docker stop`/`kill` 默认发 `SIGTERM` → 进程被直接终止，`finally`（`main.py:365-373`）中的 `memory.close()` / `attachment_store.flush()` / `channel.close()` 全不执行，sessions/attachments 可能丢写。**修复**：同时 `loop.add_signal_handler(signal.SIGTERM, _on_sigint)`。

---

### H12. hashline：CRLF 被静默改写成 LF

`connectclaw/coding/tools/hash_edit.py:187,204,299`、`hash_read.py:107`
```python
187: async with aiofiles.open(absolute_path, "r") as f:   # 缺 newline="" → universal newlines 折叠 \r\n
...
204: original_ending = detect_line_ending(raw_text)     # 恒返回 "\n"
```
读入时 `\r\n`/`\r` 已被折叠为 `\n`，`restore_line_endings` 永远还原不了。已复现：编辑 CRLF 文件任一行 → 整文件行尾变 LF（全文件级 diff）；`has_mixed_line_endings` 因输入已折叠而恒为 False（死代码）。**修复**：读/写均加 `newline=""`。

---

### H13. hashline：replace 与同边界 insert 字符偏移错位 → 静默写坏

`connectclaw/hashline/apply.py:793-801,820`

`_assert_no_conflicting_spans` 只拦 `ins.start < rep.end`（:758），恰好允许 `ins.start == rep.end`。排序把 replace 排在 insert 前先应用；insert 随后仍用**原始坐标**的 `span.start`（:820）落在已变长的串中间。已复现：
```
content "aa\nbb\ncc\n" + [replace 行1→"XXXX"], [append 行1→"Q"]
期望 "XXXX\nQ\nbb\ncc\n"  实际 "XX\nQXX\nbb\ncc\n"（且返回成功）
```
**修复**：同 end 时 insert 应先于 replace 应用，或对 insert 位置做偏移校正。

---

### H14. hashline：快照恢复丢文件末尾换行 / 原子写丢权限位

- **尾换行丢失** `connectclaw/hashline/merge.py:67-82`：`"\n".join(...)` 拼回时末尾换行永久丢失。已端到端复现（`recovered=True` 但写入 `b'X\nb'`，少了结尾 `\n`）。
- **权限位丢失** `connectclaw/coding/tools/hash_edit.py:357-373`：临时文件用默认 umask 创建后 `os.replace`，迁移的是**临时文件**的 mode。已复现 `0o755 → 0o664`。编辑 shell 脚本/hook 会静默清掉可执行位。

**修复**：merge 保留 `original.endswith("\n")`；写前 `os.stat` 原文件 mode 并 `os.chmod(tmp, mode)`。

---

## 二、中严重问题

### provider / agent / harness

| # | 位置 | 问题 | 修复 |
|---|---|---|---|
| M1 | `stream.py:118-133,163-343` | 重试未重置 `partial`/`_tool_call_slots`：中途失败重试会把两次生成的文本**拼接**、tool 参数 JSON **损坏**（静默内容损坏） | 循环内重建 `partial`/slots |
| M2 | `stream.py:323-331` | 吞掉 `asyncio.CancelledError` 且不重新抛出，破坏取消语义；取消中继续 `yield` 亦可能再抛 | 收尾后 `raise` |
| M3 | `agent_loop.py:509-579,583-585` | 并行执行前**无 `signal.is_set()` 前置检查**（顺序路径有）；Phase 3 提前 `break` **丢弃已执行工具的结果**，`/stop` 后工具仍在跑 | Phase 2 前检查；Phase 3 不丢结果 |
| M4 | `agent_loop.py:487-504` | 顺序模式工具预处理失败时漏发 `tool_execution_end` → `pending_tool_calls`（`agent.py:369`）永久残留（与并行路径行为不一致） | 异常分支补 emit |
| M5 | `agent_loop.py:839-863` | `_strip_orphan_tool_calls` 只要有**一个** tool_call 无结果，就剥掉整条消息**全部** tool_calls → 反把"孤立 toolResult"变 400 | 只删孤儿 tool_call，保留其余 |
| M6 | `compaction.py:299-318` | `_extract_file_ops` 恒空：`ToolResultMessage` **无 `details` 字段**，落盘只带 `content`。摘要里 Files Read/Modified 永远为空（死逻辑） | 构造时带 `details`，或删掉 |
| M7 | `session.py:135,184,194,324,333` | aiofiles 未指定 `encoding="utf-8"`；非 UTF-8 locale 下写中文 `UnicodeEncodeError`（消息静默丢失）或读回乱码 | 显式 `encoding="utf-8"` |
| M8 | `agent_harness.py:479-496` | 被 abort 的轮次 `prompt()` 返回**上一条** assistant 消息（abort 不 emit `message_end`，只能从尾部找）→ 契约错误 | abort 时返回 `None`/本轮 aborted |
| M9 | `agent_harness.py:250-260` | `context_state()` 折叠 bug：`first_kept_entry_id`（消息 id）在 **context/compaction id 列表**里查，恒 None → **保留区新增的 context op 被丢弃** → 下一轮重复注入（破坏前缀缓存）。真实会话 5 次压缩 5 次命中 | 按路径位置切片，非 id 命中 |
| M10 | `compaction.py:419,464` | 摘要 LLM 调用未传 `signal` → 压缩无法被 `/stop` 中断 | 透传 signal |
| M11 | `session.py:140-157` | 损坏 JSON 行静默 `continue`（无日志）；`SessionHeader(**data)`/`MessageEntry(**data)` 在 `try` **之外**，遇未知字段抛 `TypeError` 整个会话打不开 | 记 warning；过滤未知字段 |
| M12 | `rag/embedding_store.py:26-90` | `search()` 不触发初始化；`open_table`/`search` 异常被静默降级为"空库" | 区分"空"与"故障" |
| M13 | `prompt_builder.py:69` | 对模板整体 `str.format()`：用户自定义 prompt 里任何 `{` 都会抛 `KeyError/IndexError`，无捕获 | 转义或改用占位符替换 |
| M14 | `provider/embedding.py:44-67` | `ensure_loaded`/`embed` 无锁，并发首次调用重复加载模型（RAG 与 memory 共享实例） | 加锁 |
| M15 | `rag/document_store.py:70-72` | 文本模式 `f.seek(8192)` 可能落多字节 UTF-8 字符中间，产生替换字符甚至误判二进制 | 二进制读 + 解码 |

### coding / safety

| # | 位置 | 问题 | 修复 |
|---|---|---|---|
| M16 | `tools/bash.py:24-56` | BashGuard 黑名单可绕过/漏判：`(rm -rf /)`、`rm -rf /;`、`rm -rf /*`、`rm -fr /`、`rm -rf $HOME` → **SUSPICIOUS/可授权**；`dd of=/dev/sda`、`echo > /etc/passwd`、`chmod -R 000 /`、`truncate -s 0 /etc/passwd`、`crontab -r` 等 → **SAFE** | 改结构化解析（shlex+命令名归一化），或文档明确其"启发式可绕过"定位 |
| M17 | `sandbox.py:443-457` | 超时后 `await proc.wait()` **无超时**（SIGKILL 卡 D 状态即永久阻塞）；且只 kill 单进程、未杀进程组，孙进程残留 | 补超时 + `killpg` |
| M18 | `lightpanda.py:490-508` | 就绪探测失败时 `_kill_shared_proc()` 读的是全局 `_shared_proc`（此刻仍为 None）→ **新 spawn 的 serve 进程泄漏**（正是文件头说要避免的 orphan 堆积） | 先登记再探测（对照 `LightpandaEngine.start:139-153`） |
| M19 | `sandbox.py:348` | `NamespaceSandbox` 把 `cwd` 直接插进 shell 命令 `f"cd {self.cwd}; {command}"`，未引用 → cwd 含空格/`;`/反引号即注入 | `shlex.quote` |
| M20 | `agents.py:448-518` | DAG 一轮 ready 任务全量 `asyncio.gather`，**并发无上限**、`_run_task` **未传 `signal`** → `/stop` 对 DAG 子 agent 失效；模型传 100 任务即 100 并发 Agent | 加信号量上限 + 传 signal |
| M21 | `agents.py:337-339` + `subagent.py:51-54` + `session.py:336` | 子 agent `tasks[].id` 未校验字符集，直接拼 `session_id` = `sub-{pid}-{tid}`，`os.path.join(dir, f"{sid}.jsonl")` → `..` **路径穿越**，可越目录写任意 `.jsonl`、覆盖其它 session | 校验 id 字符集 |
| M22 | `coding_agent.py:158-162,360` | `_active_chat`/`_last_sender` 是**全局单值**，不同会话并发时串台：A 的 `send_file` 把文件发到 B；`/whoami` 报错别人的 open_id（可能被误填进 shell 白名单） | 随会话/请求传递 |
| M23 | `coding_agent.py:884-897` + `write.py:86-100` | 写逃逸授权是**死代码**：`args["_unsandboxed_retry"]=True` 全仓无读取方，授权后 write 仍被自身 cwd 检查拒绝 → 授权弹窗无效且误导 | 打通或删除该授权路径 |
| M24 | `write.py:71,88-89,153-156` | `write` 自身 cwd 检查被绝对路径 `..` 前缀绕过：`/proj/../proj2/f` 的 `startswith("/proj/")` 为 True → 放行（`_resolve_path` 对绝对路径不 normpath） | 先 `normpath`/`realpath` 再比较 |
| M25 | `bg_tasks.py:10-12,78,84,226` | 文档称后台任务"存活到 bot 退出"，但 `DirectSandbox`/`mode=none` 无 `--die-with-parent` 却用 `start_new_session=True` → 成孤儿；**无 shutdown/atexit kill**；`self._tasks` 只增不减（内存缓慢增长）；`tail` 的 `max_bytes` 非数字入参 `ValueError` 未捕获 | 补 shutdown + 复用判断 + 类型防护 |
| M26 | `attach_image.py:63-75,160-164,251-265` | 无"仅限附件目录"约束，manifest 的 `path` 被信任 → 伪造 manifest 可让**任意可读文件**（如 `~/.ssh/id_rsa`）被 base64 内联进模型请求；`_evict` 还会 `os.remove` 条目文件 | 校验 path 在 `attachments_dir` 内 |
| M27 | `lightpanda.py:785-806` / `read.py:66` / `hash_read.py:107` / `image_analyze.py:156` | 无大小上限整文件/整页入内存（HTTP 快路径 `resp.text` 读完才截断）；沙箱明确取消 RLIMIT 且无 cgroup 内存上限 → 15.5G OOM 事故根因未全面封堵 | 加 max_bytes / 流式读取 |
| M28 | `coding_agent.py:883-886` | 授权判定基准 `os.path.abspath(file_path)`（进程 cwd）与 `WriteTool`（agent cwd）不一致 → 进程 cwd ≠ agent cwd 时漏弹/多弹授权 | 统一基准 |

### memory

| # | 位置 | 问题 | 修复 |
|---|---|---|---|
| M29 | `retriever.py:231` | BM25 融合被 `min_similarity` **绝对闸提前砍掉**：低余弦候选在用到 BM25 前就 `continue`，与"补精确术语盲区"的设计矛盾（已复现 ID/错误码被丢弃）。BM25 语料又仅限 embedding top-33 | BM25 命中豁免绝对闸 |
| M30 | `retriever.py:403-417` | token 预算对 `summary` 级条目**完全失效**：超预算时 `if detail_level=="full"` 为假 → 照样 append；`len//3` 对中文严重低估 | 修裁剪分支 + 用真实分词 |
| M31 | `consolidator.py:201,234` | 衰减/强化只覆盖前 **1000** 条（`list_all` 默认 `LIMIT 1000`，按 importance 排序）→ 尾部记忆永不衰减，而 `cleanup` 走 SQL 会误删 | 分页处理全量 |
| M32 | `extractor.py:143,153`、`consolidator.py:420-430` | LLM 输出 `content:null`/`importance:null`/item 为 `str` 时崩溃：extractor 侧整批新记忆丢弃，consolidator 侧整轮 dream 中止；两处防御不一致（`_apply_curation` 有 `or 0.6`，`_apply_consolidation` 没有） | 统一类型防御 |
| M33 | `consolidator.py:296-298` | `_parse_json` 失败返回 `{}` 仍**无条件衰减全部 episode**（且触发 H7 复活） | 解析失败即 return |
| M34 | `store.py:248-250` | embedding 维度不匹配时 `np.dot` 抛 `ValueError`，无校验、无降级 → 冒到 recall 的 except → 触发 H6 崩溃 | 维度校验 + 降级关键词 |
| M35 | `coding_agent.py:476,509`、`commands.py:200` | fire-and-forget 任务不持引用（可能被 GC）；dream 无互斥锁，`/dream` 与周期任务并发跑衰减/整理，重复衰减互相覆盖 | 存引用 + 加锁 |

### channel / 入口 / 配置

| # | 位置 | 问题 | 修复 |
|---|---|---|---|
| M36 | `config.py:291-293,341,392-434` | 数值字段 `int(...)` 无保护（`context_window="64k"` → `ValueError` 打挂进程）；`reasoning` 无 bool 归一（`"false"` 当 True）；`enabled` 系列 `== "1"`/`!= "0"`/裸透传 三种写法不一致 | 加安全转换 + 统一 bool |
| M37 | `injection.py:60-66,167` | 记忆/清单内容**不转义**逐行拼进 `<remembered-context>`，`catalogs` 更是裸拼 → 记忆内容含 `</remembered-context>` 即可**穿出**注入伪指令；群聊任一成员的话会被提取成记忆注入他人上下文（提示注入链） | 转义包裹 |
| M38 | `model_registry.py:140-148,207-223` | `set_active`/`_save` 写盘失败静默 `pass/return`，而调用方仍回报"✅ 已切换并持久化，无需重启"→ 重启回退旧值 | 失败上报 |
| M39 | `main.py:96-117` | 图片下载 `gather` 无 `return_exceptions`；`open().write()`/`store.register`/`xxhash` 在 try 外 → 单张图失败**打断整条消息**（文本也不进 agent） | 隔离每张图 |
| M40 | `feishu.py:315-323` | 最终回复无长度上限/无分片（思考卡片有 22k 预算），超长答案 `markdown`+`text` 双失败 → 用户端"完全没回复"；`_split_for_streaming`（:773-787）是死代码 | 复用分片 |
| M41 | `coding_agent.py:576-600` | `apply_model_profile` 无锁：并发 `/model` 点击时 `self._model` 与 `llm.api_key` 分两步落地 → 可能形成"A 的 base_url + B 的 api_key"（正是注释要防的 403/串 key） | 加 `asyncio.Lock` 原子更新 |
| M42 | `config.py:26-32` | `${VAR}` 缺失时静默变空串，无变量名提示 | 缺失时 warning |
| M43 | `config.py:46-53` | TOML 语法错误被吞、照常启动（可能既没 key 又连错端点） | 关键配置 fail-fast |

---

## 三、低严重问题（摘要）

- `stream.py:200` 参数 `reasoning` 被就地改写为 int（变量遮蔽隐患）；`:213-216` 重试未重置 `_first_token_at` → TTFT 失真
- `agent/types.py:45` `AgentTool.parameters` 用 `field(default_factory=...)` 但类非 `@dataclass` → 类属性是 `Field` 对象（子类各自覆盖才没爆）
- `agent_loop.py:371` `case "toolcall_delta"` 是死分支（`stream_simple` 从不 yield 该类型）
- `agent_harness.py:323-326` `before_agent_start` 钩子未 await；其 `messages` 修改在压缩重建后丢弃
- `session.py:250-294` `get_path_to_root` 忽略 `leaf_id` 做全量 DFS；`:348` `list_sessions` 按随机文件名而非 `created_at` 排序
- `bg` 模式丢失用户 PATH（`bash.py:173` vs `:203`）；`shellpath.py:62` PATH 未引号转义
- `coding_agent.py:476/509`、`feishu.py:199/495/750`、`commands.py:200` 大量 fire-and-forget 任务未持有/未 join
- `main.py:352-357` `stop_event` 是死代码（从未被消费）
- `feishu.py:764-766` 授权按钮后点覆盖先点；`:655` `rid` 仅 32bit 截断 UUID；授权/model 卡片**不校验点击者身份**（群聊任意成员可批准逃逸授权）
- `feishu.py:213-215` 注释描述的 `_on_p2_card_action_trigger` 机制不存在（误导维护者）
- `injection.py:98-101` `ContextState.apply` 对畸形快照 `dict(v)` 抛异常（每轮必经路径）
- `logging.py:35-43` `_NameShortener` 原地改 `record.name`（多 handler 会二次截断）
- `onboard.py:262-276` 拼接 TOML 未转义 + `open(...,"w")` 后 `chmod 600` 存在密钥可读窗口
- `main.py:346-348`、`commands.py:52-54` 异常原文回吐聊天
- `memory`: `retriever.py:354-355` 未来时间戳致除零/放大；`subsystem.py:69,263` `_turn_counter` 无界且 `/new` 不重置；`commands.py:148-154` 非法类型静默返回全部；`store.py:346` `len(blob)//4` 非对齐静默截断；`subsystem.py:484-492` `clear_all` 绕过 persona 保护且无二次确认；`consolidator.py:198` `halflife_days=0` 除零；`prompts.py:118/191` 重复定义
- `hashline`: `apply.py:269` 幻影行可作合法锚点；`format.py:122` `compute_changed_line_range` 可返回 first>last；`apply.py:798` `-(s.boundary or -1)` falsy 陷阱；`hash_read.py:237` BOM 误报为非 UTF-8；`config.py:12` 默认哈希长度 2 字符（256 值）碰撞偏高；`hash_read.py:20` `DEFAULT_MAX_LINES` 仅是文案未生效；`parse.py:441` `file_path` 别名对 `hash_edit` 不可达；`edits` 与顶层 `oldText/newText` 同给时顶层被静默丢弃
- `lightpanda.py:415-424` `configure_pool` 仅首个 server 生效但每次构造都调用

---

## 四、值得肯定（不是客套）

- **provider 层重试/可观测性扎实**：`AssistantMessage` 带 `duration_ms/ttft_ms/retry_count/usage(cached/reasoning)`，同时兼容两种网关 usage 布局；toolcall 参数截断显式化（`args_error` 让模型重试）而非静默空参执行。
- **有界缓存意识**：`_base64_cache`（64MB）、`_token_memo`（4MB）、`_client_cache_max=8` 都做了淘汰——只是 H1 用错了容器类型。
- **bwrap 参数模型正确**：`--ro-bind / /` + 选择性写区 + 对 protected 文件附加 `--ro-bind`，并预 touch 不存在的 registry 防"RO bind 只影子化已存在文件"的绕过；能力探测指纹化（采纳 Codex 的 user-namespace 失败指纹表），非指纹失败时**保留隔离而非误降级**。
- **飞书接入的工程细节**：同 chat 串行 + `/stop` 显式旁路锁；思考卡片熔断 + 体积预算；`_silence_sdk_bg_tasks` 精准点名 SDK 后台任务且不取消根 `_select`；重启走根协程 `SystemExit(42)` 避开 asyncio 吞异常。
- **hashline 设计细腻**：3 行窗口哈希 + `text_hint` 反碰撞 + 模糊 Unicode 宽容；快照有硬上限（8 路径×4 版本 + 32Mi）；原子写入 + 异常清理临时文件；不允许清空非空文件。
- **记忆子系统**：SQL 全参数绑定（无注入面）；余弦除零已防；纯 numpy KMeans 边界处理得当；`recall_eval` 用强引用防 GC；"增量注入 op 而非改写历史"思路正确。
- **前缀缓存纪律**：system prompt/tools 数组刻意字节稳定，易变上下文走结构化独立条目（`injection.py` op→fold→render）。方向对，M9/M37 是实现细节掉链子。

---

## 五、修复优先级（建议）

**P0 — 正确性/安全，尽快**
1. **H1** `_client_cache` 改 `OrderedDict`（一行，必崩）
2. **H6** `recall` 错误路径返回 `[]`（一行，禁用记忆即崩）
3. **H2 + H3** 文件工具 cwd/受保护约束（`hash_edit`/`read`/`hash_read`）+ `realpath` 归一（沙箱逃逸链）
4. **H7 + H8** 做梦删除顺序 + 软删地板（记忆数据正确性）
5. **H5** 会话落盘持引用 + 加锁（丢消息/破坏会话树）
6. **H9 + H10** 授权卡超时 return + `try/finally`（用户可见 bug + 泄漏）
7. **H11** 补 SIGTERM
8. **H12–H14** hashline 编码/权限/尾换行（静默文件损坏）

**P1 — 健壮性/安全**
9. **H4** `web_fetch` URL 校验（SSRF）
10. **M16/M17/M18/M27** BashGuard、进程组、Lightpanda 泄漏、内存上限
11. **M22/M23/M24/M41** 会话串台、死代码授权、write `..` 绕过、模型切换竞态
12. **M29–M35** 记忆召回质量与后台任务
13. **M36–M43** 配置/注入转义/持久化失败上报

**P2 — 其余中等项与低严重项**

**建议补测试**：现有 394 个用例几乎全是 happy path（`test_hashline_tooling.py` 9 passed 却不含 H12–H14）。至少为 H1/H2/H6/H7/H9/H12/H13 加回归测试——它们都是"一次输入就触发、且当前无覆盖"的类型。
