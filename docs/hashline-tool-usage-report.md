# hashline 工具（hash_read / hash_edit）使用情况调研

- 调研时间：2026-10-10
- 数据源：`~/.connectclaw/sessions/*.jsonl`（24 个 session，全量解析 toolCall/toolResult）
- 方法：脚本统计调用量、报错、重试模式、强制重读链路；关键结论均回源码验证

## 1. 总览

| 工具 | 调用 | 报错 | 报错率 | 备注 |
|---|---|---|---|---|
| hash_read | 161 | 0 | 0% | 126 次分页读 / 35 次整读 |
| hash_edit | 79 | 5 | 6.3% | **38 次成功结果 "Anchors omitted"（48%）** |
| read（旧代） | 89 | 0 | 0% | hashline 时代仍在大量混用 |
| write（旧代） | 8 | 1 | 12.5% | 唯一 1 次为历史 bug（已修） |

hashline 工具实际只在 5 个 session 里用过（都是代码编辑密集会话）。
全部工具结果 ~290 条里 **is_error=true 的只有 1 条**（write 崩溃那次）——见 F3。

## 2. 发现（按影响排序）

### F1「Anchors omitted」占成功编辑的 48% —— 多行编辑强制重读
- 38/79 次 hash_edit 成功结果返回 `Anchors omitted; use hash_read for subsequent edits`。
- 根因（`connectclaw/hashline/format.py:13`）：`ANCHOR_MAX_OUTPUT_LINES = 12`。
  `compute_affected_line_range` 要求「改动行 + 两侧各 2 行上下文 ≤ 12 行」，
  即**一次编辑净改动超过约 8 行就省略锚点**；另有 50KB 字节预算兜底（几乎不触发）。
- 实际代价（会话链路实测）：38 次 omitted 中
  - 17 次（45%）紧接着对同一文件又 hash_read 一次（纯浪费：一次往返 + 该区域文本重新进上下文）；
  - 10 次在同文件继续 hash_edit 且没重读——这些是靠「同快照多 edit 一次提交」或
    `replace_text`（无需锚点）才成立的，属于 agent 自己绕过去的；
  - 9 次任务正好结束，无代价。
- 多行替换恰恰是代码编辑的主体形态，这个阈值等于把最常用路径推向了最贵的恢复方式。

### F2 双代工具并存，两套快照互不相通
- 同一会话里 `read`/`write`（LINE:content / 整文件）与 `hash_read`/`hash_edit`（LINE#HASH）同时可用，
  agent 混用：89 次 plain read；7 次 hash_edit 目标文件的「最近一次读」是 plain read。
- 隐患：`write` 的 read-before-write 检查认的是 `read_tool.recently_read`（write.py:84），
  而 **hash_read 不往这个集合里登记**（grep 全库确认）。即 `hash_read → write` 会被
  「写入前必须先用 read 工具读取该文件」拒绝。历史会话里 agent 没踩中，但陷阱存在。
- write 工具描述与 system prompt 里的规则只提旧代 read，未说明哪种读建立哪种快照。

### F3 工具层错误不打 is_error 标记
- hash_edit 的 5 次错误（E_BAD_REF、未知字段 op_note、空 edits、2 次受保护文件）
  全部以 `is_error=false` 的普通文本返回；全库 290 条结果里 is_error=true 仅 1 条。
- 后果：harness 无法按 is_error 统一注入「上一工具调用失败，请修正后重试」之类的提醒
  （实测 bash 非零退出会触发 SYSTEM WARNING，hash_edit 文本错误不会），纠错全靠模型自己读文本。

### F4 E_BAD_REF 报错不易自纠
- 现文案：`无效的行引用 "99#placeholder"：哈希必须正好是 来自 ZPMQVRWSNKTXJBYH 的 2 个字符`。
- 问题：不区分「该行不存在」/「哈希不匹配（文件已变）」/「格式错误」三种情况，
  也不给出该行**当前的真实锚点**，agent 只能盲目重新 hash_read。
- 本次会话中该错误源于模型幻觉占位符（把 "placeholder" 当锚点填了），错误信息若带上
  「第 99 行当前锚点是 99#BT: …」可一步自纠。

### F5 大 payload 并行调用会损坏（低概率高代价）
- 实测两例：
  1. 同一并行块 3 个 hash_edit（各含多行代码 payload），其中一个的 `edits` 数组
     到达工具时为空 → `错误：未提供任何编辑`，同块另两个正常。
  2. 本次调研中，write 工具两次以约 8KB 的 markdown 参数调用，参数整体丢失
     （只收到空的 `file_path`）——与 hash_edit 空 edits 同模式。
- 小参数（<~2KB）从未出现此问题。怀疑长输出被截断/分片时 JSON 结构被破坏，
  值得在 harness 层加「必填参数缺失即整轮回调重试」的保护。

### F6 分页走读是正常行为，不是 bug
- 15 段「连续 3+ 次同文件分页 hash_read」，全部是 agent 主动用小窗口（10~30 行）
  省 token；截断提示 `使用 offset=X 继续` 表达清晰，无需改动。

### F7（历史）write 工具 `'CodingAgent' object has no attribute '_cwd'`
- 09-25 会话出现 1 次，现版 write.py 已有 `_cwd`，确认已修复，仅存档。

## 3. 建议修改方向（按 ROI 排序）

| # | 方向 | 具体做法 | 预期收益 |
|---|---|---|---|
| R1 | 放宽锚点省略阈值 | `ANCHOR_MAX_OUTPUT_LINES` 12 → 40~60（50KB 字节预算已兜底）；或改为「只回改动行的新锚点，不加 2 行上下文」 | 48% 的多行编辑省掉强制重读：每省 1 次 = 1 轮往返 + 一段重复上下文，是全报告里最大的单项浪费 |
| R2 | 打通双代快照 | 最小改动：hash_read 成功后同时 `read_tool.recently_read.add(path)`；或收敛工具只留 hashline 代，write 描述与 system prompt 同步更新 | 消除「hash_read 后 write 被拒」的隐性陷阱与两套心智负担 |
| R3 | 错误统一打标 | hash_edit / hash_read 的所有拒绝与校验失败（含受保护文件）返回 `details.is_error=True` | harness 可统一注入纠错提醒，与 bash 非零退出的待遇对齐 |
| R4 | E_BAD_REF 三分支 + 自纠提示 | 区分 行不存在 / 哈希不匹配 / 格式错误；哈希不匹配时附上「该行当前锚点与内容」 | 把 F4 类错误从「盲目重读整段」变成「一步替换锚点」 |
| R5 | 大调用保护 | 工具入口检测「必填参数缺失/为空」（edits、content、file_path）时，返回带「请拆分或重试」的明确错误；harness 层对此类失败自动重试一次 | 兜住 F5 的静默损坏，避免模型把「空参数」当成自己逻辑错误去绕 |
| R6 | （可选）受保护文件提前提示 | 受保护路径在 hash_read 阶段就标注「只读/不可编辑」，而不是 hash_edit 时才被拒 | 省一次注定失败的编辑调用 |

## 4. 附：hashline 时代 session 分布

| session | hash_edit | hash_read | 说明 |
|---|---|---|---|
| 6d3c7476b767 (09-25, 2.0MB) | 30 | 45 | 重度开发会话 |
| 9df57c903e83 (09-25) | 25 | 73 | 重度开发会话 |
| cba6615054b4 (10-10, 本次) | 14 | 9 | 逃生通道改造 |
| d95542f054b6 (09-25) | 8 | 30 | |
| 991c802736d7 (09-25) | 2 | 4 | |

错误清单（hashline 全部 6 条非成功文本）：
1. `⛔ 受保护文件（逃生配置，agent 不可修改）：config.toml`（正常拦截，is_error=false）
2. `⛔ 受保护文件（逃生配置，agent 不可修改）：models.toml`（同上）
3. `错误：编辑 1 包含未知或不支持的字段：op_note。`（模型幻觉字段，拒绝正确）
4. `错误：未提供任何编辑。`（F5 payload 损坏）
5. `错误：[E_BAD_REF] 无效的行引用 "99#placeholder"…`（模型幻觉锚点，文案见 F4）
6. `write: 'CodingAgent' object has no attribute '_cwd'`（历史 bug，已修）

## 5. 修复状态（2026-10-10 已完成，commit 见 git log）

| # | 修复 | 落地位置 |
|---|---|---|
| R1 | `ANCHOR_MAX_OUTPUT_LINES` 12 → 48（50KiB 字节预算保留为硬上限）；10 行级编辑现在直接带回锚点块，链式编辑免重读 | `hashline/format.py` |
| R2 | `hash_read` 成功后登记进 legacy `read_tool.recently_read`，`hash_read → write` 不再被拒；write 的报错文案同步改为「read 或 hash_read」 | `tools/hash_read.py` + `coding_agent.py` wiring |
| R3 | **根因修复**：`agent_loop` 两处把工具结果硬编码 `is_error=False`，现改为尊重 `details['is_error']`（`_result_is_error`）；hash_read/hash_edit/read/write 全部硬失败路径（不存在/目录/二进制/读取失败/校验失败/OOB/受保护/写入失败）补标 `is_error=True`。E_DUPLICATE_EDIT 与 E_NOOP_LOOP 属「已生效/行为警告」而非失败，刻意不打标，避免 harness 的「请重试」提醒诱发更深的 noop 循环 | `agent/agent_loop.py` + 四个 tool 文件 |
| R4 | E_RANGE_OOB 现附文件真实行数 + 边界行（首/末可见行）的当前锚点，一步自纠；注意行数用 `visible_line_count`（不含 EOF 哨兵行） | `hashline/apply.py` |
| R5 | hash_edit 缺 `path`、空 `edits`，write 缺 `file_path`/`content` 均给出「参数可能在传输中被截断，请重试或拆小」的明确错误 + is_error（不再裸 KeyError） | 三个 tool 文件 |
| R6 | `hash_read` 对受保护文件在输出顶部标注「只读受保护，勿再尝试编辑」，省掉一次注定失败的 hash_edit | `tools/hash_read.py` |

测试：新增 `tests/coding/test_hashline_tooling.py`（9 用例覆盖 R1-R6 + `_result_is_error`），
全量 345 passed；另跑 7 步端到端冒烟（分页读→10行替换→链式编辑→过期锚点→OOB→write 放行→空 edits 提示）全绿。

附带收益：飞书工具卡片的 ✅/❌ 状态现在会如实反映工具层失败（之前失败也显示 ✅）。
