# ConnectClaw 架构文档

> 59 个 Python 源文件 | Python 3.14 + asyncio | DeepSeek + 飞书 Lark | CardKit 流式输出

## 〇、变更历史

- 新增 **hashline 哈希锚定编辑协议**（`hashline/` 12 文件 + `tools/hash_read.py` / `hash_edit.py`）
- 新增 **记忆混合检索**（`memory/bm25.py` BM25 + BGE-M3 语义，召回与使用分离 `confirm_usage`）
- 新增 **记忆去重**（`memory/clustering.py` 纯 numpy KMeans + 簇内合并）
- Agent 自带记忆管理工具 `tools/memory.py`（search / soft-forget）
- 配置去 vendor：`[llm]/[vision]` 替代 `[deepseek]/[mimo]`（§十三）
- 压缩管道改为 entry-based + `before_compact` hook（§九）
- 压缩预算加固：tiktoken 真实分词（CJK 感知回退）、预算贴合 keep window、benefit guard、摘要/文件列表上限（§九）
- 记忆检索加入新鲜度加成 + 自动 importance 提升 + `force_learn`（§十四）

## 一、项目定位

ConnectClaw 是一个**通用 AI 助手**，通过飞书 IM 交互。代码能力只是它的众多能力之一。

核心设计理念：**主 agent 所见一切皆为工具**，且可以自行编写新工具、生成子 agent、创建脚本。

## 二、分层架构

```
┌────────────────────────────────────────────┐
│           Channel 层 (channel/)             │
│      飞书 WebSocket + 交互卡片授权           │
├────────────────────────────────────────────┤
│         Coding Agent 层 (coding/)           │
│  工具 · agents元工具(子agent编队) · 沙箱 · 记忆  │
├────────────────────────────────────────────┤
│          Harness 层 (agent/harness/)        │
│   编排器 · 会话持久化 · 上下文压缩 · RAG      │
├────────────────────────────────────────────┤
│           Agent 层 (agent/)                 │
│       Agent 类 · AgentLoop 双循环引擎        │
├────────────────────────────────────────────┤
│          Provider 层 (provider/)            │
│    DeepSeek API · Embedding · Rerank        │
├────────────────────────────────────────────┤
│          Hashline 层 (hashline/)            │
│   哈希锚定编辑协议 (hash_read / hash_edit)  │
└────────────────────────────────────────────┘
```

## 三、目录结构

```
connectclaw/
├── main.py                    # CLI: connectclaw / connectclaw onboard
├── config.py                  # TOML 配置管理 (env var > config.toml > 默认值)
├── commands.py                # 斜杠命令 (/memory /dream /forget /new /stop)
├── logging.py                 # 统一日志系统 (DEBUG/INFO/WARNING/ERROR)
├── onboard.py                 # 交互式向导 (lark_oapi.aregister_app 扫码创建)
│
├── provider/                  # LLM API 抽象层
│   ├── types.py               # Message / Model / Context / StreamEvent + normalize_message()
│   ├── deepseek.py            # DeepSeekProvider (OpenAI SDK)
│   ├── stream.py              # stream_simple() 异步流式生成器
│   ├── embedding.py           # BGE-M3 嵌入 (懒加载)
│   └── rerank.py              # BGE-Reranker-v2-m3 重排序 (懒加载)
│
├── hashline/                  # 哈希锚定编辑协议 (自 pi-hashline-edit, MIT)
│   ├── hash.py                # xxHash32 逐行内容哈希
│   ├── parse.py               # 锚点解析 / 编辑请求规范化 / 校验
│   ├── apply.py               # 编辑执行引擎 (锚点验证 + 区间解析 + 组装)
│   ├── snapshot.py            # 读快照 LRU 存储 (过期锚点恢复)
│   ├── guard.py               # 幂等去重 / noop loop guard
│   ├── diff_util.py           # diff 生成 / 行尾处理
│   ├── format.py              # 哈希锚定区域渲染
│   ├── merge.py               # 三方合并 (过期锚点恢复)
│   └── config.py              # 轻量配置 (env 驱动)
│
├── agent/                     # Agent 框架层
│   ├── types.py               # AgentMessage / AgentTool / AgentEvent / AgentState
│   ├── agent.py               # Agent 类 (状态管理 + 事件总线 + steer/follow-up 队列)
│   ├── agent_loop.py          # 双循环引擎 (内层 tool-call/steering + 外层 follow-up)
│   │
│   └── harness/               # Harness 编排层
│       ├── agent_harness.py   # 高级编排器 (会话 · 压缩 · 钩子 · Agent 复用)
│       ├── session.py         # JSONL 树形会话持久化
│       ├── compaction.py      # 上下文压缩 (pi-mono parity: usage锚点估算 + split-turn)
│       ├── messages.py        # AgentMessage → LLM Message 转换
│       ├── prompt_builder.py  # 轻量 system prompt (模板 + skills XML + RAG)
│       ├── prompts/system.md  # 可编辑的 prompt 模板文件
│       └── rag/               # RAG 子系统 (可选，懒加载)
│           ├── document_store.py   # 文档摄入 + 分块 (~500 token/块, ~50 token 重叠)
│           ├── embedding_store.py  # LanceDB 向量存储
│           ├── retriever.py        # embed → search → rerank 流水线
│           └── subsystem.py        # RAG 总装 (RAGConfig + RAGSubsystem)
│
├── coding/                    # 应用层
│   ├── coding_agent.py        # 组装 AgentHarness + 工具 + RAG + 安全
│   │
│   ├── tools/                 # 工具集 (一切皆为工具)
│   │   ├── read.py            # 文件读取 (带行号 + recently_read 记录)
│   │   ├── write.py           # 文件写入 (已存在必须先 read + 原子写入)
│   │   ├── hash_read.py       # 哈希锚定读 (带 LINE#HASH 锚点, hash_edit 的唯一寻址方式)
│   │   ├── hash_edit.py       # 哈希锚定改 (replace/append/prepend/replace_text, 读快照校验)
│   │   ├── bash.py            # Shell 执行 (BashGuard 三级 + 三层沙箱)
│   │   ├── web_search.py      # Lightpanda 无头浏览器，Bing 引擎，免费
│   │   ├── image_analyze.py   # 子agent: Mimo 视觉分析
│   │   ├── memory.py          # agent 可用记忆工具 (search / soften 软遗忘, persona 受保护)
│   │   ├── agents.py          # agents 元工具 (list/describe/run/create) — 子 agent 编队 + DAG
│   │   ├── named_agents.py    # 命名 agent 加载 (~/.connectclaw/agents/*.md)
│   │   ├── subagent.py        # 子 agent 执行引擎
│   │   └── lightpanda.py      # Lightpanda CDP 引擎 (web_search/web_fetch 底层)
│   │
│   └── safety/
│       └── sandbox.py         # 三层沙箱 (bwrap → unshare → rlimit)
│
├── channel/                   # IM 接入层
│   ├── base.py                # Channel 抽象接口
│   └── feishu.py              # 飞书实现 (lark_oapi.channel.FeishuChannel + CardKit 流式 + 卡片授权)
│
├── memory/                    # 分层记忆子系统 (可选，SQLite 单文件，无感)
│   ├── types.py               # MemoryEntry / MemoryType (semantic/episodic/procedural)
│   ├── store.py               # SQLite 存储 + numpy 余弦相似度检索
│   ├── retriever.py           # 混合检索 (embedding+BM25 融合) + 分级细节 + confirm_usage
│   ├── bm25.py                # BM25 关键词检索 (补 embedding 在术语/路径/错误码上的盲区)
│   ├── clustering.py          # 纯 numpy KMeans (余弦距离, 定种子, 无 sklearn) — 做梦去重
│   ├── extractor.py           # 对话后自动提取记忆 (LLM，节流，带已有记忆去重)
│   ├── consolidator.py        # "做梦" 整合 (衰减 / 增强 / 聚类合并 / 情景→语义 / 清理)
│   ├── prompts.py             # 提取 / 整合 prompt 模板
│   └── subsystem.py           # 总装 (MemoryConfig + MemorySubsystem + force_learn)
│
└── utils/                     # 预留
```

## 四、核心数据流

```mermaid
sequenceDiagram
    participant User as 飞书用户
    participant Feishu as feishu.py (ws.Client)
    participant CA as CodingAgent
    participant Harness as AgentHarness
    participant Agent as Agent
    participant Loop as AgentLoop
    participant Stream as stream_simple
    participant LLM as DeepSeek API

    User->>Feishu: 发消息
    Feishu->>Feishu: SDK Channel 接收 → InboundMessage
    Feishu->>Feishu: msg.chat_id / msg.content_text
    Feishu->>CA: on_message(chat_id, text)

    CA->>CA: _refresh_tools() (base + agents 元工具)
    CA->>CA: RAG.search(text) + memory.recall(text) → 动态上下文
    CA->>CA: 拼到 user message 前 (system prompt 稳定 → 前缀缓存命中)
    CA->>Harness: handle_message → prompt(memory + rag + text)

    Harness->>Harness: build system prompt (仅模板，字节稳定不变)
    Harness->>Harness: check compaction (should_compact?)
    Harness->>Agent: agent.prompt(user_message)

    Agent->>Loop: run_agent_loop(messages, context)

    loop 内层: tool calls + steering
        Loop->>Stream: stream_simple(model, context)
        Stream->>LLM: POST /v1/chat/completions (SSE)
        LLM-->>Stream: SSE 事件流
        Stream-->>Loop: text_delta / thinking_delta / toolcall_delta / done

        alt tool calls
            Loop->>Loop: 并行执行工具 (asyncio.gather)
            alt bash: SUSPICIOUS / allow_network / unsandboxed
                CA->>Feishu: 飞书授权卡片 (approve/deny)
                Feishu-->>User: 交互按钮
                User->>Feishu: 点击
            end
            alt agents(run): DAG 子 agent
                Loop->>Loop: spawn N 个子 agent (并发)
            end
            Loop->>Loop: 添加 tool result → 继续循环
        end
    end

    opt 上下文溢出
        Harness->>Harness: prepare_compaction() → compact() → session 持久化
    end

    Loop-->>Agent: agent_end (消息列表)
    Agent-->>Harness: 最终消息
    Harness->>Harness: session.append_message() (自动持久化)
    Harness-->>CA: AssistantMessage
    CA-->>Feishu: response text
    Feishu->>Feishu: _stream_text() → CardKit 流式卡片
    Feishu-->>User: 逐段流式回复
```

## 五、AgentLoop 双循环引擎

参考 pi-mono，核心采用双循环：

```
outer: while (有 follow-up 消息):
  inner: while (有 tool calls 或 steering 消息):
    1. 注入 pending messages (steering)
    2. stream_simple() → 流式渲染
    3. if tool calls: 并行执行 (asyncio.gather)
    4. 添加 tool result 到上下文
    5. check steering queue
  检查 follow-up queue
```

- **内层循环**：处理工具调用和 steering 消息（用户中途插入的指令）
- **外层循环**：处理 follow-up 消息（对话结束后追加的任务）
- **并行执行**：多个 tool call 并发执行，全部完成后统一返回
- **before_tool_call / after_tool_call**：钩子机制，用于安全拦截和飞书授权

## 六、工具系统

### 6.1 内置工具

| 工具 | 能力 | 安全机制 |
|------|------|---------|
| `read` | 读取文件，带行号，支持 offset/limit | 仅读，记录 recently_read |
| `write` | 写入文件，原子操作 (tmp → rename) | 已存在文件必须先 read；超出 cwd 需飞书卡片授权 |
| `hash_read` | 哈希锚定读，输出 LINE#HASH 锚点 | 记录读快照 (篡改检测基准) |
| `hash_edit` | 哈希锚定改 (replace/append/prepend/replace_text) | 锚点哈希预检 + 幂等去重 + 读快照校验 |
| `bash` | 执行 shell 命令 | BashGuard 三级 + 三层沙箱 |
| `web_search` | Lightpanda 无头浏览器 + Bing 引擎搜索 | 免费，无需 API key |
| `web_fetch`  | Lightpanda 无头浏览器抓取 URL 纯文本 | 免费，无需 API key |
| `image_analyze` | Mimo 视觉模型分析图片 | API key 可选 |
| `memory` | agent 自动作记忆 (search / forget 软遗忘) | persona 级记忆受保护，只能 /forget id 显式删 |

### 6.2 编排工具

**`agents`** — 子 agent 编队的单一入口（list / describe / run / create）：

```json
agents(action="run", tasks=[
  {"prompt": "检查类型错误", "tools": ["read", "bash"]},
  {"prompt": "运行测试", "tools": ["read", "bash"], "depends_on": ["task1"]},
])
→ 独立任务并发、有依赖的拓扑分层 → 前驱产出注入后继 → 聚合结果
```

`action="create"` 写 `~/.connectclaw/agents/*.md` 定义命名 agent（自然语言 system prompt），当轮即可 `run`（每次调用实时 re-scan，不是启动时冻结）。子 agent 用受限工具集，通过 `asyncio.gather` 并行。

### 6.3 工具刷新流程

```
每次 handle_message():
  _refresh_tools()
    → base: [read, write, hash_read, hash_edit, bash, web_search, web_fetch, image_analyze, memory]
    → agents: 元工具 (list/describe/run/create)，单实例常驻
    → 返回完整列表 (可在 config.agent.tools 白名单裁剪，缺省暴露全部)
    → harness.set_tools(最新列表)
```

### 6.4 hashline 哈希锚定编辑协议（关键）

根除 LLM 编辑中的**行号漂移与幻觉修改**。来源为 pi-hashline-edit（MIT），移植到纯 Python。

**核心思想**：不用行号或原始文本锚定，而是**每行算内容哈希**——编辑指令携带 `LINE#HASH` 锚点，编辑器先验证当前行哈希是否匹配，匹配才执行：

```
hash_read  → 输出带 LINE#HASH 锚点的行视图（记录读快照）
hash_edit  → 编辑指令携带锚点 → 预检哈希 → 底向上应用 → 输出新锚点
```

**三层防护（针对行号漂移的完整方案）**：

1. **锚点哈希预检**：编辑前验证每一行哈希，不匹配直接拦截，不碰文件（不像行号方案会错误改到别的行）
2. **幂等去重**：`guard.py` 的 noop loop guard + 重复 payload 检测——同一条编辑不会执行两次，模型在重复轮次上不浪费
3. **读快照校验**：`snapshot.py` 记录 hash_read 时的多版本快照，过期锚点通过三方合并（`merge.py`）恢复，而不是粗暴失败

**与 `write` 的互补（重要）**：hash_edit 处理精准局部修改；当 JSON 编辑失败或需要大段重写时，模型退到 `write` 全文原子写入。两路都通，不让模型在单个编辑通道上死磕。实际使用中 hash_edit 的精确编辑 + write 的全文降级配合良好。

**实现位置**：`hashline/`（hash/parse/apply/snapshot/guard/diff_util/format/merge/config）+ 工具 `hash_read.py` / `hash_edit.py`。

## 七、沙箱系统

三层自动降级：

| 层 | 实现 | 文件隔离 | 网络隔离 | 依赖 |
|---|------|---------|---------|------|
| 1 | BwrapSandbox | `--ro-bind / /` 全局只读 + `--bind $cwd` 项目可写 | `--unshare-net` | bubblewrap |
| 2 | NamespaceSandbox | unshare --mount + tmpfs | `--net` | util-linux |
| 3 | RlimitSandbox | setrlimit (内存/CPU/进程) | 无 | 无 |

沙箱提权（需飞书卡片授权，60s 超时）：

| 参数 | 卡片标题 | 卡片样式 | 效果 |
|------|---------|---------|------|
| 默认 | — | — | 完整沙箱隔离 |
| `allow_network: true` | Network Access Required | info | 跳过网络隔离 |
| `unsandboxed: true` | Sandbox Escape Authorization | danger | 仅 rlimit，无隔离 |

## 八、Bash 安全

三级分类 + 三层沙箱（BashGuard 可配置，`config.toml` 可追加自定义档位）：

```
BashGuard.check(command):
  DANGEROUS:    rm -rf /, mkfs, dd to /dev, fork bomb, shutdown, iptables...
                → 直接拒绝，不执行

  SUSPICIOUS:   rm, mv, chmod, chown, eval, curl pipe to shell...
                → 飞书授权卡片 (approve/deny)

  SAFE:         → 进入沙箱执行
```

> 保持无状态、不做持久白名单——高风险命令每次强制授权，用户随时可反悔。可配置性通过 BashGuard 风险分级清单实现（如把 `docker system prune` 提为 DANGEROUS）而不必 fork 类。

## 九、上下文压缩

与 pi-mono 保持一致：

- **异步管道**: 全异步调用链——`prepare_compaction()`（纯索引，不阻塞）→ `entry_based_compact()`（async LLM 总结）→ `session.append_compaction()`（async I/O 持久化）
- **`first_kept_entry_id`**: 压缩时记录保留的起始消息 ID，`build_session_context()` 在该 ID 之前的消息替换为摘要、之后的消息保留原样。避免旧版暴力清空丢失多层 CompactionSummaryMessage。
- **`before_compact` hook**: 压缩前触发，用于强制记忆提取（`force_learn()`），在原始对话细节被总结覆盖前捕获。钩子支持 async handler。
- **Token 估算**: provider usage 作为锚点 + trailing 消息估算；单条消息用 tiktoken cl100k_base 真实分词（DeepSeek 兼容），离线回退 CJK 感知启发式。原 chars/4 对中文低估 2-4x，曾导致 keep 窗口超预算、压缩每次都不收效、每轮活锁
- **预算贴合**: `prepare_compaction()` 给定 context_window 时，把 keep-recent 窗口收缩到压缩后（保留 + 摘要）落在 `window - reserve` 内（有界 8 次扫描，窗口下限 max(2048, keep/4)）；压缩结果附带 liberated_tokens / post_compact_estimate / fits_budget 供上层预警
- **Benefit guard**: 无可压缩内容（切点保留全部）时返回 None，拒绝追加只增不清的无效摘要
- **摘要上限**: 最终摘要硬 cap `reserve*0.8` token；`<conversation>` 序列化限输入预算，超限丢最旧消息并标注；文件列表截断（≤100 个文件、路径 ≤200 字符）
- **合法切分点**: user 消息、branch summary、compaction（不切 toolResult 和 mid-turn）
- **Split-turn**: 超过预算的单轮拆分为前缀摘要 + 保留后缀
- **增量摘要**: `UPDATE_SUMMARIZATION_PROMPT` 合并进已有摘要
- **文件追踪**: `_extract_file_ops()` 记录 readFiles / modifiedFiles 注入摘要
- **结构化格式**: Goal / Progress (Done / In Progress) / Key Decisions / Next Steps / Critical Context

## 十、会话持久化

JSONL 树形结构，每行一个 JSON 对象：

```jsonl
{"type":"session","version":3,"id":"abc","created_at":"...","cwd":"..."}
{"type":"message","id":"m1","parent_id":null,"message":{"role":"user","content":"hi"}}
{"type":"message","id":"m2","parent_id":"m1","message":{"role":"assistant","content":[...]}}
{"type":"compaction","id":"c1","parent_id":"m10","summary":"...","first_kept_entry_id":"m5","tokens_before":50000}
```

支持：创建、打开、分支（parent_id 链）、压缩（CompactionEntry）、列出全部会话。

## 十一、飞书接入

- **SDK Channel**: `lark_oapi.channel.FeishuChannel` — 封装 WebSocket 连接、消息接收、去重、发送
- **连接**: `sdk.connect_until_ready()` 异步启动（后台线程运行 WS）
- **消息接收**: `sdk.on("message", handler)` → `InboundMessage`（`chat_id`、`content_text`）
- **流式回复**: `sdk.stream(chat_id, {"markdown": producer})` → CardKit 流式卡片，逐段推送
- **普通回复**: `sdk.send(chat_id, {"text": "..."})` → 文本消息
- **卡片发送**: `sdk.send(chat_id, {"card": {...}})` → 交互卡片（授权按钮）
- **交互卡片**: approve/deny 按钮 + callback value 传递 request_id + 60s 超时
- **一键配置**: `connectclaw onboard` → `lark_oapi.aregister_app()` 扫码创建应用

## 十二、Prompt 系统

轻量化设计：

- **系统 prompt**: `~/.connectclaw/prompts/system.md`，~480 chars，仅身份 + 环境 + 规则，**每轮字节稳定**
- **工具描述**: 由 LLM function-calling schema 提供，不在 prompt 中重复
- **Skills**: 以 XML 块注入 `<available_skills><skill><name>...</name></skill></available_skills>`
- **动态上下文 (RAG + 记忆)**: 每轮从检索器获取，注入到 **user message**（而非 system prompt）——保持 system prompt 稳定，让 DeepSeek 前缀缓存持续命中，详见 §十四
- **可编辑**: 用户 `vim ~/.connectclaw/prompts/system.md` 即可自定义

## 十三、配置管理

```
优先级: 环境变量 > config.toml > 默认值

~/.connectclaw/config.toml:
  [llm]          api_key / base_url / model_id        # 替代旧 [deepseek]
  [feishu]       app_id / app_secret
  [vision]       api_key / base_url / model_id        # 替代旧 [mimo]
  [agent]        cwd / thinking_level / tools(白名单) / tool_session_idle_timeout
  [session]      dir
  [rag]          enabled / docs_dir / db_path / top_k / top_n
  [web_search]   max_chars / timeout / pool_size
  [compaction]   enabled / reserve_tokens / keep_recent_tokens
  [memory]       enabled / db_path / extract_min_turns / extract_interval_turns
                 / max_context_tokens / recency_threshold_days / use_embeddings
                 / dream_interval_hours / decay_halflife_days / consolidation_enabled
```

## 十四、分层记忆系统

模仿人类认知的三层记忆，用户**无感**——自动从对话提取、检索、整合，无需说"记住这个"。与 RAG 互补：RAG 是外部知识（文档/代码），记忆是"你和用户之间发生过什么、了解用户什么"。

### 三层记忆

| 类型 | 对应认知 | 例子 |
|------|---------|------|
| 🧠 语义 semantic | 稳定事实 / 偏好 | "用户偏好 Python 类型注解" |
| 📅 情景 episodic | 具体事件 / 决策 | "上周修了 auth 模块循环引用 bug" |
| 🔧 程序 procedural | 工作模式 / 习惯 | "用户习惯先 read 再 write" |

每条记忆带 `importance`（重要性）和 `strength`（强度，随时间衰减），存于 SQLite 单文件 `~/.connectclaw/memory.db`，零外部依赖（numpy 做余弦相似度；无 embedding provider 时退化为关键词检索）。

### 数据流

```
recall   每轮对话前：query → embedding+BM25 混合检索 → 分类型配额 → 打分(相似度+新鲜度+重要+强度) → 分级细节 → 注入 user message
learn    每轮对话后：后台 asyncio.create_task 提取，每 N 轮节流一次（省 API 成本）
force_learn  压缩前触发：跳过 learn 的节流，确保原始对话细节在压缩前被提取（hook 于 before_compact）
confirm  回复产生后：扫描回复内容匹配已召回记忆 → 命中则 touch() + strength/importance 提升（召回≠使用）
dream    定时 / 手动：衰减 → 强化 → 情景→语义整合 → 聚类合并(KMeans) → 清理
```

### 混合检索：BM25 + BGE-M3（关键）

检索不再只靠语义——`memory/bm25.py` 提供关键词信号，补上 embedding 在**精确术语（名字 / 路径 / 错误码）**上的盲区：

- **embedding**：BGE-M3 语义相关，余弦相似度（硬门槛 0.45）
- **BM25**：精确关键词命中，名字 / 路径 / ID 这类语义匹配不到的也能召回
- **融合**：`relevance = max(sim, bm25_norm)` 取两者之强，再叠上新鲜度×时效 + 重要性 + strength
- **类型配额**：semantic/episodic/procedural 各保底名额（2/1/1），避免某类垄断 TopK
- **召回≠使用**：`confirm_usage()` 在回复产生后判断记忆内容是否被引用，命中才 touch + 提升，未被使用的记忆继续衰减

### 梦境去重：KMeans 聚类 + 簇内合并

记忆量一大就爆上下文。`memory/clustering.py` 用**纯 numpy KMeans**（余弦距离、固定 seed、无 sklearn）把语义相近的记忆聚成簇，再 `consolidate_by_clustering()` 簇内合并：

- 保留簇内 strength 最高的第一条，其余 content 折叠进它，删除冗余
- 无 embedding 的记忆退化为 singleton，不强制合并
- 确定性可测，`tests/memory/test_consolidator.py` 覆盖聚类分簇、缺失 embedding、合并删除、跨簇不误并、keeper 加强
- 解决"全量丢给大模型"的问题——先聚类成小簇，簇内再交给 LLM（或确定性合并）

### 缓存友好设计（关键）

DeepSeek / OpenAI-compatible provider 按**请求前缀**缓存：system prompt 改一个 token，整段缓存（含全部历史）从位置 0 失效，每轮按未命中价计费（约 10×）。因此：

- **system prompt 保持字节稳定** —— `CodingAgent.build_system_prompt()` 刻意无参，只含环境 + 规则。
- **所有每轮变化的上下文（记忆 + RAG）注入 user message**，顺序 记忆 → RAG → 用户问题，空块跳过以免污染纯对话。
- **持久化进历史反而最优**：动态上下文成为下一轮的固定前缀，让缓存前缀持续增长命中；"临时注入不持久化"反而会让倒数第二条 user message 分叉、命中更差。
- 历史膨胀由上下文压缩（§九）兜底。

### 向量检索：BGE-M3 + GPU

语义召回用 BGE-M3 embedding（`provider/embedding.py`，RAG 与记忆**共享同一实例**，避免加载两份 ~2.3GB 模型），**自动检测 GPU**（有 CUDA 用显存，否则 CPU）。依赖 `sentence-transformers`（在 `[optional] rag` 组）。

缺依赖时记忆退化为关键词检索——但**中文关键词召回基本失效**（按空格分词，中文整句成一个 token），所以中文场景强烈建议启用 embedding。

首次加载从 HuggingFace 拉 BGE-M3（~2.3GB）；`main.py` 启动时若 `HF_ENDPOINT` 未设会自动指向 `hf-mirror.com`，避免连 huggingface.co 卡住。模型缓存后可 `export HF_HUB_OFFLINE=1` 跳过更新检查。

**相关性硬门槛**：cosine similarity < `min_similarity`（默认 0.45）直接判为不相关丢弃。实测 BGE-M3 中文——相关命中 0.50–0.73，不相关 query 峰值 <0.45。没有这道门槛时，新记忆靠 recency/importance/strength 就能凑够综合分，导致无关 query 也召回记忆。

### 检索：模糊记忆 + 新鲜度保障

距离近的清晰、远的模糊。近期（<7 天）且重要/强 → 展开 `detail`（full）；否则只给一行摘要（summary）。通过门槛后按综合分排序 = 相似度×0.5 + 时效×0.25 + 重要×0.15 + 强度×0.1，并受 token 预算约束（超预算时 full 降级 summary）。

**新鲜度加成**：创建 7 天内的记忆获得最高 +30% 分数加成，随天数线性衰减。解决新记忆 embedding 冷启动无法被召回的死锁问题。

**自动 importance 提升**：`confirm_usage()` 扫描回复内容，被引用的记忆 importance 每次 +0.02（上限 0.8）。累计达 0.7（persona 阈值）后自动变为常备上下文，每轮无条件注入。全程零用户干预。

**压缩前强制提取**：`force_learn()` 跳过 learn 的节流逻辑，注册在 `before_compact` hook 上。上下文压缩前调用，确保原始对话细节被提取到记忆库后再被压缩丢弃。

### 查看与管理（飞书斜杠命令）

| 命令 | 作用 |
|------|------|
| `/memory` | 统计概览 + 最重要的记忆 |
| `/memory list [类型]` | 列出记忆（可按 semantic/episodic/procedural 过滤）|
| `/memory <关键词>` | 关键词搜索记忆 |
| `/dream` | 立即触发整合（做梦）|
| `/forget <关键词>` | 按关键词/类型/ID 选择性删除（persona 受保护）|
| `/new` | 新会话 |
| `/stop` | 中断当前 agent |
| `/restart` | 重启进程（守护进程拉起）|
| `/cmd` | 命令帮助 |

无需 sqlite / 文本工具翻 db 和 jsonl，直接在飞书对话里查看。另外 agent 可通过 `memory` 工具自行搜索 / 软遗忘（soft-forget, strength→0, 下次 dream 回收）记忆，persona 级受保护只能 `/forget id` 显式删。

## 十五、技术栈

```
Python 3.14 + asyncio · uv 包管理
DeepSeek (openai SDK) · lark-oapi + lark-channel-sdk (WebSocket + HTTP)
LanceDB · BGE-M3 · BGE-Reranker-v2-m3 (RAG, 可选)
SQLite · numpy (分层记忆) · xxhash (hashline)
bubblewrap · unshare (沙箱) · lightpanda-py (无头浏览器)
openai · tiktoken · aiofiles · pyyaml · questionary · qrcode · websockets · httpx · torch
```
