# ConnectClaw 架构文档

> 40 个 Python 源文件 | Python 3.14 + asyncio | DeepSeek + 飞书 Lark | CardKit 流式输出

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
│  工具 · 动态工具 · task子agent · 沙箱 · 记忆  │
├────────────────────────────────────────────┤
│          Harness 层 (agent/harness/)        │
│   编排器 · 会话持久化 · 上下文压缩 · RAG      │
├────────────────────────────────────────────┤
│           Agent 层 (agent/)                 │
│       Agent 类 · AgentLoop 双循环引擎        │
├────────────────────────────────────────────┤
│          Provider 层 (provider/)            │
│    DeepSeek API · Embedding · Rerank        │
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
│   │   ├── bash.py            # Shell 执行 (BashGuard + 三层沙箱)
│   │   ├── web_search.py      # glyph 浏览器，Bing 引擎，免费
│   │   ├── image_analyze.py   # 子agent: Mimo 视觉分析
│   │   ├── task.py            # DAG 并行子 agent (asyncio.gather 全部完成后聚合)
│   │   └── dynamic.py         # 动态工具加载 (~/.connectclaw/tools/*.tool.json)
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
│   ├── extractor.py           # 对话后自动提取记忆 (LLM，节流)
│   ├── retriever.py           # 分级检索 (近期清晰 / 久远模糊)
│   ├── consolidator.py        # "做梦" 整合 (衰减 / 合并 / 遗忘)
│   ├── prompts.py             # 提取 / 整合 prompt 模板
│   └── subsystem.py           # 总装 (MemoryConfig + MemorySubsystem)
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

    CA->>CA: _refresh_tools() (base + task + dynamic)
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
            alt task: DAG 子 agent
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
| `write` | 写入文件，原子操作 (tmp → rename) | 已存在文件必须先 read |
| `bash` | 执行 shell 命令 | BashGuard 两级 + 三层沙箱 |
| `web_search` | glyph 浏览器 + Bing 引擎搜索 | 免费，无需 API key |
| `web_fetch`  | glyph 浏览器抓取 URL 纯文本 | 免费，无需 API key |
| `image_analyze` | Mimo 视觉模型分析图片 | API key 可选 |

### 6.2 编排工具

**`task`** — DAG 并行子 agent：

```json
task(tasks=[
  {"prompt": "检查类型错误", "tools": ["read", "bash"]},
  {"prompt": "运行测试", "tools": ["read", "bash"]},
])
→ 3 个子 agent 并发执行 → 全部完成 → 聚合结果
```

子 agent 使用受限工具集，通过 `asyncio.gather` 并行执行。

### 6.3 动态工具

Agent 可以**自行创建工具**，写入 `~/.connectclaw/tools/*.tool.json`：

```json
{
  "name": "check_types",
  "description": "检查 Python 类型错误",
  "command": "python3 -m pyright {path}",
  "parameters": {
    "path": {"type": "string", "description": "要检查的路径"}
  }
}
```

每轮对话自动扫描目录（`load_dynamic_tools()`），新工具立即可用，无需重启。

### 6.4 工具刷新流程

```
每次 handle_message():
  _refresh_tools()
    → base: [read, write, bash, web_search, web_fetch, image_analyze]
    → task: 更新 _all_tools 引用 (子 agent 可用所有工具)
    → dynamic: 扫描 ~/.connectclaw/tools/*.tool.json
    → 返回完整列表
    → harness.set_tools(最新列表)
```

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

两级检测 + 沙箱：

```
BashGuard.check(command):
  DANGEROUS:    rm -rf /, mkfs, dd to /dev, fork bomb, shutdown, iptables...
                → 直接拒绝，不执行

  SUSPICIOUS:   rm, mv, chmod, chown, eval, curl pipe to shell...
                → 飞书授权卡片 (approve/deny)

  SAFE:         → 进入沙箱执行
```

## 九、上下文压缩

与 pi-mono 保持一致：

- **Token 估算**: provider usage 作为锚点 + trailing 消息估算（比纯 chars/4 精确得多）
- **合法切分点**: user 消息、branch summary、compaction（不切 toolResult 和 mid-turn）
- **Split-turn**: 超过预算的单轮拆分为前缀摘要 + 保留后缀
- **增量摘要**: `UPDATE_SUMMARIZATION_PROMPT` 合并进已有摘要
- **文件追踪**: `_extract_file_ops()` 记录 readFiles / modifiedFiles 注入摘要
- **结构化格式**: Goal / Progress (Done / In Progress) / Key Decisions / Next Steps / Critical Context
- **完整流水线**: `prepare_compaction()` → `compact()` → 持久化 `CompactionEntry`

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
  [deepseek]     api_key / base_url / model_id
  [feishu]       app_id / app_secret
  [mimo]         api_key / base_url / model_id
  [agent]        cwd / thinking_level
  [session]      dir
  [rag]          enabled / docs_dir / db_path / top_k / top_n
  [web_search]   glyph_bin / max_chars / timeout
  [compaction]   enabled / reserve_tokens / keep_recent_tokens
  [memory]       enabled / db_path / extract_interval_turns / use_embeddings / dream_interval_hours
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
recall  每轮对话前：query → (embedding | 关键词) → 打分 → 分级细节 → 注入 user message
learn   每轮对话后：后台 asyncio.create_task 提取，每 N 轮节流一次（省 API 成本）
dream   定时 / 手动：衰减 → 强化 → 情景→语义整合 → 合并 → 清理
```

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

### 检索：模糊记忆

距离近的清晰、远的模糊。近期（<7 天）且重要/强 → 展开 `detail`（full）；否则只给一行摘要（summary）。通过门槛后按综合分排序 = 相似度×0.5 + 时效×0.25 + 重要×0.15 + 强度×0.1，并受 token 预算约束（超预算时 full 降级 summary）。

### 查看与管理（飞书斜杠命令）

| 命令 | 作用 |
|------|------|
| `/memory` | 统计概览 + 最重要的记忆 |
| `/memory list [类型]` | 列出记忆（可按 semantic/episodic/procedural 过滤）|
| `/memory <关键词>` | 关键词搜索记忆 |
| `/dream` | 立即触发整合（做梦）|
| `/forget` | 清空所有记忆 |

无需 sqlite / 文本工具翻 db 和 jsonl，直接在飞书对话里查看。

## 十五、技术栈

```
Python 3.14 + asyncio · uv 包管理
DeepSeek (openai SDK) · lark-oapi (WebSocket + HTTP)
LanceDB · BGE-M3 · BGE-Reranker-v2-m3 (RAG, 可选)
SQLite · numpy (分层记忆)
bubblewrap · unshare (沙箱)
tiktoken · aiofiles · questionary · qrcode
```
