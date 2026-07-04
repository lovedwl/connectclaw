# ConnectClaw 架构文档

> 32 个 Python 源文件 | Python 3.14 + asyncio | DeepSeek + 飞书 Lark | CardKit 流式输出

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
│   内置工具 · 动态工具 · task子agent · 沙箱   │
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
│   │   ├── web_search.py      # 子agent: Bing 搜索
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
    CA->>CA: RAG.search(text) → rag_context
    CA->>Harness: handle_message → prompt(text)

    Harness->>Harness: build system prompt (模板 + RAG)
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
| `web_search` | Bing 搜索（子 agent） | API key 可选，无 key 返回 placeholder |
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
    → base: [read, write, bash, web_search, image_analyze]
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

- **系统 prompt**: `~/.connectclaw/prompts/system.md`，~480 chars，仅身份 + 环境 + 规则
- **工具描述**: 由 LLM function-calling schema 提供，不在 prompt 中重复
- **Skills**: 以 XML 块注入 `<available_skills><skill><name>...</name></skill></available_skills>`
- **RAG 上下文**: 每个 turn 从检索器获取最新文档，注入 prompt 末尾
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
  [web_search]   bing_api_key
  [compaction]   enabled / reserve_tokens / keep_recent_tokens
```

## 十四、技术栈

```
Python 3.14 + asyncio · uv 包管理
DeepSeek (openai SDK) · lark-oapi (WebSocket + HTTP)
LanceDB · BGE-M3 · BGE-Reranker-v2-m3 (RAG, 可选)
bubblewrap · unshare (沙箱)
tiktoken · aiofiles · questionary · qrcode
```
