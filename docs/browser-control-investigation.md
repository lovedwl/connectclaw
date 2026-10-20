# AI Agent 操控真实浏览器：主流工程做法调研报告（面向 Rust 实现）

> 调研时间：2026-10；调研方式：web_search + web_fetch（部分源站 github.com / openai.com / anthropic 被本环境网络拦截，已用 crates.io API、jsDelivr / GitHub REST API、官方 docs 站点交叉取证）。凡"本环境未能在线核验"的内容一律标注 **待验证**，标注背后是常识级知识，动手前请再核一次官方源。
>
> 一句话结论先说在前面：
> **主流方案不是"装插件"，而是 CDP（Chrome DevTools Protocol）**——用 CDP 附加/启动 Chrome，优先用 **Accessibility tree** 做元素定位、用 **坐标点击**（Input.dispatchMouseEvent）做交互、截图只按需用于视觉确认。想要"保留登录态、操控用户日常浏览器"的主流做法是两条 CDP 路径（复用 profile 重启 / 附加正在运行的 Chrome），扩展桥（如 Browser MCP）是"可选通道"而非默认。Rust 侧 `chromiumoxide` 成熟可用；自己用 `tokio-tungstenite` 写一个极简 CDP 客户端也完全现实。

---

## 0. 总览表：本轮把"谁用什么方案"核了个底

| 项目 | 语言 | 控制通道 | DOM/视觉策略 | 登录态/用户浏览器 | 备注 |
|---|---|---|---|---|---|
| **browser-use**（开源库，11.4 万+ star） | Python | Playwright 驱动本地 Chromium（CDP 底层）；新 **Actor** 直接裸 CDP | 自建 DOM 提取（注入 JS 序列化）→ 文本给 LLM；截图是 31 个动作之一，默认非 VLM | `Browser.from_system_chrome()` 复用真实 profile；Browser Harness 附加你正在用的 Chrome | 主 repo **没有扩展**；控制=CDP |
| **Browser Harness**（browser-use 官方 CLI，1.7 万 star） | Python CLI | **纯 CDP**，一个 daemon 持有到整台 Chrome 的 websocket | **Accessibility tree 优先**（`Accessibility.getFullAXTree`），再 `DOM.getBoxModel` 取坐标 → `click_at_xy`；AX 树缺失才回退 JS；截图按需 | 附加你正在用的 Chrome（`chrome://inspect/#remote-debugging` 一次性授权，无扩展）；或 cloud/远程 | 给我们的设计参考价值最高 |
| **Browser MCP**（browsermcp/mcp） | TS/Node MCP + Chrome 扩展 | **Chrome 扩展桥**：扩展在用户真实浏览器里执行，接本地 MCP server | **accessibility snapshot** + 截图；click/hover/type/drag | 天然保留（就住你的 profile 里） | **"装插件"路线的事实标杆** |
| opencode（sst→anomalyco/opencode，TS，21 万 star） | TS | **无内置浏览器工具**，只有 `webfetch`（纯 HTTP→markdown）和 `websearch` | - | - | 浏览器能力靠 MCP（Browser Harness MCP / browser-mcp / Playwright MCP） |
| aider | Python | **无真实浏览器控制**；`/web <url>` + `aider.scrape` 是纯 HTTP 抓转 markdown | - | - | 达不到"pluggable 交互"诉求 |
| Claude Code / Anthropic SDK | TS/Python | 内置只有 web_search/web_fetch；新 **browser toolset**（31 个浏览器动作）由**驱动程序实现**（browser-use 提供官方驱动），经 CDP 跑 | 动作级（navigate/read_page/left_click/type…） | 本地/cloud/已有 CDP 浏览器三选一 | 动作分类可当我们的 `NavigatorTool` 字典 |
| Anthropic computer use（`computer use tool`） | API | **截图驱动**：截图回传模型 → 模型输出鼠标/键盘动作 → 驱动执行 | **视觉唯一**（整机/VNC，无 DOM） | - | 是"用鼠标键盘点整台机器"的 CUA，不是浏览器专用 |
| OpenAI computer-use（`computer_use_preview`，Operator） | API | 截图驱动 CUA | 视觉 | Operator 跑**隔离 Chromium**，不碰你的浏览器 | 官方文档被 CDN 拦截，细节待验证 |
| chromiumoxide | Rust | **CDP**（tokio + async-tungstenite），官方生成全部 domain 类型 | 高层：find_element(css)/content()/screenshot/evaluate；a11y 无高层包装（需 `Page::execute` 手调，待验证） | `Browser::launch`（可配 user profile）或 `Browser::connect` 附加 | 0.9.1（2026-02），~477 万下载，活跃 |

---

## 1. browser-use（最流行）机制拆解

### 1.1 用 CDP 还是扩展？—— CDP，不用扩展
- 开源库通过 Playwright 驱动本地 Chromium（`browser_use` 依赖安装 Playwright；`Browser(headless=True/False, window_size, proxy, cdp_url=...)` 都走 Playwright/CDP）。[来源：browser-use 官方文档-Browser 配置页](https://docs.browser-use.com/open-source/customize/browser/basics.md)
- 新增的底层 **Actor** 明确写"Not Playwright: built on CDP"。[来源：Actor Basics](https://docs.browser-use.com/open-source/legacy/actor/basics.md)
- 我拉取了主 repo 根目录列表（GitHub REST API），**根下没有 extension 目录**（只有 `browser_use/`、`examples/`、`skills/`、`docker/` 等）。控制通道=CDP 实锤。[来源：browser-use 仓库](https://github.com/browser-use/browser-use)
- 远程/已有浏览器用 `Browser(cdp_url="http://remote-server:9222")` 直接连。[来源：Remote Browser 文档](https://docs.browser-use.com/open-source/customize/browser/remote.md)

### 1.2 DOM 如何提取 & 怎么点击输入
- 给 LLM 的主导输入是**文本化的、可交互的 DOM 视图**（browser-use 内部叫 DOM mining：在页面注入 JS 构建一棵"交互树"再序列化，元素以 index 引用），配一张整页截图，VLM 不是默认路径：模型记为"按 index click / input selector"。
- 内置工具集（官方文档列全）：`search / navigate / go_back / wait / click(index) / input / upload_file / scroll / find_text / send_keys / evaluate(JS) / switch / close / extract / screenshot / dropdown_options / select_dropdown / write_file / read_file / replace_file / done`。[来源：Available Tools](https://docs.browser-use.com/open-source/customize/tools/available.md)
- 点击/输入是**对元素做 CDP 坐标或 DOM 级操作**（点击由框模型中心/元素句柄触发），输入用 `Input.insertText` 一类，而非对页面注入脚本。

### 1.3 保留登录态 / 控制用户日常浏览器的方案（回答你问的 desktop/extension）
- **复用系统 Chrome profile**：`Browser.from_system_chrome(profile_directory='Profile 5')` 自动探测 Chrome 可执行路径与 `~/.config/google-chrome`（mac/win/linux）等 user data dir，以该真实 profile 重启 Chrome（所以要"先完全关闭 Chrome"），登录态、cookie、扩展**全部保住**。[来源：Real Browser 文档](https://docs.browser-use.com/open-source/customize/browser/real-browser.md)
- **附加你正在用的浏览器**：Browser Harness 连到运行中的 Chrome（新版 Chrome 在 `chrome://inspect/#remote-debugging` 勾一次"Allow remote debugging for this browser instance"即可，**无需插件**、无需注入）。[来源：Browser Harness SKILL.md / install.md](https://cdn.jsdelivr.net/gh/browser-use/browser-harness@main/SKILL.md)、[install.md](https://cdn.jsdelivr.net/gh/browser-use/browser-harness@main/install.md)
- Cloud/profile 同步是另一条路（cookie 可同步，localStorage/IndexedDB/扩展不保证）。[来源：browser-use PyPI FAQ](https://pypi.org/project/browser-use/)

> 补充：确实存在"browser-use 装插件"的说法——历史上/别的渠道有过 dashboard 用的扩展，但**当前官方主路径是 CDP**，不推荐也不依赖插件。[来源：browser-use 仓库根目录](https://github.com/browser-use/browser-use)、[browser-use.com 产品地图](https://browser-use.com/llms.txt)

### 1.4 browser-use 相关的"给 agent 用浏览器"三个形态（产品地图）
1. 开源库（自建 benchmark/agent）→ CDP 本地/远程浏览器。
2. **Browser Harness** CLI/skill → "coding agent 附加你的真实 Chrome"（我们重点抄这个）。
3. Cloud（单独产品）。
`browser-use.com/llms.txt` 里还提到 Claude Code/Codex/Gemini CLI 都通过 Browser Harness / Browser Use MCP 拿到真实浏览器。[来源：browser-use.com Product map](https://browser-use.com/llms.txt)

---

## 2. 其他 agent 的浏览器方案 & 取舍

### 2.1 opencode —— 现在没有内置 browser 工具
- 我核了当前 `packages/opencode/src/tool/` 目录（dev 分支，GitHub REST API）：**只有 webfetch / websearch / bash / edit / read…，没有 browser.ts**。
- `webfetch.txt` 明文："Fetches content from a URL… converts to markdown by default; read-only"。也就是**纯 HTTP 抓取**，不是真实浏览器。[来源：opencode 源码 webfetch.txt](https://github.com/anomalyco/opencode/blob/dev/packages/opencode/src/tool/webfetch.txt)、[官方 Tools 文档](https://opencode.ai/docs/tools/)
- 它想要"真浏览器"时走 MCP（Browser Harness MCP / Playwright MCP / browser-mcp）。历史版本曾有过 Playwright 的 browser 工具（已下线，历史 commit 未逐一考据，**待验证**）。

### 2.2 aider —— 纯抓取，无浏览器
- `/web <url>` 与 `python -m aider.scrape <url>`，把 URL 转 markdown 进上下文；图片靠用户 /paste 截图。[来源：aider 文档 Images & web pages](https://aider.chat/docs/usage/images-urls.html)。达不到你要的"导航/点击/输入"。

### 2.3 Claude Code / Anthropic SDK —— 浏览器工具集（对我方最有参考价值）
- 内置只有 web_search/web_fetch；**新的 `anthropic.tools.browser` 浏览器工具集**：SDK 的 tool runner 把 Claude 的动作调用送给一个**驱动（driver）**执行，driver 一般跑 CDP。
- browser-use 发布了官方驱动 `browser_use.integrations.toolsets_for_claude`，可把本地 Chromium / Cloud / 已有 CDP 浏览器交给 Claude。**31 个规范动作**如下（这就是一套现成的 `NavigatorTool` 动作字典）：
  - 导航：`navigate, new_tab, list_tabs, switch_tab, close_tab`
  - 页面状态：`screenshot, zoom, read_page, find, get_page_text, wait`
  - 指针：`left_click, right_click, middle_click, double_click, triple_click, hover, mouse_move, left_mouse_down, left_mouse_up, left_click_drag, scroll, scroll_to`
  - 输入：`type, key, hold_key, form_input, file_upload`
  - 诊断：`read_console, read_network, javascript_exec`
[来源：Browser Use toolsets for Claude 集成文档](https://docs.browser-use.com/open-source/customize/integrations/toolsets-for-claude.md)（上游 SDK 合约见 anthropics/claude-quickstarts#browser-toolset）

### 2.4 Anthropic computer use（computer use tool）
- **机制是"截图→视觉→鼠标键盘坐标"的 CUA（computer-use agent）**：宿主把屏幕/screenshot 交给模型，模型输出 computer 工具调用（经典动作含 `screenshot / cursor_position / left_click / right_click / double_click / drag / key / type / scroll / wait`），还要配合 `text_editor`、`bash`；需要宿主循环截图回传下一轮。
- 取舍：适合"整机/远程 VM/任何软件"，但**每轮都吞截图、坐标命中不稳、很贵**；对浏览器单页的精确交互不如 DOM/a11y 文本路线。官方文档入口（本环境被 CDP 拦截，**具体函数清单待验证**）：[Anthropic Computer Use](https://docs.anthropic.com/en/docs/agents-and-tools/computer-use)

### 2.5 OpenAI computer-use（computer_use_preview / Operator）
- 同样是截图驱动（工具名 `computer_use_preview`，历史动作为 `click(x,y)/double_click/scroll/type/keypress/wait/move/cursor_position/screenshot`，**待验证**）。Operator 一个关键事实：跑了**隔离的 Chromium**，不登录、不碰你的日常浏览器。官方文档入口被 Cloudflare 拦：[OpenAI Tools: Computer Use](https://platform.openai.com/docs/guides/tools-computer-use)

### 2.6 Browser Harness —— 文档里最有工程细节的"CDP + a11y tree + 自愈 helper"
要点（全部来自 SKILL.md，已抓取原文）：
- 附加本地正在运行的 Chrome/Chromium（CDP），无需 profile 选择；`new_tab(url)`/`switch_tab`/`list_tabs`/`activate_tab`（默认**不主动把 Chrome 带到前台**，可后台操作）。
- **元素定位第一优先 = accessibility tree**：`cdp("Accessibility.getFullAXTree")["nodes"]` 给出 role / name / backendDOMNodeId；点击 = AX 节点 → `DOM.getBoxModel` 算盒子中心坐标 → `click_at_xy(x,y)` → 短 JS/`page_info()` 验证。
- 坐标点击是 CDP 合成器级鼠标事件，**天然穿透 iframe / shadow DOM / 跨域**（无需逐帧注入脚本）。
- AX 树缺的元素（canvas、自定义组件）才回退 `js(...)`；"screenshot when layout or imagery matters"。
- 登录墙：停下来问用户；已登录 SSO 可自动走，密码/MFA/同意永远问人。
- 把"整台 Chrome 的 CDP 连接"做成**本地 daemon**，agent 每次通过 CLI 通话（一个 daemon 共享串行使用）。[来源：SKILL.md](https://cdn.jsdelivr.net/gh/browser-use/browser-harness@main/SKILL.md)、[README](https://cdn.jsdelivr.net/gh/browser-use/browser-harness@main/README.md)

### 2.7 Browser MCP —— "装插件"方案的标杆
- **Chrome 扩展 + 本地 MCP server**：用户装扩展、点一下"Connect"把当前 tab 交给 agent；全部动作在**用户自己的浏览器、自己的登录态**里执行（"Logged In / Stealth / Private(local)"是它主打卖点）。
- 工具集：`navigate / go back / go forward / wait / press key / snapshot(accessibility) / click / drag & drop / hover / type text / get console logs / screenshot`——**snapshot 也是 accessibility**。[来源：browsermcp.io](https://browsermcp.io/)、[setup-extension 文档](https://docs.browsermcp.io/setup-extension)、[仓库](https://github.com/browsermcp/mcp)
- 取舍（相对 CDP attach）：用户侧"零重装 Chrome / 零关浏览器"，体验最顺；但多一个扩展面（权限模型、扩展端执行细节、Chrome Web Store 政策），且动作仍是一次 tab 一次授权，不适合高频/多 tab 自动化。

---

## 3. Rust 生态落地方案

### 3.1 `chromiumoxide` —— 成熟度 / 维护 / 主要 API
查证（crates.io API，非道听途说）：
- 最新 `0.9.1`（2026-02-25 发布），共 20 个版本，**总下载 ~477 万、近 90 天 ~205 万**，维护者 mattsse（Matthias Seitz，reth 作者），MIT OR Apache-2.0，**只支持 tokio**。[来源：crates.io API](https://crates.io/api/v1/crates/chromiumoxide)
- 结构：`chromiumoxide`（高层）+ `chromiumoxide_cdp`（从 Chromium PDL 生成全部 domain 类型，编译较重 ~6 万行）+ `chromiumoxide_fetcher`（可选 `fetcher` 特性帮下载 Chromium，需配 `rustls`/`native-tls` + `zip0`/`zip8`）。

**已确认 API（来自 README 原文 + 官方 example）：**
- 启动：`Browser::launch(BrowserConfig::builder().with_head().build()?)`（默认 headless；`chrome_executable` 可指定路径）；`Browser::connect(ws)` 附加/连接已运行实例。
- 页面：`browser.new_page("https://...")`；`page.find_element("css")` → `Element`（`.click()`、`.type_str("...")`、`.press_key("Enter")`）；`page.wait_for_navigation()`；`page.content()` 取 HTML；`page.save_screenshot(ScreenshotParams / format(CaptureScreenshotFormat::Png), "file.png")`、元素级截图也可；`page.pdf(...) -> Vec<u8>`。
- 万能口：`page.execute(任意 cdp::*Params)` / `page.evaluate(...)` / `evaluate_on_new_document`，所有 domain 的 Command 类型都有 `<Name>Params::builder()`。
[来源：chromiumoxide README（crates.io 官方渲染）](https://static.crates.io/readmes/chromiumoxide/chromiumoxide-0.9.1.html)、[screenshot example](https://cdn.jsdelivr.net/gh/mattsse/chromiumoxide@master/examples/screenshot.rs)

**满足你问题 3 的逐项结论：**
- navigate ✅ `new_page/goto`；click/type ✅ `find_element(...).click()/.type_str()`；screenshot ✅；dump dom ✅ `content()` / `evaluate`。
- a11y tree：CDP domain 有（`Accessibility.getFullAXTree` 类型在 `chromiumoxide_cdp`），**但没有高层包装**（要 `page.execute(AccessibilityGetFullAXTreeParams)` 手调），**待验证**。
- 坐标级鼠标（AX 命中→`Input.dispatchMouseEvent`）：没有高层方法，走 `page.execute(InputDispatchMouseEventParams)`，**待验证**。
- 默认 0.8.0 起默认 runtime 变 tokio（0.8 changelog 可见 tokio-runtime 成为 default；适合我们）。

**替代/参考 crate：** `spider`（成熟 Rust 爬虫框架，含 chromiumoxide 驱动的 web 自动化，可用作对照实现）[来源：spider crate，待验证其 chromiumoxide 集成的当前版本](https://crates.io/crates/spider)；`rust-headless-chrome`（chromiumoxide 的配置/按键来源，维护度较低）。

### 3.2 自己用 tokio-tungstenite 写 CDP 客户端？—— 现实，且对"想完全掌控"的我们很合适
CDP 其实是个 **JSON-over-WebSocket 的请求-响应 + 事件推送**协议，核心握手和要点：
1. 发现：起 Chrome 带 `--remote-debugging-port=9222`（或 `=0` 让 Chrome 随机端口并打印到 stderr，chromiumoxide 也是解析 stderr 拿端口）。
   - `GET http://127.0.0.1:9222/json/version` → `{ "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/browser/<id>" }`（浏览器级）；
   - `GET /json/list` → 各页 target（page/extension 等）的 ws URL。
2. 附加页：`Target.attachToTarget { targetId, flatten:true }` → 返回 `sessionId`；之后**带 sessionId** 的命令在该页会话内执行。
3. 常用命令/事件：
   - `Page.enable` / `Page.navigate {url}` / `Page.captureScreenshot {format:"png", fromSurface}`；`Page.loadEventFired` 事件判定加载。
   - `Runtime.evaluate {expression, returnByValue:true, awaitPromise:true}` —— 跑提取脚本/读状态（页面 World）。
   - `DOM.getDocument` + `DOM.querySelector`/`DOM.getBoxModel {nodeId/backendNodeId}` 拿元素坐标（不需要注入脚本）。
   - **`Accessibility.getFullAXTree`**（DOM 域外的一条，返回带 `backendDOMNodeId` 的语义树，通常数千节点，要过滤）。
   - `Input.dispatchMouseEvent {type:mousePressed/mouseReleased/mouseMoved, x, y, button:"left", clickCount}`、`Input.insertText {text}`、`Input.dispatchKeyEvent`；`Emulation.setFocusEmulationEnabled` 让隐藏页可操作。
   - `Target.createTarget {url}` 开新页；`Target.getTargets`/`Target.setDiscoverTargets` 枚举 tab。
   - 下载：`Browser.setDownloadBehavior {behavior:"allow", downloadPath}`。
   参考协议规范（官方，本环境未抓正文，字段以 devtools-protocol 仓库为准）：[Chrome DevTools Protocol](https://chromedevtools.github.io/devtools-protocol/)
4. 实现量：一个 `CdpConn`（tungstenite stream + 自增 id + `tokio::sync::oneshot` 收 `/map` 的 Future 表 + 事件 channel）约 300–500 行；命令类型可先只声明你要的十几个（serde struct），不必生成全量。
   取舍：**优点** = 无 5MB 生成代码、编译快、依赖少、完全掌控（想加任意 domain 随手加）、便于把 a11y/坐标点击做成自己的高层 API；**缺点** = 要自己维护重连/超时/事件路由，遇到 Chromium 行为差异自己排。

> 建议：优先**自己写薄 CDP 客户端**（本项目是 research/coding agent，掌控与可读性重要）；若想一周出 MVP、少踩 ws 重连的坑，`chromiumoxide` 直接可用。二者甚至可以共存：先用 chromiumoxide 快速验证，再逐步替换为自研薄客户端。

---

## 4. 用户侧"给浏览器装插件"路径的实际做法（如果要走的第二轨道）

### 4.1 事实：主流已不需要插件
- browser-harness 用 **Chrome 内置的一次性授权**：`chrome://inspect/#remote-debugging` 里勾 "Allow remote debugging for this browser instance"，之后本机 CDP 就能附加到你**正在用的那个 Chrome**（Mac 上还会弹一次系统级授权）。无需插件、无需注入。[来源：browser-harness install.md](https://cdn.jsdelivr.net/gh/browser-use/browser-harness@main/install.md)
- 这条途径的工程形态 = **本地守护进程**（daemon 持有到浏览器进程级 ws 的长连接）+ 每次 agent 调用复用该 daemon（见 2.6），多 agent 串行共享一个 daemon、避免重复弹授权。

### 4.2 扩展桥（Browser MCP 形态）怎么做
- 结构：Chrome 扩展（MV3）常驻你的浏览器 → 通过**本地传输**与"本地 agent 服务"通信。实际工程上有两种主流传输：
  1. **localhost WebSocket/HTTP**（Browser MCP 模式）：扩展里 `service worker/background` 连 `ws://127.0.0.1:<port>` 或 fetch 本地 HTTP；用户点扩展按钮"Connect"锁定当前 tab/CSP 权限；此后 agent 端经本地 server 把动作发给扩展执行（扩展用 `chrome.scripting`/自身权限读 DOM、做 click/type/snapshot）。无需系统级安装。
  2. **Native Messaging**：扩展 `chrome.runtime.connectNative("com.example.agent")` 连到 `manifest.json` 注册的原生 host 可执行文件（host 负责把它变成 agent 的 stdin/stdout 或再转 ws）。需要安装一个 native host（注册表/JSON manifest 白名单），分发更重，但对"扩展端有原生能力、无需开放本地端口"更安全。
- **登录态 / 持久 profile**：扩展方案**天然就是用户的浏览器**——不用 sync cookie，不丢 localStorage/IndexedDB/扩展；这是它对比 cloud profile 的最大优势（browser-use 自己都注明 profile 同步"只传 cookie，不传 localStorage/IndexedDB/扩展"）。[来源：browser-use PyPI FAQ](https://pypi.org/project/browser-use/)
- **安全边界注意**：
  - 页内注入（page JS）→ 越少越好；CDP/扩展的**原生输入与坐标**（`Input.dispatchMouseEvent` / `chrome.scripting` 的合成事件）不触碰页面 JS 上下文、难被反爬检测、也被 CSP 拦不住；只有要读 React 内部 state 等才注入 `Runtime.evaluate`。
  - 扩展：最小权限（`activeTab` / `scripting` / 限定 `host_permissions`），只对你点击 Connect 的那个 tab 生效；不要把 `"<all_urls>"` 权限常开。
  - 本地 server 只 bind `127.0.0.1`，配随机端口 + 一次性 token/origin 校验（防跨站 WebSocket 连接）；native messaging 的 native host manifest 白名单防第三方程序自注册。
  - CDP attach 的边界：能连到 9222 的本机进程 = 你的整个浏览器（含 cookie/密码）。所以**默认 bind 127.0.0.1、不复用正在使用的端口暴露公网**；新版 Chrome 的"允许远程调试"弹窗本质就是想让你在授权时刻点一下确认——这是个挺好的安全信号，别关掉它。

---

## 5. 综合推荐（Rust 直接开工的路径）

### 5.1 两条场景，两条主路径（都绕开 lightpanda 静态 fetch）
- **场景 A：本地 / 无头自动化（隔离测试、爬虫、自建 task runner）**
  1. 依赖：`tokio` + `tokio-tungstenite` + `serde/serde_json` + `url`（自写薄客户端）；或直接 `chromiumoxide`（含 fetcher 特性可自动下 Chromium）二选一；日志 `tracing`；`base64`（解码截图）。
  2. 起浏览器：找一个独立 profile 目录（`--user-data-dir=<tmp>`），`--headless=new` 或 headful，`--remote-debugging-port=0`（解析 stderr 端口）或固定 9222；`--no-first-run --no-default-browser-check --disable-gpu`（按需）。**别复用用户真实 profile**（隔离）。
  3. 连接模型：本地 daemon 持有浏览器级 ws，agent 的 `NavigatorTool` 通过该 daemon 发命令（可抄 browser-harness 的"一个 daemon 串行"思路）。
- **场景 B：操控用户日常浏览器（保留登录态）——用户点名要的**
  1. **首选：CDP 附加**（无插件）。
     - Chrome ≥ 136 且正开着：引导用户去 `chrome://inspect/#remote-debugging` 勾一次授权，然后 `Browser::connect`（chromiumoxide）或 `/json/version` 通配附加。
     - 否则：`from_system_chrome` 式——探测系统 Chrome 可执行路径 + `~/.config/google-chrome`（Linux，mac/win 参考 1.3 表格），**先要求用户完全关闭 Chrome**，再用真实 `--user-data-dir` + `--remote-debugging-port` 重启，登录态全保留。
  2. **可选第二通道：扩展桥**（Browser MCP 模式）——如果最终交付里用户强烈要"零重启、装个扩展就完事"，再上；交给本地 agent 用 MCP 动作集（navigate/snapshot/click/type/screenshot…）。
  3. 安全默认：仅 `127.0.0.1`；敏感页（支付/密码/MFA）弹确认；不改用户书签/设置。

### 5.2 `NavigatorTool` 工具参数（直接参考 Anthropic 31 动作 + browser-use 动作集收敛）
工具名 `browser`（一次调用=一个动作；保持原子、可重试、可并 tab）：

```jsonc
{
  "name": "browser",
  "description": "驱动真实 Chrome 执行导航/点击/输入/读取等。元素默认按 accessibility-tree 中的 index 或坐标定位。",
  "parameters": {
    "type": "object",
    "properties": {
      "action": {
        "type": "string",
        "enum": [
          "open", "click", "type", "scroll", "read", "screenshot",
          "back", "forward", "refresh", "new_tab", "switch_tab", "list_tabs", "close_tab",
          "wait", "find", "get_text", "key", "form_input", "upload_file", "eval_js", "done"
        ]
      },
      "url":      { "type": "string", "description": "open 时必填" },
      "index":    { "type": "integer", "description": "click/type/hover 的目标，accessibility 树中的元素序号" },
      "selector": { "type": "string", "description": "兜底的 CSS 选择器（index 缺失时）" },
      "text":     { "type": "string", "description": "type/form_input 要输入的文本" },
      "key":      { "type": "string", "description": "key 动作的特殊键，如 Enter/Escape/Tab" },
      "dx_dy":    { "type": "object", "description": "scroll: {x,y} 或 {amount}" ,
                    "properties": { "x":{"type":"integer"}, "y":{"type":"integer"} } },
      "wait_ms":  { "type": "integer", "description": "wait 毫秒数" },
      "format":   { "type": "string", "enum":["text","markdown","a11y","html"], "description": "read 的输出格式，默认 a11y" },
      "full_page":{"type":"boolean", "description": "screenshot 是否整页" },
      "js":       { "type": "string", "description": "eval_js 的表达式" },
      "file":     { "type": "string", "description": "upload_file 的本机路径" }
    },
    "required": ["action"]
  }
}
```

**返回约定**：`{ ok, url, title, content?, screenshot?("data:image/png;base64,..."), a11y_hint?, tab, error? }`。`read` 默认返回**过滤后的 accessibility 文本**（带序号），模型据此选 index/坐标。

### 5.3 DOM 提取策略（结论）
- **第一优先 = Accessibility tree**（`Accessibility.getFullAXTree`）：
  - 优点：轻量（千级节点 vs DOM 万级）、语义（role/name/value/checked）、天然过滤隐藏元素、与屏幕阅读器一致、**节点带 `backendDOMNodeId` 可直接 `DOM.getBoxModel(backendNodeId=...)` 反查坐标**。
  - 做法：过滤出可交互节点（button/link/checkbox/combobox/textbox/… + 有 name/值）→ 编号 → 序列化成模型的"菜单"。
  - 实测口径来自 browser-harness：点击=AX节点→盒子中心→坐标点击→短 JS 校验。
- **兜底 = JS 提取**（canvas、复杂自定义组件、需要读内部 state 时）：`Runtime.evaluate` 跑自建脚本（可见文本/文档树/React props），以及 `page.content()`/读 `document.body.innerText`。
- **截图 + 多模态 = 按需**：视觉确认布局/CAPTCHA/图像型内容时才 `Page.captureScreenshot` 附给视觉模型；默认不给 LLM 喂图（省 token、稳）。browser-use 与 browser-harness 都是这个口径。
- 点击输入用坐标/原生 CDP，**避免默认注入页面 JS**（2.4/4.2 安全道理同上）。

### 5.4 最小实现步骤（延伸你的 repo）
1. `browser_cdp` crate：连接层（ws、请求-响应、事件、重连）+ 启动层（找 chrome、起进程、解析端口）。
2. driver 层：`navigate / get_a11y / box_of(index) / click_xy / type_text / scroll / screenshot / eval`。
3. `NavigatorTool`（tool 层）包装 driver，输出 a11y 文本视图 + 动作字典（5.2）。
4. daemon 化：长驻 ws 连接（browser-harness 思路），多 agent 串行共享。
5. 安全与登录态：场景 B 用真实 profile/CDP attach，敏感动作走 human-in-loop。
6. 可选：MCP adapter（对外暴露 browser 工具），方便与 Claude Code/其他 MCP 客户端复用。

---

## 6. 主要来源（按主题）

**browser-use 生态**
- 仓库（11.4 万+ star）：https://github.com/browser-use/browser-use
- PyPI 详情/FAQ/安装（含"Playwright、真实浏览器复用 profile、profile 同步局限"）：https://pypi.org/project/browser-use/
- 官方文档索引：https://docs.browser-use.com/llms.txt
- Real Browser / from_system_chrome：https://docs.browser-use.com/open-source/customize/browser/real-browser.md
- Remote / CDP URL：https://docs.browser-use.com/open-source/customize/browser/remote.md
- Browser 配置（Playwright launch）：https://docs.browser-use.com/open-source/customize/browser/basics.md
- 内置工具清单：https://docs.browser-use.com/open-source/customize/tools/available.md
- Actor（裸 CDP）说明：https://docs.browser-use.com/open-source/legacy/actor/basics.md
- 产品地图（Cloud/Harness/开源三分）：https://browser-use.com/llms.txt

**Browser Harness（CDP+a11y 主参考）**
- SKILL.md（a11y 优先、坐标点击、daemon、登录墙策略）：https://cdn.jsdelivr.net/gh/browser-use/browser-harness@main/SKILL.md
- install.md（chrome://inspect 授权、Windows/macOS 差异）：https://cdn.jsdelivr.net/gh/browser-use/browser-harness@main/install.md
- README（"一条 CDP websocket 直连你浏览器"）：https://cdn.jsdelivr.net/gh/browser-use/browser-harness@main/README.md
- 仓库：https://github.com/browser-use/browser-harness

**Anthropic 浏览器工具集（31 动作字典）**
- Browser Use toolsets for Claude：https://docs.browser-use.com/open-source/customize/integrations/toolsets-for-claude.md
- Anthropic computer use（待验证，被拦截）：https://docs.anthropic.com/en/docs/agents-and-tools/computer-use

**OpenAI / Operator**
- OpenAI Tools: Computer Use（待验证，被 Cloudflare 拦）：https://platform.openai.com/docs/guides/tools-computer-use

**opencode / aider**
- opencode 仓库（anomalyco/opencode，dev 分支）：https://github.com/anomalyco/opencode
- opencode Tools 文档（含 webfetch/websearch）：https://opencode.ai/docs/tools/
- opencode webfetch 源码（纯 HTTP→markdown 实证）：https://github.com/anomalyco/opencode/blob/dev/packages/opencode/src/tool/webfetch.txt
- aider 文档（/web + scrape，纯抓取）：https://aider.chat/docs/usage/images-urls.html

**Rust 生态**
- chromiumoxide crates.io（版本/下载/维护者实据）：https://crates.io/api/v1/crates/chromiumoxide
- chromiumoxide README（API 实证）：https://static.crates.io/readmes/chromiumoxide/chromiumoxide-0.9.1.html
- chromiumoxide 仓库与 example：https://github.com/mattsse/chromiumoxide （screenshot: https://cdn.jsdelivr.net/gh/mattsse/chromiumoxide@master/examples/screenshot.rs ）
- CDP 协议规范（字段以 devtools-protocol 官方排期为准）：https://chromedevtools.github.io/devtools-protocol/

**扩展桥**
- Browser MCP：https://browsermcp.io/ ；文档 https://docs.browsermcp.io/setup-extension ；仓库 https://github.com/browsermcp/mcp
- Chrome 远程调试（含 chrome://inspect / 端口，被拦截未抓正文）：https://developer.chrome.com/docs/devtools/remote-debugging/
