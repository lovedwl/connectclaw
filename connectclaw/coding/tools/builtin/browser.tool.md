---
name: browser
desc: 操控无头浏览器(Lightpanda)做搜索/取页/点击/输入,会话内保持 cookie 与登录态。ops：search=Bing 搜索、goto=打开URL并读正文、read=读当前页、click=点击 CSS 选择器、type=在选择器里输入(带 selector+text)
runtime: python
stateful: true
params: { op: "search | goto | read | click | type", arg: "query(search) / url(goto) / css selector(click)", selector: "for type", text: "for type" }
---
import sys, os, json, asyncio, urllib.parse

# stateful 脚本范式:响应直写 fd1(SessionRuntime 读的行协议通道),并把库的杂散
# stdout(如 Lightpanda 的启动 banner)引到 stderr,避免污染协议、造成响应错位。
sys.stdout = sys.stderr


def _respond(obj):
    os.write(1, (json.dumps(obj) + "\n").encode())


from connectclaw.coding.tools.lightpanda import LightpandaEngine, _cap


async def _ensure(st):
    if st["eng"] is None:
        eng = LightpandaEngine()
        await eng.start()
        st["eng"] = eng
        st["sid"] = await eng.open_page()


async def _reset(st):
    if st["eng"] is not None:
        try:
            await st["eng"].close()
        except Exception:
            pass
    st["eng"] = None
    st["sid"] = None


async def _handle(st, req):
    op = (req.get("op") or "").strip()
    arg = req.get("arg") or ""
    await _ensure(st)
    eng, sid = st["eng"], st["sid"]
    if op in ("goto", "fetch", "open"):
        await eng.navigate(sid, arg)
        return await eng.read_markdown(sid)
    if op == "read":
        return await eng.read_markdown(sid)
    if op == "search":
        await eng.navigate(sid, "https://www.bing.com/search?q=" + urllib.parse.quote(arg))
        return await eng.read_markdown(sid)
    if op == "click":
        return f"clicked={await eng.click(sid, arg)}"
    if op == "type":
        return f"typed={await eng.type(sid, req.get('selector') or arg, req.get('text') or '')}"
    return f"unknown op '{op}'. use: search | goto | read | click | type"


async def main():
    st = {"eng": None, "sid": None}
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        try:
            req = json.loads(line)
        except Exception:
            _respond({"ok": False, "error": "bad json line"})
            continue
        try:
            out = await _handle(st, req)
            _respond({"ok": True, "result": _cap(out or "", 8000)})
        except Exception as e:
            await _reset(st)  # engine may have crashed — rebuild on next call
            _respond({"ok": False, "error": str(e)[:300]})


asyncio.run(main())
