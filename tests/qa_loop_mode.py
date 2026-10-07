"""A manual-style pass over loop mode: every control, both directions, two viewports."""
import functools, http.server, threading, sys, json
from playwright.sync_api import sync_playwright
# the built site by default; "embed" serves the artifact body inside an
# unsized wrapper div, the way an embedding host presents the page
import os
ROOT = "/home/user/minihill/site"
if len(sys.argv) > 1 and sys.argv[1] == "embed":
    ROOT = "/tmp/claude-0/-home-user-minihill/d254be37-1566-5bd0-97bf-17a3d00b2c7e/scratchpad/embed"
h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=ROOT); h.log_message = lambda *a, **k: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h); threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{srv.server_address[1]}/"
fails = []
def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else "   <- " + json.dumps(detail, default=str)))
    if not cond: fails.append(name)

STATE = """() => { const f = App.family, u = f && f.unique[App.state.loop ? App.state.loopIdx : 0];
  const g = (id) => document.getElementById(id);
  return { loop: App.state.loop, loopMi: App.state.loopMi, idx: App.state.loopIdx, outBack: App.state.outBack, mode: App.state.mode, calm: App.state.calm,
    from: App.state.from && App.state.from.label, to: App.state.to && App.state.to.label, toField: g('to').value,
    hash: location.hash, famLoop: !!(f && f.loop), n: f ? f.unique.length : 0, targetM: f && f.targetM, kind: u && u.kind,
    inView: App.inView(), share: +App.viewShare().toFixed(2), zoom: App.map.getZoom(),
    status: g('status').textContent, delta: g('delta').textContent, turnsHidden: g('turns').hidden, profHidden: g('prof').hidden, resultHidden: g('result').hidden,
    obrow: !g('obrow').hidden, obChecked: g('outback').checked, calmrow: !g('calmrow').hidden, slMin: +g('sl').min, slMax: +g('sl').max, slVal: +g('sl').value, slpos: g('slpos').textContent,
    toWidth: g('tofield').getBoundingClientRect().width, swapWidth: g('swap').getBoundingClientRect().width, markers: document.querySelectorAll('.pin-icon').length,
    shareHidden: g('share').hidden, scanning: !!App._scan, dist: u && u.stats.distance_m, gain: u && u.stats.elev_gain_m, nextloop: !!g('nextloop') && g('nextloop').textContent }; }"""
def run(viewport, label):
    print(f"\n### {label} {viewport}")
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome", args=["--no-sandbox"])
        ctx = b.new_context(viewport=viewport, is_mobile=viewport["width"] < 600, has_touch=viewport["width"] < 600)
        page = ctx.new_page(); errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        st = lambda: page.evaluate(STATE)
        settle = lambda: (page.wait_for_function("App.family && !App.family.partial && !App._scan", timeout=120000), page.wait_for_timeout(450))
        page.goto(BASE, wait_until="load"); settle()
        s0 = st()
        check("default trip loads as point to point", not s0["loop"] and s0["n"] > 1 and s0["markers"] == 2 and s0["hash"].startswith("#t~"), s0)
        # 1. loop on
        page.click("#loopbtn"); settle(); s = st()
        check("loop on: state, hash, family", s["loop"] and s["famLoop"] and s["hash"].startswith("#l~") and s["n"] >= 1, s)
        check("loop on: To folded, swap folded, one marker", s["toWidth"] < 4 and s["swapWidth"] < 4 and s["markers"] == 1, s)
        check("loop on: rows (out-and-back shown, calm hidden on foot)", s["obrow"] and not s["obChecked"] and not s["calmrow"], s)
        check("loop on: slider 1..15 at 4 mi", s["slMin"] == 1 and s["slMax"] == 15 and s["slVal"] == 4 and s["slpos"] == "4 mi loop", s)
        check("loop on: fitted", s["inView"] and s["share"] >= 0.35, s)
        check("loop on: texts", "loops tried" in s["status"] and "loop" in s["delta"] and not s["turnsHidden"] and not s["profHidden"] and not s["resultHidden"] and not s["shareHidden"], s)
        check("loop on: length within 0.25 mi", abs(s["dist"] - 4 * 1609.344) <= 0.25 * 1609.344, s)
        # 2. slider up to 10
        page.evaluate("() => { const s = document.getElementById('sl'); s.value = 10; s.dispatchEvent(new Event('input')); s.dispatchEvent(new Event('change')); }"); settle(); s = st()
        check("10 mi: recomputed, hash, fitted", s["loopMi"] == 10 and "~10~" in s["hash"] and abs(s["dist"] - 10 * 1609.344) <= 0.25 * 1609.344 and s["inView"], s)
        check("10 mi: label", s["slpos"] == "10 mi loop", s)
        # 3. slider down to 2: zoom back in
        page.evaluate("() => { const s = document.getElementById('sl'); s.value = 2; s.dispatchEvent(new Event('input')); s.dispatchEvent(new Event('change')); }"); settle(); s = st()
        check("2 mi: zoomed back in", s["inView"] and s["share"] >= 0.35, s)
        # 4. another loop
        n = s["n"]
        if n > 1:
            d0 = s["dist"]; page.click("#nextloop"); page.wait_for_timeout(500); s = st()
            check("another loop: index, hash, label", s["idx"] == 1 and s["hash"].split("~")[5] == "1" and "(2 of" in s["nextloop"], s)
        # 5. out-and-back on
        page.click("#outback"); settle(); s = st()
        check("out-and-back on: flattest is one, said so, hash flag, reframed, index reset", s["outBack"] and s["kind"] == "outback" and s["delta"].startswith("Out and back.") and s["hash"].split("~")[3] == "wo" and s["inView"] and s["idx"] == 0, s)
        # 6. bike
        page.click("#mode button[data-v=bike]"); settle(); s = st()
        check("bike in loop mode: calm row shows, loops recomputed, hash bo", s["mode"] == "bike" and s["calmrow"] and s["obrow"] and s["famLoop"] and s["hash"].split("~")[3] == "bo" and s["inView"], s)
        page.click("#calm"); settle(); s = st()
        check("calm off: hash bxo, still a loop", s["hash"].split("~")[3] == "bxo" and s["famLoop"] and not s["calm"], s)
        page.click("#outback"); settle(); s = st()
        check("out-and-back off on bike: hash bx, no outback shown", s["hash"].split("~")[3] == "bx" and s["kind"] != "outback" and not s["delta"].startswith("Out and back"), s)
        page.click("#mode button[data-v=walk]"); settle(); s = st()
        check("back to walk: calm row hidden, hash w", not s["calmrow"] and s["hash"].split("~")[3] == "w" and s["famLoop"], s)
        # 7. click the map far away: moves the start
        page.evaluate("() => App.map.fire('click', { latlng: L.latLng(37.8060, -122.4435) })"); settle(); s = st()
        check("map click in loop mode moves the start and recomputes", s["from"] and s["from"] != s0["from"] and s["famLoop"] and s["inView"] and s["markers"] == 1, s)
        # 8. type a new start
        page.fill("#from", "Dolores Park"); page.wait_for_timeout(400)
        page.keyboard.press("Enter"); settle(); s = st()
        check("typed start: label, loop recomputed, fitted", "Dolores" in (s["from"] or "") and s["famLoop"] and s["inView"], s)
        # 9. share link text matches the hash
        url = page.evaluate("() => App.shareUrl()")
        check("share url carries the loop hash", url.endswith(s["hash"]), [url, s["hash"]])
        # 10. loop off
        page.click("#loopbtn"); settle(); s = st()
        check("loop off: To restored, route back, hash t, two markers", not s["loop"] and s["to"] == s0["to"] and s["toField"] == s0["to"] and not s["famLoop"] and s["hash"].startswith("#t~") and s["markers"] == 2, s)
        check("loop off: slider back, rows hidden, fitted", s["slMin"] == 0 and s["slMax"] == 1 and not s["obrow"] and s["inView"] and s["toWidth"] > 100, s)
        check("loop off: texts are the route's", "routes" in s["status"] and ("shortest" in s["delta"].lower()), s)
        # 11. reload from a loop link with out-and-back and an index
        page.goto("about:blank"); page.goto(BASE + "#l~-122.44350~37.80600~wo~3.5~1~Marina_20Green", wait_until="load"); settle(); s = st()
        check("loop link: everything restored", s["loop"] and s["outBack"] and s["obChecked"] and s["loopMi"] == 3.5 and s["idx"] == min(1, s["n"] - 1) and s["from"] == "Marina Green" and s["kind"] in ("outback", "petal", "triangle", "polygon"), s)
        check("loop link: fitted on load", s["inView"] and s["share"] >= 0.35, s)
        page.goto("about:blank"); page.goto(BASE + "#l~-122.44350~37.80600~w~4~9~Marina_20Green", wait_until="load"); settle(); s = st()
        check("loop link with a too-big index clamps", s["idx"] == s["n"] - 1 and not s["outBack"], s)
        # 12. a rapid burst: loop on, slider, toggle, loop off, before anything finishes
        page.goto("about:blank"); page.goto(BASE, wait_until="load"); settle()
        page.evaluate("""() => { document.getElementById('loopbtn').click();
          const s = document.getElementById('sl'); s.value = 12; s.dispatchEvent(new Event('input')); s.dispatchEvent(new Event('change'));
          document.getElementById('outback').click(); document.getElementById('loopbtn').click(); }""")
        settle(); s = st()
        check("rapid burst ends as a consistent route", not s["loop"] and not s["famLoop"] and s["to"] == s0["to"] and "routes" in s["status"] and not s["scanning"] and not s["obrow"], s)
        page.evaluate("""() => { document.getElementById('loopbtn').click();
          const s = document.getElementById('sl'); s.value = 6; s.dispatchEvent(new Event('input')); s.dispatchEvent(new Event('change'));
          const cb = document.getElementById('outback'); if (!cb.checked) cb.click(); }""")
        settle(); s = st()
        check("rapid burst ends as a consistent loop", s["loop"] and s["famLoop"] and s["loopMi"] == 6 and s["outBack"] and s["kind"] == "outback" and "loops tried" in s["status"] and s["delta"].startswith("Out and back."), s)
        # 13. the edge of the city at 15 mi
        page.goto("about:blank"); page.goto(BASE + "#l~-122.50300~37.71500~w~15~0~Fort_20Funston", wait_until="load"); settle(); s = st()
        check("15 mi from Fort Funston: a result and a sane message", s["famLoop"] and s["n"] >= 1 and (("loops tried" in s["status"]) or ("closest" in s["status"])) and len(s["delta"]) > 10 and s["inView"], s)
        # 14. reduced motion
        ctx2 = b.new_context(viewport=viewport, reduced_motion="reduce"); p2 = ctx2.new_page(); e2 = []
        p2.on("pageerror", lambda e: e2.append(str(e)))
        p2.goto(BASE, wait_until="load"); p2.wait_for_function("App.family && !App.family.partial", timeout=120000)
        p2.click("#loopbtn"); p2.wait_for_function("App.family && App.family.loop && !App._scan", timeout=60000); p2.wait_for_timeout(300)
        p2.click("#outback"); p2.wait_for_function("App.family && App.family.loop && App.family.unique[0].kind === 'outback'", timeout=60000); p2.wait_for_timeout(300)
        t = p2.evaluate("() => [document.getElementById('status').textContent, document.getElementById('delta').textContent, document.getElementById('turns').hidden, getComputedStyle(document.getElementById('delta')).opacity]")
        check("reduced motion: texts land, nothing stuck faded", "loops tried" in t[0] and t[1].startswith("Out and back.") and not t[2] and t[3] == "1" and not e2, [t, e2])
        ctx2.close()
        check("no page errors", not errs, errs[:5])
        b.close()
run({"width": 1280, "height": 800}, "desktop")
run({"width": 390, "height": 844}, "phone")
srv.shutdown()
print("\nFAILED:", fails if fails else "none")
