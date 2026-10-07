"""Record every loop transition and look for hard cuts in the map: a cut is
one frame whose map pixels change far more than its neighbours'."""
import functools, http.server, threading, os, glob, shutil, subprocess, sys
from playwright.sync_api import sync_playwright
from PIL import Image, ImageChops
import numpy as np
SP = "/tmp/claude-0/-home-user-minihill/d254be37-1566-5bd0-97bf-17a3d00b2c7e/scratchpad/loops/cuts_" + (sys.argv[2] if len(sys.argv) > 2 else "a")
FF = "/usr/local/lib/python3.11/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
ROOT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/minihill/site"
h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=ROOT); h.log_message = lambda *a, **k: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h); threading.Thread(target=srv.serve_forever, daemon=True).start()
FPS = 20
def run(vp, label):
    shutil.rmtree(SP, ignore_errors=True); os.makedirs(SP)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome", args=["--no-sandbox"])
        ctx = b.new_context(viewport=vp, record_video_dir=SP, record_video_size=vp, is_mobile=vp["width"] < 600, has_touch=vp["width"] < 600)
        page = ctx.new_page()
        page.goto(f"http://127.0.0.1:{srv.server_address[1]}/", wait_until="load")
        settle = lambda: (page.wait_for_function("window.App && App.family && !App.family.partial && !App._scan", timeout=120000), page.wait_for_timeout(1200))
        settle(); page.evaluate("""() => { window._ev = []; const m = document.createElement('div'); m.id = 'qamark';
          Object.assign(m.style, { position: 'fixed', right: '0', top: '0', width: '24px', height: '24px', background: '#000', zIndex: 99999, display: 'none' }); document.body.appendChild(m); }""")
        def mark(name):
            page.evaluate(f"() => {{ _ev.push('{name}'); const m = document.getElementById('qamark'); m.style.display = 'block'; setTimeout(() => m.style.display = 'none', 160); }}")
            page.wait_for_timeout(220)
        mark("loop on"); page.click("#loopbtn"); settle()
        mark("slider 8"); page.evaluate("() => { const s = document.getElementById('sl'); s.value = 8; s.dispatchEvent(new Event('input')); s.dispatchEvent(new Event('change')); }"); settle()
        for k in range(2):
            mark("another loop"); page.click("#nextloop"); page.wait_for_timeout(1200)
        mark("slider 2"); page.evaluate("() => { const s = document.getElementById('sl'); s.value = 2; s.dispatchEvent(new Event('input')); s.dispatchEvent(new Event('change')); }"); settle()
        mark("out-and-back on"); page.click("#outback"); settle()
        mark("loop off"); page.click("#loopbtn"); settle()
        ev = page.evaluate("() => _ev")
        ctx.close(); b.close()
    v = glob.glob(SP + "/*.webm")[0]
    subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", "-i", v, "-vf", f"fps={FPS}", f"{SP}/f%04d.png"], check=True)
    fs = sorted(glob.glob(f"{SP}/f*.png"))
    # the map region: right of the card on wide screens, above it on phones
    wide = vp["width"] > 640
    crop = (440, 0, vp["width"], vp["height"]) if wide else (0, 0, vp["width"], int(vp["height"] * 0.42))
    prev = None; d = []; marks = []
    for i, f in enumerate(fs):
        full = Image.open(f).convert("L")
        corner = np.asarray(full.crop((vp["width"] - 20, 2, vp["width"] - 4, 18))).mean()
        if corner < 60 and (not marks or i - marks[-1] > 6): marks.append(i)
        im = full.crop(crop).resize(((crop[2]-crop[0])//2, (crop[3]-crop[1])//2))
        d.append(0 if prev is None else float(np.asarray(ImageChops.difference(im, prev)).mean())); prev = im
    d = np.array(d)
    print(f"\n### {label} {vp}: {len(fs)} frames, {len(marks)} marks for {len(ev)} events")
    bad = False
    for j, name in enumerate(ev):
        if j >= len(marks): break
        f0 = marks[j] + 4; f1 = marks[j + 1] if j + 1 < len(marks) else len(d)
        win = d[f0:f1]
        if not len(win): continue
        peak = win.max(); frames = (win > 1.0).sum()
        # a glide spreads change over several frames; a cut puts it in one
        kind = "cut" if peak > 5 and frames <= 2 else ("glide" if frames >= 3 else "still")
        verdict = "  <-- CUT" if kind == "cut" else ""
        if kind == "cut": bad = True
        print(f"  {name:16s} peak {peak:5.1f}  changing frames {frames:2d}  {kind}{verdict}")
    return bad
bad = run({"width": 1280, "height": 800}, "desktop") | run({"width": 390, "height": 844}, "phone")
srv.shutdown()
print("\nHARD CUTS FOUND" if bad else "\nno hard cuts")
