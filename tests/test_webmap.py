"""End-to-end test of the interactive map's in-browser router.

The map does its own routing, so the thing most worth guarding is that it
agrees with Python.  This test drives the real HTML file in a headless
browser, hands the JavaScript router the *exact* arc sequences Python chose,
and checks two things:

1. the metrics JavaScript computes for Python's own path match Python's
   figures (this isolates quantisation error from route choice);
2. the route JavaScript finds for itself is never more expensive, under its
   own cost model, than the one Python found (this is the optimality check).

Routes may legitimately differ where two paths tie on cost -- the two
implementations break ties differently -- so path identity is reported but
not asserted.

Skipped unless Playwright, a Chromium build and a built map are all present:

    pip install playwright && playwright install chromium
    python -m flatten_seattle map
"""
from __future__ import annotations

import glob
import json
import os
import tempfile
from pathlib import Path

import pytest

from flatten_seattle.config import PROCESSED_DIR
from flatten_seattle.viz_interactive import INTERACTIVE_HTML

playwright = pytest.importorskip("playwright.sync_api",
                                 reason="playwright is not installed")

#: Pairs chosen to span flat and hilly parts of the city, and both modes.
PAIRS = [
    ("Mission", "Outer Sunset"), ("Noe Valley", "Financial District"),
    ("Inner Richmond", "Downtown/Civic Center"), ("Bayview", "Golden Gate Park"),
    ("Bernal Heights", "Marina"), ("Chinatown", "Inner Sunset"),
    ("Excelsior", "South of Market"), ("Potrero Hill", "Western Addition"),
    ("Presidio", "Visitacion Valley"), ("Twin Peaks", "Marina"),
]


def _chromium() -> str | None:
    """A usable Chromium, either Playwright's own or a pre-installed one."""
    for pattern in ("/opt/pw-browsers/chromium-*/chrome-linux/chrome",
                    os.path.expanduser(
                        "~/.cache/ms-playwright/chromium-*/chrome-linux/chrome")):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[-1]
    return None


pytestmark = [
    pytest.mark.skipif(not INTERACTIVE_HTML.exists(),
                       reason="map not built; run `python -m flatten_seattle map`"),
    pytest.mark.skipif(not (PROCESSED_DIR / "edges_directed.parquet").exists(),
                       reason="processed data not built"),
]


@pytest.fixture(scope="module")
def reference():
    """Python's own routes for the sample pairs, with full arc sequences."""
    from flatten_seattle.config import ROUTING_PROFILES
    from flatten_seattle.pipeline import build_context
    from flatten_seattle.routing import route
    from flatten_seattle.utils import configure_gdal_for_proxy

    configure_gdal_for_proxy()
    ctx = build_context()
    rows = []
    for mode, graph in ctx.graphs.items():
        pts = dict(zip(ctx.points[mode]["neighborhood"], ctx.points[mode]["node"]))
        for pname, w in ROUTING_PROFILES.items():
            cost = graph.build_costs(w)
            for o, d in PAIRS:
                if o not in pts or d not in pts:
                    continue
                arcs, s = route(graph, pts[o], pts[d], w, arc_cost=cost)
                if not arcs:
                    continue
                t = graph.table.iloc[arcs]
                rows.append({
                    "mode": mode, "profile": pname, "o": o, "d": d,
                    "path": [[int(e), 1 if dr == "rev" else 0]
                             for e, dr in zip(t["edge_id"], t["direction"])],
                    "cost": float(cost[arcs].sum()),
                    "distance_m": float(s["distance_m"]),
                    "gain": float(s["elev_gain_m"]),
                    "loss": float(s["elev_loss_m"]),
                    "maxg": float(s["max_grade"]),
                    "avgg": float(s["avg_abs_grade"]),
                    "th": [float(s[f"d_above_{t_}"]) for t_ in (3, 5, 8, 10, 15)],
                })
    assert rows, "no reference routes could be computed"
    return rows


@pytest.fixture(scope="module")
def browser_results(reference):
    """Drive the real HTML file and collect the JavaScript router's answers."""
    from playwright.sync_api import sync_playwright

    exe = _chromium()
    script = _BROWSER_SCRIPT
    with tempfile.TemporaryDirectory() as tmp:
        ref_path = Path(tmp) / "ref.json"
        ref_path.write_text(json.dumps(reference))
        errors: list = []
        with sync_playwright() as pw:
            try:
                browser = pw.chromium.launch(
                    executable_path=exe,
                    args=["--no-sandbox", "--disable-gpu"])
            except Exception as exc:                       # pragma: no cover
                pytest.skip(f"no usable Chromium: {exc}")
            page = browser.new_page(viewport={"width": 1400, "height": 900})
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text)
                    if m.type == "error" and "ERR_TUNNEL" not in m.text
                    and "ERR_NAME" not in m.text and "tile" not in m.text.lower()
                    else None)
            page.goto(INTERACTIVE_HTML.resolve().as_uri(), wait_until="load",
                      timeout=240_000)
            page.wait_for_function("window.App && window.App.graph",
                                  timeout=240_000)
            out = page.evaluate(script, reference)
            info = page.evaluate(
                "() => ({nodes: App.graph.n, arcs: App.graph.m,"
                " edges: App.geom.nEdges})")
            browser.close()
    return out, info, errors


_BROWSER_SCRIPT = """(ref) => {
    const res = [];
    for (const r of ref) {
        const g = App.graph, w = App.meta.profiles[r.profile];
        const arcs = r.path.map(([e, rev]) => g.arcOf(e, rev));
        if (arcs.some(a => a < 0)) { res.push({...r, miss: 'arc'}); continue; }
        const sPy = g.summarise(arcs);
        const costPy = g.pathCost(arcs, r.mode, w);
        const pts = App.DATA.points[r.mode] || {};
        const bit = g.modeBit(r.mode);
        const f = i => (g.nodeFlags[i] & bit) !== 0;
        const src = App.nodeGrid.nearest(pts[r.o][0], pts[r.o][1], f);
        const dst = App.nodeGrid.nearest(pts[r.d][0], pts[r.d][1], f);
        const rt = g.route(src, dst, r.mode, w);
        if (!rt) { res.push({...r, miss: 'route'}); continue; }
        const sJs = g.summarise(rt.arcs);
        res.push({
            mode: r.mode, profile: r.profile, o: r.o, d: r.d,
            same: arcs.length === rt.arcs.length
                  && arcs.every((a, i) => a === rt.arcs[i]),
            onPyPath: {distance_m: sPy.distance_m, gain: sPy.elev_gain_m,
                       loss: sPy.elev_loss_m, maxg: sPy.max_grade,
                       avgg: sPy.avg_abs_grade, th: sPy.thresholds,
                       cost: costPy},
            ownRoute: {distance_m: sJs.distance_m, gain: sJs.elev_gain_m,
                       cost: rt.cost},
        });
    }
    return res;
}"""


@pytest.fixture(scope="module")
def warp_result():
    """Build the warped city in the browser and report its diagnostics."""
    from playwright.sync_api import sync_playwright
    exe = _chromium()
    errors: list = []
    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(executable_path=exe,
                                         args=["--no-sandbox", "--disable-gpu"])
        except Exception as exc:                            # pragma: no cover
            pytest.skip(f"no usable Chromium: {exc}")
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(INTERACTIVE_HTML.resolve().as_uri(), wait_until="load",
                  timeout=240_000)
        page.wait_for_function("window.App && window.App.graph", timeout=240_000)
        page.click("#warptoggle")
        page.wait_for_function(
            "App.warp && !document.getElementById('busy').classList.contains('on')",
            timeout=180_000)
        out = page.evaluate("""() => {
            const f = Warp.frame(37.76, -122.44), pts = App.DATA.points.walk, shift = {};
            for (const n of Object.keys(pts)) {
                const [lon, lat] = pts[n];
                const [wlon, wlat] = App.warp.transform(lon, lat);
                const a = f.toXY(lon, lat), b = f.toXY(wlon, wlat);
                shift[n] = Math.hypot(a[0] - b[0], a[1] - b[1]);
            }
            const c = App.network.altCoords;
            let finite = true;
            for (let i = 0; i < c.length; i++) if (!Number.isFinite(c[i])) { finite = false; break; }
            return {anchors: App.warp.anchors, stress: App.warp.stress,
                    ms: App.warp.ms, shift, finite, n: c.length};
        }""")
        browser.close()
    return out, errors


# --------------------------------------------------------------------- tests
def test_the_page_loads_without_javascript_errors(browser_results):
    _out, _info, errors = browser_results
    assert not errors, f"JavaScript errors on load: {errors[:4]}"


def test_the_browser_graph_matches_the_python_graph(browser_results):
    _out, info, _errors = browser_results
    import pandas as pd
    directed = pd.read_parquet(PROCESSED_DIR / "edges_directed.parquet")
    import geopandas as gpd
    edges = gpd.read_parquet(PROCESSED_DIR / "edges_metrics.parquet")
    n_arcs = int((directed["walk_traversable"]
                  | directed["bike_traversable"]).sum())
    assert info["edges"] == len(edges)
    # the packed arc count drops arcs outside both modes' components
    assert 0.9 * n_arcs <= info["arcs"] <= n_arcs


def test_every_python_route_resolves_in_the_browser(browser_results):
    out, _info, _errors = browser_results
    missing = [r for r in out if r.get("miss")]
    assert not missing, f"{len(missing)} routes did not resolve: {missing[:3]}"


def test_metrics_agree_on_pythons_own_path(browser_results, reference):
    """Same arcs, same numbers -- to within the packing's quantisation."""
    out, _info, _errors = browser_results
    ref = {(r["mode"], r["profile"], r["o"], r["d"]): r for r in reference}
    bad = []
    for r in out:
        py = ref[(r["mode"], r["profile"], r["o"], r["d"])]
        js = r["onPyPath"]
        # distance error accumulates one quantum per arc, so scale with length
        tol = max(0.5, 1e-4 * py["distance_m"])
        if abs(js["distance_m"] - py["distance_m"]) > tol:
            bad.append((r, f"distance {js['distance_m']:.2f} vs "
                           f"{py['distance_m']:.2f}"))
        for k, t in (("gain", 0.3), ("loss", 0.3)):
            if abs(js[k] - py[k]) > t:
                bad.append((r, f"{k} {js[k]:.3f} vs {py[k]:.3f}"))
        for k in ("maxg", "avgg"):
            if abs(js[k] - py[k]) > 5e-4:
                bad.append((r, f"{k} {js[k]:.5f} vs {py[k]:.5f}"))
        for i, (a, b) in enumerate(zip(js["th"], py["th"])):
            if abs(a - b) > max(1.0, 1e-3 * py["distance_m"]):
                bad.append((r, f"d_above[{i}] {a:.2f} vs {b:.2f}"))
    assert not bad, f"{len(bad)} metric mismatches, e.g. {bad[:4]}"


def test_the_browser_router_is_optimal(browser_results):
    """The browser's own route must never cost more than Python's."""
    out, _info, _errors = browser_results
    worse = [(r["mode"], r["profile"], r["o"], r["d"],
              r["ownRoute"]["cost"], r["onPyPath"]["cost"])
             for r in out
             if r["ownRoute"]["cost"] > r["onPyPath"]["cost"] * 1.0005 + 0.5]
    assert not worse, f"browser found costlier routes: {worse[:4]}"


def test_most_routes_are_identical_not_merely_equivalent(browser_results):
    """A sanity floor: tie-breaking should differ sometimes, not usually."""
    out, _info, _errors = browser_results
    same = sum(1 for r in out if r.get("same"))
    assert same >= 0.6 * len(out), (
        f"only {same}/{len(out)} arc sequences matched exactly, which "
        "suggests a cost-model difference rather than tie-breaking")


def test_the_warp_builds_without_errors(warp_result):
    out, errors = warp_result
    assert not errors, errors[:3]
    assert out["finite"] and out["n"] > 0


def test_the_warp_fits_the_cost_matrix_reasonably(warp_result):
    """Stress is the residual between page distance and climbing cost."""
    out, _ = warp_result
    assert 100 <= out["anchors"] <= 400
    assert out["stress"] < 0.25, f"stress {out['stress']:.3f} is too high to read"


def test_hilly_neighborhoods_move_more_than_flat_ones(warp_result):
    """The whole point: a ridge pushes places apart; the flats stay put."""
    out, _ = warp_result
    s = out["shift"]
    hilly = max(s.get("Twin Peaks", 0), s.get("West of Twin Peaks", 0))
    flat = min(s.get("Mission", 1e9), s.get("South of Market", 1e9),
               s.get("Financial District", 1e9))
    assert hilly > 2 * flat, f"hilly {hilly:.0f} m vs flat {flat:.0f} m"
    assert hilly > 1500
