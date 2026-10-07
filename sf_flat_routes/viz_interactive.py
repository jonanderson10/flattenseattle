"""The two web maps: the analysis explorer and the simple route page.

Both are single self-contained HTML files with Leaflet vendored and every
byte of data embedded, so they can be moved around and opened directly, and
both route in the browser over the packed graph (``webgraph.py``) with the
same cost model Python uses.

* **Explorer** (``outputs/sf_flat_routes_map.html``): every analysis layer
  (gradient-coloured network, corridors, passes, barriers, basins, bike
  facilities), the four objectives with live weight sliders, Pareto readout
  and the cost-warped city. Dense by design; this is the working view.
* **Route page**: one card with origin, destination and a shortest-to-
  flattest slider over a quiet hillshade. Place search is offline
  (intersections from the graph, Overture places and addresses packed into
  the page). Written twice: as ``outputs/sf_flat_route_finder.html``, one
  file that opens from disk, and as the static site in ``site/`` (HTML, CSS,
  JS, the gzipped graph and the hillshade as separate cacheable files),
  which GitHub Pages serves as the demo.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .config import (CITY_NAME, OUTPUT_DIR, PRODUCT_NAME, REPO_URL, SITE_DIR,
                     SITE_DOMAIN, SITE_URL)
from .utils import get_logger, human_bytes, step

log = get_logger("sf_flat_routes.viz_interactive")

INTERACTIVE_HTML = OUTPUT_DIR / "sf_flat_routes_map.html"
#: The route finder as one self-contained file, and as a static site.
SIMPLE_HTML = OUTPUT_DIR / "sf_flat_route_finder.html"
SITE_INDEX = SITE_DIR / "index.html"
WEB_DIR = Path(__file__).resolve().parent / "web"
VENDOR_DIR = Path(__file__).resolve().parent / "vendor"

# --------------------------------------------------------------------------
# layer preparation
# --------------------------------------------------------------------------
def _round_geometry(geo: dict, ndigits: int = 5) -> dict:
    """Round every coordinate in a __geo_interface__ mapping.

    Full-precision floats serialise to ~17 characters each; five decimal
    places is about 1 m at this latitude, which is finer than the source
    data's own accuracy, and shrinks the embedded GeoJSON several-fold.
    """
    def walk(c):
        if isinstance(c, (list, tuple)):
            if c and isinstance(c[0], (int, float)):
                return [round(float(v), ndigits) for v in c[:2]]
            return [walk(x) for x in c]
        return c
    return {"type": geo["type"], "coordinates": walk(geo["coordinates"])}


def _geojson(gdf, props: list[str], simplify: float = 4.0) -> dict:
    """GeoDataFrame -> GeoJSON dict in WGS84, geometry simplified in metres."""
    g = gdf.copy()
    if simplify:
        g["geometry"] = g.geometry.simplify(simplify, preserve_topology=False)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    g = g.to_crs("EPSG:4326")
    keep = [c for c in props if c in g.columns]
    feats = []
    for _, r in g.iterrows():
        p = {}
        for c in keep:
            v = r[c]
            if isinstance(v, (np.floating, float)):
                v = None if not np.isfinite(v) else round(float(v), 5)
            elif isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.bool_, bool)):
                v = bool(v)
            elif v is not None and not isinstance(v, str):
                v = str(v)
            p[c] = v
        feats.append({"type": "Feature", "properties": p,
                      "geometry": _round_geometry(r.geometry.__geo_interface__)})
    return {"type": "FeatureCollection", "features": feats}


def build_layers(ctx, corridors, passes, barriers, basins):
    """The small vector overlays. The street network is *not* here.

    The network is served from the packed graph instead, so its geometry
    exists exactly once in the file and the gradient display can never
    disagree with what the router uses.
    """
    edges = ctx.edges
    layers = {
        "neighborhoods": _geojson(ctx.neighborhoods,
                                  ["neighborhood", "area_km2"], simplify=12.0),
        "corridors": _geojson(
            corridors.to_crs(edges.crs) if corridors.crs != edges.crs else corridors,
            ["corridor_id", "corridor_name", "street_names", "mode",
             "length_km", "mean_abs_grade", "max_grade", "gain_per_km",
             "pair_count_max", "neighborhood_span", "climb_saved_m",
             "elev_min_m", "elev_max_m", "neighborhoods", "total_score"],
            simplify=5.0),
        "passes": _geojson(
            passes.to_crs(edges.crs) if passes.crs != edges.crs else passes,
            ["edge_id", "name", "neighborhood", "pass_elev_m", "pass_elev_ft",
             "pairs_served", "max_abs_grade", "neighborhoods_separated"],
            simplify=2.0),
        "barriers": _geojson(
            (barriers.to_crs(edges.crs) if barriers.crs != edges.crs
             else barriers).head(350),
            ["edge_id", "name", "neighborhood", "max_abs_grade", "length_m",
             "shortest_use", "flat_use_per_objective", "unavoidability"],
            simplify=2.0),
    }
    bike = edges[edges["bike_facility"].astype(bool) & (edges["bike_facility"] != "")]
    layers["bike_network"] = _geojson(
        bike, ["name", "cls", "bike_facility", "length_m", "max_abs_grade"],
        simplify=5.0)
    carfree = edges[edges["cls"].isin(["pedestrian", "living_street"])
                    | (edges["bike_facility"] == "car_free_street")]
    layers["low_stress"] = _geojson(
        carfree, ["name", "cls", "bike_facility", "length_m"], simplify=5.0)
    if basins is not None and len(basins):
        b = basins.to_crs(edges.crs) if basins.crs != edges.crs else basins
        agg = b.dissolve(by="basin", aggfunc={"basin_label": "first",
                                              "length_m": "sum"}).reset_index()
        agg["length_km"] = agg["length_m"] / 1000.0
        layers["basins"] = _geojson(agg, ["basin", "basin_label", "length_km"],
                                    simplify=14.0)
    return layers


#: Guided examples, so that opening the map demonstrates the findings
#: without anyone having to know which neighborhoods to pick. Notes are
#: filled in from the analysis outputs at build time.
_EXAMPLE_PAIRS = (
    ("Mission", "Outer Sunset", "walk", "min_climb",
     "crossing the city east to west"),
    ("Noe Valley", "Financial District", "walk", "min_climb",
     "almost all the climbing is optional"),
    ("Bayview", "Golden Gate Park", "walk", "min_climb",
     "the biggest single saving in the city"),
    ("Mission", "Marina", "bike", "balanced",
     "by bicycle, over the northern saddles"),
    ("West of Twin Peaks", "Downtown/Civic Center", "walk", "grade_averse",
     "behind the Twin Peaks barrier: no cheap way over"),
)


def _examples(points: dict) -> list[dict]:
    """Attach measured savings to each guided example."""
    from .pairs import PAIRS_PARQUET
    import pandas as pd

    out = []
    table = None
    if PAIRS_PARQUET.exists():
        table = pd.read_parquet(PAIRS_PARQUET)
    for o, d, mode, prof, blurb in _EXAMPLE_PAIRS:
        if o not in points.get(mode, {}) or d not in points.get(mode, {}):
            continue
        note = blurb
        if table is not None:
            sel = table[(table["mode"] == mode) & (table["profile"] == prof)
                        & (table["origin"] == o) & (table["destination"] == d)]
            if len(sel):
                r = sel.iloc[0]
                note = (f"{blurb} &mdash; "
                        f"{r['shortest_gain_m']*3.28084:.0f} ft of climbing "
                        f"becomes {r['elev_gain_m']*3.28084:.0f} ft")
        out.append({"o": o, "d": d, "mode": mode, "profile": prof,
                    "note": note})
    return out


def _asset(name: str) -> str:
    return (WEB_DIR / name).read_text(encoding="utf-8")


def _vendor(name: str) -> str:
    """Read a vendored asset for inlining (see ``vendor/README.md``)."""
    return (VENDOR_DIR / name).read_text(encoding="utf-8")


def _render(payload: dict) -> str:
    """The explorer page."""
    html = _asset("index.html")
    html = html.replace("/*__LEAFLET_CSS__*/", _vendor("leaflet-1.9.4.css"))
    html = html.replace("/*__APP_CSS__*/", _asset("app.css"))
    html = html.replace("/*__LEAFLET_JS__*/", _vendor("leaflet-1.9.4.min.js"))
    html = html.replace("/*__APP_JS__*/", _asset("engine.js") + "\n" + _asset("app.js")
                        + "\n" + _asset("warp.js"))
    return html.replace("/*__DATA__*/", json.dumps(payload, separators=(",", ":")))


def _route_page_html(payload: dict, linked: bool, assets: dict | None = None) -> str:
    """The route page: assets inlined (one file) or linked (the site).

    ``assets`` maps each plain asset name to its content-hashed file name.
    """
    html = _asset("simple.html").replace("/*__REPO_URL__*/", REPO_URL)
    if linked:
        a = assets or {}
        html = html.replace('<style>/*__LEAFLET_CSS__*/</style>',
                            f'<link rel="stylesheet" href="{a.get("leaflet.css", "leaflet.css")}">')
        html = html.replace('<style>/*__APP_CSS__*/</style>',
                            f'<link rel="stylesheet" href="{a.get("app.css", "app.css")}">')
        html = html.replace('<script>/*__LEAFLET_JS__*/</script>',
                            f'<script src="{a.get("leaflet.js", "leaflet.js")}"></script>')
        html = html.replace('<script>/*__APP_JS__*/</script>',
                            f'<script src="{a.get("app.js", "app.js")}"></script>')
        extra = "\n".join([
            f'<link rel="canonical" href="{SITE_URL}">',
            '<link rel="icon" href="favicon.svg" type="image/svg+xml">',
            '<meta property="og:type" content="website">',
            f'<meta property="og:title" content="{PRODUCT_NAME}">',
            f'<meta property="og:site_name" content="{PRODUCT_NAME}">',
            f'<meta property="og:description" content="{_DESCRIPTION}">',
            f'<meta property="og:url" content="{SITE_URL}">',
            f'<meta property="og:image" content="{SITE_URL}preview.jpg">',
            f'<meta property="og:image:secure_url" content="{SITE_URL}preview.jpg">',
            '<meta property="og:image:type" content="image/jpeg">',
            '<meta property="og:image:width" content="1200">',
            '<meta property="og:image:height" content="630">',
            f'<meta property="og:image:alt" content="A map of {CITY_NAME} with a fan of '
            'routes between two places, from the shortest to the flattest">',
            '<meta name="twitter:card" content="summary_large_image">',
            f'<meta name="twitter:title" content="{PRODUCT_NAME}">',
            f'<meta name="twitter:description" content="{_DESCRIPTION}">',
            f'<meta name="twitter:image" content="{SITE_URL}preview.jpg">',
            f'<link rel="preload" href="{payload["bundle_url"]}" as="fetch" crossorigin>',
        ])
        html = html.replace("<!--__HEAD_EXTRA__-->", extra)
    else:
        html = html.replace("<!--__HEAD_EXTRA__-->\n", "")
        html = html.replace("/*__LEAFLET_CSS__*/", _vendor("leaflet-1.9.4.css"))
        html = html.replace("/*__APP_CSS__*/", _asset("simple.css"))
        html = html.replace("/*__LEAFLET_JS__*/", _vendor("leaflet-1.9.4.min.js"))
        html = html.replace("/*__APP_JS__*/", _asset("engine.js") + "\n" + _asset("simple.js"))
    return html.replace("/*__DATA__*/", json.dumps(payload, separators=(",", ":")))


_DESCRIPTION = ("The flattest walking or cycling route between any two places i"
                "n Seattle, and every route between it and the shortest one.")

_FAVICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
<rect width="32" height="32" rx="7" fill="#0f8f6f"/>
<path d="M5 22 C10 22 11 12 16 12 S22 20 27 10" fill="none" stroke="#fff" stroke-width="3.2" stroke-linecap="round"/>
</svg>
"""


#: Where the route page opens before anyone types: the Fremont Troll to
#: Pike Place Market, across the Ship Canal and past Queen Anne hill.
#: Each entry is (label, place names to try in order, fallback), where the
#: fallback is a neighborhood access point or a (lon, lat) pair for a spot
#: the index does not carry.
_DEFAULT_TRIP = (
    ("Fremont Troll", ("Fremont Troll",), (-122.3473, 47.6510)),
    ("Pike Place Market", ("Pike Place Market",), (-122.3422, 47.6097)),
)


def _default_trip(places: dict | None, points: dict) -> list[dict]:
    out = []
    for label, names, fallback in _DEFAULT_TRIP:
        hit = None
        if places:
            for want in names:
                for i, n in enumerate(places["names"]):
                    if n == want:
                        hit = {"label": label, "lon": places["lon"][i], "lat": places["lat"][i]}
                        break
                if hit:
                    break
        if hit is None and isinstance(fallback, tuple):
            hit = {"label": label, "lon": fallback[0], "lat": fallback[1]}
        if hit is None and fallback in points.get("walk", {}):
            lon, lat = points["walk"][fallback]
            hit = {"label": label, "lon": lon, "lat": lat}
        if hit is None:
            return []
        out.append(hit)
    return out


def _labels(ctx) -> list[dict]:
    """Sparse neighborhood labels for the route page's base map."""
    nb = ctx.neighborhoods.to_crs("EPSG:4326")
    out = []
    for _, r in nb.iterrows():
        p = r.geometry.representative_point()
        out.append({"n": r["neighborhood"], "lon": round(p.x, 5), "lat": round(p.y, 5)})
    return out


def _write_route_page(ctx, graph: dict, pts: dict) -> Path:
    """Pack the place index and hillshade with the graph; write both forms."""
    import hashlib
    import shutil

    from .download import ADDRESSES_PARQUET, PLACES_PARQUET
    from .places import build_addresses, build_hillshade, build_places
    from .webgraph import bundle

    strings = {"geom": graph["geom"]}
    arrays = dict(graph["arrays"])
    places = None
    if PLACES_PARQUET.exists():
        places = build_places()
        strings["places"] = json.dumps(places, separators=(",", ":"))
    else:
        log.warning("no places parquet; the route page will search intersections only")
    if ADDRESSES_PARQUET.exists():
        addr = build_addresses()
        strings["addr_streets"] = json.dumps(addr["streets"], separators=(",", ":"))
        for k in ("street", "number", "lon", "lat"):
            arrays["addr_" + k] = addr[k]
        addr_meta = {"origin": addr["origin"], "step": 1e-5}
    else:
        addr_meta = None
    try:
        hillshade = build_hillshade()
    except Exception as exc:  # the page works without it, on streets alone
        log.warning("hillshade unavailable (%s)", exc)
        hillshade = None

    with step("bundling the route page payload", log):
        packed = bundle(dict(graph, arrays=arrays), strings)
    common = {
        "manifest": packed["manifest"], "meta": packed["meta"], "addr": addr_meta,
        "labels": _labels(ctx), "default": _default_trip(places, pts),
    }

    # one file that opens from disk: everything inline
    inline = dict(common, bundle=packed["b64"],
                  hillshade=hillshade and {"bounds": hillshade["bounds"],
                                           "data_uri": hillshade["data_uri"]})
    SIMPLE_HTML.write_text(_route_page_html(inline, linked=False), encoding="utf-8")

    # the static site: fetchable, cacheable files. Everything the page
    # references carries a content hash in its name, so a browser that cached
    # yesterday's app.js cannot run it against today's index.html.
    data_dir = SITE_DIR / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)
    for stale in SITE_DIR.glob("*-*.??*"):
        if re.match(r"^(app|leaflet)-[0-9a-f]{10}\.(js|css)$", stale.name):
            stale.unlink()

    def hashed(stem: str, ext: str, text: str) -> str:
        name = f"{stem}-{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}.{ext}"
        (SITE_DIR / name).write_text(text, encoding="utf-8")
        return name

    assets = {
        "app.css": hashed("app", "css", _asset("simple.css")),
        "app.js": hashed("app", "js", _asset("engine.js") + "\n" + _asset("simple.js")),
        "leaflet.css": hashed("leaflet", "css", _vendor("leaflet-1.9.4.css")),
        "leaflet.js": hashed("leaflet", "js", _vendor("leaflet-1.9.4.min.js")),
    }
    gz = packed["gz"]
    gz_name = f"graph-{hashlib.sha1(gz).hexdigest()[:10]}.bin.gz"
    (data_dir / gz_name).write_bytes(gz)
    linked = dict(common, bundle_url="data/" + gz_name, bundle_bytes=len(gz))
    if hillshade:
        png = hillshade["png"]
        png_name = f"hillshade-{hashlib.sha1(png).hexdigest()[:10]}.png"
        (data_dir / png_name).write_bytes(png)
        linked["hillshade"] = {"bounds": hillshade["bounds"], "url": "data/" + png_name}
    else:
        linked["hillshade"] = None
    SITE_INDEX.write_text(_route_page_html(linked, linked=True, assets=assets),
                          encoding="utf-8")
    for plain in ("app.css", "app.js", "leaflet.css", "leaflet.js"):
        (SITE_DIR / plain).unlink(missing_ok=True)
    (SITE_DIR / "favicon.svg").write_text(_FAVICON, encoding="utf-8")
    (SITE_DIR / ".nojekyll").write_text("", encoding="utf-8")
    # GitHub Pages reads the custom domain from here on branch deploys and
    # from Settings -> Pages on Actions deploys; shipping it covers both.
    # Without a domain there must be no CNAME, or Pages redirects to nothing.
    if SITE_DOMAIN:
        (SITE_DIR / "CNAME").write_text(SITE_DOMAIN + "\n", encoding="utf-8")
    else:
        (SITE_DIR / "CNAME").unlink(missing_ok=True)
    site_bytes = sum(f.stat().st_size for f in SITE_DIR.rglob("*") if f.is_file())
    log.info("wrote %s (%s) and the site in %s (%s)", SIMPLE_HTML.name,
             human_bytes(SIMPLE_HTML.stat().st_size), SITE_DIR.name, human_bytes(site_bytes))
    return SIMPLE_HTML


def _points(ctx) -> dict:
    """Neighborhood access points per mode, as {name: [lon, lat]}."""
    pts = {}
    for mode, gdf in ctx.points.items():
        g = gdf.to_crs("EPSG:4326")
        pts[mode] = {r["neighborhood"]: [round(r.geometry.x, 6),
                                         round(r.geometry.y, 6)]
                     for _, r in g.iterrows()}
    return pts


def make_route_page(ctx) -> Path:
    """Write the route finder (one file and the site) without the analysis
    layers the explorer needs."""
    from .webgraph import build_payload

    graph = build_payload(ctx.edges, ctx.directed)
    with step("writing the route page", log):
        return _write_route_page(ctx, graph, _points(ctx))


def make_interactive_map(ctx, corridors, passes, barriers, pairs_df=None,
                         arc_store=None, basins=None) -> Path:
    """Assemble and write the interactive map.

    ``pairs_df`` and ``arc_store`` are accepted for call-site compatibility
    but are no longer needed: the map routes for itself rather than looking
    up precomputed answers.
    """
    from .webgraph import build_payload, bundle

    with step("preparing interactive map layers", log):
        layers = build_layers(ctx, corridors, passes, barriers, basins)

    graph = build_payload(ctx.edges, ctx.directed)
    pts = _points(ctx)
    names = sorted(set(pts.get("walk", {})) | set(pts.get("bike", {})))

    with step("bundling and compressing the map payload", log):
        packed = bundle(graph, {
            "geom": graph["geom"],
            "layers": json.dumps(layers, separators=(",", ":")),
        })
    payload = {
        "manifest": packed["manifest"], "bundle": packed["b64"],
        "meta": packed["meta"], "points": pts, "neighborhood_names": names,
        "examples": _examples(pts),
    }

    with step("writing the explorer HTML", log):
        INTERACTIVE_HTML.parent.mkdir(parents=True, exist_ok=True)
        INTERACTIVE_HTML.write_text(_render(payload), encoding="utf-8")
    log.info("wrote %s (%s)", INTERACTIVE_HTML.name,
             human_bytes(INTERACTIVE_HTML.stat().st_size))
    with step("writing the route page", log):
        _write_route_page(ctx, graph, pts)
    return INTERACTIVE_HTML
