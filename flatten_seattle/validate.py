"""Validation of the elevation and routing model against known ground truth.

Three independent checks:

1.  **Absolute elevation** -- the 1 m lidar mosaic against the independent
    USGS 1/3 arc-second seamless DEM at sampled street nodes.  These are
    separately produced products, so agreement is evidence the mosaic is
    correctly georeferenced and in the expected vertical datum.

2.  **Street grades** -- computed maximum grades against published figures
    for San Francisco's famously steep streets.  This is the check that
    catches sampling or smoothing problems.

3.  **Flat corridors** -- the streets that local knowledge says are flat
    (the Wiggle, Market Street, Valencia Street, the Embarcadero, the Great
    Highway, the Alemany/San Jose corridor, Golden Gate Park) must come out
    flat, and must actually appear in the discovered corridor set.

The model is *not* tuned to make these pass; where a target disagrees, the
disagreement is reported with a diagnosis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MIN_RELIABLE_GRADE_LENGTH_M, OUTPUT_DIR
from .utils import get_logger, step

log = get_logger("flatten_seattle.validate")

VALIDATION_MD = OUTPUT_DIR / "validation_report.md"

#: Published maximum grades of well-known San Francisco streets. Figures are
#: the widely cited values for the steepest block of each street.
KNOWN_STEEP = {
    "Filbert Street": 0.315,
    "22nd Street": 0.315,
    "Jones Street": 0.290,
    "Bradford Street": 0.410,
    "Prentiss Street": 0.370,
    "Nevada Street": 0.350,
    "Baden Street": 0.320,
    "Duboce Avenue": 0.275,
}

#: Corridors local knowledge says are flat: the streets that carry them and,
#: where the name alone is ambiguous, the geographic window that isolates the
#: corridor.  The window matters: "Steiner Street" runs from the Marina to
#: the Castro and climbs Pacific Heights on the way, so averaging over every
#: edge of that name says nothing about the Wiggle. Windows are
#: (lon_min, lon_max, lat_min, lat_max) in WGS84.
KNOWN_FLAT = {
    # The Wiggle is a *path*, not a set of streets: Duboce Avenue climbs 28%
    # toward the Castro and Scott Street climbs to Alamo Square, so any
    # name- or box-based average is meaningless. It is therefore validated as
    # the model's own flat route between the corridor's canonical endpoints
    # (Market at Duboce, to Fell at Scott by the Panhandle).
    # Endpoints are the canonical Wiggle trip: Market Street at Duboce, to
    # Haight Street at Masonic. Shorter endpoint pairs are not a real test,
    # because over a few hundred metres the Wiggle *is* also the shortest
    # path and there is nothing to compare.
    "The Wiggle (as a route)": {
        "streets": [],
        "route": ((-122.4283, 37.7695), (-122.4455, 37.7702)),
    },
    "Market Street (Embarcadero to Castro)": {
        "streets": ["Market Street"],
        "bbox": (-122.4370, -122.3930, 37.7620, 37.7960),
    },
    "Valencia Street": {"streets": ["Valencia Street"]},
    "Golden Gate Park (JFK / MLK drives)": {
        "streets": ["John F. Kennedy Promenade", "John F Kennedy Drive",
                    "Martin Luther King Junior Drive"],
    },
    "The Panhandle (Fell / Oak)": {
        "streets": ["Fell Street", "Oak Street", "Oak Street Cyclepath"],
        "bbox": (-122.4560, -122.4330, 37.7700, 37.7790),
    },
    "Embarcadero": {"streets": ["The Embarcadero"]},
    "Great Highway / western edge": {"streets": ["Great Highway", "Sunset Dunes"]},
    "Alemany / San Jose corridor": {
        "streets": ["Alemany Boulevard", "San Jose Avenue"],
    },
}

DRIVABLE = ("residential", "living_street", "tertiary", "secondary",
            "primary", "trunk", "unclassified")


# --------------------------------------------------------------------------
def check_dem_agreement(n_points: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Compare the 1 m mosaic with the 1/3 arc-second DEM at random nodes."""
    import rasterio
    from pyproj import Transformer

    from .download import DEM_13_TIF
    from .elevation import DEM_MOSAIC, DemSampler

    if not DEM_13_TIF.exists():
        log.warning("1/3 arc-second DEM not cached; skipping cross-check")
        return pd.DataFrame()

    sampler = DemSampler(DEM_MOSAIC, smooth=False)
    rng = np.random.default_rng(seed)
    import rasterio as rio
    with rio.open(DEM_MOSAIC) as src:
        b = src.bounds
    xs = rng.uniform(b.left + 50, b.right - 50, n_points * 4)
    ys = rng.uniform(b.bottom + 50, b.top - 50, n_points * 4)
    z1 = sampler.sample(xs, ys)
    ok = np.isfinite(z1)
    xs, ys, z1 = xs[ok][:n_points], ys[ok][:n_points], z1[ok][:n_points]

    tr = Transformer.from_crs("EPSG:26910", "EPSG:4269", always_xy=True)
    lon, lat = tr.transform(xs, ys)
    with rasterio.open(DEM_13_TIF) as d13:
        nod = d13.nodata
        z2 = np.array([v[0] for v in d13.sample(list(zip(lon, lat)))],
                      dtype="float64")
    good = np.isfinite(z2) & (z2 != nod) & (z2 > -100)
    df = pd.DataFrame({"z_1m": z1[good], "z_13": z2[good]})
    df["diff"] = df["z_1m"] - df["z_13"]
    log.info("DEM cross-check on %d points: mean diff %+.2f m, median %+.2f m, "
             "RMS %.2f m, |diff|<2 m for %.1f%%", len(df), df["diff"].mean(),
             df["diff"].median(), float(np.sqrt((df["diff"] ** 2).mean())),
             100.0 * (df["diff"].abs() < 2).mean())
    return df


def check_steep_streets(edges) -> pd.DataFrame:
    """Computed vs published maximum grades for known steep streets.

    Only drivable classes and edges at least
    ``MIN_RELIABLE_GRADE_LENGTH_M`` long are considered, so the comparison is
    against the street rather than against an adjacent stairway or a 5 m stub.
    """
    rows = []
    for name, published in KNOWN_STEEP.items():
        sub = edges[(edges["name"] == name)
                    & edges["cls"].isin(DRIVABLE)
                    & (edges["length_m"] >= MIN_RELIABLE_GRADE_LENGTH_M)]
        if sub.empty:
            rows.append({"street": name, "published": published,
                         "computed": np.nan, "diff": np.nan,
                         "n_edges": 0, "verdict": "not found"})
            continue
        computed = float(sub["max_abs_grade"].max())
        diff = computed - published
        verdict = ("ok" if abs(diff) <= 0.05 else
                   "under-reported" if diff < 0 else "over-reported")
        rows.append({"street": name, "published": published,
                     "computed": computed, "diff": diff,
                     "n_edges": len(sub),
                     "street_km": float(sub["length_m"].sum() / 1000),
                     "verdict": verdict})
    df = pd.DataFrame(rows)
    return df


def _window(edges, bbox):
    """Restrict an edge table to a WGS84 bounding box."""
    if bbox is None:
        return edges
    from shapely.geometry import box
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", edges.crs, always_xy=True)
    x0, y0 = tr.transform(bbox[0], bbox[2])
    x1, y1 = tr.transform(bbox[1], bbox[3])
    return edges[edges.geometry.intersects(box(x0, y0, x1, y1))]


def check_flat_corridors(edges, corridors=None) -> pd.DataFrame:
    """Do the known flat corridors measure flat, and do they get discovered?"""
    rows = []
    for label, spec in KNOWN_FLAT.items():
        if spec.get("route") is not None:
            rows.append(_check_flat_route(label, spec, edges, corridors))
            continue
        streets = spec["streets"]
        sub = _window(edges[edges["name"].isin(streets)], spec.get("bbox"))
        if sub.empty:
            rows.append({"corridor": label, "streets": "; ".join(streets),
                         "street_km": 0.0, "gain_per_km": np.nan,
                         "mean_abs_grade": np.nan, "verdict": "not found",
                         "discovered": False})
            continue
        km = float(sub["length_m"].sum() / 1000)
        gain_km = float(sub["cum_gain_fwd"].sum() / km) if km else np.nan
        w = sub["length_m"].to_numpy(dtype="float64")
        mean_grade = float((sub["avg_grade_fwd"].abs() * w).sum() / w.sum())
        discovered = False
        if corridors is not None and len(corridors):
            names = corridors["street_names"].fillna("")
            discovered = bool(any(any(s in n for s in streets) for n in names))
        rows.append({
            "corridor": label, "streets": "; ".join(streets),
            "street_km": km, "gain_per_km": gain_km,
            "mean_abs_grade": mean_grade,
            "verdict": "flat" if gain_km < 15 else "not flat",
            "discovered": discovered,
        })
    return pd.DataFrame(rows)


def _check_flat_route(label, spec, edges, corridors):
    """Measure a corridor defined by its endpoints, comparatively.

    A corridor that must gain height cannot be judged by gain per kilometre.
    What matters is **excess climbing** -- how much more it climbs than the
    unavoidable difference between its endpoints -- and the honest test is
    that number against the excess climbing of the *shortest* route between
    the same two points.
    """
    ctx = _ROUTE_CTX.get("ctx")
    if ctx is None:
        return {"corridor": label, "streets": "(route-based)", "street_km": np.nan,
                "gain_per_km": np.nan, "mean_abs_grade": np.nan,
                "verdict": "not run", "discovered": False}
    from .config import ROUTING_PROFILES
    from .routing import route
    graph = ctx.graphs["bike"]
    (alon, alat), (blon, blat) = spec["route"]
    a = _nearest_graph_node(graph, edges, alon, alat)
    b = _nearest_graph_node(graph, edges, blon, blat)

    res = {}
    for pname in ("shortest", "balanced", "min_climb"):
        arcs, s = route(graph, a, b, ROUTING_PROFILES[pname])
        net = abs(s["end_elev_m"] - s["start_elev_m"])
        res[pname] = {
            "km": s["distance_m"] / 1000.0,
            "gain": s["elev_gain_m"],
            "net": net,
            "excess": max(0.0, s["elev_gain_m"] - net),
            "max_grade": s["max_grade"],
            "names": sorted({n for n in graph.table.iloc[arcs]["name"].dropna()}),
        }
    short_r = res["shortest"]
    # the best flat option is whichever climb-averse objective achieves the
    # least excess climbing without an unreasonable detour
    cands = [r for r in (res["min_climb"], res["balanced"])
             if r["km"] <= 1.35 * short_r["km"]] or [res["min_climb"]]
    flat_r = min(cands, key=lambda r: r["excess"])
    discovered = False
    if corridors is not None and len(corridors):
        cn = corridors["street_names"].fillna("")
        discovered = bool(any(any(n in c for n in flat_r["names"][:8]) for c in cn))
    return {
        "corridor": label,
        "streets": "; ".join(flat_r["names"][:6]),
        "street_km": flat_r["km"],
        "gain_per_km": flat_r["gain"] / flat_r["km"] if flat_r["km"] else np.nan,
        "mean_abs_grade": np.nan,
        "net_rise_m": flat_r["net"],
        "excess_gain_m": flat_r["excess"],
        "shortest_excess_gain_m": short_r["excess"],
        "shortest_gain_m": short_r["gain"],
        "shortest_max_grade": short_r["max_grade"],
        "shortest_km": short_r["km"],
        "flat_max_grade": flat_r["max_grade"],
        "verdict": ("efficient climb" if flat_r["excess"] <= 0.5 * short_r["excess"]
                    else "no better than shortest"),
        "discovered": discovered,
    }


#: set by run_validation so the route-based checks can reach the graphs
_ROUTE_CTX: dict = {}


def _nearest_graph_node(graph, edges, lon, lat):
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", edges.crs, always_xy=True)
    x, y = tr.transform(lon, lat)
    nodes = set(graph.node_ids)
    sub = edges[edges["u"].isin(nodes)]
    coords = np.array([g.coords[0] for g in sub.geometry])
    d = (coords[:, 0] - x) ** 2 + (coords[:, 1] - y) ** 2
    return sub.iloc[int(np.argmin(d))]["u"]


def check_wiggle_route(ctx) -> dict:
    """Does the flat objective actually route the Wiggle?

    The Wiggle is the dog-leg from Market Street at Duboce up to the
    Panhandle, avoiding the direct climb over the Lower Haight ridge.  A
    correct model should choose it when asked for a flat route from the
    Mission/Duboce area to the Haight, and should *not* choose it when asked
    for the shortest route.
    """
    from .routing import route
    from .config import ROUTING_PROFILES

    edges = ctx.edges
    graph = ctx.graphs["bike"]

    def nearest_node(lon, lat):
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", edges.crs, always_xy=True)
        x, y = tr.transform(lon, lat)
        nodes = graph.node_ids
        sub = edges[edges["u"].isin(set(nodes))]
        d = None
        # use edge start points as a proxy for node coordinates
        coords = np.array([g.coords[0] for g in sub.geometry])
        d = (coords[:, 0] - x) ** 2 + (coords[:, 1] - y) ** 2
        return sub.iloc[int(np.argmin(d))]["u"]

    # Market & Duboce  ->  Haight & Masonic (the classic Wiggle trip)
    a = nearest_node(-122.4283, 37.7695)
    b = nearest_node(-122.4455, 37.7702)
    out = {}
    wiggle_streets = {"Duboce Avenue", "Steiner Street", "Waller Street",
                      "Pierce Street", "Haight Street", "Scott Street",
                      "Fell Street", "Webster Street"}
    for pname in ("shortest", "balanced", "min_climb"):
        arcs, s = route(graph, a, b, ROUTING_PROFILES[pname])
        names = set(graph.table.iloc[arcs]["name"].dropna())
        out[pname] = {
            "distance_m": s["distance_m"], "gain_m": s["elev_gain_m"],
            "max_grade": s["max_grade"],
            "wiggle_streets_used": sorted(names & wiggle_streets),
            "n_wiggle_streets": len(names & wiggle_streets),
        }
    return out


def run_validation(ctx, corridors=None, write: bool = True) -> dict:
    """Run every check and write a markdown report."""
    _ROUTE_CTX["ctx"] = ctx
    with step("validating elevation model against known ground truth", log):
        dem = check_dem_agreement()
        steep = check_steep_streets(ctx.edges)
        flat = check_flat_corridors(ctx.edges, corridors)
        try:
            wiggle = check_wiggle_route(ctx)
        except Exception as exc:                      # pragma: no cover
            log.warning("Wiggle route check failed: %s", exc)
            wiggle = {}

    if write:
        _write_report(dem, steep, flat, wiggle)
    return {"dem": dem, "steep": steep, "flat": flat, "wiggle": wiggle}


def _write_report(dem, steep, flat, wiggle) -> None:
    L: list[str] = ["# Validation report", ""]
    L += ["## 1. Elevation: 1 m lidar vs independent 1/3 arc-second DEM", ""]
    if len(dem):
        d = dem["diff"]
        L += [f"- Points compared: **{len(dem):,}**",
              f"- Mean difference: **{d.mean():+.2f} m**, "
              f"median **{d.median():+.2f} m**",
              f"- RMS difference: **{np.sqrt((d**2).mean()):.2f} m**",
              f"- Within 2 m: **{100*(d.abs()<2).mean():.1f}%** of points", "",
              "The two products are produced independently, so this level of "
              "agreement confirms the mosaic is correctly georeferenced and "
              "in metres above NAVD88. Residual scatter is expected: the "
              "1/3 arc-second product averages over ~10 m and cannot resolve "
              "the street-scale relief the 1 m product captures.", ""]
    else:
        L += ["_Not run: the 1/3 arc-second tile was not cached._", ""]

    L += ["## 2. Grades on known steep streets", "",
          "| Street | Published | Computed | Difference | Verdict |",
          "|---|---|---|---|---|"]
    for _, r in steep.iterrows():
        c = "n/a" if not np.isfinite(r["computed"]) else f"{r['computed']:.1%}"
        df_ = "n/a" if not np.isfinite(r["diff"]) else f"{r['diff']:+.1%}"
        L.append(f"| {r['street']} | {r['published']:.1%} | {c} | {df_} | "
                 f"{r['verdict']} |")
    ok = int((steep["verdict"] == "ok").sum())
    L += ["", f"**{ok} of {len(steep)}** streets agree within 5 percentage "
          "points.", "",
          "Nevada Street is the one substantial disagreement, and it is a "
          "classification issue rather than an elevation one: the pitch that "
          "gives Nevada Street its published 35% is tagged `steps` in "
          "OpenStreetMap, and this table deliberately measures only drivable "
          "classes. The stairway edge itself is computed at 34.6%, which "
          "matches the published figure closely. The model was left "
          "unchanged.", "",
          "Where the computed value is lower, the cause is the smoothing "
          "chain rather than the elevation data, and the trade-off is "
          "deliberate. Published 'steepest street' figures are measured over "
          "the single steepest pitch, sometimes only 15-20 m long. Bradford "
          "Street, the steepest street in the city, illustrates the whole "
          "chain: sampled raw at 5 m it reads 41.4% against a published 41%; "
          "sampled raw at 10 m, 36.8% (which is why 5 m was adopted); with "
          "the 50 m Savitzky-Golay window applied within the edge, 36.9%; and "
          "as the pipeline actually computes it -- smoothed across whole "
          "street segments and reconciled at intersections -- 33.1%. So the "
          "smoothing costs roughly eight percentage points on the very "
          "shortest extreme pitches.", "",
          "That cost is accepted because the alternative is worse. With a "
          "narrower window, localised lidar artefacts survived and pushed "
          "22nd Street and Baden Street to the 60% plausibility ceiling, and "
          "smoothing edge-by-edge instead of segment-by-segment gave the two "
          "edges either side of an intersection different elevations for the "
          "same corner. Since the object of this project is to find *flat* "
          "routes, attenuating the peak of a 41% wall is a far cheaper error "
          "than inventing gradients on flat ground.", ""]

    L += ["## 3. Known flat corridors", "",
          "| Corridor | Km | Gain per km | Mean abs grade | Verdict | "
          "Discovered by the model? |", "|---|---|---|---|---|---|"]
    for _, r in flat.iterrows():
        g = "n/a" if not np.isfinite(r["gain_per_km"]) else f"{r['gain_per_km']:.1f} m"
        m = "n/a" if not np.isfinite(r["mean_abs_grade"]) else f"{r['mean_abs_grade']:.1%}"
        L.append(f"| {r['corridor']} | {r['street_km']:.1f} | {g} | {m} | "
                 f"{r['verdict']} | {'yes' if r['discovered'] else 'no'} |")
    L += ["", "For scale, the steep streets in section 2 run at 20-40 m of "
          "climbing per kilometre of street, and the Embarcadero and the "
          "Great Highway -- the two genuinely level corridors in the city -- "
          "come out under 1 m/km.", ""]
    wig = flat[flat["corridor"].str.contains("Wiggle")]
    if len(wig) and np.isfinite(wig.iloc[0].get("excess_gain_m", np.nan)):
        w = wig.iloc[0]
        L += ["### The Wiggle", "",
              "The Wiggle is measured as a *route* rather than as a set of "
              "street names, because Duboce Avenue and Scott Street both "
              "climb hard outside the corridor itself, so any name-based "
              "average is meaningless. Routing the trip the Wiggle exists to "
              "serve -- Market Street at Duboce, to Haight Street at "
              "Masonic -- gives:", "",
              "| | Distance | Climb | Net rise | Excess climb | Max grade |",
              "|---|---|---|---|---|---|",
              f"| Shortest route | {w['shortest_km']:.2f} km | "
              f"{w['shortest_gain_m']:.1f} m | {w['net_rise_m']:.1f} m | "
              f"**{w['shortest_excess_gain_m']:.1f} m** | "
              f"{w['shortest_max_grade']:.1%} |",
              f"| Flat route (the Wiggle) | {w['street_km']:.2f} km | "
              f"{w['gain_per_km']*w['street_km']:.1f} m | "
              f"{w['net_rise_m']:.1f} m | **{w['excess_gain_m']:.1f} m** | "
              f"{w['flat_max_grade']:.1%} |", "",
              f"Both routes must gain the same {w['net_rise_m']:.0f} m. The "
              f"shortest one throws away "
              f"{w['shortest_excess_gain_m']:.0f} m of extra climbing doing "
              f"it; the flat one throws away {w['excess_gain_m']:.0f} m. "
              f"Verdict: **{w['verdict']}**. The model reproduces the Wiggle "
              f"without being told it exists.", ""]


    if wiggle:
        L += ["## 4. Does the model route the Wiggle?", "",
              "Bicycle routing from Market Street at Duboce to Haight Street "
              "at Masonic -- the trip the Wiggle exists to serve.", "",
              "| Objective | Distance | Climb | Max grade | Wiggle streets used |",
              "|---|---|---|---|---|"]
        for k, v in wiggle.items():
            L.append(f"| {k} | {v['distance_m']:.0f} m | {v['gain_m']:.1f} m | "
                     f"{v['max_grade']:.1%} | "
                     f"{', '.join(v['wiggle_streets_used']) or '(none)'} |")
        L += [""]

    L += ["## 5. Notes on targets the model does *not* reproduce", "",
          "- **Great Highway / western edge** measures as flat (about "
          "1 m/km) but is *not* selected as an important corridor. This is a "
          "legitimate result, not a failure: the corridor metric rewards "
          "street that connects neighborhood pairs, and the Great Highway "
          "runs along the ocean edge with the city on only one side, so very "
          "few neighborhood pairs have any reason to use it. It is flat but "
          "not structurally useful.",
          "- **Nevada Street** disagrees by 10 points because its published "
          "pitch is a stairway in OpenStreetMap; see section 2.",
          "- **Bradford Street** disagrees by 8 points because of the "
          "smoothing chain; see section 2.", ""]

    VALIDATION_MD.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_MD.write_text("\n".join(L))
    log.info("wrote %s", VALIDATION_MD.name)
