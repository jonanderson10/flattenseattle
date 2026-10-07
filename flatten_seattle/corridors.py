"""Discovery of San Francisco's low-elevation corridors.

The question is not "which streets are flat" -- thousands are -- but "which
flat streets does the city's geography force low-gradient traffic onto".  A
street earns corridor status by being *used*, repeatedly, by good flat routes
between different parts of the city, and by saving climbing when it is used.

Importance score
----------------
For each undirected edge we accumulate, over every ordered neighborhood pair
and every climb-averse objective:

``pair_count``
    number of distinct neighborhood pairs whose route uses the edge --
    raw betweenness over the pair set.

``neighborhood_span``
    number of *distinct neighborhoods* appearing as an endpoint of a route
    through the edge.  This separates a genuinely citywide corridor from a
    street that merely sits on many routes between the same two districts.

``climb_saved``
    sum over those routes of the climbing avoided versus the shortest path,
    apportioned to the edge by its share of the route's length.  An edge only
    scores here if the routes through it actually avoid hills.

``pareto_weight``
    usage on Pareto-efficient routes only, which filters out edges that are
    used solely by absurd detours.

``detour_efficiency``
    mean of ``climb_saved_per_extra_m`` over the routes using the edge -- how
    cheaply, in extra distance, those routes buy their flatness.

The composite score multiplies a usage term by a benefit term, so an edge
must be both well used and genuinely hill-avoiding:

    score = log1p(pair_count) * log1p(neighborhood_span)
          * (1 + climb_saved_norm) * (1 + pareto_share)

and edges whose own gradient disqualifies them as "flat" (average grade above
``corridor_max_avg_grade``, or a maximum above ``corridor_max_edge_grade``)
are excluded from corridor material regardless of usage -- otherwise the
unavoidable climbs *out* of a corridor get absorbed into it.

Merging
-------
High-scoring edges are grouped into connected components of the subgraph they
induce, then split by street name so that corridors are reportable in human
terms ("Duboce Ave - Steiner St - Waller St"), and components shorter than
``corridor_min_length_m`` are dropped.
"""
from __future__ import annotations

import collections

import numpy as np
import pandas as pd

from .config import ANALYSIS, OUTPUT_DIR, PROCESSED_DIR
from .utils import get_logger, progress, step

log = get_logger("flatten_seattle.corridors")

EDGE_SCORES_PARQUET = PROCESSED_DIR / "edge_corridor_scores.parquet"
CORRIDORS_GEOJSON = OUTPUT_DIR / "flat_corridors.geojson"
CORRIDORS_GPKG = OUTPUT_DIR / "flat_corridors.gpkg"
CORRIDORS_CSV = OUTPUT_DIR / "flat_corridors.csv"

#: Objectives whose routes count towards corridor importance. The shortest
#: path is deliberately excluded: it says nothing about avoiding hills.
CORRIDOR_PROFILES = ("min_climb", "grade_averse", "balanced")


def accumulate_edge_usage(graph, arc_store: dict, pairs_df: pd.DataFrame,
                          pareto_df: pd.DataFrame | None = None,
                          profiles=CORRIDOR_PROFILES) -> pd.DataFrame:
    """Aggregate route usage onto undirected edges for one mode."""
    mode = graph.mode
    arc_edge = graph.table["edge_id"].to_numpy()
    arc_len = graph.table["length_m"].to_numpy(dtype="float64")

    pair_count: dict[int, int] = collections.Counter()
    endpoints: dict[int, set] = collections.defaultdict(set)
    climb_saved: dict[int, float] = collections.defaultdict(float)
    eff_sum: dict[int, float] = collections.defaultdict(float)
    eff_n: dict[int, int] = collections.Counter()
    pareto_count: dict[int, int] = collections.Counter()

    lookup = pairs_df[pairs_df["mode"] == mode].set_index(
        ["profile", "origin", "destination"])

    seen_pairs: dict[int, set] = collections.defaultdict(set)

    with step(f"accumulating edge usage over neighborhood routes [{mode}]", log):
        for (pname, o, d), arcs in progress(arc_store.items(),
                                            total=len(arc_store),
                                            desc="  routes", unit="route"):
            if pname not in profiles or not arcs:
                continue
            try:
                row = lookup.loc[(pname, o, d)]
            except KeyError:
                continue
            saved = float(row.get("gain_saved_m", 0.0) or 0.0)
            eff = row.get("climb_saved_per_extra_m", np.nan)
            total_len = float(row.get("distance_m", 0.0) or 0.0)

            eids = arc_edge[arcs]
            lens = arc_len[arcs]
            share = lens / total_len if total_len > 0 else np.zeros_like(lens)
            for eid, sh in zip(eids, share):
                eid = int(eid)
                seen_pairs[eid].add((o, d))
                endpoints[eid].add(o); endpoints[eid].add(d)
                if saved > 0:
                    climb_saved[eid] += saved * sh
                if eff is not None and np.isfinite(eff):
                    eff_sum[eid] += float(eff); eff_n[eid] += 1

    for eid, s in seen_pairs.items():
        pair_count[eid] = len(s)

    if pareto_df is not None and not pareto_df.empty:
        log.info("  (Pareto usage is folded in via the frontier route set)")

    rows = []
    for eid, n in pair_count.items():
        rows.append({
            "edge_id": eid, "mode": mode,
            "pair_count": n,
            "neighborhood_span": len(endpoints[eid]),
            "climb_saved_m": climb_saved.get(eid, 0.0),
            "detour_efficiency": (eff_sum[eid] / eff_n[eid]) if eff_n[eid] else 0.0,
            "pareto_count": pareto_count.get(eid, 0),
        })
    df = pd.DataFrame(rows)
    log.info("  %d edges used by at least one climb-averse route", len(df))
    return df


def score_edges(usage: pd.DataFrame, edges, cfg=ANALYSIS) -> pd.DataFrame:
    """Combine usage terms into the corridor-importance score."""
    cols = ["edge_id", "name", "cls", "length_m", "max_abs_grade",
            "avg_grade_fwd", "cum_gain_fwd", "elev_min", "elev_max",
            "low_stress", "bike_facility", "geometry"]
    if "grade_reliable" in edges.columns:
        cols.append("grade_reliable")
    df = usage.merge(edges[cols], on="edge_id", how="left")
    if "grade_reliable" not in df.columns:
        df["grade_reliable"] = True

    # Only genuinely low-gradient street may be corridor material. The
    # maximum-grade test is applied only where the edge is long enough for
    # that maximum to mean anything; short stubs are judged on average grade,
    # so a 5 m DEM artefact cannot disqualify an otherwise flat block.
    df["flat_enough"] = (
        (df["avg_grade_fwd"].abs() <= cfg.corridor_max_avg_grade)
        & (~df["grade_reliable"].fillna(True)
           | (df["max_abs_grade"] <= cfg.corridor_max_edge_grade)))

    cs = df["climb_saved_m"].to_numpy(dtype="float64")
    denom = np.percentile(cs[cs > 0], 90) if (cs > 0).any() else 1.0
    df["climb_saved_norm"] = np.clip(cs / (denom or 1.0), 0, 4)

    pc = df["pair_count"].to_numpy(dtype="float64")
    ns = df["neighborhood_span"].to_numpy(dtype="float64")
    df["score"] = (np.log1p(pc) * np.log1p(ns)
                   * (1.0 + df["climb_saved_norm"].to_numpy()))
    df.loc[~df["flat_enough"], "score"] = 0.0
    return df


# --------------------------------------------------------------------------
# merging into named corridors
# --------------------------------------------------------------------------
def _clean_names(names) -> list[str]:
    """Drop NaN/None/blank street names (Arrow nulls arrive as float NaN)."""
    return [n for n in names if isinstance(n, str) and n.strip()]


def _canonical_name(names) -> str:
    """Human-readable corridor label from its constituent street names."""
    counts = collections.Counter(_clean_names(names))
    if not counts:
        return "unnamed corridor"
    top = [n for n, _ in counts.most_common(4)]
    return " - ".join(top)


def _bridge_gaps(core: set[int], candidate_edges, edges, max_gap_edges: int = 3,
                 max_gap_m: float = 220.0) -> set[int]:
    """Close short gaps between high-scoring components along the same street.

    Thresholding a continuous corridor edge-by-edge inevitably punches holes
    in it: one block of Valencia Street may fall a hair below the cut and
    split the corridor in two.  Any chain of at most ``max_gap_edges``
    eligible edges (short, flat, and already scoring at least a fraction of
    the cut) that joins two *different* core components is therefore adopted
    into the corridor.  This is a morphological closing on the graph, and it
    only ever connects material that already qualifies as flat.
    """
    import networkx as nx

    topo = edges.set_index("edge_id")[["u", "v", "length_m"]]
    core_g = nx.Graph()
    for eid in core:
        r = topo.loc[eid]
        core_g.add_edge(r["u"], r["v"], edge_id=int(eid))
    comp_of: dict = {}
    for ci, comp in enumerate(nx.connected_components(core_g)):
        for n in comp:
            comp_of[n] = ci
    if len(set(comp_of.values())) <= 1:
        return set()

    gap_g = nx.Graph()
    for eid in candidate_edges:
        if eid in core:
            continue
        r = topo.loc[eid]
        gap_g.add_edge(r["u"], r["v"], edge_id=int(eid), weight=float(r["length_m"]))

    added: set[int] = set()
    core_nodes = [n for n in comp_of if n in gap_g]
    seen_pairs = set()
    for src in core_nodes:
        # bounded BFS through gap-eligible edges only
        lengths, paths = nx.single_source_dijkstra(
            gap_g, src, cutoff=max_gap_m, weight="weight")
        for dst, path in paths.items():
            if dst not in comp_of or comp_of[dst] == comp_of[src]:
                continue
            if len(path) - 1 > max_gap_edges:
                continue
            key = tuple(sorted((comp_of[src], comp_of[dst])))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            for a, b in zip(path[:-1], path[1:]):
                added.add(int(gap_g[a][b]["edge_id"]))
    return added


def _merge_same_street(records: list[dict], max_join_m: float = 600.0) -> list[dict]:
    """Fuse components that are clearly the same named street.

    After gap closing, a long corridor can still arrive as two pieces when the
    interruption is a large intersection or a park crossing.  Components whose
    dominant street name matches and whose geometries lie within
    ``max_join_m`` of each other describe one corridor to a reader, so they
    are reported as one.
    """
    from shapely.ops import unary_union

    def dominant(rec):
        names = [n.strip() for n in rec["street_names"].split(";") if n.strip()]
        return rec["corridor_name"].split(" - ")[0] if names else ""

    groups: list[list[dict]] = []
    for rec in records:
        key = dominant(rec)
        placed = False
        if key:
            for grp in groups:
                if dominant(grp[0]) != key:
                    continue
                if any(rec["geometry"].distance(o["geometry"]) <= max_join_m
                       for o in grp):
                    grp.append(rec); placed = True; break
        if not placed:
            groups.append([rec])

    merged = []
    for grp in groups:
        if len(grp) == 1:
            merged.append(grp[0]); continue
        total_len = sum(g["length_m"] for g in grp)
        w = [g["length_m"] / total_len for g in grp]
        base = dict(grp[0])
        base.update({
            "geometry": unary_union([g["geometry"] for g in grp]),
            "n_edges": sum(g["n_edges"] for g in grp),
            "length_m": total_len,
            "length_km": total_len / 1000.0,
            "total_score": sum(g["total_score"] for g in grp),
            "mean_score": sum(g["mean_score"] * wi for g, wi in zip(grp, w)),
            "pair_count_max": max(g["pair_count_max"] for g in grp),
            "neighborhood_span": max(g["neighborhood_span"] for g in grp),
            "climb_saved_m": sum(g["climb_saved_m"] for g in grp),
            "mean_abs_grade": sum(g["mean_abs_grade"] * wi for g, wi in zip(grp, w)),
            "max_grade": max(g["max_grade"] for g in grp),
            "elev_min_m": min(g["elev_min_m"] for g in grp),
            "elev_max_m": max(g["elev_max_m"] for g in grp),
            "gain_per_km": sum(g["gain_per_km"] * wi for g, wi in zip(grp, w)),
            "low_stress_share": sum(g["low_stress_share"] * wi
                                    for g, wi in zip(grp, w)),
            "street_names": "; ".join(sorted({n for g in grp
                                              for n in g["street_names"].split("; ")
                                              if n})),
            "n_pieces": len(grp),
        })
        base["elev_range_m"] = base["elev_max_m"] - base["elev_min_m"]
        merged.append(base)
    return merged


def merge_corridors(scored: pd.DataFrame, edges, cfg=ANALYSIS,
                    quantile: float | None = None, grow_factor: float = 0.45):
    """Group high-scoring edges into contiguous, named corridors."""
    import geopandas as gpd
    import networkx as nx
    from shapely.ops import linemerge, unary_union

    quantile = cfg.corridor_score_quantile if quantile is None else quantile
    pos = scored[scored["score"] > 0]
    if pos.empty:
        return gpd.GeoDataFrame(columns=["corridor_id", "geometry"],
                                geometry="geometry", crs=edges.crs)
    cut = float(np.quantile(pos["score"], quantile))
    core = set(scored.loc[scored["score"] >= cut, "edge_id"].astype(int))
    eligible = set(scored.loc[(scored["score"] >= cut * grow_factor)
                              & scored["flat_enough"], "edge_id"].astype(int))
    log.info("corridor threshold: score >= %.2f (q%.2f) keeps %d core edges",
             cut, quantile, len(core))

    bridged = _bridge_gaps(core, eligible, edges)
    if bridged:
        log.info("  gap closing adopted %d additional edges", len(bridged))
    keep_ids = core | bridged
    top = scored[scored["edge_id"].isin(keep_ids)].copy()
    log.info("  corridor material: %d edges, %.0f km",
             len(top), top["length_m"].sum() / 1000)

    topo = edges.set_index("edge_id")[["u", "v"]]
    top = top.join(topo, on="edge_id")

    g = nx.Graph()
    for eid, u, v in zip(top["edge_id"], top["u"], top["v"]):
        g.add_edge(u, v, edge_id=int(eid))

    records = []
    for comp_id, comp in enumerate(nx.connected_components(g)):
        sub = g.subgraph(comp)
        eids = [d["edge_id"] for _, _, d in sub.edges(data=True)]
        part = top[top["edge_id"].isin(eids)]
        length = float(part["length_m"].sum())
        if length < cfg.corridor_min_length_m:
            continue
        geoms = list(part.geometry.values)
        try:
            geom = linemerge(geoms)
        except Exception:
            geom = unary_union(geoms)
        names = part["name"].tolist()
        w = part["length_m"].to_numpy(dtype="float64")
        records.append({
            "corridor_id": comp_id,
            "corridor_name": _canonical_name(names),
            "street_names": "; ".join(sorted(set(_clean_names(names)))),
            "n_edges": len(part),
            "length_m": length,
            "length_km": length / 1000.0,
            "mean_score": float(part["score"].mean()),
            "total_score": float(part["score"].sum()),
            "pair_count_max": int(part["pair_count"].max()),
            "neighborhood_span": int(part["neighborhood_span"].max()),
            "climb_saved_m": float(part["climb_saved_m"].sum()),
            "mean_abs_grade": float((part["avg_grade_fwd"].abs() * w).sum() / w.sum()),
            "max_grade": float(part["max_abs_grade"].max()),
            "elev_min_m": float(part["elev_min"].min()),
            "elev_max_m": float(part["elev_max"].max()),
            "elev_range_m": float(part["elev_max"].max() - part["elev_min"].min()),
            "gain_per_km": float(part["cum_gain_fwd"].sum() / (length / 1000.0)),
            "low_stress_share": float((part["low_stress"] * w).sum() / w.sum()),
            "mode": part["mode"].iloc[0],
            "n_pieces": 1,
            "geometry": geom,
        })

    records = _merge_same_street(records)
    records = [r for r in records if r["length_m"] >= cfg.corridor_min_length_m]
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=edges.crs)
    if len(gdf):
        gdf = gdf.sort_values("total_score", ascending=False).reset_index(drop=True)
        gdf["corridor_id"] = np.arange(len(gdf))
    log.info("merged into %d corridors >= %.0f m", len(gdf),
             cfg.corridor_min_length_m)
    return gdf


def attribute_corridors(corridors, neighborhoods):
    """Add the list of neighborhoods each corridor passes through."""
    import geopandas as gpd
    if corridors.empty:
        corridors["neighborhoods"] = []
        return corridors
    joined = gpd.sjoin(corridors[["corridor_id", "geometry"]],
                       neighborhoods[["neighborhood", "geometry"]],
                       how="left", predicate="intersects")
    nb = (joined.groupby("corridor_id")["neighborhood"]
          .apply(lambda s: "; ".join(sorted({x for x in s if isinstance(x, str)}))))
    corridors = corridors.merge(nb.rename("neighborhoods"),
                                left_on="corridor_id", right_index=True, how="left")
    return corridors


def corridor_endpoints(corridors):
    """Add each corridor's two extreme endpoints, in lon/lat.

    A corridor's extent is characterised by its two most widely separated
    terminal points, which for a branching corridor is more informative than
    the first and last vertex of whatever order the geometry happens to be
    in.
    """
    from pyproj import Transformer
    from shapely.geometry import MultiLineString

    if corridors.empty:
        for c in ("end_a_lon", "end_a_lat", "end_b_lon", "end_b_lat"):
            corridors[c] = []
        return corridors

    tr = Transformer.from_crs(corridors.crs, "EPSG:4326", always_xy=True)
    rows = []
    for _, r in corridors.iterrows():
        g = r.geometry
        if g is None or g.is_empty:
            rows.append((np.nan,) * 4)
            continue
        if isinstance(g, MultiLineString):
            pts = [c for line in g.geoms
                   for c in (line.coords[0], line.coords[-1])]
        else:
            pts = [g.coords[0], g.coords[-1]]
        arr = np.asarray(pts)[:, :2]
        d2 = ((arr[:, None, :] - arr[None, :, :]) ** 2).sum(-1)
        i, j = np.unravel_index(np.argmax(d2), d2.shape)
        a = tr.transform(*arr[i])
        b = tr.transform(*arr[j])
        rows.append((round(a[0], 6), round(a[1], 6),
                     round(b[0], 6), round(b[1], 6)))
    corridors = corridors.copy()
    corridors["end_a_lon"] = [r[0] for r in rows]
    corridors["end_a_lat"] = [r[1] for r in rows]
    corridors["end_b_lon"] = [r[2] for r in rows]
    corridors["end_b_lat"] = [r[3] for r in rows]
    return corridors


def run_corridor_analysis(graphs: dict, arc_store: dict, pairs_df: pd.DataFrame,
                          edges, neighborhoods, force: bool = False):
    """Full corridor pipeline across modes."""
    import geopandas as gpd

    if CORRIDORS_GPKG.exists() and EDGE_SCORES_PARQUET.exists() and not force:
        log.info("cached corridors")
        return gpd.read_file(CORRIDORS_GPKG), pd.read_parquet(EDGE_SCORES_PARQUET)

    all_scores, all_corridors = [], []
    for mode, graph in graphs.items():
        usage = accumulate_edge_usage(graph, arc_store[mode], pairs_df)
        scored = score_edges(usage, edges)
        all_scores.append(scored)
        cor = merge_corridors(scored, edges)
        cor = attribute_corridors(cor, neighborhoods)
        cor = corridor_endpoints(cor)
        all_corridors.append(cor)

    scores = pd.concat(all_scores, ignore_index=True)
    corridors = pd.concat(all_corridors, ignore_index=True)
    corridors["corridor_uid"] = [f"{m}-{i}" for i, m in
                                 zip(range(len(corridors)), corridors["mode"])]

    scores.drop(columns=["geometry"]).to_parquet(EDGE_SCORES_PARQUET)
    corridors.to_file(CORRIDORS_GPKG, driver="GPKG")
    corridors.to_crs("EPSG:4326").to_file(CORRIDORS_GEOJSON, driver="GeoJSON")
    corridors.drop(columns=["geometry"]).to_csv(CORRIDORS_CSV, index=False)
    log.info("wrote %s, %s, %s", CORRIDORS_GPKG.name, CORRIDORS_GEOJSON.name,
             CORRIDORS_CSV.name)
    return corridors, scores
