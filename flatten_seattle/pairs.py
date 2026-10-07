"""Neighborhood-to-neighborhood routing, the results matrix and Pareto fronts.

All **ordered** pairs are computed, not just unordered ones: climbing is
direction dependent, so Bernal Heights -> Marina and Marina -> Bernal Heights
are different problems with different answers.

One Dijkstra per origin yields every destination at once, so the whole
36 x 35 matrix under four objectives and two modes costs a few hundred
shortest-path solves.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from .config import (GRADE_THRESHOLDS, OUTPUT_DIR, PARETO_LAMBDA_SWEEP,
                     PROCESSED_DIR, ROUTING_PROFILES, with_scale)
from .routing import (RouteGraph, arcs_from_predecessors, shortest_paths,
                      summarise_route)
from .utils import get_logger, progress, step

log = get_logger("flatten_seattle.pairs")

_TH = [int(t * 100) for t in GRADE_THRESHOLDS]

PAIRS_PARQUET = PROCESSED_DIR / "neighborhood_pairs.parquet"
PAIRS_CSV = OUTPUT_DIR / "neighborhood_pairs.csv"
PARETO_PARQUET = PROCESSED_DIR / "pareto_frontier.parquet"
PARETO_CSV = OUTPUT_DIR / "pareto_frontier.csv"
ROUTE_ARCS_PICKLE = PROCESSED_DIR / "route_arcs.pkl"


def _origin_targets(graph: RouteGraph, points: pd.DataFrame):
    """Map neighborhoods to node indices, dropping any not in the component."""
    idx, names = [], []
    for name, node in zip(points["neighborhood"], points["node"]):
        i = graph.index_of(node)
        if i is None:
            log.warning("%s: representative node not in the %s component; skipped",
                        name, graph.mode)
            continue
        idx.append(i); names.append(name)
    return np.asarray(idx, dtype=np.int64), names


def compute_pair_routes(graph: RouteGraph, points: pd.DataFrame,
                        profiles: dict | None = None):
    """Route every ordered neighborhood pair under every objective.

    Returns ``(DataFrame, {(profile, origin, dest): arc_list})``.
    """
    profiles = profiles or ROUTING_PROFILES
    node_idx, names = _origin_targets(graph, points)
    rows: list[dict] = []
    arc_store: dict[tuple, list[int]] = {}

    for pname, weights in profiles.items():
        arc_cost = graph.build_costs(weights)
        with step(f"routing {len(names)}x{len(names)-1} ordered pairs "
                  f"[{graph.mode}/{pname}]", log):
            dist, pred, winner = shortest_paths(graph, arc_cost, node_idx)
            for si, oname in enumerate(progress(names, desc=f"  {pname}",
                                                unit="origin")):
                for ti, dname in zip(node_idx, names):
                    if dname == oname:
                        continue
                    arcs = arcs_from_predecessors(graph, pred[si], winner,
                                                  int(node_idx[si]), int(ti))
                    if not arcs:
                        log.warning("no route %s -> %s (%s/%s)", oname, dname,
                                    graph.mode, pname)
                        continue
                    s = summarise_route(graph, arcs)
                    s.update(mode=graph.mode, profile=pname,
                             origin=oname, destination=dname,
                             cost=float(dist[si, ti]))
                    rows.append(s)
                    arc_store[(pname, oname, dname)] = arcs

    df = pd.DataFrame(rows)
    return df, arc_store


def add_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """Add distance penalty and elevation saved relative to the shortest path."""
    base = (df[df["profile"] == "shortest"]
            .set_index(["mode", "origin", "destination"])
            [["distance_m", "elev_gain_m", "max_grade"]]
            .rename(columns={"distance_m": "shortest_distance_m",
                             "elev_gain_m": "shortest_gain_m",
                             "max_grade": "shortest_max_grade"}))
    out = df.merge(base, left_on=["mode", "origin", "destination"],
                   right_index=True, how="left")
    out["detour_ratio"] = out["distance_m"] / out["shortest_distance_m"]
    out["distance_penalty_m"] = out["distance_m"] - out["shortest_distance_m"]
    out["gain_saved_m"] = out["shortest_gain_m"] - out["elev_gain_m"]
    out["gain_saved_pct"] = np.where(
        out["shortest_gain_m"] > 0,
        100.0 * out["gain_saved_m"] / out["shortest_gain_m"], 0.0)
    # metres of climbing avoided per extra metre walked -- the efficiency of
    # the detour, and the single most useful number in the whole matrix
    out["climb_saved_per_extra_m"] = np.where(
        out["distance_penalty_m"] > 1.0,
        out["gain_saved_m"] / out["distance_penalty_m"], np.nan)
    return out


def run_pair_analysis(graphs: dict[str, RouteGraph], points_by_mode: dict,
                      force: bool = False):
    """Full pair matrix across modes; caches results and the arc store."""
    if PAIRS_PARQUET.exists() and ROUTE_ARCS_PICKLE.exists() and not force:
        log.info("cached %s", PAIRS_PARQUET.name)
        with open(ROUTE_ARCS_PICKLE, "rb") as fh:
            store = pickle.load(fh)
        return pd.read_parquet(PAIRS_PARQUET), store

    frames, store = [], {}
    for mode, graph in graphs.items():
        df, arcs = compute_pair_routes(graph, points_by_mode[mode])
        frames.append(df)
        store[mode] = arcs
    df = add_comparisons(pd.concat(frames, ignore_index=True))

    # tidy column order for the published CSV
    lead = ["mode", "profile", "origin", "destination", "distance_m",
            "elev_gain_m", "elev_loss_m", "net_change_m", "max_grade",
            "avg_abs_grade", "p95_grade"]
    rest = [c for c in df.columns if c not in lead]
    df = df[lead + rest]

    df.to_parquet(PAIRS_PARQUET)
    df.to_csv(PAIRS_CSV, index=False)
    with open(ROUTE_ARCS_PICKLE, "wb") as fh:
        pickle.dump(store, fh)
    log.info("wrote %s (%d routes) and %s", PAIRS_PARQUET.name, len(df),
             PAIRS_CSV.name)
    return df, store


# --------------------------------------------------------------------------
# Pareto analysis
# --------------------------------------------------------------------------
def _non_dominated(points: np.ndarray) -> np.ndarray:
    """Boolean mask of Pareto-optimal rows (all objectives minimised)."""
    n = len(points)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = np.all(points <= points[i], axis=1) & np.any(
            points < points[i], axis=1)
        if dominated.any():
            keep[i] = False
    return keep


def pareto_for_pairs(graph: RouteGraph, points: pd.DataFrame, pairs,
                     lambdas=PARETO_LAMBDA_SWEEP, base_profile: str = "balanced"):
    """Trace the distance / climbing / max-grade frontier for given pairs.

    A single scalar scales all of the balanced profile's climbing and grade
    terms from zero (pure distance) upwards, and the pure minimum-climbing
    objective is added as the far anchor.  Each weighting yields one optimal
    route; duplicates are merged and the non-dominated subset over
    (distance, gain, max grade) is the frontier.
    """
    node_idx, names = _origin_targets(graph, points)
    name_to_idx = dict(zip(names, node_idx))
    wanted = [(o, d) for o, d in pairs
              if o in name_to_idx and d in name_to_idx]
    if not wanted:
        return pd.DataFrame()

    base = ROUTING_PROFILES[base_profile]
    origins = sorted({o for o, _ in wanted})
    src = np.asarray([name_to_idx[o] for o in origins], dtype=np.int64)

    sweep = [(lam, with_scale(base, lam)) for lam in lambdas]
    sweep.append((float("inf"), ROUTING_PROFILES["min_climb"]))
    records: list[dict] = []
    with step(f"Pareto sweep: {len(sweep)} weights x {len(wanted)} pairs "
              f"[{graph.mode}]", log):
        for alpha, w in progress(sweep, desc="  weight sweep", unit="weight"):
            arc_cost = graph.build_costs(w)
            dist, pred, winner = shortest_paths(graph, arc_cost, src)
            for si, oname in enumerate(origins):
                for dname in [d for o, d in wanted if o == oname]:
                    ti = int(name_to_idx[dname])
                    arcs = arcs_from_predecessors(graph, pred[si], winner,
                                                  int(src[si]), ti)
                    if not arcs:
                        continue
                    s = summarise_route(graph, arcs)
                    s.update(mode=graph.mode, origin=oname, destination=dname,
                             alpha=alpha,
                             route_key=hash(tuple(arcs)))
                    records.append(s)

    df = pd.DataFrame(records)
    if df.empty:
        return df
    # one row per distinct route per pair (the smallest alpha that produced it)
    df = (df.sort_values("alpha")
            .drop_duplicates(["mode", "origin", "destination", "route_key"])
            .reset_index(drop=True))

    keep_rows = []
    for (m, o, d), grp in df.groupby(["mode", "origin", "destination"]):
        obj = grp[["distance_m", "elev_gain_m", "max_grade"]].to_numpy(float)
        mask = _non_dominated(obj)
        g = grp.copy()
        g["pareto_optimal"] = mask
        keep_rows.append(g)
    out = pd.concat(keep_rows, ignore_index=True)
    log.info("Pareto: %d distinct routes, %d on the frontier",
             len(out), int(out["pareto_optimal"].sum()))
    return out


def run_pareto(graphs: dict[str, RouteGraph], points_by_mode: dict, pairs,
               force: bool = False):
    if PARETO_PARQUET.exists() and not force:
        log.info("cached %s", PARETO_PARQUET.name)
        return pd.read_parquet(PARETO_PARQUET)
    frames = [pareto_for_pairs(g, points_by_mode[m], pairs)
              for m, g in graphs.items()]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    df.to_parquet(PARETO_PARQUET)
    df.to_csv(PARETO_CSV, index=False)
    log.info("wrote %s and %s", PARETO_PARQUET.name, PARETO_CSV.name)
    return df
