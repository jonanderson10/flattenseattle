"""Routing over the directed, elevation-attributed street graph.

Cost model
----------
The cost of a directed edge, in **equivalent metres** (the distance a
traveller would consider "as bad as" this edge), is

    cost = length * mode_multiplier(class)
         + alpha * cumulative_gain
         + beta  * SUM_k  penalty_k * distance_above_threshold_k
         + gamma * extreme_extra * distance_above_highest_threshold

The threshold terms are *cumulative*: 100 m at a 12% grade incurs the 3%, 5%,
8% and 10% penalties simultaneously, so the marginal cost of extra steepness
rises with grade rather than staying linear.  With the default balanced
weights, a metre travelled at 12% costs about 8 equivalent metres of grade
penalty on top of the climbing term, while a metre at 4% costs 0.25.

``alpha`` is the substitution rate between climbing and distance.  Naismith's
rule for walking implies roughly 8 m of flat walking per metre climbed; the
"min_climb" profile pushes it to 120 to express near-lexicographic preference
for avoiding climbing, and the sweep in ``PARETO_LAMBDA_SWEEP`` traces the
whole frontier between the two.

The per-class comfort multipliers (a protected cycleway at 0.85, 19th Avenue
at 1.9) express stress rather than distance, so they are switched **off** for
the ``shortest`` objective, which minimises pure distance.  That keeps
``shortest`` a meaningful baseline: every distance-penalty and
elevation-saved figure in the results matrix is measured against a genuine
shortest path.

Only ``cumulative_gain`` enters the cost, never ``net_change``: a route that
climbs 120 m and descends 120 m is not flat, and the analytical point of the
project is that net elevation difference is the wrong variable.

Descent is not free in reality (steep descents are hard on knees and brakes),
but it is charged only through the grade-threshold terms, which are computed
on the *climbing* grade in the direction of travel, plus the class
multipliers.  This keeps the cost function monotone in climbing, which the
Pareto analysis relies on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CostWeights, GRADE_THRESHOLDS, MODES
from .utils import get_logger

log = get_logger("flatten_seattle.routing")

_TH = [int(t * 100) for t in GRADE_THRESHOLDS]


# --------------------------------------------------------------------------
# cost computation
# --------------------------------------------------------------------------
def edge_costs(directed: pd.DataFrame, weights: CostWeights,
               mode: str = "walk") -> np.ndarray:
    """Vectorised traversal cost, in equivalent metres, for every directed edge."""
    cfg = MODES[mode]
    length = directed["length_m"].to_numpy(dtype="float64")

    mult = np.ones_like(length)
    if cfg.class_multiplier and getattr(weights, "use_class_multiplier", True):
        cls = directed["cls"].to_numpy()
        for klass, m in cfg.class_multiplier.items():
            mult[cls == klass] = m

    cost = length * mult
    cost = cost + weights.alpha * directed["cum_gain"].to_numpy(dtype="float64")

    if weights.beta:
        pen = np.zeros_like(length)
        for th, w in zip(_TH, weights.threshold_penalties):
            if w:
                pen += w * directed[f"d_above_{th}"].to_numpy(dtype="float64")
        cost = cost + weights.beta * pen

    if weights.gamma and weights.extreme_extra:
        cost = cost + (weights.gamma * weights.extreme_extra
                       * directed[f"d_above_{_TH[-1]}"].to_numpy(dtype="float64"))

    return np.maximum(cost, 1e-6)


# --------------------------------------------------------------------------
# graph container
# --------------------------------------------------------------------------
@dataclass
class RouteGraph:
    """Node/arc structure for one travel mode, backed by ``scipy.csgraph``.

    Shortest paths run through ``scipy.sparse.csgraph.dijkstra``, which needs
    a node-by-node sparse matrix.  Two nodes can be joined by more than one
    arc (two different streets between the same pair of intersections), and a
    COO->CSR conversion would *sum* such duplicates, silently inventing a
    more expensive street.  Arcs are therefore grouped by ordered node pair,
    and for each cost vector the cheapest arc in each group is selected, so
    the matrix holds a true minimum and the winning arc is recoverable.
    """
    mode: str
    node_ids: np.ndarray                  # index -> node id
    node_index: dict                      # node id -> index
    table: pd.DataFrame                   # directed arcs, positional order
    arc_pair: np.ndarray                  # arc -> pair index
    pair_from: np.ndarray                 # pair -> tail node index
    pair_to: np.ndarray                   # pair -> head node index
    pair_lookup: dict                     # (tail, head) -> pair index

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_arcs(self) -> int:
        return len(self.table)

    def build_costs(self, weights: CostWeights) -> np.ndarray:
        """Cost per arc, in equivalent metres."""
        return edge_costs(self.table, weights, self.mode)

    def pair_costs(self, arc_cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Collapse arc costs to one cost per node pair, keeping the winner."""
        n_pairs = len(self.pair_from)
        best = np.full(n_pairs, np.inf)
        np.minimum.at(best, self.arc_pair, arc_cost)
        # recover which arc achieved the minimum for each pair
        winner = np.full(n_pairs, -1, dtype=np.int64)
        is_min = arc_cost <= best[self.arc_pair] + 0.0
        idx = np.flatnonzero(is_min)
        winner[self.arc_pair[idx]] = idx
        return best, winner

    def matrix(self, arc_cost: np.ndarray):
        """CSR cost matrix plus the arc chosen for each retained pair."""
        import scipy.sparse as sp
        best, winner = self.pair_costs(arc_cost)
        m = sp.csr_matrix((best, (self.pair_from, self.pair_to)),
                          shape=(self.n_nodes, self.n_nodes))
        return m, winner

    def index_of(self, node_id):
        return self.node_index.get(node_id)


def build_route_graph(directed: pd.DataFrame, mode: str = "walk") -> RouteGraph:
    """Assemble the routable graph for a mode, restricted to the largest
    strongly connected component (so every origin can reach every target)."""
    import scipy.sparse as sp

    col = f"{mode}_traversable"
    sub = directed[directed[col]].reset_index(drop=True)
    if sub.empty:
        raise RuntimeError(f"no traversable directed edges for mode {mode!r}")

    def _index(frame):
        nodes = pd.unique(pd.concat([frame["from_node"], frame["to_node"]],
                                    ignore_index=True))
        return nodes, {n: i for i, n in enumerate(nodes)}

    nodes, node_index = _index(sub)
    fi = sub["from_node"].map(node_index).to_numpy(dtype=np.int64)
    ti = sub["to_node"].map(node_index).to_numpy(dtype=np.int64)
    adj = sp.coo_matrix((np.ones(len(fi)), (fi, ti)),
                        shape=(len(nodes), len(nodes))).tocsr()
    ncomp, labels = sp.csgraph.connected_components(adj, directed=True,
                                                    connection="strong")
    if ncomp > 1:
        biggest = int(np.bincount(labels).argmax())
        keep = labels == biggest
        mask = keep[fi] & keep[ti]
        log.info("mode %s: largest strongly connected component keeps "
                 "%d/%d nodes and %d/%d arcs", mode, int(keep.sum()),
                 len(nodes), int(mask.sum()), len(fi))
        sub = sub[mask].reset_index(drop=True)
        nodes, node_index = _index(sub)
        fi = sub["from_node"].map(node_index).to_numpy(dtype=np.int64)
        ti = sub["to_node"].map(node_index).to_numpy(dtype=np.int64)

    # group arcs by ordered node pair
    key = fi.astype(np.int64) * len(nodes) + ti.astype(np.int64)
    uniq, arc_pair = np.unique(key, return_inverse=True)
    pair_from = (uniq // len(nodes)).astype(np.int64)
    pair_to = (uniq % len(nodes)).astype(np.int64)
    pair_lookup = {(int(a), int(b)): i for i, (a, b) in
                   enumerate(zip(pair_from, pair_to))}

    g = RouteGraph(mode=mode, node_ids=np.asarray(nodes), node_index=node_index,
                   table=sub, arc_pair=arc_pair.astype(np.int64),
                   pair_from=pair_from, pair_to=pair_to,
                   pair_lookup=pair_lookup)
    n_parallel = len(arc_pair) - len(uniq)
    log.info("mode %s graph: %d nodes, %d arcs (%d parallel), %.0f km",
             mode, g.n_nodes, g.n_arcs, n_parallel,
             sub["length_m"].sum() / 1000)
    return g


# --------------------------------------------------------------------------
# shortest paths
# --------------------------------------------------------------------------
def shortest_paths(graph: RouteGraph, arc_cost: np.ndarray, sources,
                   targets=None):
    """Multi-source Dijkstra. Returns (dist, predecessors, winner_arc).

    ``dist`` and ``predecessors`` are (n_sources, n_nodes) as returned by
    scipy; ``winner_arc`` maps a pair index to the arc that realises it.
    """
    from scipy.sparse.csgraph import dijkstra as sp_dijkstra

    m, winner = graph.matrix(arc_cost)
    src = np.atleast_1d(np.asarray(sources, dtype=np.int64))
    dist, pred = sp_dijkstra(m, directed=True, indices=src,
                             return_predecessors=True)
    if dist.ndim == 1:
        dist = dist[None, :]
        pred = pred[None, :]
    return dist, pred, winner


def arcs_from_predecessors(graph: RouteGraph, pred_row: np.ndarray,
                           winner: np.ndarray, source: int,
                           target: int) -> list[int]:
    """Rebuild the arc sequence of one path from a scipy predecessor row."""
    if source == target:
        return []
    path_nodes = [int(target)]
    v = int(target)
    guard = 0
    while v != source:
        p = int(pred_row[v])
        if p < 0:
            return []                       # unreachable
        path_nodes.append(p)
        v = p
        guard += 1
        if guard > graph.n_nodes + 5:       # pathological safety net
            return []
    path_nodes.reverse()

    arcs: list[int] = []
    for a, b in zip(path_nodes[:-1], path_nodes[1:]):
        pi = graph.pair_lookup.get((a, b))
        if pi is None or winner[pi] < 0:
            return []
        arcs.append(int(winner[pi]))
    return arcs


def route(graph: RouteGraph, source_node, target_node, weights: CostWeights,
          arc_cost: np.ndarray | None = None):
    """Route between two node ids under one cost profile. Returns (arcs, summary)."""
    si = graph.index_of(source_node)
    ti = graph.index_of(target_node)
    if si is None or ti is None:
        raise KeyError("source or target node is not in the routable component")
    c = arc_cost if arc_cost is not None else graph.build_costs(weights)
    dist, pred, winner = shortest_paths(graph, c, si)
    arcs = arcs_from_predecessors(graph, pred[0], winner, si, ti)
    return arcs, summarise_route(graph, arcs)


# --------------------------------------------------------------------------
# route summarisation
# --------------------------------------------------------------------------
def summarise_route(graph: RouteGraph, arcs: list[int]) -> dict:
    """Aggregate metrics for a route given as a list of arc indices."""
    if not arcs:
        return {"n_edges": 0, "distance_m": 0.0, "elev_gain_m": 0.0,
                "elev_loss_m": 0.0, "max_grade": 0.0, "net_change_m": 0.0,
                "avg_abs_grade": 0.0,
                **{f"d_above_{t}": 0.0 for t in _TH}}
    t = graph.table.iloc[arcs]
    length = t["length_m"].to_numpy(dtype="float64")
    total = float(length.sum())
    mean_abs = t["mean_abs_grade"].to_numpy(dtype="float64")
    out = {
        "n_edges": len(arcs),
        "distance_m": total,
        "elev_gain_m": float(t["cum_gain"].sum()),
        "elev_loss_m": float(t["cum_loss"].sum()),
        "net_change_m": float(t["net_change"].sum()),
        "max_grade": float(t["max_grade"].max()),
        "max_abs_grade_any": float(t["max_grade"].abs().max()),
        "avg_abs_grade": float((mean_abs * length).sum() / total) if total else 0.0,
        "p95_grade": float(np.percentile(t["p95_grade"].to_numpy(), 95))
        if len(t) else 0.0,
        "start_elev_m": float(t["start_elev"].iloc[0]),
        "end_elev_m": float(t["end_elev"].iloc[-1]),
    }
    for th in _TH:
        out[f"d_above_{th}"] = float(t[f"d_above_{th}"].sum())
    return out


def route_geometry(graph: RouteGraph, arcs: list[int], edges_gdf):
    """Merged LineString for a route, oriented in travel order."""
    from shapely.geometry import LineString

    if not arcs:
        return None
    geom_by_id = edges_gdf.set_index("edge_id").geometry
    coords: list[tuple] = []
    t = graph.table.iloc[arcs]
    for eid, direction in zip(t["edge_id"], t["direction"]):
        g = geom_by_id.loc[eid]
        cs = list(g.coords)
        if direction == "rev":
            cs = cs[::-1]
        if coords and coords[-1] == cs[0]:
            coords.extend(cs[1:])
        else:
            coords.extend(cs)
    if len(coords) < 2:
        return None
    return LineString(coords)


def route_profile(graph: RouteGraph, arcs: list[int], profiles: dict):
    """Concatenated (cumulative_distance, elevation) profile for a route."""
    if not arcs:
        return np.array([]), np.array([])
    t = graph.table.iloc[arcs]
    dists: list[np.ndarray] = []
    elevs: list[np.ndarray] = []
    base = 0.0
    for eid, direction in zip(t["edge_id"], t["direction"]):
        d, z = profiles[int(eid)]
        d = np.asarray(d, dtype="float64"); z = np.asarray(z, dtype="float64")
        if direction == "rev":
            d = d[-1] - d[::-1]
            z = z[::-1]
        if dists:
            d = d[1:]; z = z[1:]
        dists.append(base + d)
        elevs.append(z)
        base = dists[-1][-1] if dists[-1].size else base
    return np.concatenate(dists), np.concatenate(elevs)


