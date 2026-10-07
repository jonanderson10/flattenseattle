"""Passes, saddles and barriers: where San Francisco's geography forces a climb.

Method
------
The key question -- "if I want to get from this part of the city to that one,
what is the *lowest* summit I can possibly cross?" -- is a **minimax (bottleneck)
path** problem, not a shortest-path problem.  For two nodes *s* and *t* the
answer is

    pass_height(s, t) = min over all paths P from s to t of max elevation on P

and the edge realising that maximum is the pass itself.  This has an exact,
elegant solution: sort every edge by its crest (the highest elevation reached
anywhere along it), add edges to a union-find structure in increasing crest
order, and the crest of the edge that first connects *s* to *t* is
``pass_height(s, t)``.  The resulting structure is a minimum bottleneck
spanning tree, and its critical edges are precisely the city's passes.

Rather than asking about arbitrary node pairs, the analysis first identifies
the **lowland basins**: connected components of the street network lying
entirely below a low-elevation threshold.  These are the flat districts that
people actually travel between -- the northeastern waterfront plain, the
Mission/SoMa flats, the Sunset, the Richmond, the Bayview flats, and so on.
The pass tree over those basins then answers, for every pair of flat
districts, how much climbing is geometrically unavoidable and exactly which
block you must climb it on.

Barriers are the complementary view: steep edges that carry a large share of
inter-neighborhood traffic because no flatter alternative exists.  A street
with a high grade *and* high usage is a genuine wall in the city's
topography; a steep street nobody routes over is merely steep.
"""
from __future__ import annotations

import collections

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR
from .utils import get_logger, step

log = get_logger("flatten_seattle.passes")

PASSES_GEOJSON = OUTPUT_DIR / "passes.geojson"
PASSES_CSV = OUTPUT_DIR / "passes.csv"
BARRIERS_GEOJSON = OUTPUT_DIR / "barriers.geojson"
BARRIERS_CSV = OUTPUT_DIR / "barriers.csv"
BASINS_GEOJSON = OUTPUT_DIR / "lowland_basins.geojson"

#: Elevation (m) below which street is considered "lowland" when delineating
#: basins.  25 m was tried first and proved too generous: the 25 m contour
#: links the Mission, SoMa, the northeastern waterfront and the Bayview into
#: a single 817 km "basin", which says nothing useful.  15 m separates the
#: real flat districts while still following the valley floors.
BASIN_ELEV_M = 15.0
#: A basin must contain at least this much street to count as a district.
BASIN_MIN_KM = 2.0


# --------------------------------------------------------------------------
# node elevations
# --------------------------------------------------------------------------
def node_elevations(directed: pd.DataFrame) -> pd.Series:
    """Elevation per graph node, from the structure-corrected edge profiles."""
    a = directed[["from_node", "start_elev"]].rename(
        columns={"from_node": "node", "start_elev": "z"})
    b = directed[["to_node", "end_elev"]].rename(
        columns={"to_node": "node", "end_elev": "z"})
    both = pd.concat([a, b], ignore_index=True)
    return both.groupby("node")["z"].median()


# --------------------------------------------------------------------------
# union-find
# --------------------------------------------------------------------------
class _DSU:
    __slots__ = ("parent", "rank")

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return int(x)

    def union(self, a: int, b: int) -> tuple[int, int] | None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return None
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra, rb


# --------------------------------------------------------------------------
# basins
# --------------------------------------------------------------------------
def find_basins(edges, node_z: pd.Series, threshold: float = BASIN_ELEV_M,
                min_km: float = BASIN_MIN_KM, mode: str = "walk"):
    """Connected lowland components of the network below ``threshold``."""
    import networkx as nx

    sub = edges[edges[f"{mode}_ok"]]
    zu = sub["u"].map(node_z)
    zv = sub["v"].map(node_z)
    low = sub[(zu <= threshold) & (zv <= threshold)]
    g = nx.Graph()
    for u, v, L in zip(low["u"], low["v"], low["length_m"]):
        if g.has_edge(u, v):
            g[u][v]["length"] = min(g[u][v]["length"], L)
        else:
            g.add_edge(u, v, length=L)

    basins = {}
    labels: dict[str, int] = {}
    bid = 0
    for comp in nx.connected_components(g):
        km = sum(d["length"] for _, _, d in g.subgraph(comp).edges(data=True)) / 1000
        if km < min_km:
            continue
        for n in comp:
            labels[n] = bid
        basins[bid] = {"n_nodes": len(comp), "length_km": km}
        bid += 1
    log.info("lowland basins below %.0f m: %d districts (>= %.1f km of street)",
             threshold, len(basins), min_km)
    return labels, basins


def name_basins(labels: dict, edges, neighborhoods, node_z: pd.Series):
    """Label each basin with the neighborhoods it covers, by street length."""
    import geopandas as gpd

    sub = edges[edges["u"].map(labels).notna() | edges["v"].map(labels).notna()].copy()
    sub["basin"] = sub["u"].map(labels)
    sub = sub[sub["basin"].notna()]
    mid = sub.geometry.interpolate(0.5, normalized=True)
    pts = gpd.GeoDataFrame({"basin": sub["basin"].astype(int).values,
                            "length_m": sub["length_m"].values},
                           geometry=mid.values, crs=edges.crs)
    j = gpd.sjoin(pts, neighborhoods[["neighborhood", "geometry"]],
                  how="left", predicate="within")
    out = {}
    for b, grp in j.groupby("basin"):
        top = (grp.groupby("neighborhood")["length_m"].sum()
               .sort_values(ascending=False))
        out[int(b)] = {
            "label": " / ".join(top.head(3).index.tolist()),
            "neighborhoods": "; ".join(sorted(top[top > 200].index.tolist())),
            "street_km": float(grp["length_m"].sum() / 1000),
        }
    return out


# --------------------------------------------------------------------------
# pass tree
# --------------------------------------------------------------------------
def find_passes(edges, node_z: pd.Series, basin_labels: dict,
                mode: str = "walk"):
    """Minimum bottleneck spanning tree over basins -> the city's passes.

    Returns one record per pass: the crest elevation at which two groups of
    basins first become connected, and the edge on which that happens.
    """
    sub = edges[edges[f"{mode}_ok"]].copy()
    # crest of an edge = highest elevation reached anywhere along it
    sub["crest"] = sub[["elev_max"]].max(axis=1)
    sub = sub.sort_values("crest", kind="stable")

    nodes = pd.unique(pd.concat([sub["u"], sub["v"]], ignore_index=True))
    nidx = {n: i for i, n in enumerate(nodes)}
    dsu = _DSU(len(nodes))
    # which basins each component currently contains
    comp_basins: dict[int, set] = collections.defaultdict(set)
    for n, b in basin_labels.items():
        i = nidx.get(n)
        if i is not None:
            comp_basins[i].add(b)

    passes = []
    u_idx = sub["u"].map(nidx).to_numpy()
    v_idx = sub["v"].map(nidx).to_numpy()
    crest = sub["crest"].to_numpy(dtype="float64")
    eid = sub["edge_id"].to_numpy()

    with step(f"building the minimum bottleneck spanning tree [{mode}]", log):
        for k in range(len(sub)):
            a, b = int(u_idx[k]), int(v_idx[k])
            ra, rb = dsu.find(a), dsu.find(b)
            if ra == rb:
                continue
            sa, sb = comp_basins.get(ra, set()), comp_basins.get(rb, set())
            merged = dsu.union(a, b)
            if merged is None:
                continue
            root, child = merged
            new = sa | sb
            comp_basins[root] = new
            comp_basins.pop(child, None)
            # a pass exists only where two *different* basin groups join
            if sa and sb and not (sa & sb):
                passes.append({
                    "edge_id": int(eid[k]),
                    "pass_elev_m": float(crest[k]),
                    "basins_a": sorted(sa),
                    "basins_b": sorted(sb),
                    "n_basins_joined": len(sa) + len(sb),
                })
    log.info("  %d passes found linking the lowland districts", len(passes))
    return pd.DataFrame(passes)


def describe_passes(passes: pd.DataFrame, edges, neighborhoods,
                    basin_names: dict):
    """Attach geometry, street name, neighborhood and basin labels."""
    import geopandas as gpd

    if passes.empty:
        return gpd.GeoDataFrame(columns=["edge_id", "geometry"],
                                geometry="geometry", crs=edges.crs)
    e = edges.set_index("edge_id")
    keep = [c for c in ["name", "cls", "length_m", "max_abs_grade", "elev_min",
                        "elev_max", "geometry"] if c in e.columns]
    df = passes.join(e[keep], on="edge_id")

    def label(bs):
        return " + ".join(basin_names.get(int(b), {}).get("label", f"basin {b}")
                          for b in bs)

    df["side_a"] = df["basins_a"].apply(label)
    df["side_b"] = df["basins_b"].apply(label)
    df["basins_a"] = df["basins_a"].apply(lambda x: ",".join(map(str, x)))
    df["basins_b"] = df["basins_b"].apply(lambda x: ",".join(map(str, x)))

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=edges.crs)
    mid = gdf.geometry.interpolate(0.5, normalized=True)
    pt = gpd.GeoDataFrame(geometry=mid.values, crs=edges.crs)
    j = gpd.sjoin(pt, neighborhoods[["neighborhood", "geometry"]], how="left",
                  predicate="within")
    gdf["neighborhood"] = j["neighborhood"].to_numpy()[:len(gdf)]
    gdf["pass_elev_ft"] = gdf["pass_elev_m"] * 3.28084
    return gdf.sort_values("pass_elev_m").reset_index(drop=True)




# --------------------------------------------------------------------------
# minimum bottleneck spanning tree + LCA: pass height for any pair of points
# --------------------------------------------------------------------------
def build_bottleneck_tree(edges, mode: str = "walk"):
    """Kruskal merge tree over crest-sorted edges (a Cartesian/merge tree).

    Each union creates an internal node labelled with the crest elevation at
    which the two sides became connected, and with the edge responsible.  The
    minimax pass height between any two graph nodes is then the label of their
    lowest common ancestor in this tree, and the pass itself is that
    ancestor's edge.  This is exact, and one construction answers every pair.
    """
    sub = edges[edges[f"{mode}_ok"]].copy()
    sub["crest"] = sub["elev_max"]
    sub = sub.sort_values("crest", kind="stable")

    nodes = pd.unique(pd.concat([sub["u"], sub["v"]], ignore_index=True))
    nidx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # merge-tree arrays: leaves 0..n-1 are graph nodes, internals follow
    parent_tree = np.full(2 * n, -1, dtype=np.int64)
    label = np.full(2 * n, -np.inf)
    tree_edge = np.full(2 * n, -1, dtype=np.int64)
    comp_root = np.arange(n, dtype=np.int64)      # DSU root -> tree node
    dsu = _DSU(n)
    next_node = n

    u_idx = sub["u"].map(nidx).to_numpy()
    v_idx = sub["v"].map(nidx).to_numpy()
    crest = sub["crest"].to_numpy(dtype="float64")
    eid = sub["edge_id"].to_numpy()

    with step(f"building the minimum bottleneck spanning tree [{mode}]", log):
        for k in range(len(sub)):
            a, b = int(u_idx[k]), int(v_idx[k])
            ra, rb = dsu.find(a), dsu.find(b)
            if ra == rb:
                continue
            ta, tb = int(comp_root[ra]), int(comp_root[rb])
            t = next_node; next_node += 1
            parent_tree[ta] = t
            parent_tree[tb] = t
            label[t] = float(crest[k])
            tree_edge[t] = int(eid[k])
            merged = dsu.union(a, b)
            root, child = merged
            comp_root[root] = t
    log.info("  merge tree: %d leaves, %d internal nodes", n, next_node - n)
    return {"nidx": nidx, "parent": parent_tree[:next_node],
            "label": label[:next_node], "edge": tree_edge[:next_node],
            "n_leaves": n}


def _ancestors(tree, leaf: int) -> dict[int, int]:
    """Map ancestor -> depth for one leaf (root last)."""
    out = {}
    v = leaf
    d = 0
    par = tree["parent"]
    while v != -1:
        out[v] = d
        v = int(par[v])
        d += 1
    return out


def pass_height(tree, node_a, node_b):
    """Minimax pass elevation and the responsible edge for one pair of nodes."""
    ia = tree["nidx"].get(node_a)
    ib = tree["nidx"].get(node_b)
    if ia is None or ib is None:
        return None
    anc = _ancestors(tree, ia)
    v = ib
    par = tree["parent"]
    while v != -1:
        if v in anc:
            return float(tree["label"][v]), int(tree["edge"][v])
        v = int(par[v])
    return None


def neighborhood_pass_matrix(tree, points: pd.DataFrame):
    """Minimax pass elevation for every ordered pair of neighborhood points.

    This is the direct answer to "how much climbing is unavoidable between
    these two parts of the city, and where".
    """
    names = list(points["neighborhood"])
    nodes = list(points["node"])
    rows = []
    with step(f"minimax pass heights for {len(names)} neighborhoods", log):
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r = pass_height(tree, nodes[i], nodes[j])
                if r is None:
                    continue
                h, e = r
                rows.append({"neighborhood_a": names[i], "neighborhood_b": names[j],
                             "pass_elev_m": h, "pass_elev_ft": h * 3.28084,
                             "pass_edge_id": e})
    return pd.DataFrame(rows)


def rank_critical_passes(pass_matrix: pd.DataFrame, edges, neighborhoods):
    """Aggregate the pass matrix onto edges: the city's true saddles.

    An edge that is the binding constraint for many neighborhood pairs is a
    pass that San Francisco's topography genuinely forces traffic over.
    """
    import geopandas as gpd

    if pass_matrix.empty:
        return gpd.GeoDataFrame(columns=["edge_id", "geometry"],
                                geometry="geometry", crs=edges.crs)
    agg = (pass_matrix.groupby("pass_edge_id")
           .agg(pairs_served=("pass_elev_m", "size"),
                pass_elev_m=("pass_elev_m", "first"))
           .reset_index().rename(columns={"pass_edge_id": "edge_id"}))
    # which neighborhoods sit on each side
    sides = (pass_matrix.groupby("pass_edge_id")
             .apply(lambda g: "; ".join(sorted(set(g["neighborhood_a"])
                                               | set(g["neighborhood_b"]))),
                    include_groups=False)
             .rename("neighborhoods_separated"))
    agg = agg.merge(sides, left_on="edge_id", right_index=True, how="left")

    e = edges.set_index("edge_id")
    cols = [c for c in ["name", "cls", "length_m", "max_abs_grade", "elev_min",
                        "elev_max", "geometry"] if c in e.columns]
    agg = agg.join(e[cols], on="edge_id")
    agg["pass_elev_ft"] = agg["pass_elev_m"] * 3.28084
    gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=edges.crs)
    mid = gdf.geometry.interpolate(0.5, normalized=True)
    j = gpd.sjoin(gpd.GeoDataFrame(geometry=mid.values, crs=edges.crs),
                  neighborhoods[["neighborhood", "geometry"]], how="left",
                  predicate="within")
    gdf["neighborhood"] = j["neighborhood"].to_numpy()[:len(gdf)]
    return gdf.sort_values("pairs_served", ascending=False).reset_index(drop=True)



# --------------------------------------------------------------------------
# barriers
# --------------------------------------------------------------------------
def find_barriers(graph, arc_store: dict, edges, neighborhoods,
                  min_grade: float = 0.08, profile: str = "shortest"):
    """Steep edges that inter-neighborhood traffic cannot avoid.

    Usage is counted on the **shortest**-path routes, because those describe
    what the street grid forces on someone who is not trying to avoid hills;
    a steep edge with high shortest-path usage is a true barrier in the city's
    fabric.  Each barrier is then annotated with how much of that traffic
    survives on the climb-averse routes -- a barrier that vanishes under the
    flat objective has an alternative, while one that persists does not.
    """
    import geopandas as gpd

    arc_edge = graph.table["edge_id"].to_numpy()
    short_use: collections.Counter = collections.Counter()
    flat_use: collections.Counter = collections.Counter()
    for (pname, o, d), arcs in arc_store.items():
        if not arcs:
            continue
        target = short_use if pname == profile else flat_use
        for eid in np.unique(arc_edge[arcs]):
            target[int(eid)] += 1

    steep = edges[edges["max_abs_grade"] >= min_grade].copy()
    steep["shortest_use"] = steep["edge_id"].map(short_use).fillna(0).astype(int)
    steep["flat_use"] = steep["edge_id"].map(flat_use).fillna(0).astype(int)
    steep = steep[steep["shortest_use"] > 0].copy()
    # ``flat_use`` is summed over the three climb-averse objectives, so it is
    # not directly comparable with ``shortest_use`` until it is averaged.
    n_flat_profiles = len({p for (p, _o, _d) in arc_store if p != profile}) or 1
    steep["flat_use_per_objective"] = steep["flat_use"] / n_flat_profiles
    # a barrier is unavoidable to the extent flat routes still have to use it
    steep["unavoidability"] = (steep["flat_use_per_objective"]
                               / steep["shortest_use"].clip(lower=1)).clip(0, 1)
    steep["barrier_score"] = (steep["shortest_use"]
                              * steep["max_abs_grade"]
                              * steep["length_m"] / 100.0)
    steep = steep.sort_values("barrier_score", ascending=False)

    gdf = gpd.GeoDataFrame(steep, geometry="geometry", crs=edges.crs)
    mid = gdf.geometry.interpolate(0.5, normalized=True)
    j = gpd.sjoin(gpd.GeoDataFrame(geometry=mid.values, crs=edges.crs),
                  neighborhoods[["neighborhood", "geometry"]], how="left",
                  predicate="within")
    gdf["neighborhood"] = j["neighborhood"].to_numpy()[:len(gdf)]
    log.info("barriers: %d steep edges (>=%.0f%%) carry shortest-path traffic",
             len(gdf), min_grade * 100)
    return gdf


def run_pass_analysis(graphs: dict, arc_store: dict, edges, directed,
                      neighborhoods, points=None, force: bool = False):
    """Full pass/barrier/basin pipeline (walking network)."""
    import geopandas as gpd

    if PASSES_GEOJSON.exists() and BARRIERS_GEOJSON.exists() and not force:
        log.info("cached passes and barriers")
        return (gpd.read_file(PASSES_GEOJSON), gpd.read_file(BARRIERS_GEOJSON),
                gpd.read_file(BASINS_GEOJSON) if BASINS_GEOJSON.exists() else None)

    node_z = node_elevations(directed)

    # --- lowland basins (descriptive geography) ---
    labels, basins = find_basins(edges, node_z, mode="walk")
    basin_names = name_basins(labels, edges, neighborhoods, node_z)
    for b, info in sorted(basins.items()):
        log.info("  basin %-2d %-46s %6.1f km of street", b,
                 basin_names.get(b, {}).get("label", "?"), info["length_km"])

    # --- minimax passes between every neighborhood pair ---
    tree = build_bottleneck_tree(edges, mode="walk")
    pass_matrix = neighborhood_pass_matrix(tree, points) if points is not None \
        else pd.DataFrame()
    passes_gdf = rank_critical_passes(pass_matrix, edges, neighborhoods)
    if not pass_matrix.empty:
        pass_matrix.to_csv(OUTPUT_DIR / "pass_matrix.csv", index=False)
        log.info("  %d distinct critical passes over %d neighborhood pairs",
                 len(passes_gdf), len(pass_matrix))

    # --- basin-to-basin passes (kept as a coarser summary) ---
    basin_passes = find_passes(edges, node_z, labels, mode="walk")
    if not basin_passes.empty:
        describe_passes(basin_passes, edges, neighborhoods, basin_names) \
            .drop(columns=["geometry"]).to_csv(
                OUTPUT_DIR / "basin_passes.csv", index=False)

    barriers = find_barriers(graphs["walk"], arc_store["walk"], edges,
                             neighborhoods)

    sub = edges[edges["u"].map(labels).notna()].copy()
    sub["basin"] = sub["u"].map(labels).astype("Int64")
    basins_gdf = gpd.GeoDataFrame(
        sub[["basin", "name", "length_m", "geometry"]], geometry="geometry",
        crs=edges.crs)
    basins_gdf["basin_label"] = basins_gdf["basin"].map(
        lambda b: basin_names.get(int(b), {}).get("label", "") if pd.notna(b) else "")

    for gdf, gj, csv in ((passes_gdf, PASSES_GEOJSON, PASSES_CSV),
                         (barriers.head(400), BARRIERS_GEOJSON, BARRIERS_CSV)):
        if len(gdf):
            gdf.to_crs("EPSG:4326").to_file(gj, driver="GeoJSON")
            gdf.drop(columns=["geometry"]).to_csv(csv, index=False)
    if len(basins_gdf):
        basins_gdf.to_crs("EPSG:4326").to_file(BASINS_GEOJSON, driver="GeoJSON")
    log.info("wrote passes, barriers and basins")
    return passes_gdf, barriers, basins_gdf
