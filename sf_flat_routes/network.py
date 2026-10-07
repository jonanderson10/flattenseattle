"""Build a routable street graph from Overture transportation segments.

Topology comes from Overture ``connectors``: every segment lists the
connector IDs it touches together with the fractional position ``at`` along
its own geometry.  Splitting each segment at those positions and keying nodes
by connector ID yields exact topology with no geometric snapping tolerance,
and grade-separated crossings correctly stay unconnected.

Access is derived from Overture ``access_restrictions``.  The rule shapes
actually present in San Francisco are:

* ``denied`` + ``heading=backward`` (7,034 segments) -- one-way streets.
  Enforced for bicycles, ignored for pedestrians, since OSM ``oneway``
  describes vehicle movement.
* ``denied``/``allowed``/``designated`` + ``mode=[foot|bicycle|...]`` --
  explicit per-mode permissions.
* ``recognized=[as_private]`` or ``using=[as_customer|at_destination]`` --
  private or destination-only access; excluded from through routing.

A rule that names a mode outranks a rule that applies to every mode,
whatever order they appear in.  San Francisco's Slow Streets (Page,
Shotwell, Cabrillo, 12th Avenue) arrive as "foot allowed, bicycle
designated, motor vehicles at destination only, everything at destination
only", in that order; reading them in document order let the final
all-modes rule close the street to walking, which is the opposite of what a
Slow Street is.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (BIKE_FORBIDDEN_CLASSES, CRS_GEOGRAPHIC, CRS_PROJECTED,
                     MODES, NEVER_ROUTABLE_CLASSES, NEVER_ROUTABLE_SUBCLASSES,
                     PROCESSED_DIR, SIDEWALK_MAX_ANGLE_DEG, SIDEWALK_MIN_HITS,
                     SIDEWALK_OFFSET_M, SIDEWALK_SAMPLES)
from .utils import get_logger, progress, step

log = get_logger("sf_flat_routes.network")

EDGES_PARQUET = PROCESSED_DIR / "edges_raw.parquet"

#: access_type values that permit travel.
_PERMIT = {"allowed", "designated"}
#: conditional qualifiers that mean "not available for through travel".
_PRIVATE_RECOGNIZED = {"as_private", "as_employee", "as_student", "as_permitted"}
_DESTINATION_USING = {"at_destination", "as_customer", "as_delivery"}
#: mode tokens that subsume the pedestrian / bicycle modes.
_MODE_PARENTS = {
    "foot": {"foot", "pedestrian"},
    "bicycle": {"bicycle"},
}


def _as_list(value) -> list:
    """Coerce an Arrow-derived value (ndarray / list / None / NaN) to a list.

    Arrow list columns arrive as numpy object arrays after ``to_pandas``, and
    ``bool(ndarray)`` raises, so every nested column is funnelled through here.
    """
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, float) and np.isnan(value):
        return []
    return [value]


def _as_dict(value) -> dict:
    """Coerce an Arrow struct value to a dict."""
    if isinstance(value, dict):
        return value
    return {}


# --------------------------------------------------------------------------
# access parsing
# --------------------------------------------------------------------------
def _rule_modes(rule: dict) -> set[str] | None:
    modes = _as_list(_as_dict(rule.get("when")).get("mode"))
    return set(modes) if modes else None


def _rule_applies(rule: dict, mode: str) -> bool:
    """Does an access rule constrain ``mode``?"""
    modes = _rule_modes(rule)
    if modes is None:
        return True                      # applies to all modes
    return bool(modes & _MODE_PARENTS[mode])


def evaluate_access(restrictions, mode: str, default: bool = True) -> dict:
    """Resolve Overture access rules for one travel mode.

    Returns a dict with ``allowed`` (through travel permitted at all),
    ``oneway_forward_only`` (backward travel prohibited) and ``restricted``
    (access is conditional/private).

    Rules that apply to every mode are resolved first and rules that name
    ``mode`` after them, each group in document order, so a mode-specific
    rule always has the last word over a general one.  Rules carrying
    ``between`` apply to only part of the segment; they are recorded but do
    not veto the whole segment, because vetoing would delete usable street
    from the network.
    """
    allowed = default
    oneway_forward_only = False
    restricted = False
    partial = False

    rules = [_as_dict(r) for r in _as_list(restrictions)]
    rules.sort(key=lambda r: 0 if _rule_modes(r) is None else 1)   # stable
    for rule in rules:
        if not _rule_applies(rule, mode):
            continue
        when = _as_dict(rule.get("when"))
        if when.get("during"):
            # time-conditional (e.g. peak-hour bans); ignored, noted only
            continue
        is_partial = len(_as_list(rule.get("between"))) > 0
        atype = rule.get("access_type")
        heading = when.get("heading")

        if heading:
            # directional restriction == one-way
            if atype == "denied" and heading == "backward":
                oneway_forward_only = True
            elif atype in _PERMIT and heading == "backward":
                oneway_forward_only = False   # e.g. contraflow bike lane
            continue

        recognized = set(_as_list(when.get("recognized")))
        using = set(_as_list(when.get("using")))
        if recognized & _PRIVATE_RECOGNIZED or using & _DESTINATION_USING:
            # conditional access -- treat as unavailable for through routing
            if atype in _PERMIT:
                restricted = True
                if not is_partial:
                    allowed = False
            continue

        if is_partial:
            partial = True
            continue
        allowed = atype in _PERMIT
        if allowed and _rule_modes(rule) is not None:
            # an unconditional permit naming this mode also lifts a general
            # private/destination restriction read earlier
            restricted = False

    return {"allowed": allowed, "oneway_forward_only": oneway_forward_only,
            "restricted": restricted, "partial_rules": partial}


def _flags(road_flags) -> set[str]:
    out: set[str] = set()
    for entry in _as_list(road_flags):
        for v in _as_list(_as_dict(entry).get("values")):
            out.add(v)
    return out


# --------------------------------------------------------------------------
# segment loading and splitting
# --------------------------------------------------------------------------
def load_segments(path: Path) -> pd.DataFrame:
    """Load the cached Overture segment extract into a DataFrame."""
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    log.info("loaded %d raw Overture segments", table.num_rows)
    return table.to_pandas()


def _base_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Drop everything that is never part of a walk/bike street network."""
    n0 = len(df)
    df = df[df["subtype"] == "road"].copy()
    log.info("  subtype=road: %d -> %d", n0, len(df))

    df["cls"] = df["class"].fillna("unknown")
    df = df[~df["cls"].isin(NEVER_ROUTABLE_CLASSES)]
    df = df[~df["subclass"].fillna("").isin(NEVER_ROUTABLE_SUBCLASSES)]
    log.info("  after class/subclass filter: %d", len(df))

    df["flags"] = df["road_flags"].apply(_flags)
    bad = df["flags"].apply(
        lambda f: bool(f & {"is_abandoned", "is_under_construction", "is_proposed"}))
    df = df[~bad]
    # Freeway ramps are tagged is_link on a motorway-ish class; the class
    # filter already removed motorway_link, but some links hang off trunk
    # roads and are genuinely walkable, so is_link alone is not disqualifying.
    log.info("  after condition filter: %d", len(df))
    return df


def _split_geometries(df: pd.DataFrame):
    """Split each segment at its connectors, returning an edge-level table."""
    import shapely
    from shapely.ops import substring

    geoms = shapely.from_wkb(df["geometry"].values)
    rows: list[dict] = []
    n_synthetic = 0

    for i, (idx, row) in enumerate(progress(df.iterrows(), desc="  splitting segments",
                                            total=len(df), unit="seg")):
        line = geoms[i]
        if line is None or line.is_empty or line.length == 0:
            continue
        conns = [_as_dict(c) for c in _as_list(row["connectors"])]
        cuts = sorted({float(np.clip(c["at"], 0.0, 1.0)): c["connector_id"]
                       for c in conns if c.get("connector_id")}.items())
        # ensure both ends are nodes
        if not cuts or cuts[0][0] > 1e-9:
            cuts.insert(0, (0.0, f"synth:{row['id']}:start"))
            n_synthetic += 1
        if cuts[-1][0] < 1.0 - 1e-9:
            cuts.append((1.0, f"synth:{row['id']}:end"))
            n_synthetic += 1
        for k in range(len(cuts) - 1):
            a_at, a_id = cuts[k]
            b_at, b_id = cuts[k + 1]
            if b_at - a_at <= 1e-9 or a_id == b_id:
                continue
            piece = substring(line, a_at, b_at, normalized=True)
            if piece.is_empty or piece.length == 0:
                continue
            rows.append({
                "segment_id": row["id"],
                "u": a_id, "v": b_id,
                "part": k,
                "start_at": a_at, "end_at": b_at,
                "cls": row["cls"],
                "subclass": row["subclass"],
                "name": _as_dict(row["names"]).get("primary"),
                "flags": row["flags"],
                "access_restrictions": row["access_restrictions"],
                "geometry": piece,
            })
    log.info("  split into %d edges (%d synthetic end nodes)", len(rows), n_synthetic)
    return rows


#: classes that are not streets, for the purpose of finding sidewalks beside them
_NON_STREET = frozenset({"footway", "path", "cycleway", "steps", "pedestrian",
                         "track", "bridleway"})


def _sidewalk_hits(cand, streets) -> tuple[np.ndarray, np.ndarray]:
    """Count sample points per candidate that are near, and parallel to, a street."""
    import shapely

    k = SIDEWALK_SAMPLES
    frac = np.tile(np.linspace(0.1, 0.9, k), len(cand))
    lines = np.repeat(cand.geometry.values, k)
    # a short tangent either side of each sample point, at most 2 m long
    h = np.minimum(2.0 / np.maximum(shapely.length(lines), 1e-6), 0.05)
    pts = shapely.line_interpolate_point(lines, frac, normalized=True)
    a = shapely.line_interpolate_point(lines, frac - h, normalized=True)
    b = shapely.line_interpolate_point(lines, frac + h, normalized=True)

    tree = shapely.STRtree(streets.geometry.values)
    ip, js = tree.query_nearest(pts, max_distance=SIDEWALK_OFFSET_M, all_matches=False)
    sg = streets.geometry.values[js]
    s = shapely.line_locate_point(sg, pts[ip])
    sa = shapely.line_interpolate_point(sg, s - 2.0)
    sb = shapely.line_interpolate_point(sg, s + 2.0)

    def vec(p, q):
        return np.c_[shapely.get_x(q) - shapely.get_x(p), shapely.get_y(q) - shapely.get_y(p)]

    v1, v2 = vec(a[ip], b[ip]), vec(sa, sb)
    cos = np.abs((v1 * v2).sum(1)) / (np.linalg.norm(v1, axis=1)
                                      * np.linalg.norm(v2, axis=1) + 1e-9)
    near = np.zeros(len(pts), dtype=bool)
    near[ip] = True
    par = np.zeros(len(pts), dtype=bool)
    par[ip[cos >= np.cos(np.radians(SIDEWALK_MAX_ANGLE_DEG))]] = True
    return near.reshape(-1, k).sum(1), par.reshape(-1, k).sum(1)


def _keep_shortcuts(gdf, usable, is_cand, sidewalk, snap_m: float = 30.0,
                    ratio: float = 1.25, slack_m: float = 25.0) -> int:
    """Put back sidewalks the street network cannot stand in for.

    A footway can run beside a street without being interchangeable with it:
    the walkway across the Ballard Locks parallels a service road that stops
    at the water. Each end of a dropped sidewalk is snapped to the nearest
    node of the remaining network; if walking between those two nodes is much
    longer than the sidewalk itself (or an end has nothing nearby), the
    sidewalk is a shortcut and stays. Modifies ``sidewalk`` in place.
    """
    import networkx as nx
    from scipy.spatial import cKDTree

    u, v = gdf["u"].to_numpy(), gdf["v"].to_numpy()
    length = gdf["length_m"].to_numpy()
    start = np.array([g.coords[0] for g in gdf.geometry])
    end = np.array([g.coords[-1] for g in gdf.geometry])

    g = nx.Graph()
    for i in np.flatnonzero(usable & ~sidewalk):
        if not g.has_edge(u[i], v[i]) or g[u[i]][v[i]]["w"] > length[i]:
            g.add_edge(u[i], v[i], w=float(length[i]))
    # snap only to nodes of streets, paths and steps, not to other footway stubs
    solid = usable & ~is_cand
    xy = pd.concat([pd.DataFrame({"n": u[solid], "x": start[solid, 0], "y": start[solid, 1]}),
                    pd.DataFrame({"n": v[solid], "x": end[solid, 0], "y": end[solid, 1]})]
                   ).drop_duplicates("n")
    xy = xy[xy["n"].isin(g)]
    tree = cKDTree(xy[["x", "y"]].to_numpy())
    nodes = xy["n"].to_numpy()

    idx = np.flatnonzero(sidewalk)
    da, ia = tree.query(start[idx])
    db, ib = tree.query(end[idx])
    a, b = nodes[ia], nodes[ib]
    # an end that is still part of the network is its own snap point
    for ends, snap, d in ((u[idx], a, da), (v[idx], b, db)):
        live = np.fromiter((n in g for n in ends), bool, len(ends))
        snap[live] = ends[live]
        d[live] = 0.0
    budget = ratio * (length[idx] + da + db) + slack_m
    keep = (da > snap_m) | (db > snap_m)

    order = pd.DataFrame({"a": a, "k": np.arange(len(idx))})
    for src, grp in order[~keep].groupby("a"):
        ks = grp["k"].to_numpy()
        dist = nx.single_source_dijkstra_path_length(
            g, src, cutoff=float(budget[ks].max()), weight="w")
        for k in ks:
            if dist.get(b[k], np.inf) > budget[k]:
                keep[k] = True
    sidewalk[idx[keep]] = False
    return int(keep.sum())


def drop_untagged_sidewalks(gdf):
    """Remove sidewalks that Overture leaves as plain, untagged footways.

    The analysis routes along street centrelines, and a sidewalk on each side
    of every street triples the graph without changing a single route. A
    footway is taken to be a sidewalk when it runs alongside a walkable street
    (see ``SIDEWALK_OFFSET_M``). Three passes then repair the damage:

    * A sidewalk that is shorter than any way round by street is kept (see
      ``_keep_shortcuts``).
    * Steps, trails and named walkways often join the sidewalk rather than the
      street. Where dropping sidewalks would strand them, the shortest chain of
      dropped sidewalk that reconnects them to the network is put back.
    * Crossings and curb links are left dangling off the street once the
      sidewalks they joined are gone. Untagged footway stubs that dead-end
      inside a street's corridor are pruned, repeatedly, until none remain.
    """
    import networkx as nx

    if SIDEWALK_OFFSET_M is None:
        return gdf
    usable = (gdf["walk_ok"] | gdf["bike_ok"]).to_numpy()
    streets = gdf[~gdf["cls"].isin(_NON_STREET) & gdf["walk_ok"]]
    is_cand = ((gdf["cls"] == "footway") & gdf["subclass"].isna()
               & gdf["name"].isna() & ~gdf["is_structure"]).to_numpy() & usable
    near, par = _sidewalk_hits(gdf[is_cand], streets)
    sidewalk = np.zeros(len(gdf), dtype=bool)
    stub = np.zeros(len(gdf), dtype=bool)
    sidewalk[np.flatnonzero(is_cand)[par >= SIDEWALK_MIN_HITS]] = True
    stub[np.flatnonzero(is_cand)[near >= SIDEWALK_MIN_HITS]] = True
    stub &= ~sidewalk

    u, v = gdf["u"].to_numpy(), gdf["v"].to_numpy()
    length = gdf["length_m"].to_numpy()
    detours = _keep_shortcuts(gdf, usable, is_cand, sidewalk)

    # Reconnect: kept edges cost nothing, dropped sidewalk costs its length,
    # so the shortest path from the main component to a stranded node runs
    # over as little sidewalk as possible.
    g = nx.Graph()
    for i in np.flatnonzero(usable & ~sidewalk):
        g.add_edge(u[i], v[i], w=0.0, eid=-1)
    for i in np.flatnonzero(sidewalk):
        if not g.has_edge(u[i], v[i]) or g[u[i]][v[i]]["w"] > length[i]:
            g.add_edge(u[i], v[i], w=float(length[i]), eid=int(i))
    kept = g.edge_subgraph((a, b) for a, b, d in g.edges(data=True) if d["eid"] < 0)
    main = max(nx.connected_components(kept), key=len)
    pred, _ = nx.dijkstra_predecessor_and_distance(g, next(iter(main)), weight="w")
    important = usable & ~is_cand
    anchors = set(u[important]) | set(v[important])
    restored = 0
    seen: set = set()
    for node in anchors - main:
        while node in pred and pred[node] and node not in seen:
            seen.add(node)
            prev = pred[node][0]
            eid = g[prev][node]["eid"]
            if eid >= 0 and sidewalk[eid]:
                sidewalk[eid] = False
                restored += 1
            elif eid < 0 and prev in main:
                break
            node = prev

    # Prune crossing and curb-link stubs left dead-ending off the street.
    alive = usable & ~sidewalk
    pruned = 0
    while True:
        ends = pd.Series(np.r_[u[alive], v[alive]]).value_counts()
        deg_u = ends.reindex(u).fillna(0).to_numpy()
        deg_v = ends.reindex(v).fillna(0).to_numpy()
        dead = stub & alive & ((deg_u <= 1) | (deg_v <= 1))
        if not dead.any():
            break
        alive &= ~dead
        pruned += int(dead.sum())

    drop = usable & ~alive
    log.info("  untagged sidewalks: dropped %d edges (%.0f km); kept %d that are "
             "shortcuts and %d that connect steps and paths; pruned %d dangling stubs",
             int(sidewalk.sum()), length[sidewalk].sum() / 1000, detours, restored,
             pruned)
    return gdf[~drop].reset_index(drop=True)


def merge_pass_through_nodes(gdf):
    """Join consecutive pieces of one segment where nothing else meets them.

    Segments are split at every connector, including the ones where a
    driveway, crossing or sidewalk stub (all since removed) used to attach.
    Those leave a street in pieces at nodes that join nothing but the two
    pieces. Pieces of one segment share every attribute, so they can be put
    back together without changing a route.
    """
    import shapely

    deg = pd.concat([gdf["u"], gdf["v"]]).value_counts()
    s = gdf.sort_values(["segment_id", "part"])
    prev = s.groupby("segment_id")[["v"]].shift(1)["v"]
    joins = (prev == s["u"]).to_numpy() & (deg.reindex(s["u"]).to_numpy() == 2)
    run = np.cumsum(~joins)
    first = s.groupby(run)["u"].transform("first").to_numpy()
    last = s.groupby(run)["v"].transform("last").to_numpy()
    # never fold a piece of a loop back onto itself
    loop = pd.Series(first == last).groupby(run).transform("any").to_numpy()
    multi = pd.Series(run).map(pd.Series(run).value_counts()).to_numpy() > 1
    whole = ~multi | loop
    if (~whole).sum() == 0:
        return gdf
    # a looped run is left in pieces: give each piece its own run id
    run = np.where(loop, -np.arange(1, len(run) + 1), run)

    s = s.assign(_run=run)
    singles = s[whole]
    merged = s[~whole]
    rows = []
    for _, grp in merged.groupby("_run", sort=False):
        coords = [np.asarray(g.coords) for g in grp.geometry]
        line = np.vstack([coords[0]] + [c[1:] for c in coords[1:]])
        row = grp.iloc[0].copy()
        row["v"] = grp["v"].iloc[-1]
        row["geometry"] = shapely.LineString(line)
        row["length_m"] = grp["length_m"].sum()
        rows.append(row)
    out = pd.concat([singles, pd.DataFrame(rows)], ignore_index=True)
    out = out.drop(columns="_run").sort_values(["segment_id", "part"]).reset_index(drop=True)
    log.info("  merged %d pieces into %d edges at pass-through nodes (%d -> %d edges)",
             len(merged), len(rows), len(gdf), len(out))
    return out.set_geometry("geometry", crs=gdf.crs)


def build_edges(segments_path: Path, force: bool = False,
                clip_to_city: bool = True) -> "pd.DataFrame":
    """Produce the undirected edge table with geometry, class and access.

    The Overture extract covers a bounding box, which also catches the Marin
    headlands (reachable only across the Golden Gate Bridge and partly
    outside the lidar footprint) and northern San Mateo County.  Clipping to
    the union of the San Francisco neighborhood polygons keeps the analysis
    to the city, which is what the corridor and pass analysis is about.
    """
    import geopandas as gpd

    if EDGES_PARQUET.exists() and not force:
        log.info("cached %s", EDGES_PARQUET.name)
        return gpd.read_parquet(EDGES_PARQUET)

    df = load_segments(segments_path)
    with step("filtering segments", log):
        df = _base_filter(df)
    with step("splitting segments at connectors", log):
        rows = _split_geometries(df)

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_GEOGRAPHIC)
    gdf = gdf.to_crs(CRS_PROJECTED)
    gdf["length_m"] = gdf.geometry.length
    gdf = gdf[gdf["length_m"] > 0.5].reset_index(drop=True)
    log.info("  %d edges after dropping sub-metre slivers", len(gdf))

    if clip_to_city:
        from .neighborhoods import city_boundary
        with step("clipping network to the city boundary", log):
            boundary = city_boundary(buffer_m=250.0)
            mid = gdf.geometry.interpolate(0.5, normalized=True)
            keep = gpd.GeoSeries(mid, crs=gdf.crs).within(boundary)
            n_before = len(gdf)
            gdf = gdf[keep.to_numpy()].reset_index(drop=True)
            log.info("  %d -> %d edges inside the city (+250 m)", n_before, len(gdf))

    # structure flags
    gdf["is_bridge"] = gdf["flags"].apply(lambda f: "is_bridge" in f)
    gdf["is_tunnel"] = gdf["flags"].apply(lambda f: "is_tunnel" in f)
    gdf["is_structure"] = gdf["is_bridge"] | gdf["is_tunnel"]

    # per-mode access
    with step("evaluating per-mode access restrictions", log):
        for mode_name, mode in MODES.items():
            res = [evaluate_access(r, mode.access_modes[0]) for r in gdf["access_restrictions"]]
            gdf[f"{mode_name}_allowed"] = [r["allowed"] for r in res]
            gdf[f"{mode_name}_oneway"] = [r["oneway_forward_only"] for r in res]
            gdf[f"{mode_name}_restricted"] = [r["restricted"] for r in res]
            # class eligibility
            ok = gdf["cls"].isin(mode.allowed_classes)
            if mode_name == "bike":
                ok &= ~gdf["cls"].isin(BIKE_FORBIDDEN_CLASSES)
            gdf[f"{mode_name}_ok"] = ok & gdf[f"{mode_name}_allowed"]
            log.info("  %s: %d of %d edges routable", mode_name,
                     int(gdf[f"{mode_name}_ok"].sum()), len(gdf))

    with step("dropping untagged sidewalks", log):
        gdf = drop_untagged_sidewalks(gdf)
    with step("merging pass-through nodes", log):
        gdf = merge_pass_through_nodes(gdf)

    # bicycle infrastructure / low-stress proxy (SFMTA data unavailable)
    gdf["bike_facility"] = np.where(
        gdf["cls"] == "cycleway", "dedicated_cycleway",
        np.where(gdf["cls"].isin(["path", "footway"]) & gdf["bike_allowed"],
                 "shared_path",
                 np.where(gdf["cls"] == "living_street", "living_street",
                          np.where(gdf["cls"] == "pedestrian", "car_free_street", ""))))
    gdf["low_stress"] = gdf["bike_facility"].astype(bool) | gdf["cls"].isin(
        ["residential", "living_street", "pedestrian"])

    gdf["edge_id"] = np.arange(len(gdf), dtype=np.int64)
    keep = ["edge_id", "segment_id", "u", "v", "part", "cls", "subclass", "name",
            "length_m", "is_bridge", "is_tunnel", "is_structure",
            "walk_ok", "walk_oneway", "walk_restricted",
            "bike_ok", "bike_oneway", "bike_restricted",
            "bike_facility", "low_stress", "geometry"]
    gdf = gdf[keep]
    gdf.to_parquet(EDGES_PARQUET)
    log.info("wrote %s (%d edges)", EDGES_PARQUET.name, len(gdf))
    return gdf


# --------------------------------------------------------------------------
# graph assembly
# --------------------------------------------------------------------------
def largest_component(edges, mode: str):
    """Restrict an edge table to the largest connected component for a mode."""
    import networkx as nx
    sub = edges[edges[f"{mode}_ok"]]
    g = nx.Graph()
    g.add_edges_from(zip(sub["u"], sub["v"]))
    if g.number_of_nodes() == 0:
        raise RuntimeError(f"no routable edges for mode {mode}")
    comp = max(nx.connected_components(g), key=len)
    keep = sub["u"].isin(comp) & sub["v"].isin(comp)
    log.info("mode %s: largest component has %d nodes, %d of %d edges",
             mode, len(comp), int(keep.sum()), len(sub))
    return sub[keep].copy(), comp
