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
                     PROCESSED_DIR)
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
