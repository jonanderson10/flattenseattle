"""Neighborhood boundaries, city boundary and representative access points.

Choosing an origin/destination point per neighborhood matters more than it
looks.  A polygon centroid can easily land in the middle of a park, on a
hillside with no street, in the water, or (for a concave neighborhood like
the Presidio or Lakeshore) outside the neighborhood altogether.  Routing from
such a point produces garbage distances.

The representative point is therefore chosen as follows:

1.  Take every graph node inside the neighborhood that lies on the mode's
    largest connected component.
2.  Compute a **street-weighted centre**: the mean node position weighted by
    the length of street incident on each node.  Street length is a far
    better proxy for where people actually start journeys than polygon area,
    so parks, cliffs and water pull the centre much less than they pull a
    geometric centroid.
3.  Snap to the nearest *qualifying intersection* -- a node of degree >= 3 on
    a named street of an ordinary urban class.  Falling back progressively to
    looser criteria guarantees a usable point for every neighborhood.

The offset between the geometric centroid and the chosen point is recorded so
the choice is auditable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (ANALYSIS, CRS_PROJECTED, EXCLUDED_NEIGHBORHOODS,
                     PROCESSED_DIR)
from .download import NEIGHBORHOODS_GEOJSON
from .utils import get_logger, step

log = get_logger("sf_flat_routes.neighborhoods")

NEIGHBORHOODS_GPKG = PROCESSED_DIR / "neighborhoods.gpkg"
POINTS_GPKG = PROCESSED_DIR / "neighborhood_points.gpkg"

#: Classes that make a node a credible journey origin (not a park path,
#: alley, driveway or freeway ramp).
_GOOD_CLASSES = ("residential", "living_street", "tertiary", "secondary",
                 "primary", "unclassified", "trunk", "pedestrian")


def load_neighborhoods(path: Path = NEIGHBORHOODS_GEOJSON, force: bool = False):
    """Load neighborhood polygons, projected, with cleaned attributes."""
    import geopandas as gpd

    if NEIGHBORHOODS_GPKG.exists() and not force:
        return gpd.read_file(NEIGHBORHOODS_GPKG)

    gdf = gpd.read_file(path)
    # Seattle's Neighborhood Map Atlas names the polygon S_HOOD
    gdf = gdf.rename(columns={"S_HOOD": "neighborhood", "name": "neighborhood"})
    gdf = gdf[["neighborhood", "geometry"]].copy()
    gdf = gdf.to_crs(CRS_PROJECTED)
    gdf["geometry"] = gdf.geometry.buffer(0)          # repair any self-touching rings
    gdf["area_km2"] = gdf.geometry.area / 1e6
    gdf = gdf.sort_values("neighborhood").reset_index(drop=True)
    NEIGHBORHOODS_GPKG.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(NEIGHBORHOODS_GPKG, driver="GPKG")
    log.info("loaded %d neighborhoods, %.1f km2 total",
             len(gdf), gdf["area_km2"].sum())
    return gdf


def city_boundary(neighborhoods=None, buffer_m: float = 250.0):
    """Union of the neighborhood polygons, optionally buffered."""
    from shapely.ops import unary_union

    nb = load_neighborhoods() if neighborhoods is None else neighborhoods
    geom = unary_union(nb.geometry.values)
    if buffer_m:
        geom = geom.buffer(buffer_m)
    return geom


def analysis_neighborhoods(neighborhoods=None):
    """Neighborhoods used for pair analysis (excludes unreachable islands)."""
    nb = load_neighborhoods() if neighborhoods is None else neighborhoods
    return nb[~nb["neighborhood"].isin(EXCLUDED_NEIGHBORHOODS)].reset_index(drop=True)


# --------------------------------------------------------------------------
# representative points
# --------------------------------------------------------------------------
def _node_table(edges) -> pd.DataFrame:
    """Node coordinates, incident street length and best incident class."""
    rows = []
    for u, v, geom, length, cls, name in zip(
            edges["u"], edges["v"], edges.geometry, edges["length_m"],
            edges["cls"], edges["name"]):
        c = geom.coords
        good = cls in _GOOD_CLASSES
        named = bool(name)
        rows.append((u, c[0][0], c[0][1], length, good, named))
        rows.append((v, c[-1][0], c[-1][1], length, good, named))
    df = pd.DataFrame(rows, columns=["node", "x", "y", "length_m", "good", "named"])
    agg = df.groupby("node").agg(
        x=("x", "first"), y=("y", "first"),
        street_len=("length_m", "sum"), degree=("length_m", "size"),
        good=("good", "max"), named=("named", "max")).reset_index()
    return agg


def choose_representative_points(edges, neighborhoods=None, mode: str = "walk",
                                 valid_nodes=None, force: bool = False):
    """Pick one representative, network-snapped access point per neighborhood.

    ``valid_nodes`` restricts candidates to nodes that are actually routable
    for this mode (the largest strongly connected component of the mode's
    graph).  Without it a neighborhood such as the Presidio can be handed a
    node that exists only on a footpath, which is unreachable by bicycle.
    """
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.prepared import prep

    out_path = POINTS_GPKG.with_name(f"neighborhood_points_{mode}.gpkg")
    if out_path.exists() and not force:
        return gpd.read_file(out_path)

    nb = analysis_neighborhoods(neighborhoods)
    mode_edges = edges[edges[f"{mode}_ok"]] if f"{mode}_ok" in edges.columns else edges
    nodes = _node_table(mode_edges)
    if valid_nodes is not None:
        before = len(nodes)
        nodes = nodes[nodes["node"].isin(set(valid_nodes))].reset_index(drop=True)
        log.info("%s: %d of %d nodes are in the routable component",
                 mode, len(nodes), before)
    node_pts = gpd.GeoSeries(gpd.points_from_xy(nodes["x"], nodes["y"]),
                             crs=CRS_PROJECTED)
    sidx = node_pts.sindex

    records = []
    with step(f"choosing representative points for {len(nb)} neighborhoods ({mode})",
              log):
        for _, row in nb.iterrows():
            poly = row.geometry
            cand_idx = list(sidx.query(poly, predicate="intersects"))
            if not cand_idx:
                log.warning("%s: no network nodes inside polygon; using centroid",
                            row["neighborhood"])
                p = poly.representative_point()
                records.append({"neighborhood": row["neighborhood"], "node": None,
                                "geometry": p, "offset_m": 0.0,
                                "selection": "centroid_no_nodes"})
                continue
            pr = prep(poly)
            inside = [i for i in cand_idx if pr.contains(node_pts.iloc[i])]
            if not inside:
                inside = cand_idx
            sub = nodes.iloc[inside]

            # street-length-weighted centre of the neighborhood's network
            w = sub["street_len"].to_numpy()
            w = w / w.sum() if w.sum() > 0 else np.full(len(sub), 1 / len(sub))
            cx = float((sub["x"].to_numpy() * w).sum())
            cy = float((sub["y"].to_numpy() * w).sum())

            # snap to the nearest qualifying intersection, loosening criteria
            for crit, label in (
                ((sub["degree"] >= 3) & sub["good"] & sub["named"], "intersection"),
                ((sub["degree"] >= 3) & sub["good"], "intersection_unnamed"),
                (sub["good"], "street_node"),
                (sub["degree"] >= 3, "any_intersection"),
                (pd.Series(True, index=sub.index), "any_node"),
            ):
                pool = sub[crit.to_numpy()] if hasattr(crit, "to_numpy") else sub[crit]
                if len(pool):
                    d2 = ((pool["x"] - cx) ** 2 + (pool["y"] - cy) ** 2).to_numpy()
                    order = np.argsort(d2, kind="stable")
                    best = pool.iloc[int(order[min(ANALYSIS.point_rank, len(order) - 1)])]
                    centroid = poly.centroid
                    pt = Point(best["x"], best["y"])
                    records.append({
                        "neighborhood": row["neighborhood"],
                        "node": best["node"],
                        "geometry": pt,
                        "offset_m": float(pt.distance(centroid)),
                        "centroid_inside": bool(poly.contains(centroid)),
                        "selection": label,
                        "degree": int(best["degree"]),
                    })
                    break

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=CRS_PROJECTED)
    gdf.to_file(out_path, driver="GPKG")
    far = gdf[gdf["offset_m"] > 400]
    log.info("representative points: %d chosen; %d more than 400 m from the "
             "polygon centroid", len(gdf), len(far))
    for _, r in far.iterrows():
        log.info("   %-26s %6.0f m from centroid (%s)", r["neighborhood"],
                 r["offset_m"], r["selection"])
    return gdf


def assign_edges_to_neighborhoods(edges, neighborhoods=None):
    """Label each edge with the neighborhood containing its midpoint."""
    import geopandas as gpd

    nb = load_neighborhoods() if neighborhoods is None else neighborhoods
    mid = edges.geometry.interpolate(0.5, normalized=True)
    pts = gpd.GeoDataFrame({"edge_id": edges["edge_id"].values},
                           geometry=mid.values, crs=CRS_PROJECTED)
    joined = gpd.sjoin(pts, nb[["neighborhood", "geometry"]], how="left",
                       predicate="within")
    joined = joined.drop_duplicates("edge_id")
    return joined.set_index("edge_id")["neighborhood"]
