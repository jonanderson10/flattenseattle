"""Directed edge metrics: climbing, grade distribution and steep distance.

Every undirected edge becomes two directed edges, because climbing cost is
direction dependent: the uphill direction of Filbert Street is a wall and the
downhill direction is a brake test, and a routing model that averages the two
is useless.

Definitions
-----------
``cum_gain`` / ``cum_loss``
    Sum of positive / negative steps between the profile's turning points
    after every reversal smaller than the dead-band ``db`` has been pruned
    away (see :func:`prune_reversals`).  DEM noise cannot invent climbing,
    because an oscillation below ``db`` is deleted outright; and a genuine
    climb is never shortened, because pruning only ever removes interior
    wiggles and never moves the endpoints.  The latter property matters: an
    earlier backlash-operator implementation charged one dead-band per edge
    and so under-reported long climbs by up to 17 m over Twin Peaks.

    The filter runs **once**, in the geometric direction of the edge, and the
    reverse direction's gain is *defined* as the forward direction's loss.
    That makes the pair exactly consistent: gain(u->v) == loss(v->u) always,
    which a direction-by-direction re-filter would not guarantee.

``avg_grade``
    Signed net elevation change over length -- the average grade experienced
    travelling in that direction.

``max_grade``
    Steepest single sampled interval *in the direction of travel* (positive
    = climbing).  Clipped at 60% to reject DEM artefacts; the steepest
    drivable street in San Francisco is about 31.5%.

``p95_grade``
    Length-weighted 95th percentile of the signed interval grade -- a
    robust "how steep is this really" number that a single bad sample cannot
    dominate.

``d_above_XX``
    Distance travelled while *climbing* at or above the threshold.  This is
    the quantity the routing penalties act on.

``d_abs_above_XX``
    Distance at or above the threshold in absolute value, regardless of
    direction; used for the grade map and descriptive statistics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (ELEVATION, GRADE_PERCENTILE, GRADE_THRESHOLDS,
                     MIN_RELIABLE_GRADE_LENGTH_M, PROCESSED_DIR)
from .utils import get_logger, progress, step

log = get_logger("flatten_seattle.metrics")

DIRECTED_PARQUET = PROCESSED_DIR / "edges_directed.parquet"
UNDIRECTED_PARQUET = PROCESSED_DIR / "edges_metrics.parquet"
#: GeoPackage mirror of the processed network, for use in desktop GIS.
NETWORK_GPKG = PROCESSED_DIR / "street_network.gpkg"


# --------------------------------------------------------------------------
# core numerics (pure functions -- these are what the tests exercise)
# --------------------------------------------------------------------------
def deadband_filter(z: np.ndarray, deadband: float) -> np.ndarray:
    """Backlash / hysteresis filter (retained for comparison and tests).

    ``zf[i]`` tracks ``z[i]`` but refuses to move until the input has moved
    more than ``deadband`` from the current output.  Simple and monotone, but
    it costs one whole ``deadband`` at the start of every monotonic run,
    which is why it is *not* used for the published gain figures -- see
    :func:`prune_reversals`.
    """
    z = np.asarray(z, dtype="float64")
    if deadband <= 0 or z.size == 0:
        return z.copy()
    out = np.empty_like(z)
    out[0] = z[0]
    prev = z[0]
    for i in range(1, z.size):
        zi = z[i]
        if zi > prev + deadband:
            prev = zi - deadband
        elif zi < prev - deadband:
            prev = zi + deadband
        out[i] = prev
    return out


def _to_reversals(z: np.ndarray) -> list[float]:
    """Collapse a profile to its alternating sequence of turning points."""
    ext: list[float] = [float(z[0])]
    for v in z[1:]:
        v = float(v)
        if len(ext) == 1:
            if v != ext[0]:
                ext.append(v)
            continue
        if (ext[-1] - ext[-2]) * (v - ext[-1]) >= 0:
            ext[-1] = v                 # still going the same way
        else:
            ext.append(v)               # direction changed
    return ext


def prune_reversals(z: np.ndarray, deadband: float) -> list[float]:
    """Dead-band filter that removes small oscillations without shortening climbs.

    The profile is reduced to its turning points, and then any *interior*
    reversal whose amplitude is below ``deadband`` is deleted, merging the
    monotonic runs either side of it.  Deletion repeats, smallest first, until
    every surviving reversal is larger than the dead-band.

    This is the filter used for the published ``cum_gain``/``cum_loss``,
    because unlike the backlash operator it charges *nothing* for a genuine
    sustained climb: a clean 100 m climb returns exactly 100 m, whether it is
    measured in one piece or split across fifty consecutive edges.  The
    backlash operator loses one dead-band per edge, which systematically
    under-reported long climbs -- on a route over Twin Peaks the error reached
    17 m.  Endpoints are never removed, so gain minus loss always equals the
    true net elevation change.
    """
    z = np.asarray(z, dtype="float64")
    if z.size == 0:
        return []
    if z.size == 1:
        return [float(z[0])]
    ext = _to_reversals(z)
    if deadband <= 0:
        return ext
    while len(ext) > 2:
        best_i, best_a = -1, np.inf
        for i in range(1, len(ext) - 2):
            a = abs(ext[i + 1] - ext[i])
            if a < deadband and a < best_a:
                best_i, best_a = i, a
        if best_i < 0:
            break
        del ext[best_i:best_i + 2]
        ext = _to_reversals(np.asarray(ext))
    return ext


def _reversal_indices(z: np.ndarray) -> list[int]:
    """Indices of the alternating turning points of a profile."""
    idx: list[int] = [0]
    for i in range(1, z.size):
        if len(idx) == 1:
            if z[i] != z[idx[0]]:
                idx.append(i)
            continue
        if (z[idx[-1]] - z[idx[-2]]) * (z[i] - z[idx[-1]]) >= 0:
            idx[-1] = i
        else:
            idx.append(i)
    if idx[-1] != z.size - 1:
        idx.append(z.size - 1)
    return idx


def prune_reversal_indices(z: np.ndarray, deadband: float,
                           protected: set[int] | None = None) -> list[int]:
    """Turning-point indices surviving the dead-band, endpoints always kept.

    ``protected`` indices are never pruned.  Callers pass the stations that
    fall on an edge boundary, so that the filtered profile passes exactly
    through the measured elevation at every intersection.  That makes
    ``cum_gain - cum_loss`` equal ``net_change`` exactly for each edge, and
    makes the per-edge figures telescope exactly along a route.
    """
    z = np.asarray(z, dtype="float64")
    protected = protected or set()
    if z.size < 2:
        return list(range(z.size))
    idx = _reversal_indices(z)
    if deadband <= 0:
        return sorted(set(idx) | {i for i in protected if 0 <= i < z.size})
    while len(idx) > 2:
        best_i, best_a = -1, np.inf
        for k in range(1, len(idx) - 2):
            if idx[k] in protected or idx[k + 1] in protected:
                continue
            a = abs(z[idx[k + 1]] - z[idx[k]])
            if a < deadband and a < best_a:
                best_i, best_a = k, a
        if best_i < 0:
            break
        del idx[best_i:best_i + 2]
        # re-collapse any now-monotonic neighbours
        keep = [idx[0]]
        for k in range(1, len(idx) - 1):
            prev, cur, nxt = z[keep[-1]], z[idx[k]], z[idx[k + 1]]
            if (cur - prev) * (nxt - cur) >= 0 and len(keep) >= 1:
                continue
            keep.append(idx[k])
        keep.append(idx[-1])
        idx = keep
    if protected:
        idx = sorted(set(idx) | {i for i in protected if 0 <= i < z.size})
    return idx


def rectify_profile(dist: np.ndarray, z: np.ndarray, deadband: float,
                    protected: set[int] | None = None) -> np.ndarray:
    """Profile with sub-dead-band oscillations removed, sampled everywhere.

    Between two surviving turning points the profile is replaced by its
    **monotone envelope**, clamped to the run's endpoints: a running maximum
    on a climbing run, a running minimum on a descending one.  Small
    oscillations are flattened against the running extremum while the shape
    of the run is otherwise untouched.

    The properties the metrics rely on:

    * an oscillation smaller than ``deadband`` contributes *no* gain;
    * a sustained climb is preserved to the millimetre;
    * gain is **additive**: the gains of any partition sum to the gain of the
      whole, so per-edge figures can be summed along a route without drift;
    * **the shape within a run is preserved**, so slicing the profile gives
      each edge the climbing that actually happens on it.

    An earlier version joined the surviving turning points with straight
    lines instead.  That got the total right but redistributed it: on a
    500 m segment that is level for 400 m and then climbs hard, every edge
    of the level part was charged with a share of the climb, with errors
    reaching 51 m.
    """
    dist = np.asarray(dist, dtype="float64")
    z = np.asarray(z, dtype="float64")
    if z.size < 2:
        return z.copy()
    idx = prune_reversal_indices(z, deadband, protected)
    out = z.copy()
    for k in range(len(idx) - 1):
        a, b = idx[k], idx[k + 1]
        seg = z[a:b + 1]
        if z[b] >= z[a]:
            out[a:b + 1] = np.minimum(np.maximum.accumulate(seg), z[b])
        else:
            out[a:b + 1] = np.maximum(np.minimum.accumulate(seg), z[b])
    return out


def cumulative_gain_loss(z: np.ndarray, deadband: float,
                         dist: np.ndarray | None = None) -> tuple[float, float]:
    """Cumulative gain and loss of a profile, small oscillations removed."""
    z = np.asarray(z, dtype="float64")
    if z.size < 2:
        return 0.0, 0.0
    if dist is None:
        dist = np.arange(z.size, dtype="float64")
    zr = rectify_profile(dist, z, deadband)
    d = np.diff(zr)
    return float(d[d > 0].sum()), float(-d[d < 0].sum())


def interval_grades(dist: np.ndarray, z: np.ndarray,
                    max_grade: float = ELEVATION.max_plausible_grade):
    """Signed grade and length of each sampled interval.

    Returns ``(grades, lengths)``. Grades are clipped to +/-``max_grade``.
    """
    dd = np.diff(np.asarray(dist, dtype="float64"))
    dz = np.diff(np.asarray(z, dtype="float64"))
    ok = dd > 1e-9
    g = np.zeros_like(dd)
    g[ok] = dz[ok] / dd[ok]
    return np.clip(g, -max_grade, max_grade), dd


def weighted_percentile(values: np.ndarray, weights: np.ndarray,
                        pct: float) -> float:
    """Length-weighted percentile (linear interpolation on the CDF)."""
    values = np.asarray(values, dtype="float64")
    weights = np.asarray(weights, dtype="float64")
    if values.size == 0 or weights.sum() <= 0:
        return 0.0
    order = np.argsort(values)
    v = values[order]; w = weights[order]
    cw = np.cumsum(w)
    cutoff = (pct / 100.0) * cw[-1]
    idx = int(np.searchsorted(cw, cutoff, side="left"))
    return float(v[min(idx, v.size - 1)])


def distance_above(grades: np.ndarray, lengths: np.ndarray,
                   threshold: float, absolute: bool = False) -> float:
    """Distance travelled at (signed or absolute) grade >= ``threshold``."""
    g = np.abs(grades) if absolute else grades
    return float(lengths[g >= threshold].sum())


def directional_metrics(dist: np.ndarray, elev: np.ndarray,
                        deadband: float = ELEVATION.gain_deadband_m,
                        thresholds=GRADE_THRESHOLDS,
                        rectified: np.ndarray | None = None) -> dict:
    """All metrics for both directions of one edge, from one profile pass.

    ``rectified`` lets the caller supply the dead-band-filtered profile
    computed over the whole parent street segment, so that the filter is not
    restarted at every edge boundary.  Grades are always taken from the
    unrectified (but smoothed) profile, because rectification deliberately
    flattens short features and would understate real pitches.
    """
    dist = np.asarray(dist, dtype="float64")
    elev = np.asarray(elev, dtype="float64")
    length = float(dist[-1]) if dist.size else 0.0

    if rectified is None:
        gain_f, loss_f = cumulative_gain_loss(elev, deadband, dist)
    else:
        d = np.diff(np.asarray(rectified, dtype="float64"))
        gain_f = float(d[d > 0].sum())
        loss_f = float(-d[d < 0].sum())
    grades, lens = interval_grades(dist, elev)

    z0, z1 = float(elev[0]), float(elev[-1])
    net_f = z1 - z0
    abs_max = float(np.max(np.abs(grades))) if grades.size else 0.0

    out = {
        "length_m": length,
        "max_abs_grade": abs_max,
        "elev_min": float(np.min(elev)), "elev_max": float(np.max(elev)),
    }
    for th in thresholds:
        out[f"d_abs_above_{int(th*100)}"] = distance_above(grades, lens, th, True)

    for direction, sign in (("fwd", 1.0), ("rev", -1.0)):
        g = grades * sign
        if sign < 0:
            g = g[::-1]; L = lens[::-1]
        else:
            L = lens
        d = {
            "start_elev": z0 if sign > 0 else z1,
            "end_elev": z1 if sign > 0 else z0,
            "net_change": net_f * sign,
            "cum_gain": gain_f if sign > 0 else loss_f,
            "cum_loss": loss_f if sign > 0 else gain_f,
            "avg_grade": (net_f * sign / length) if length > 0 else 0.0,
            "max_grade": float(np.max(g)) if g.size else 0.0,
            "p95_grade": weighted_percentile(g, L, GRADE_PERCENTILE),
            "mean_abs_grade": (float(np.sum(np.abs(g) * L) / L.sum())
                               if L.sum() > 0 else 0.0),
        }
        for th in thresholds:
            d[f"d_above_{int(th*100)}"] = distance_above(g, L, th, False)
        out[direction] = d
    return out


# --------------------------------------------------------------------------
# table construction
# --------------------------------------------------------------------------
_DIR_FIELDS = ("start_elev", "end_elev", "net_change", "cum_gain", "cum_loss",
               "avg_grade", "max_grade", "p95_grade", "mean_abs_grade")


def contiguous_runs(edges) -> list[np.ndarray]:
    """Edge ids grouped into runs of consecutive parts of one street segment.

    Clipping to the city boundary (and dropping sub-metre slivers) can delete
    a middle piece of a segment, leaving its surviving parts non-contiguous.
    Anything that concatenates edge profiles must respect that, or it splices
    two unrelated ends together and invents a cliff.
    """
    order = edges[["edge_id", "segment_id", "part"]].sort_values(
        ["segment_id", "part"])
    runs: list[np.ndarray] = []
    for _seg, grp in order.groupby("segment_id", sort=False):
        parts = grp["part"].to_numpy()
        eids = grp["edge_id"].to_numpy()
        breaks = np.flatnonzero(np.diff(parts) != 1) + 1
        for chunk in np.split(eids, breaks):
            if len(chunk):
                runs.append(chunk)
    return runs


def _rectify_by_segment(edges, profiles: dict) -> dict[int, np.ndarray]:
    """Dead-band filter each *original street segment*, then split it back up.

    Edges are pieces of Overture segments, cut at intersections, so a single
    continuous climb is spread over many edges.  Applying the dead-band to
    each edge separately restarts the filter constantly, which is how the
    earlier implementation lost up to 17 m of real climbing on a route over
    Twin Peaks.  Rectifying the concatenated segment profile and then slicing
    it keeps the filter's noise rejection while making per-edge gains sum
    correctly along the street.
    """
    deadband = ELEVATION.gain_deadband_m
    out: dict[int, np.ndarray] = {}
    for eids in contiguous_runs(edges):
        if len(eids) == 1:
            d, z = profiles[int(eids[0])]
            out[int(eids[0])] = rectify_profile(d, z, deadband,
                                                protected={0, len(z) - 1})
            continue
        # concatenate, dropping the duplicated joint station each time
        segs, base, spans = [], 0.0, []
        zs = []
        for k, eid in enumerate(eids):
            d, z = profiles[int(eid)]
            d = np.asarray(d, dtype="float64")
            z = np.asarray(z, dtype="float64")
            if k > 0:
                d = d[1:]; z = z[1:]
            segs.append(base + d)
            zs.append(z)
            spans.append(len(d))
            base = segs[-1][-1] if segs[-1].size else base
        cat_d = np.concatenate(segs)
        cat_z = np.concatenate(zs)
        # protect every edge boundary station, including both extremes
        bounds, acc = {0, cat_z.size - 1}, 0
        for n in spans:
            acc += n
            bounds.add(min(max(acc - 1, 0), cat_z.size - 1))
        cat_r = rectify_profile(cat_d, cat_z, deadband, protected=bounds)
        # slice back, re-attaching the shared joint station
        pos = 0
        for k, eid in enumerate(eids):
            n = spans[k]
            lo = pos if k == 0 else pos - 1
            hi = pos + n
            out[int(eid)] = cat_r[lo:hi]
            pos = hi
    return out


def compute_edge_metrics(edges, profiles: dict, force: bool = False):
    """Attach undirected metrics to the edge table and build the directed table.

    Returns ``(undirected_gdf, directed_df)``.
    """
    import geopandas as gpd

    if (UNDIRECTED_PARQUET.exists() and DIRECTED_PARQUET.exists() and not force):
        log.info("cached edge metrics")
        return (gpd.read_parquet(UNDIRECTED_PARQUET),
                pd.read_parquet(DIRECTED_PARQUET))

    th_names = [int(t * 100) for t in GRADE_THRESHOLDS]
    und_rows: list[dict] = []
    dir_rows: list[dict] = []

    usable = np.array([np.all(np.isfinite(profiles[int(e)][1]))
                       for e in edges["edge_id"].values])
    if not usable.all():
        log.warning("dropping %d edges whose elevation could not be "
                    "determined at all", int((~usable).sum()))
        edges = edges[usable].reset_index(drop=True)

    with step("rectifying elevation profiles per street segment", log):
        rect = _rectify_by_segment(edges, profiles)

    with step(f"computing metrics for {len(edges):,} edges "
              f"({2*len(edges):,} directed)", log):
        for edge_id, length in progress(
                zip(edges["edge_id"].values, edges["length_m"].values),
                desc="  metrics", total=len(edges), unit="edge"):
            dist, elev = profiles[int(edge_id)]
            m = directional_metrics(dist, elev, rectified=rect.get(int(edge_id)))
            und = {"edge_id": int(edge_id),
                   "max_abs_grade": m["max_abs_grade"],
                   "elev_min": m["elev_min"], "elev_max": m["elev_max"],
                   "cum_gain_fwd": m["fwd"]["cum_gain"],
                   "cum_loss_fwd": m["fwd"]["cum_loss"],
                   "net_change_fwd": m["fwd"]["net_change"],
                   "avg_grade_fwd": m["fwd"]["avg_grade"],
                   "n_samples": int(len(dist))}
            for t in th_names:
                und[f"d_abs_above_{t}"] = m[f"d_abs_above_{t}"]
            und_rows.append(und)

            for direction in ("fwd", "rev"):
                d = m[direction]
                row = {"edge_id": int(edge_id), "direction": direction}
                row.update({k: d[k] for k in _DIR_FIELDS})
                for t in th_names:
                    row[f"d_above_{t}"] = d[f"d_above_{t}"]
                dir_rows.append(row)

    und = pd.DataFrame(und_rows)
    gdf = edges.merge(und, on="edge_id", how="left")
    # a very short edge cannot support a trustworthy maximum grade
    gdf["grade_reliable"] = gdf["length_m"] >= MIN_RELIABLE_GRADE_LENGTH_M
    gdf["max_abs_grade_reliable"] = np.where(
        gdf["grade_reliable"], gdf["max_abs_grade"], np.nan)
    log.info("  %d of %d edges are long enough (>=%.0f m) for a reliable "
             "maximum grade", int(gdf["grade_reliable"].sum()), len(gdf),
             MIN_RELIABLE_GRADE_LENGTH_M)

    directed = pd.DataFrame(dir_rows)
    # attach topology / class / access for each direction
    base = edges[["edge_id", "u", "v", "length_m", "cls", "subclass", "name",
                  "is_structure", "walk_ok", "walk_oneway", "bike_ok",
                  "bike_oneway", "bike_facility", "low_stress"]]
    directed = directed.merge(base, on="edge_id", how="left")
    fwd = directed["direction"] == "fwd"
    directed["from_node"] = np.where(fwd, directed["u"], directed["v"])
    directed["to_node"] = np.where(fwd, directed["v"], directed["u"])
    # one-way handling: pedestrians ignore it, bicycles do not
    directed["walk_traversable"] = directed["walk_ok"]
    directed["bike_traversable"] = directed["bike_ok"] & ~(
        directed["bike_oneway"] & ~fwd)
    directed = directed.drop(columns=["u", "v"])

    gdf.to_parquet(UNDIRECTED_PARQUET)
    directed.to_parquet(DIRECTED_PARQUET)
    # A GeoPackage mirror so the processed network opens directly in QGIS or
    # ArcGIS; list columns have no GeoPackage equivalent, so they are dropped.
    try:
        export = gdf.drop(columns=[c for c in gdf.columns
                                   if gdf[c].dtype == object
                                   and c not in ("cls", "subclass", "name",
                                                 "segment_id", "u", "v",
                                                 "bike_facility", "geometry")])
        export.to_file(NETWORK_GPKG, layer="streets", driver="GPKG")
        log.info("wrote %s (%d features)", NETWORK_GPKG.name, len(export))
    except Exception as exc:                        # pragma: no cover
        log.warning("could not write the GeoPackage mirror: %s", exc)
    log.info("wrote %s and %s", UNDIRECTED_PARQUET.name, DIRECTED_PARQUET.name)
    log.info("directed edges traversable: walk %d, bike %d",
             int(directed["walk_traversable"].sum()),
             int(directed["bike_traversable"].sum()))
    return gdf, directed
