"""Elevation sampling from the USGS 3DEP 1 m lidar DEM.

Accuracy of the climbing metrics rests entirely on this module, so the
method is spelled out here and mirrored in the README.

1.  **Mosaic.**  The four 1 m 3DEP tiles are merged and clipped to the study
    area once, in their native EPSG:26910, and cached.  No raster
    reprojection is ever performed, so no resampling error is introduced.

2.  **Noise suppression (spatial).**  A Gaussian filter of sigma = 3 m is
    applied to the DEM before sampling.  A bare-earth lidar DEM still
    contains decimetre-scale artefacts from curbs, parked vehicles,
    vegetation misclassification and interpolation over occlusions.  A 3 m
    sigma is far narrower than a San Francisco street (15-25 m kerb to kerb)
    and far narrower than the ~100 m block scale on which real street grade
    varies, so block-scale slope is preserved while artefacts are damped.
    Smoothing the raster rather than the per-edge profile means the result is
    continuous across edge boundaries and is well defined even for edges only
    a few metres long.

3.  **Sampling.**  Each edge geometry is densified at 10 m spacing
    (endpoints always included, minimum 3 samples) and sampled with bilinear
    interpolation.

4.  **Structures.**  Where an edge is flagged ``is_bridge`` or ``is_tunnel``
    the DEM describes the ground or water surface under the deck, not the
    deck.  Elevations on such edges are replaced by a linear ramp between
    their endpoints.  Endpoint elevations that are themselves unreliable
    (nodes touched only by structure edges, e.g. mid-viaduct) are recovered
    by solving a discrete Laplace problem over the structure sub-graph with
    the reliable nodes as boundary conditions -- i.e. the deck is modelled as
    the smoothest ramp consistent with where it meets the ground.

5.  **Noise suppression (profile).**  Profiles with at least 5 samples get a
    Savitzky-Golay filter (order 2) over a ~50 m window.

6.  **Dead-band.**  Cumulative gain and loss ignore consecutive elevation
    changes smaller than 0.5 m.  Without this, residual DEM noise makes a
    genuinely flat street accumulate tens of metres of phantom climbing over
    a few kilometres.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import CITY_BBOX, CRS_PROJECTED, ELEVATION, PROCESSED_DIR
from .download import DEM_DIR
from .utils import get_logger, progress, step

log = get_logger("sf_flat_routes.elevation")

DEM_MOSAIC = PROCESSED_DIR / "dem_sf_1m.tif"
PROFILES_NPZ = PROCESSED_DIR / "edge_profiles.npz"

#: Standard deviation (m) of the Gaussian pre-filter applied to the DEM
#: (configured in ``config.ElevationConfig.dem_sigma_m``).
DEM_SMOOTH_SIGMA_M = ELEVATION.dem_sigma_m
WATER_BELOW_M = ELEVATION.water_below_m


# --------------------------------------------------------------------------
# DEM mosaic
# --------------------------------------------------------------------------
def build_dem_mosaic(force: bool = False) -> Path:
    """Merge and clip the 1 m tiles to the study area (cached)."""
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import transform_bounds

    if DEM_MOSAIC.exists() and not force:
        log.info("cached %s", DEM_MOSAIC.name)
        return DEM_MOSAIC

    tiles = sorted(DEM_DIR.glob("*.tif"))
    if not tiles:
        raise FileNotFoundError(
            f"no DEM tiles in {DEM_DIR}; run `python -m sf_flat_routes download`")

    lon_min, lon_max, lat_min, lat_max = CITY_BBOX
    bounds = transform_bounds("EPSG:4326", CRS_PROJECTED,
                              lon_min, lat_min, lon_max, lat_max)
    # pad so that edge densification never samples outside the mosaic
    pad = 300.0
    bounds = (bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad)

    srcs = [rasterio.open(t) for t in tiles]
    try:
        with step(f"merging {len(srcs)} 1 m DEM tiles over the study area", log):
            arr, transform = merge(srcs, bounds=bounds, res=(1.0, 1.0),
                                   nodata=srcs[0].nodata)
        profile = srcs[0].profile.copy()
    finally:
        for s in srcs:
            s.close()

    profile.update(height=arr.shape[1], width=arr.shape[2], transform=transform,
                   count=1, dtype="float32", compress="lzw", tiled=True,
                   blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
    DEM_MOSAIC.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEM_MOSAIC.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(arr[0].astype("float32"), 1)
    tmp.replace(DEM_MOSAIC)
    log.info("wrote %s  shape=%s  bounds=%s", DEM_MOSAIC.name, arr.shape[1:],
             [round(b) for b in bounds])
    return DEM_MOSAIC


class DemSampler:
    """In-memory, pre-smoothed DEM with bilinear point sampling."""

    def __init__(self, path: Path = DEM_MOSAIC, sigma_m: float = DEM_SMOOTH_SIGMA_M,
                 smooth: bool = True, water_below_m: float | None = WATER_BELOW_M):
        import rasterio
        from scipy.ndimage import gaussian_filter

        with rasterio.open(path) as src:
            self.transform = src.transform
            self.crs = src.crs
            self.nodata = src.nodata
            self.res = src.res[0]
            data = src.read(1).astype("float32")

        self.valid = np.isfinite(data)
        if self.nodata is not None:
            self.valid &= data != self.nodata
        # Hydro-flattened water: real values, so they stay in the data for
        # smoothing (the shoreline keeps its true height), but samples that
        # land on them are invalid and get solved from their neighbours.
        water = (self.valid & (data < water_below_m)) if water_below_m is not None else None
        # Fill nodata with the mean so the Gaussian does not smear -999999
        # across the coastline; invalid cells are masked again after sampling.
        fill = float(data[self.valid].mean()) if self.valid.any() else 0.0
        data = np.where(self.valid, data, fill)

        if smooth and sigma_m > 0:
            sigma_px = sigma_m / self.res
            with step(f"pre-smoothing DEM (Gaussian sigma={sigma_m:g} m)", log):
                data = gaussian_filter(data, sigma=sigma_px, mode="nearest")
        self.data = data
        if water is not None:
            self.valid &= ~water
            log.info("DEM: %.1f%% of cells are water below %g m, treated as nodata",
                     100.0 * water.mean(), water_below_m)
        self.sigma_m = sigma_m if smooth else 0.0
        log.info("DEM sampler ready: %s cells, %.1f%% valid",
                 f"{data.size:,}", 100.0 * self.valid.mean())

    # ---- sampling ----------------------------------------------------
    def _to_pixel(self, x: np.ndarray, y: np.ndarray):
        t = self.transform
        # inverse of an axis-aligned affine transform
        col = (x - t.c) / t.a
        row = (y - t.f) / t.e
        return row, col

    def sample(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinear sample; returns NaN outside the raster or over nodata."""
        x = np.asarray(x, dtype="float64")
        y = np.asarray(y, dtype="float64")
        row, col = self._to_pixel(x, y)
        h, w = self.data.shape

        r0 = np.floor(row - 0.5).astype(np.int64)
        c0 = np.floor(col - 0.5).astype(np.int64)
        fr = (row - 0.5) - r0
        fc = (col - 0.5) - c0

        inside = (r0 >= 0) & (c0 >= 0) & (r0 + 1 < h) & (c0 + 1 < w)
        r0c = np.clip(r0, 0, h - 2)
        c0c = np.clip(c0, 0, w - 2)

        d = self.data
        v00 = d[r0c, c0c]; v01 = d[r0c, c0c + 1]
        v10 = d[r0c + 1, c0c]; v11 = d[r0c + 1, c0c + 1]
        top = v00 * (1 - fc) + v01 * fc
        bot = v10 * (1 - fc) + v11 * fc
        out = top * (1 - fr) + bot * fr

        # invalidate points whose nearest cell was nodata
        rn = np.clip(np.round(row - 0.5).astype(np.int64), 0, h - 1)
        cn = np.clip(np.round(col - 0.5).astype(np.int64), 0, w - 1)
        ok = inside & self.valid[rn, cn]
        return np.where(ok, out, np.nan)


# --------------------------------------------------------------------------
# densification
# --------------------------------------------------------------------------
def densify(geom, spacing: float, min_samples: int):
    """Return (distance_along, x, y) sample arrays for one LineString."""
    length = geom.length
    n = max(min_samples, int(np.ceil(length / spacing)) + 1)
    dist = np.linspace(0.0, length, n)
    pts = [geom.interpolate(float(d)) for d in dist]
    x = np.fromiter((p.x for p in pts), dtype="float64", count=n)
    y = np.fromiter((p.y for p in pts), dtype="float64", count=n)
    return dist, x, y


# --------------------------------------------------------------------------
# structure (bridge/tunnel) elevation recovery
# --------------------------------------------------------------------------
def _solve_structure_nodes(edges, node_elev: dict, reliable: set,
                           subset=None, label: str = "structure-only") -> dict:
    """Laplace-smooth elevations for nodes with no usable DEM value.

    Nodes on a viaduct have no DEM-derived elevation worth using, and a
    handful of nodes (piers, DEM holes) have none at all.  Treating the
    relevant sub-graph as a resistor network with the reliable nodes as
    Dirichlet boundary conditions produces the smoothest surface consistent
    with where it meets solid, measured ground.

    ``subset`` selects the edges spanned; the default is the structure
    sub-graph.
    """
    import collections

    struct = edges[edges["is_structure"]] if subset is None else subset
    adj: dict[str, list[tuple[str, float]]] = collections.defaultdict(list)
    for u, v, L in zip(struct["u"], struct["v"], struct["length_m"]):
        adj[u].append((v, max(L, 1.0)))
        adj[v].append((u, max(L, 1.0)))
    unknown = [n for n in adj if n not in reliable]
    if not unknown:
        return node_elev

    # initialise from any nearby reliable value, then iterate
    est = {n: node_elev.get(n, np.nan) for n in adj}
    for n in unknown:
        if not np.isfinite(est.get(n, np.nan)):
            est[n] = np.nan
    seed = np.nanmean([v for n, v in est.items() if n in reliable and np.isfinite(v)]) \
        if any(n in reliable and np.isfinite(v) for n, v in est.items()) else 0.0
    for n in unknown:
        if not np.isfinite(est[n]):
            est[n] = seed

    for _ in range(400):
        delta = 0.0
        for n in unknown:
            num = den = 0.0
            for m, L in adj[n]:
                w = 1.0 / L
                val = node_elev[m] if m in reliable else est[m]
                if np.isfinite(val):
                    num += w * val
                    den += w
            if den > 0:
                new = num / den
                delta = max(delta, abs(new - est[n]))
                est[n] = new
        if delta < 1e-4:
            break

    # A connected piece of the sub-graph that touches no reliable node has
    # nothing to anchor it: the iteration above just leaves it at the seed
    # (the mean of every reliable node, tens of metres off). Leave those
    # unknown instead, so the whole-graph DEM-void pass solves them from
    # their real neighbours. This happens to footbridges whose only landing
    # sits over (hydro-flattened) water.
    anchored: set[str] = set()
    for start in adj:
        if start in anchored or start in reliable:
            continue
        comp, stack = {start}, [start]
        while stack:
            for m, _ in adj[stack.pop()]:
                if m not in comp:
                    comp.add(m); stack.append(m)
        if any(m in reliable and np.isfinite(node_elev.get(m, np.nan)) for m in comp):
            anchored |= comp
    out = dict(node_elev)
    floating = 0
    for n in unknown:
        if n in anchored:
            out[n] = est[n]
        else:
            out[n] = np.nan
            floating += 1
    log.info("  recovered elevations for %d %s nodes%s", len(unknown) - floating, label,
             f"; {floating} with no reliable neighbour left for later" if floating else "")
    return out


def compute_node_elevations(edges, sampler: DemSampler) -> tuple[dict, set]:
    """DEM elevation at every graph node, plus the set of reliable nodes."""
    import collections
    xs: dict[str, list[float]] = collections.defaultdict(list)
    ys: dict[str, list[float]] = collections.defaultdict(list)
    solid: set[str] = set()

    for u, v, geom, is_struct in zip(edges["u"], edges["v"], edges.geometry,
                                     edges["is_structure"]):
        c = geom.coords
        xs[u].append(c[0][0]); ys[u].append(c[0][1])
        xs[v].append(c[-1][0]); ys[v].append(c[-1][1])
        if not is_struct:
            solid.add(u); solid.add(v)

    nodes = list(xs)
    px = np.array([np.mean(xs[n]) for n in nodes])
    py = np.array([np.mean(ys[n]) for n in nodes])
    z = sampler.sample(px, py)
    node_elev = {n: float(zz) for n, zz in zip(nodes, z)}
    reliable = {n for n in solid if np.isfinite(node_elev[n])}
    log.info("node elevations: %d nodes, %d reliable (non-structure, valid DEM)",
             len(nodes), len(reliable))
    return node_elev, reliable


# --------------------------------------------------------------------------
# profile extraction
# --------------------------------------------------------------------------
def _smooth_profile(z: np.ndarray, spacing: float) -> np.ndarray:
    """Savitzky-Golay along the profile where enough samples exist."""
    from scipy.signal import savgol_filter
    n = z.size
    if n < 5:
        return z
    win = int(round(2 * ELEVATION.smooth_window_m / max(spacing, 1e-6))) + 1
    win = min(win, n if n % 2 == 1 else n - 1)
    if win < 5:
        return z
    if win % 2 == 0:
        win -= 1
    try:
        return savgol_filter(z, win, ELEVATION.smooth_polyorder, mode="interp")
    except Exception:
        return z


def sample_edge_profiles(edges, sampler: DemSampler | None = None,
                         force: bool = False) -> dict[int, tuple]:
    """Elevation profile for every edge, cached to ``edge_profiles.npz``.

    Returns ``{edge_id: (distance_along_m, elevation_m)}``.
    """
    if PROFILES_NPZ.exists() and not force:
        with np.load(PROFILES_NPZ, allow_pickle=False) as npz:
            ids = npz["edge_ids"]
            offsets = npz["offsets"]
            dist = npz["dist"]
            elev = npz["elev"]
        out = {int(e): (dist[offsets[i]:offsets[i + 1]], elev[offsets[i]:offsets[i + 1]])
               for i, e in enumerate(ids)}
        log.info("cached %s (%d profiles)", PROFILES_NPZ.name, len(out))
        return out

    sampler = sampler or DemSampler(build_dem_mosaic())
    cfg = ELEVATION

    with step("computing node elevations", log):
        node_elev, reliable = compute_node_elevations(edges, sampler)
    with step("recovering bridge/tunnel deck elevations", log):
        node_elev = _solve_structure_nodes(edges, node_elev, reliable)
        reliable = reliable | {n for n in node_elev
                               if np.isfinite(node_elev.get(n, np.nan))}
    missing = [n for n in node_elev if not np.isfinite(node_elev.get(n, np.nan))]
    if missing:
        # A few nodes sit where the DEM has no data at all (piers over water,
        # interpolation holes). Assigning them 0 m would be catastrophic: it
        # once put an 18 m footway 65 m below the street it joins. Solve them
        # from their neighbours over the whole graph instead.
        with step(f"recovering {len(missing)} nodes with no DEM coverage", log):
            node_elev = _solve_structure_nodes(edges, node_elev, reliable,
                                               subset=edges,
                                               label="DEM-void")

    # one vectorised DEM read for all sample points
    with step("densifying edge geometries", log):
        all_dist, all_x, all_y, counts = [], [], [], []
        for geom in progress(edges.geometry, desc="  densify", total=len(edges),
                             unit="edge"):
            d, x, y = densify(geom, cfg.sample_spacing_m, cfg.min_samples)
            all_dist.append(d); all_x.append(x); all_y.append(y)
            counts.append(d.size)
    counts = np.asarray(counts)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    cat_x = np.concatenate(all_x); cat_y = np.concatenate(all_y)
    cat_dist = np.concatenate(all_dist)
    with step(f"sampling DEM at {cat_x.size:,} points", log):
        cat_z = sampler.sample(cat_x, cat_y)

    edge_ids = np.asarray(edges["edge_id"].values)
    is_struct = np.asarray(edges["is_structure"].values)
    us = list(edges["u"]); vs = list(edges["v"])
    row_of = {int(e): i for i, e in enumerate(edge_ids)}

    # Structure ramps, gap filling and smoothing are all applied to the
    # *concatenated* profile of each contiguous run of a street segment, not
    # edge by edge. Smoothing an edge in isolation gives the two edges either
    # side of an intersection different elevations for the same corner, and
    # in San Francisco that happens every 80 m; the discontinuities then
    # propagate into the climbing figures.
    from .metrics import contiguous_runs

    out_dist: list[np.ndarray] = [None] * len(edge_ids)
    out_elev: list[np.ndarray] = [None] * len(edge_ids)
    n_struct_fixed = n_gapfilled = n_unresolved = 0

    runs = contiguous_runs(edges)
    for eids in progress(runs, desc="  profiles", total=len(runs), unit="run"):
        rows = [row_of[int(e)] for e in eids]
        spans, segs, zs, base = [], [], [], 0.0
        for k, i in enumerate(rows):
            a, b = offsets[i], offsets[i + 1]
            d = cat_dist[a:b].copy()
            z = cat_z[a:b].copy()
            if k > 0:
                d = d[1:]; z = z[1:]
            segs.append(base + d); zs.append(z); spans.append(d.size)
            base = segs[-1][-1] if segs[-1].size else base
        run_d = np.concatenate(segs)
        run_z = np.concatenate(zs)

        # index range of each edge within the run, sharing joint stations
        ranges, pos = [], 0
        for k in range(len(rows)):
            lo = pos if k == 0 else pos - 1
            hi = pos + spans[k]
            ranges.append((lo, hi))
            pos = hi

        # structures: replace the DEM's view of the ground under the deck
        if cfg.interpolate_structures:
            for k, i in enumerate(rows):
                if not is_struct[i]:
                    continue
                lo, hi = ranges[k]
                z0 = node_elev.get(us[i], np.nan)
                z1 = node_elev.get(vs[i], np.nan)
                if np.isfinite(z0) and np.isfinite(z1):
                    dd = run_d[lo:hi]
                    spanlen = dd[-1] - dd[0]
                    frac = (dd - dd[0]) / (spanlen if spanlen > 0 else 1.0)
                    run_z[lo:hi] = z0 + (z1 - z0) * frac
                    n_struct_fixed += 1

        if not np.all(np.isfinite(run_z)):
            good = np.isfinite(run_z)
            if good.any():
                run_z = np.interp(run_d, run_d[good], run_z[good])
            else:
                ends = [node_elev.get(us[rows[0]], np.nan),
                        node_elev.get(vs[rows[-1]], np.nan)]
                if all(np.isfinite(ends)):
                    frac = ((run_d - run_d[0])
                            / max(run_d[-1] - run_d[0], 1e-9))
                    run_z = ends[0] + (ends[1] - ends[0]) * frac
                else:
                    n_unresolved += len(rows)
            n_gapfilled += 1

        # smooth the whole run, except where it is entirely a structure deck
        if np.all(np.isfinite(run_z)) and not all(is_struct[i] for i in rows):
            run_z = _smooth_profile(run_z, cfg.sample_spacing_m)

        for k, i in enumerate(rows):
            lo, hi = ranges[k]
            d = run_d[lo:hi]
            out_dist[i] = (d - d[0]).astype("float32")
            out_elev[i] = run_z[lo:hi].astype("float32")

    log.info("  linear ramp applied to %d structure edges; %d runs gap-filled",
             n_struct_fixed, n_gapfilled)
    if n_unresolved:
        log.warning("  %d edges have no recoverable elevation and carry NaN; "
                    "they are excluded when the metrics table is built",
                    n_unresolved)

    # ---- one elevation per intersection -------------------------------
    # Each street segment is smoothed independently, so two streets meeting
    # at a corner end up with slightly different elevations for that same
    # corner. Per-edge climbing is still exact, but summing it along a route
    # then drifts from the difference between the route's endpoints (a few
    # metres over a long route). Reconciling every node to a single
    # elevation and rubber-sheeting each profile onto it with a linear
    # correction removes the drift entirely while leaving the profile's
    # shape, and therefore its gradients, essentially untouched.
    with step("reconciling elevations at intersections", log):
        acc: dict[str, list[float]] = {}
        for i in range(len(edge_ids)):
            if out_elev[i] is None or not np.all(np.isfinite(out_elev[i])):
                continue
            acc.setdefault(us[i], []).append(float(out_elev[i][0]))
            acc.setdefault(vs[i], []).append(float(out_elev[i][-1]))
        canon = {n: float(np.mean(v)) for n, v in acc.items()}

        shifts = []
        for i in range(len(edge_ids)):
            z = out_elev[i]
            if z is None or z.size < 2 or not np.all(np.isfinite(z)):
                continue
            z0 = canon.get(us[i]); z1 = canon.get(vs[i])
            if z0 is None or z1 is None:
                continue
            d = out_dist[i].astype("float64")
            span = d[-1] - d[0]
            frac = (d - d[0]) / (span if span > 0 else 1.0)
            delta0 = z0 - float(z[0])
            delta1 = z1 - float(z[-1])
            out_elev[i] = (z.astype("float64")
                           + delta0 * (1.0 - frac) + delta1 * frac
                           ).astype("float32")
            shifts.append((max(abs(delta0), abs(delta1)), int(edge_ids[i])))
        if shifts:
            sh = np.asarray([s for s, _ in shifts])
            log.info("  node reconciliation shifted profile ends by "
                     "%.3f m on average, %.2f m at worst (%d edges)",
                     sh.mean(), sh.max(), sh.size)
            worst = sorted(shifts, reverse=True)[:5]
            log.debug("  largest shifts (m, edge_id): %s",
                      ", ".join(f"{s:.2f} @ {e}" for s, e in worst))

    cat_d = np.concatenate(out_dist); cat_e = np.concatenate(out_elev)
    np.savez_compressed(PROFILES_NPZ, edge_ids=edge_ids, offsets=offsets,
                        dist=cat_d, elev=cat_e)
    log.info("wrote %s", PROFILES_NPZ.name)
    return {int(e): (out_dist[i], out_elev[i]) for i, e in enumerate(edge_ids)}
