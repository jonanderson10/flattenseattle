"""Publication-quality static map: San Francisco's low-elevation backbone.

The map answers one question -- *which streets form San Francisco's
low-elevation transportation backbone?* -- so the visual hierarchy is built
to answer it and nothing else:

* a hillshade computed from the same 1 m lidar DEM the analysis uses, so the
  topography the corridors are threading through is visible;
* the full street network drawn very faintly, present for context but
  carrying almost no visual weight;
* the discovered corridors drawn boldly, with line width scaled by corridor
  importance and colour by mean gradient;
* the critical passes marked, because a backbone is defined as much by where
  it must cross a ridge as by where it runs level;
* labels for the major corridors and for the neighborhoods.
"""
from __future__ import annotations

import numpy as np

from .config import OUTPUT_DIR
from .utils import get_logger, step

log = get_logger("flatten_seattle.viz_static")

STATIC_PNG = OUTPUT_DIR / "sf_flat_backbone.png"
STATIC_PDF = OUTPUT_DIR / "sf_flat_backbone.pdf"
GRADE_PNG = OUTPUT_DIR / "sf_street_grades.png"


# --------------------------------------------------------------------------
def hillshade(dem: np.ndarray, res: float = 1.0, azimuth: float = 315.0,
              altitude: float = 45.0, z_factor: float = 1.6) -> np.ndarray:
    """Standard Horn hillshade, returned in 0..1."""
    # explicit axes: np.gradient's scalar-spacing return shape varies by
    # numpy version, and the axis form is unambiguous. Rows run north to
    # south, so dy is the southward slope, which is what the ESRI/Horn
    # aspect below expects. (The arguments were swapped once; the light then
    # came from the south-east and every hill read as a hollow.)
    dy = np.gradient(dem, res, axis=0)
    dx = np.gradient(dem, res, axis=1)
    slope = np.arctan(z_factor * np.hypot(dx, dy))
    aspect = np.arctan2(dy, -dx)
    az = np.radians(360.0 - azimuth + 90.0)
    alt = np.radians(altitude)
    hs = (np.sin(alt) * np.cos(slope)
          + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(hs, 0, 1)


def _load_hillshade(downsample: int = 3, land_mask_geom=None):
    """Hillshade of the study area, masked to land.

    The 3DEP mosaic carries plausible-looking values across the Bay floor and
    beyond the study area, which render as spurious beige shelves and a hard
    diagonal at the tile edge.  Masking to the city boundary is both more
    honest and much cleaner to look at.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import rasterize

    from .elevation import DEM_MOSAIC

    with rasterio.open(DEM_MOSAIC) as src:
        h = src.height // downsample
        w = src.width // downsample
        # read(1, ...) already returns a 2-D array; do not index it
        dem = src.read(1, out_shape=(h, w),
                       resampling=Resampling.average).astype("float64")
        nod = src.nodata
        bounds = src.bounds
        transform = src.transform * src.transform.scale(
            src.width / w, src.height / h)

    valid = np.isfinite(dem) & (dem != nod) & (dem > -50)
    if land_mask_geom is not None:
        land = rasterize([land_mask_geom], out_shape=(h, w),
                         transform=transform, fill=0, default_value=1,
                         dtype="uint8").astype(bool)
        valid &= land
    dem_f = np.where(valid, dem, np.nan)
    filled = np.where(valid, dem, np.nan)
    # interpolate across the mask edge so the hillshade has no hard rim
    filled = np.where(np.isfinite(filled), filled,
                      np.nanmedian(dem[valid]) if valid.any() else 0.0)
    hs = hillshade(filled, res=downsample, z_factor=2.1)
    return hs, dem_f, valid, bounds


def make_backbone_map(corridors, edges, neighborhoods, passes=None,
                      mode: str = "walk", top_n: int = 22,
                      label_n: int = 11) -> tuple:
    """Draw the flagship static map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D

    cor = corridors[corridors["mode"] == mode].copy()
    cor = cor.sort_values("total_score", ascending=False).head(top_n)

    with step("rendering the static backbone map", log):
        from shapely.ops import unary_union
        land_geom = unary_union(neighborhoods.geometry.values).buffer(60)
        hs, dem, valid, bounds = _load_hillshade(land_mask_geom=land_geom)
        extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)

        fig, ax = plt.subplots(figsize=(13.5, 15.5), dpi=220)
        fig.patch.set_facecolor("#f7f5f0")
        ax.set_facecolor("#dfe8ef")                       # water

        # --- terrain: hillshade tinted by elevation -------------------
        land = np.where(valid, 1.0, np.nan)
        elev_cmap = LinearSegmentedColormap.from_list(
            "sf_terrain", ["#f2efe6", "#e8e1cf", "#ddd2b6", "#cfc09b",
                           "#bfa87f"])
        ax.imshow(dem, extent=extent, origin="upper", cmap=elev_cmap,
                  norm=Normalize(-10, 250), alpha=1.0, interpolation="bilinear",
                  zorder=1)
        ax.imshow(hs * land, extent=extent, origin="upper", cmap="gray",
                  alpha=0.52, interpolation="bilinear", vmin=0.15, vmax=0.95,
                  zorder=2)

        # --- neighborhoods -------------------------------------------
        neighborhoods.boundary.plot(ax=ax, color="#ffffff", linewidth=1.0,
                                    alpha=0.8, zorder=3)

        # --- full street network: present for context, low weight -----
        net = edges[edges[f"{mode}_ok"]]
        net.plot(ax=ax, color="#59646f", linewidth=0.20, alpha=0.42, zorder=4)

        # --- steep street emphasis (the walls the backbone avoids) ----
        rel = net["grade_reliable"] if "grade_reliable" in net.columns else True
        steep = net[(net["max_abs_grade"] >= 0.12) & rel]
        steep.plot(ax=ax, color="#a8321f", linewidth=0.62, alpha=0.62, zorder=5)

        # --- the corridors -------------------------------------------
        smax = float(cor["total_score"].max()) if len(cor) else 1.0
        grade_cmap = LinearSegmentedColormap.from_list(
            "flatness", ["#08306b", "#1f78b4", "#41b6c4", "#7fcdbb"])
        for _, r in cor.iterrows():
            lw = 1.9 + 5.4 * (r["total_score"] / smax) ** 0.65
            col = grade_cmap(min(r["mean_abs_grade"] / 0.035, 1.0))
            gs = getattr(r.geometry, "geoms", [r.geometry])
            for g in gs:
                xs, ys = g.xy
                ax.plot(xs, ys, color="white", linewidth=lw + 1.7,
                        solid_capstyle="round", alpha=0.85, zorder=6)
                ax.plot(xs, ys, color=col, linewidth=lw,
                        solid_capstyle="round", zorder=7)

        # --- critical passes -----------------------------------------
        if passes is not None and len(passes):
            pz = passes.head(12)
            if pz.crs != edges.crs:
                pz = pz.to_crs(edges.crs)
            pts = pz.geometry.interpolate(0.5, normalized=True)
            ax.scatter([p.x for p in pts], [p.y for p in pts], s=54,
                       marker="^", facecolor="#ffd166", edgecolor="#4a3b10",
                       linewidth=0.8, zorder=9)

        # --- labels ---------------------------------------------------
        for _, r in neighborhoods.iterrows():
            c = r.geometry.representative_point()
            ax.text(c.x, c.y, r["neighborhood"].replace("/", "/\n"),
                    fontsize=6.2, color="#33404b", ha="center", va="center",
                    zorder=10, alpha=0.95, style="italic",
                    path_effects=_outline(1.9))

        placed: list[tuple[float, float]] = []
        used_labels: set[str] = set()
        for _, r in cor.iterrows():
            if len(used_labels) >= label_n:
                break
            label = r["corridor_name"].split(" - ")[0]
            if label in used_labels:
                continue
            g = r.geometry
            gs = list(getattr(g, "geoms", [g]))
            longest = max(gs, key=lambda x: x.length)
            # try a few positions along the corridor and keep the first that
            # is not crowding an existing label
            for frac in (0.55, 0.3, 0.75, 0.45, 0.9):
                p = longest.interpolate(frac, normalized=True)
                if all((p.x - qx) ** 2 + (p.y - qy) ** 2 > 620 ** 2
                       for qx, qy in placed):
                    ax.text(p.x, p.y + 110, label, fontsize=8.0,
                            fontweight="bold", color="#062a5c", ha="center",
                            va="bottom", zorder=11, path_effects=_outline(2.8))
                    placed.append((p.x, p.y))
                    used_labels.add(label)
                    break

        # --- frame ----------------------------------------------------
        b = neighborhoods.total_bounds
        pad = 700
        ax.set_xlim(b[0] - pad, b[2] + pad)
        ax.set_ylim(b[1] - pad, b[3] + pad)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        ax.set_title("San Francisco's low-elevation backbone",
                     fontsize=21, fontweight="bold", color="#14202b",
                     loc="left", pad=16)
        ax.text(0.0, 1.006,
                f"Streets that repeatedly carry low-gradient routes between "
                f"neighborhoods  ·  {mode}ing network",
                transform=ax.transAxes, fontsize=10.2, color="#4a5560",
                va="bottom")

        legend = [
            Line2D([], [], color="#1f78b4", lw=5.0,
                   label="Flat corridor (width = importance)"),
            Line2D([], [], color="#08306b", lw=3.0,
                   label="Corridor mean grade < 1%"),
            Line2D([], [], color="#7fcdbb", lw=3.0,
                   label="Corridor mean grade ~3.5%"),
            Line2D([], [], color="#b3402f", lw=1.6,
                   label="Street steeper than 12%"),
            Line2D([], [], color="#5d6b78", lw=0.8, alpha=0.5,
                   label="All other streets"),
            Line2D([], [], color="none", marker="^", markersize=7,
                   markerfacecolor="#ffd166", markeredgecolor="#4a3b10",
                   label="Critical pass (lowest crossing)"),
        ]
        ax.legend(handles=legend, loc="lower left", fontsize=7.6,
                  frameon=True, facecolor="white", framealpha=0.92,
                  edgecolor="#c9cdd2", borderpad=0.7)

        ax.text(0.995, -0.018,
                "Street network: Overture Maps (OpenStreetMap, ODbL)  ·  "
                "Elevation: USGS 3DEP 1 m lidar  ·  "
                "Neighborhoods: SF Planning / DataSF",
                transform=ax.transAxes, fontsize=6.4, color="#6a747d",
                ha="right", va="top")

        _scalebar(ax, b)
        fig.tight_layout()
        fig.savefig(STATIC_PNG, bbox_inches="tight", facecolor=fig.get_facecolor())
        # suppressing the PDF creation date keeps the output reproducible
        fig.savefig(STATIC_PDF, bbox_inches="tight", facecolor=fig.get_facecolor(),
                    metadata={"CreationDate": None})
        plt.close(fig)
    log.info("wrote %s and %s", STATIC_PNG.name, STATIC_PDF.name)
    return STATIC_PNG, STATIC_PDF


def _outline(lw: float):
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=lw, foreground="white", alpha=0.9)]


def _scalebar(ax, bounds, length_m: float = 2000.0):
    # bottom-right, clear of the legend
    x0 = bounds[2] - length_m - 600
    y0 = bounds[1] + 400
    ax.plot([x0, x0 + length_m], [y0, y0], color="#14202b", lw=2.4,
            solid_capstyle="butt", zorder=12)
    ax.text(x0 + length_m / 2, y0 + 150, f"{length_m/1000:g} km", fontsize=7.4,
            ha="center", color="#14202b", zorder=12,
            path_effects=_outline(2.0))


# --------------------------------------------------------------------------
def make_grade_map(edges, neighborhoods, mode: str = "walk"):
    """Secondary figure: the whole network coloured by gradient."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable

    with step("rendering the citywide grade map", log):
        net = edges[edges[f"{mode}_ok"]].copy()
        net["g"] = net["max_abs_grade"].fillna(0.0)
        bounds_g = [0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.60]
        cmap = LinearSegmentedColormap.from_list(
            "grades", ["#2c7bb6", "#8fd0c0", "#ffffbf", "#fdae61", "#e8613c",
                       "#7f1d1d"], N=6)
        norm = BoundaryNorm(bounds_g, cmap.N)

        fig, ax = plt.subplots(figsize=(12.5, 14.5), dpi=200)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#eef3f6")
        neighborhoods.plot(ax=ax, facecolor="#f8f7f4", edgecolor="#d8dde1",
                           linewidth=0.6, zorder=1)
        for lo, hi in zip(bounds_g[:-1], bounds_g[1:]):
            sel = net[(net["g"] >= lo) & (net["g"] < hi)]
            if sel.empty:
                continue
            lw = 0.22 if hi <= 0.05 else (0.38 if hi <= 0.10 else 0.62)
            sel.plot(ax=ax, color=cmap(norm(lo + 1e-9)), linewidth=lw,
                     alpha=0.95, zorder=2 + bounds_g.index(lo))

        b = neighborhoods.total_bounds
        ax.set_xlim(b[0] - 500, b[2] + 500); ax.set_ylim(b[1] - 500, b[3] + 500)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title("San Francisco street gradients", fontsize=19,
                     fontweight="bold", loc="left", pad=14)
        ax.text(0.0, 1.005, "Maximum sampled gradient per street segment, "
                            "from USGS 3DEP 1 m lidar",
                transform=ax.transAxes, fontsize=9.6, color="#4a5560",
                va="bottom")
        sm = ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.015,
                          boundaries=bounds_g, ticks=bounds_g)
        cb.ax.set_yticklabels([f"{v:.0%}" for v in bounds_g])
        cb.set_label("maximum gradient", fontsize=9)
        fig.tight_layout()
        fig.savefig(GRADE_PNG, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    log.info("wrote %s", GRADE_PNG.name)
    return GRADE_PNG
