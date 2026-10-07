"""Central configuration: paths, CRS, grade thresholds and routing weights.

Everything tunable in the analysis lives here so that experiments are
reproducible and the cost model is auditable.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("SFFR_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
#: ``SFFR_RUN_DIR`` redirects every processed and output path under one
#: directory, so an experiment (see ``sensitivity.py``) can rebuild the whole
#: pipeline with different parameters without touching the main results.
_RUN_DIR = os.environ.get("SFFR_RUN_DIR")
PROCESSED_DIR = Path(_RUN_DIR) / "processed" if _RUN_DIR else DATA_DIR / "processed"
OUTPUT_DIR = Path(_RUN_DIR) / "outputs" if _RUN_DIR else PROJECT_ROOT / "outputs"
#: The route finder as a static site, deployed to GitHub Pages from here.
SITE_DIR = Path(_RUN_DIR) / "site" if _RUN_DIR else PROJECT_ROOT / "site"
#: The product is "Flatten Seattle", a fork of flattensf; the Python package
#: keeps the upstream name so upstream changes still merge.
PRODUCT_NAME = "Flatten Seattle"
CITY_NAME = "Seattle"
#: Suffix on cached raw-data file names.
CITY_SLUG = "seattle"
REPO_URL = "https://github.com/jonanderson10/flattenseattle"
UPSTREAM_URL = "https://github.com/almostimplemented/flattensf"
#: Custom domain, once there is one: it is written to site/CNAME and used for
#: the canonical URL. ``None`` serves from the default Pages URL, no CNAME.
SITE_DOMAIN = None
SITE_URL = (f"https://{SITE_DOMAIN}/" if SITE_DOMAIN
            else "https://jonanderson10.github.io/flattenseattle/")

for _d in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Coordinate reference systems
# --------------------------------------------------------------------------
#: Geographic CRS of the source vector data (Overture) and of all map output.
CRS_GEOGRAPHIC = "EPSG:4326"
#: Projected CRS used for *every* length, slope and distance computation.
#: NAD83 / UTM zone 10N -- the native CRS of the USGS 3DEP 1 m tiles for Seattle,
#: so elevation sampling needs no reprojection of the raster.
CRS_PROJECTED = "EPSG:26910"

# --------------------------------------------------------------------------
# Study area
# --------------------------------------------------------------------------
#: Analysis bounding box (lon_min, lon_max, lat_min, lat_max).
#: The Seattle city limits (-122.436..-122.236, 47.4955..47.7342) plus ~500 m.
#: The street network is then clipped to the land inside the city, so the
#: I-90 and SR 520 floating bridges end at the shore: the route finder gets
#: you to the bridge and no further.
CITY_BBOX = (-122.4420, -122.2300, 47.4900, 47.7400)
#: Latitude at the middle of the study area, for degree <-> metre shortcuts.
CITY_LAT = (CITY_BBOX[2] + CITY_BBOX[3]) / 2
#: metres per degree of longitude at CITY_LAT
LON_M_PER_DEG = 111320.0 * math.cos(math.radians(CITY_LAT))

#: Neighborhoods left out of pair routing (none for Seattle).
EXCLUDED_NEIGHBORHOODS: tuple = ()

#: Hand-placed coordinates (lon, lat) for places whose feed location is not
#: where anyone would say they arrived. Overture puts Pike Place Market on its
#: west side above Western Avenue; people mean the Public Market sign at
#: Pike Street and Pike Place.
PLACE_OVERRIDES: dict = {
    "Pike Place Market": (-122.34000, 47.60884),
}

# --------------------------------------------------------------------------
# Elevation sampling / smoothing
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ElevationConfig:
    #: Spacing (m) of elevation samples along each edge. 5 m was chosen
    #: empirically: at 10 m spacing the short steep pitches that give San
    #: Francisco its reputation were measurably clipped (Bradford St came out
    #: at 36.8% against a documented 41%, Prentiss St at 32.9% against 37%),
    #: while 5 m recovers them. Cost is ~1M sample points citywide.
    sample_spacing_m: float = 5.0
    #: Minimum number of samples per edge (endpoints always included).
    min_samples: int = 3
    #: Half-width (m) of the Savitzky-Golay window applied along the profile.
    #: 25 m either side => ~50 m window, shorter than a city block, so real
    #: block-scale grade is preserved while curb/vehicle/vegetation artefacts
    #: in the lidar DEM are suppressed. Narrower windows were tested and
    #: rejected: at a 12.5 m half-width, localised DEM artefacts on 22nd St
    #: and Baden St survived and pushed those streets to the 60% plausibility
    #: clip, whereas 25 m returns 32.9% and 30.6% against documented values
    #: of 31.5% and 32%.
    smooth_window_m: float = 25.0
    #: Polynomial order of the Savitzky-Golay filter.
    smooth_polyorder: int = 2
    #: Dead-band (m). Elevation wiggles smaller than this are not counted as
    #: cumulative gain/loss. Standard practice in altimetry; without it a flat
    #: street accumulates tens of metres of phantom climbing from DEM noise.
    gain_deadband_m: float = 0.5
    #: Grades above this magnitude are treated as DEM artefacts and clipped.
    #: Filbert St between Hyde and Leavenworth is ~31.5%, the steepest
    #: drivable street in SF; public stairways reach ~50%+, so the clip is
    #: generous and only removes physically impossible values.
    max_plausible_grade: float = 0.60
    #: Elevation on bridges/tunnels is taken as a linear interpolation between
    #: the endpoints instead of from the DEM (the DEM samples the ground or
    #: water surface beneath the structure).
    interpolate_structures: bool = True
    #: Standard deviation (m) of the Gaussian applied to the DEM before any
    #: sampling. 3 m is far narrower than a street and far narrower than the
    #: block scale on which real gradient varies; 0 disables it.
    dem_sigma_m: float = 3.0
    #: DEM cells below this elevation (m, NAVD88) are water, treated as
    #: nodata. The King County 2021 lidar hydro-flattens Puget Sound, the
    #: Duwamish and the salt-water lock chamber at Ballard to about -0.4 to
    #: -0.9 m instead of leaving them empty, so walkways over water (the
    #: locks, piers, the Spokane St crossing) sampled the water surface and
    #: showed metres of phantom climbing. No routable ground in Seattle sits
    #: this low; ``None`` disables it.
    water_below_m: float | None = 1.0


def _overrides() -> dict:
    """Parameter overrides from ``SFFR_OVERRIDES`` (a JSON object).

    Used by the sensitivity harness to rebuild the pipeline under different
    settings. Keys are ``ElevationConfig`` or ``AnalysisConfig`` field names.
    """
    raw = os.environ.get("SFFR_OVERRIDES")
    if not raw:
        return {}
    import json
    try:
        return dict(json.loads(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SFFR_OVERRIDES is not a JSON object: {raw!r}") from exc


_OV = _overrides()


def _apply(instance, overrides: dict):
    names = {f.name for f in instance.__dataclass_fields__.values()}
    picked = {k: v for k, v in overrides.items() if k in names}
    return replace(instance, **picked) if picked else instance


ELEVATION = _apply(ElevationConfig(), _OV)

# --------------------------------------------------------------------------
# Grade thresholds
# --------------------------------------------------------------------------
#: Grade thresholds (as fractions) at which "distance above grade" is tallied.
#: 3% = noticeable on a bike; 5% = sustained-effort threshold; 8% = hard;
#: 10% = very hard / ADA-infeasible; 15% = extreme, walk-your-bike territory.
GRADE_THRESHOLDS = (0.03, 0.05, 0.08, 0.10, 0.15)

#: Percentile used for the "typical steep" grade statistic of an edge.
GRADE_PERCENTILE = 95

#: Minimum edge length (m) for its *maximum* grade to be treated as reliable.
#: Over a 5 m stub a single decimetre of DEM artefact reads as a 20% grade;
#: the worst real example found was a 5 m connector at Market and 5th which
#: reported 41%. Short edges keep their metrics but are excluded from
#: maximum-grade tests, which are instead judged on average grade.
MIN_RELIABLE_GRADE_LENGTH_M = 15.0

# --------------------------------------------------------------------------
# Routing cost model
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CostWeights:
    """Weights for the additive edge cost model.

    The cost of traversing a directed edge is, in "equivalent metres":

        cost = length_m * mode_multiplier
             + alpha * cumulative_gain_m
             + beta  * sum_k(w_k * distance_above_threshold_k)
             + gamma * extreme_grade_distance_m
             + turn/one-way handling

    ``alpha`` is the horizontal distance a walker/rider would happily trade
    for one metre of climbing.  The classic Naismith rule (walking) is ~7.9 m
    per metre climbed; cyclists are far more climb-averse, so the bicycle
    default is higher.
    """
    name: str = "balanced"
    #: metres of equivalent detour accepted per metre of climbing
    alpha: float = 8.0
    #: multiplier on the graded steep-distance penalty
    beta: float = 1.0
    #: multiplier on the extreme-grade penalty
    gamma: float = 1.0
    #: per-threshold penalty weights (equivalent metres per metre travelled
    #: above the threshold). Applied cumulatively, so a 12% grade incurs the
    #: 3%, 5%, 8% and 10% penalties -- this makes the penalty grow
    #: nonlinearly (super-linearly) with grade.
    threshold_penalties: tuple = (0.25, 0.75, 2.0, 4.0, 10.0)
    #: Extra penalty (equivalent metres per metre) for distance above the
    #: highest threshold, scaled by gamma.
    extreme_extra: float = 10.0
    #: Cap on total cost inflation relative to plain length, to stop a single
    #: objective producing absurd detours. ``None`` disables the cap.
    max_detour_ratio: float | None = None
    #: Whether the mode's per-class comfort multipliers apply. The "shortest"
    #: objective sets this False so that it minimises *distance only*, as the
    #: analysis specification requires. Without the flag, the bicycle comfort
    #: weights (19th Ave at 1.9x, protected cycleway at 0.85x) made the
    #: "shortest" bicycle route longer in real metres than the flat route,
    #: which made every distance-penalty comparison meaningless.
    use_class_multiplier: bool = True


#: The four routing objectives required by the analysis.
ROUTING_PROFILES: dict[str, CostWeights] = {
    # A. Shortest -- distance only.
    "shortest": CostWeights(name="shortest", alpha=0.0, beta=0.0, gamma=0.0,
                            threshold_penalties=(0, 0, 0, 0, 0), extreme_extra=0.0,
                            use_class_multiplier=False),
    # B. Minimum climbing -- climbing dominates, grade shape ignored.
    "min_climb": CostWeights(name="min_climb", alpha=120.0, beta=0.0, gamma=0.0,
                             threshold_penalties=(0, 0, 0, 0, 0), extreme_extra=0.0),
    # C. Grade-averse -- steep *segments* dominate, total climbing secondary.
    "grade_averse": CostWeights(name="grade_averse", alpha=4.0, beta=12.0, gamma=6.0,
                                threshold_penalties=(0.25, 1.0, 3.0, 6.0, 15.0),
                                extreme_extra=15.0),
    # D. Balanced "flat but reasonable".
    "balanced": CostWeights(name="balanced", alpha=14.0, beta=2.0, gamma=2.0,
                            threshold_penalties=(0.25, 0.75, 2.0, 4.0, 10.0),
                            extreme_extra=10.0),
}

#: Sweep used to trace the distance/climb Pareto frontier. Each value scales
#: *every* climbing and grade term of the balanced profile together (alpha,
#: beta and gamma), so 0 is pure distance, 1 is the balanced objective and
#: large values approach the flattest possible route. Scaling only alpha,
#: as an earlier version did, left beta and gamma in force at the "shortest"
#: end and the frontier never reached the true shortest path.
PARETO_LAMBDA_SWEEP = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
                       5.0, 8.0, 15.0)

# --------------------------------------------------------------------------
# Travel modes
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ModeConfig:
    name: str
    #: Overture ``class`` values allowed for this mode.
    allowed_classes: frozenset = field(default_factory=frozenset)
    #: Overture ``subclass`` values that are always excluded.
    excluded_subclasses: frozenset = field(default_factory=frozenset)
    #: Access modes checked against Overture ``access_restrictions``.
    access_modes: tuple = ()
    #: Per-class multiplier on length, expressing comfort/stress rather than
    #: distance. 1.0 = neutral.
    class_multiplier: dict = field(default_factory=dict)
    #: If True, one-way restrictions are enforced (bicycles) rather than
    #: ignored (pedestrians).
    respect_oneway: bool = False


#: Classes that are never usable on foot or by bike, in either mode.
NEVER_ROUTABLE_CLASSES = frozenset({
    "motorway",           # grade-separated freeways
    "motorway_link",
    # rail/water subtypes are filtered by subtype, these are belt-and-braces
    "light_rail", "subway", "tram", "standard_gauge", "funicular", "rail",
})

#: Subclasses excluded from the routable graph for *both* modes.
#: ``sidewalk`` and ``crosswalk`` are excluded because the analysis models
#: travel along street centrelines: including the sidewalk network would
#: represent every street two or three times, which distorts corridor
#: aggregation and street naming. See README "Limitations".
NEVER_ROUTABLE_SUBCLASSES = frozenset({"sidewalk", "crosswalk", "driveway", "parking_aisle"})

#: Many sidewalks reach Overture as plain ``footway`` with no subclass
#: (Seattle's sidewalk import is mostly untagged), so the subclass filter
#: above misses them. An unnamed, untagged footway is treated as a sidewalk
#: when at least ``SIDEWALK_MIN_HITS`` of ``SIDEWALK_SAMPLES`` points along it
#: lie within ``SIDEWALK_OFFSET_M`` of a walkable street and run within
#: ``SIDEWALK_MAX_ANGLE_DEG`` of its direction. Set ``SIDEWALK_OFFSET_M`` to
#: None to keep every footway, as upstream does.
SIDEWALK_OFFSET_M: float | None = 12.0
SIDEWALK_MAX_ANGLE_DEG = 30.0
SIDEWALK_SAMPLES = 5
SIDEWALK_MIN_HITS = 4

WALK = ModeConfig(
    name="walk",
    allowed_classes=frozenset({
        "trunk", "primary", "secondary", "tertiary", "residential",
        "living_street", "unclassified", "service", "pedestrian",
        "footway", "path", "steps", "track", "cycleway", "bridleway", "unknown",
    }),
    excluded_subclasses=NEVER_ROUTABLE_SUBCLASSES,
    access_modes=("foot",),
    class_multiplier={
        # Stairs are walkable but slow and unpleasant with any load.
        "steps": 1.6,
        # Big arterials are unpleasant but passable on foot.
        "trunk": 1.15,
        "primary": 1.05,
    },
    respect_oneway=False,   # pedestrians are not bound by one-way streets
)

BIKE = ModeConfig(
    name="bike",
    allowed_classes=frozenset({
        "trunk", "primary", "secondary", "tertiary", "residential",
        "living_street", "unclassified", "service", "pedestrian",
        "footway", "path", "cycleway", "track", "unknown",
    }),
    # Steps are excluded outright for bicycles -- a route suitable for a
    # pedestrian is emphatically not necessarily rideable.
    excluded_subclasses=NEVER_ROUTABLE_SUBCLASSES,
    access_modes=("bicycle",),
    class_multiplier={
        "cycleway": 0.85,       # protected/dedicated bike infrastructure
        "living_street": 0.9,
        "residential": 0.95,
        "trunk": 1.9,           # e.g. 19th Ave, Van Ness -- high stress
        "primary": 1.45,
        "secondary": 1.2,
        "footway": 1.5,         # rideable only where bicycles are permitted
        "path": 1.1,
        "pedestrian": 1.4,
    },
    respect_oneway=True,
)

MODES: dict[str, ModeConfig] = {"walk": WALK, "bike": BIKE}

#: Classes bicycles may never use even when the class list would allow it.
BIKE_FORBIDDEN_CLASSES = frozenset({"steps", "bridleway"})

# --------------------------------------------------------------------------
# Analysis parameters
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AnalysisConfig:
    #: Number of representative access points sampled per neighborhood.
    #: One primary point is used for the pair matrix; the rest support
    #: robustness checks.
    points_per_neighborhood: int = 1
    #: Corridor detection: keep edges whose importance score is at or above
    #: this quantile of the positive-score distribution.
    corridor_score_quantile: float = 0.93
    #: Minimum corridor length (m) after merging contiguous segments.
    corridor_min_length_m: float = 350.0
    #: Maximum average grade for an edge to be eligible as corridor material.
    corridor_max_avg_grade: float = 0.055
    #: Corridors are grown only through edges below this maximum grade.
    corridor_max_edge_grade: float = 0.09
    #: Pass detection: elevation band (m) used to group saddle candidates.
    pass_cluster_radius_m: float = 400.0
    #: Number of Pareto sample pairs reported in detail.
    pareto_detail_pairs: int = 14
    #: Which qualifying intersection to use as a neighborhood's access point:
    #: 0 is the one nearest the street-weighted centre, 1 the next nearest,
    #: and so on. Non-zero values exist for robustness checks only.
    point_rank: int = 0


ANALYSIS = _apply(AnalysisConfig(), _OV)

# --------------------------------------------------------------------------
# Representative neighborhood pairs highlighted in the written analysis
# --------------------------------------------------------------------------
FEATURED_PAIRS = (
    ("Mission", "Outer Sunset"),
    ("Inner Richmond", "Downtown/Civic Center"),
    ("Mission", "Marina"),
    ("Bayview", "Golden Gate Park"),
    ("Noe Valley", "Financial District"),
    ("Outer Richmond", "Mission"),
    ("Excelsior", "South of Market"),
    ("Haight Ashbury", "Financial District"),
    ("Parkside", "Downtown/Civic Center"),
    ("Bernal Heights", "Marina"),
    ("Potrero Hill", "Western Addition"),
    ("West of Twin Peaks", "Downtown/Civic Center"),
    ("Visitacion Valley", "Mission"),
    ("Chinatown", "Inner Sunset"),
)

#: Validation targets -- well known flat corridors and steep streets used as
#: sanity checks on the elevation model (see ``validate`` command).
VALIDATION_FLAT = (
    "The Wiggle", "Market Street", "Valencia Street", "The Embarcadero",
    "Great Highway", "Alemany Boulevard", "San Jose Avenue", "Illinois Street",
)
VALIDATION_STEEP = (
    "Filbert Street", "22nd Street", "Jones Street", "Divisadero Street",
    "Lombard Street", "Duboce Avenue",
)


def profile(name: str) -> CostWeights:
    """Look up a routing profile by name."""
    try:
        return ROUTING_PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown routing profile {name!r}; "
                       f"choose from {sorted(ROUTING_PROFILES)}") from None


def with_alpha(weights: CostWeights, alpha: float) -> CostWeights:
    """Return a copy of ``weights`` with a different climbing weight."""
    return replace(weights, alpha=alpha, name=f"{weights.name}_a{alpha:g}")


def with_scale(weights: CostWeights, lam: float) -> CostWeights:
    """Scale every climbing/grade term of ``weights`` by ``lam``.

    ``lam == 0`` is pure distance, with the comfort multipliers switched off
    so that it coincides exactly with the ``shortest`` objective.
    """
    return replace(weights, alpha=weights.alpha * lam, beta=weights.beta * lam,
                   gamma=weights.gamma * lam,
                   use_class_multiplier=(lam > 0) and weights.use_class_multiplier,
                   name=f"{weights.name}_x{lam:g}")
