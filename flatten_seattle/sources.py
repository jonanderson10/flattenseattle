"""Registry of every external dataset used by the project.

Each entry records the URL, the access date, resolution/vintage, licence and
the limitations that matter for this analysis.  ``python -m flatten_seattle
sources`` prints this table, and it is the single source of truth for the
data-provenance section of the README.
"""
from __future__ import annotations

from dataclasses import dataclass

#: Date on which every URL below was last fetched and verified.
ACCESS_DATE = "2026-09-16"

#: Overture Maps release used for the street network.
OVERTURE_RELEASE = "2026-08-19.0"
OVERTURE_BUCKET = "https://overturemaps-us-west-2.s3.amazonaws.com"
OVERTURE_PREFIX = f"release/{OVERTURE_RELEASE}/theme=transportation"
#: Places and addresses themes of the same release, used only for the route
#: page's offline place search (fetched 2026-10-04).
OVERTURE_PLACES_PREFIX = f"release/{OVERTURE_RELEASE}/theme=places"
OVERTURE_ADDRESSES_PREFIX = f"release/{OVERTURE_RELEASE}/theme=addresses"
OVERTURE_BASE_PREFIX = f"release/{OVERTURE_RELEASE}/theme=base"

#: USGS 3DEP 1 m lidar project covering Seattle (King County, flown 2021).
TNM_BUCKET = "https://prd-tnm.s3.amazonaws.com"
LIDAR_PROJECT = "WA_KingCounty_2021_B21"
LIDAR_PREFIX = f"StagedProducts/Elevation/1m/Projects/{LIDAR_PROJECT}/TIFF"
#: The eight 10 km tiles that the TNM products API returns for CITY_BBOX.
LIDAR_TILES = tuple(
    f"USGS_1M_10_x{x}y{y}_{LIDAR_PROJECT}.tif"
    for x in (54, 55) for y in (526, 527, 528, 529)
)

#: USGS 1/3 arc-second seamless DEM tile, used only to cross-validate the
#: lidar product (it is ~10 m and far too coarse for street grades).
SEAMLESS_DEM_URL = (
    f"{TNM_BUCKET}/StagedProducts/Elevation/13/TIFF/current/n48w123/"
    "USGS_13_n48w123.tif"
)

#: City of Seattle GIS (ArcGIS Online) feature services.
SEATTLE_GIS = "https://services.arcgis.com/ZOyb2t4B0UYuYNYH/arcgis/rest/services"
#: Neighborhood Map Atlas neighborhoods: 94 polygons that follow the
#: shoreline, so their union is the city's land area.
NEIGHBORHOOD_URL = (
    f"{SEATTLE_GIS}/nma_nhoods_sub/FeatureServer/0/query"
    "?where=1%3D1&outFields=S_HOOD,L_HOOD&outSR=4326&f=geojson"
)
#: SDOT bike facilities: layer 1 is multi-use trails, layer 2 on-street
#: facilities (protected and painted lanes, greenways, sharrows).
BIKE_FACILITIES_LAYER = f"{SEATTLE_GIS}/SDOT_Bike_Facilities/FeatureServer"


@dataclass(frozen=True)
class Dataset:
    key: str
    title: str
    publisher: str
    url: str
    accessed: str
    resolution: str
    licence: str
    limitations: str
    role: str
    local: str = ""
    notes: str = ""
    optional: bool = False
    substituted: bool = False
    substitution_reason: str = ""


DATASETS: tuple[Dataset, ...] = (
    Dataset(
        key="overture_segments",
        title=f"Overture Maps transportation segments (release {OVERTURE_RELEASE})",
        publisher="Overture Maps Foundation (derived from OpenStreetMap)",
        url=f"{OVERTURE_BUCKET}/{OVERTURE_PREFIX}/type=segment/",
        accessed=ACCESS_DATE,
        resolution="Vector linestrings; OSM-equivalent positional accuracy (~1-5 m)",
        licence="ODbL 1.0 (OpenStreetMap contributors); Overture schema CDLA-Permissive 2.0",
        role="Routable street network: geometry, road class, per-mode access "
             "restrictions, bridge/tunnel flags and connector topology.",
        local="data/raw/overture_segments_seattle.parquet",
        limitations=(
            "OSM-derived, so completeness and tagging quality vary by area. "
            "Road classification of arterials is inconsistent in places "
            "(state routes such as Aurora Ave N are tagged 'trunk' along "
            "stretches that are ordinary surface streets with sidewalks, so "
            "'trunk' cannot be excluded from walking/biking). Sidewalk and "
            "crosswalk geometry is present but of uneven completeness and is "
            "deliberately not used."
        ),
        notes="Read with Parquet row-group bbox pruning: only a handful of "
              "the global row groups intersect Seattle, so the whole "
              "extract costs seconds and megabytes instead of 64 GB.",
    ),
    Dataset(
        key="overture_connectors",
        title=f"Overture Maps transportation connectors (release {OVERTURE_RELEASE})",
        publisher="Overture Maps Foundation (derived from OpenStreetMap)",
        url=f"{OVERTURE_BUCKET}/{OVERTURE_PREFIX}/type=connector/",
        accessed=ACCESS_DATE,
        resolution="Vector points",
        licence="ODbL 1.0; Overture schema CDLA-Permissive 2.0",
        role="Authoritative intersection nodes. Using connector IDs for graph "
             "topology avoids geometric snapping tolerances entirely.",
        local="data/raw/overture_connectors_seattle.parquet",
        limitations="Connectors exist only where OSM ways share a node; "
                    "grade-separated crossings correctly do not connect.",
    ),
    Dataset(
        key="dem_1m",
        title=f"USGS 3DEP 1 metre bare-earth DEM, project {LIDAR_PROJECT}",
        publisher="U.S. Geological Survey, 3D Elevation Program",
        url=f"{TNM_BUCKET}/{LIDAR_PREFIX}/",
        accessed=ACCESS_DATE,
        resolution="1 m ground sample distance; NAD83/UTM 10N (EPSG:26910); "
                   "float32 metres above NAVD88",
        licence="Public domain (U.S. Government work)",
        role="Primary elevation source for all grade and climbing metrics.",
        local="data/raw/dem/*.tif",
        limitations=(
            "Bare-earth interpolation leaves artefacts on bridges, tunnels and "
            "elevated structures, where the DEM samples the ground or water "
            "surface underneath rather than the deck -- handled explicitly by "
            "interpolating elevation across segments flagged is_bridge or "
            "is_tunnel. Residual noise of a few decimetres from vehicles, "
            "curbs and vegetation misclassification is handled by "
            "Savitzky-Golay smoothing plus a gain dead-band. Eight 10 km tiles "
            "(~1.9 GB total) are cloud-optimised GeoTIFFs, so windowed reads "
            "are cheap."
        ),
    ),
    Dataset(
        key="dem_13",
        title="USGS 3DEP 1/3 arc-second seamless DEM, tile n48w123",
        publisher="U.S. Geological Survey, 3D Elevation Program",
        url=SEAMLESS_DEM_URL,
        accessed=ACCESS_DATE,
        resolution="1/3 arc-second (~10 m); EPSG:4269",
        licence="Public domain (U.S. Government work)",
        role="Independent cross-check on the 1 m lidar elevations (validation "
             "only -- too coarse for street grades).",
        local="data/raw/dem_13_n48w123.tif",
        limitations="~10 m posting smooths away street-scale relief and "
                    "systematically under-reports maximum grades.",
        optional=True,
    ),
    Dataset(
        key="neighborhoods",
        title="Seattle Neighborhood Map Atlas neighborhoods",
        publisher="City of Seattle (Department of Neighborhoods), Seattle GeoData",
        url=NEIGHBORHOOD_URL,
        accessed=ACCESS_DATE,
        resolution="Vector polygons, 94 neighborhoods in 20 districts",
        licence="Open data (City of Seattle)",
        role="City boundary (the union of the polygons, which follow the "
             "shoreline) for clipping the street network, and neighborhood "
             "labels on the route page.",
        local="data/raw/seattle_neighborhoods.geojson",
        limitations=(
            "Neighborhood names are informal and some overlap in common use. "
            "The polygons exclude open water, so bridges leaving the city "
            "(I-90, SR 520) are clipped about 250 m past the shore."
        ),
    ),
    Dataset(
        key="overture_places",
        title=f"Overture Maps places (release {OVERTURE_RELEASE})",
        publisher="Overture Maps Foundation (Meta and Microsoft POI data)",
        url=f"{OVERTURE_BUCKET}/{OVERTURE_PLACES_PREFIX}/type=place/",
        accessed="2026-10-04",
        resolution="Point features with names, categories and a confidence score",
        licence="CDLA Permissive 2.0",
        role="Offline place search in the route page (parks, landmarks, "
             "transit, schools, shops, cafes).",
        local="data/raw/overture_places_seattle.parquet",
        limitations="Point-of-interest coverage and naming are uneven; only "
                    "records with confidence >= 0.6 in routable categories "
                    "are kept. Not used by the analysis itself.",
        optional=True,
    ),
    Dataset(
        key="overture_base",
        title=f"Overture Maps base theme: land use, infrastructure, land (release {OVERTURE_RELEASE})",
        publisher="Overture Maps Foundation (derived from OpenStreetMap)",
        url=f"{OVERTURE_BUCKET}/{OVERTURE_BASE_PREFIX}/",
        accessed="2026-10-04",
        resolution="Mapped outlines and points with names and OSM-derived classes",
        licence="ODbL 1.0 (OpenStreetMap contributors)",
        role="Mapped parks, schools, hospitals, plazas, stations, piers, "
             "bridges, viewpoints, peaks and beaches for the route page's "
             "offline search; these outrank the POI feed, which places the "
             "same names unreliably.",
        local="data/raw/overture_{land_use,infrastructure,land}_seattle.parquet",
        limitations="Only named features in a fixed class list are used. "
                    "Not used by the analysis itself.",
        optional=True,
    ),
    Dataset(
        key="overture_addresses",
        title=f"Overture Maps addresses (release {OVERTURE_RELEASE})",
        publisher="Overture Maps Foundation (OpenAddresses / King County)",
        url=f"{OVERTURE_BUCKET}/{OVERTURE_ADDRESSES_PREFIX}/type=address/",
        accessed="2026-10-04",
        resolution="Address points with street number and street name",
        licence="Open (OpenAddresses sources)",
        role="Offline street-address search in the route page.",
        local="data/raw/overture_addresses_seattle.parquet",
        limitations="One point per (street, number) is kept; unit numbers "
                    "are dropped. Not used by the analysis itself.",
        optional=True,
    ),
    Dataset(
        key="bike_network",
        title="SDOT Bike Facilities (existing facilities and multi-use trails)",
        publisher="Seattle Department of Transportation, Seattle GeoData",
        url=f"{BIKE_FACILITIES_LAYER}/",
        accessed=ACCESS_DATE,
        resolution="~3,600 on-street facility segments (CATEGORY: protected, "
                   "buffered and painted lanes, climbing lanes, neighborhood "
                   "greenways, sharrows, off-street) and ~200 trail segments",
        licence="Open data (City of Seattle)",
        role="Bike-mode comfort weighting on the route page ('prefer calm "
             "streets'); see bikeways.py.",
        local="data/raw/sdot_bike_facilities.geojson",
        limitations=(
            "Matched to graph edges geometrically (within 12 m and 25 degrees, "
            "over at least half the edge), since Overture carries no SDOT "
            "segment keys. Facilities under construction are left out."
        ),
        optional=True,
    ),
)

DATASETS_BY_KEY = {d.key: d for d in DATASETS}


def format_table() -> str:
    """Human-readable provenance report."""
    lines = [f"Data sources (all URLs verified {ACCESS_DATE})", "=" * 78]
    for d in DATASETS:
        flag = " [OPTIONAL]" if d.optional else ""
        flag += " [SUBSTITUTED]" if d.substituted else ""
        lines += [
            f"\n{d.key}{flag}",
            f"  title       : {d.title}",
            f"  publisher   : {d.publisher}",
            f"  url         : {d.url}",
            f"  accessed    : {d.accessed}",
            f"  resolution  : {d.resolution}",
            f"  licence     : {d.licence}",
            f"  local cache : {d.local}",
            f"  role        : {d.role}",
            f"  limitations : {d.limitations}",
        ]
        if d.notes:
            lines.append(f"  notes       : {d.notes}")
        if d.substitution_reason:
            lines.append(f"  substitution: {d.substitution_reason}")
    return "\n".join(lines)


def markdown_table() -> str:
    """Compact markdown table for the README."""
    rows = ["| Dataset | Publisher | Resolution / vintage | Licence | Role |",
            "|---|---|---|---|---|"]
    for d in DATASETS:
        rows.append(
            f"| {d.title} | {d.publisher} | {d.resolution} | {d.licence} | {d.role} |"
        )
    return "\n".join(rows)
