"""Offline place index for the route page: addresses, places, and the base.

A static page cannot call a geocoding API without carrying a billed key in
public, and should not need one for a single city anyway, so the index is
built here and shipped with the page.
Three sources, all already in hand or fetched the same way as the streets:

* **Street intersections** are derived in the browser from the graph itself
  ("24th St & Mission St"); nothing to pack.
* **Mapped features** come from Overture's base theme, which is OpenStreetMap
  data: parks, playgrounds, schools, hospitals, plazas, stations, piers,
  bridges, viewpoints, peaks and beaches, each placed on its mapped outline.
  These are authoritative and are never pruned.
* **Places** come from Overture's places theme (Meta/Microsoft POI data):
  landmarks, museums, shops, cafes and so on. The feed is noisy -- the same
  name recurs at several spots, some of them nowhere near the real thing --
  so a record is kept only where nearby records corroborate it, and it is
  dropped when a mapped feature already carries its name.
* **Addresses** come from Overture's addresses theme (OpenAddresses data for
  San Francisco), deduplicated to one point per street number.

The hillshade base is rendered from the same lidar DEM the analysis uses and
reprojected to WGS84 so it overlays correctly, then palette-quantised: it is
a quiet grey image and does not need 24-bit colour.
"""
from __future__ import annotations

import base64
import io
import re

import numpy as np
import pandas as pd

from .config import CITY_BBOX, CITY_LAT, LON_M_PER_DEG, PROCESSED_DIR
from .download import ADDRESSES_PARQUET, BASE_PARQUETS, PLACES_PARQUET
from .utils import get_logger, step

log = get_logger("sf_flat_routes.places")

HILLSHADE_PNG = PROCESSED_DIR / "hillshade_light.png"

#: Overture primary categories kept, grouped for display. Anything not listed
#: is dropped unless it is a landmark-like category matched by _KEEP_RE.
CATEGORY_GROUPS = {
    "park": ["park", "garden", "playground", "beach", "dog_park", "hiking_trail",
             "scenic_point", "nature_preserve", "botanical_garden", "national_park",
             "state_park", "plaza", "picnic_ground"],
    "landmark": ["landmark_and_historical_building", "monument", "tourist_attraction",
                 "museum", "art_museum", "history_museum", "science_museum",
                 "aquarium", "zoo", "stadium_arena", "observatory", "lighthouse",
                 "historical_site", "memorial", "pier", "marina", "amusement_park",
                 "theatre", "performing_arts_theatre", "concert_hall", "cinema",
                 "library", "public_library"],
    "transit": ["train_station", "light_rail_station", "subway_station",
                "bus_station", "ferry_terminal", "transit_station", "cable_car_station",
                "metro_station", "public_transportation"],
    "school": ["school", "university", "college_university", "high_school",
               "elementary_school", "middle_school", "community_college", "campus"],
    "civic": ["city_hall", "courthouse", "post_office", "hospital", "fire_station",
              "police_station", "community_center", "recreation_center",
              "swimming_pool", "public_swimming_pool", "church_cathedral", "synagogue",
              "mosque", "temple", "farmers_market", "public_market"],
    "food": ["restaurant", "cafe", "coffee_shop", "bakery", "bar", "pub", "brewery",
             "ice_cream_shop", "pizza_restaurant", "taco_restaurant", "diner",
             "dessert_shop", "tea_room", "wine_bar", "cocktail_bar", "food_court"],
    "shop": ["grocery_store", "supermarket", "bookstore", "shopping_center",
             "hardware_store", "bicycle_shop", "pharmacy", "farmers_market",
             "convenience_store", "department_store", "record_store", "florist"],
    "lodging": ["hotel", "hostel", "bed_and_breakfast"],
}
_GROUP_OF = {c: g for g, cs in CATEGORY_GROUPS.items() for c in cs}

#: Overture base-theme (OpenStreetMap) classes kept, with the kind shown in
#: the search list. Keyed by (type, class).
BASE_CLASSES = {
    ("land_use", "park"): "park", ("land_use", "dog_park"): "dog park",
    ("land_use", "playground"): "playground", ("land_use", "garden"): "garden",
    ("land_use", "allotments"): "community garden", ("land_use", "plaza"): "plaza",
    ("land_use", "pedestrian"): "plaza", ("land_use", "school"): "school",
    ("land_use", "kindergarten"): "school", ("land_use", "university"): "university",
    ("land_use", "college"): "college", ("land_use", "hospital"): "hospital",
    ("land_use", "golf_course"): "golf course", ("land_use", "stadium"): "stadium",
    ("land_use", "marina"): "marina", ("land_use", "recreation_ground"): "park",
    ("land_use", "national_park"): "park", ("land_use", "military"): "landmark",
    ("land_use", "protected_landscape_seascape"): "park",
    ("infrastructure", "railway_station"): "station",
    ("infrastructure", "subway_station"): "station",
    ("infrastructure", "ferry_terminal"): "ferry", ("infrastructure", "pier"): "pier",
    ("infrastructure", "bridge"): "bridge", ("infrastructure", "viewpoint"): "viewpoint",
    ("infrastructure", "observation"): "landmark",
    ("infrastructure", "communication_tower"): "landmark",
    ("land", "peak"): "peak", ("land", "hill"): "hill", ("land", "beach"): "beach",
    ("land", "island"): "island", ("land", "islet"): "island",
}
_KEEP_RE = re.compile(r"park|garden|museum|station|landmark|monument|library|"
                      r"theat|school|universit|college|church|cathedral|hospital|"
                      r"beach|plaza|square|pier|market|stadium|trail|overlook", re.I)

_SUFFIX = {
    "ST": "St", "AVE": "Ave", "BLVD": "Blvd", "DR": "Dr", "RD": "Rd", "CT": "Ct",
    "PL": "Pl", "LN": "Ln", "TER": "Ter", "WAY": "Way", "HWY": "Hwy", "PKWY": "Pkwy",
    "CIR": "Cir", "ALY": "Aly", "SQ": "Sq", "TERR": "Ter", "STWY": "Stwy",
    "HL": "Hl", "WALK": "Walk", "LOOP": "Loop", "ROW": "Row", "PATH": "Path",
    "EXPY": "Expy", "PLZ": "Plz",
}
_DIR = {"N": "N", "S": "S", "E": "E", "W": "W"}


def _title_street(raw: str) -> str:
    """'JOHN MUIR DR' -> 'John Muir Dr'; keeps ordinals like '24TH' -> '24th'."""
    out = []
    for tok in str(raw).split():
        if tok in _SUFFIX:
            out.append(_SUFFIX[tok])
        elif re.fullmatch(r"\d+(ST|ND|RD|TH)", tok):
            out.append(tok.lower())
        elif tok in _DIR and len(out):
            out.append(tok)
        else:
            out.append(tok.capitalize() if not tok.isdigit() else tok)
    return " ".join(out)


def _support(names: pd.Series, lon: np.ndarray, lat: np.ndarray,
             all_names: pd.Series, all_lon: np.ndarray, all_lat: np.ndarray,
             radius_m: float = 300.0) -> np.ndarray:
    """How many nearby records mention each name.

    'Dolores Park' at the real park is surrounded by 'Dolores Park Cafe',
    'Dolores Park Tennis Courts' and so on; a stray 'Dolores Park' dropped in
    the Tenderloin has none of that. The feed carries several such strays
    with full confidence, so the name alone cannot pick the right one.
    """
    cell = radius_m / 111000.0
    grid: dict[tuple[int, int], list[int]] = {}
    low = all_names.str.lower().to_numpy()
    for i, (x, y) in enumerate(zip(all_lon, all_lat)):
        grid.setdefault((int(x / cell), int(y / cell)), []).append(i)
    out = np.zeros(len(names), dtype=int)
    for k, (n, x, y) in enumerate(zip(names.str.lower(), lon, lat)):
        cx, cy = int(x / cell), int(y / cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((cx + dx, cy + dy), ()):
                    o = low[j]
                    if o != n and n in o and abs(all_lon[j] - x) * LON_M_PER_DEG < radius_m \
                            and abs(all_lat[j] - y) * 111000 < radius_m:
                        out[k] += 1
    return out


_CITY_SUFFIXES = {"seattle", "seattle wa", "seattle washington", "wa",
                  "washington", "usa"}
# 'X Seattle' is a city suffix even without a comma; a bare trailing
# 'Washington' or 'WA' only after one ('Washington' alone is a street, a
# state and a university here).
_CORE_RE = re.compile(
    r"(?:[\s,\-/]+seattle(?:[\s,]+(?:wa|washington|usa))?"
    r"|[,\-/]\s*(?:wa|washington|usa))\s*$", re.I)


def _core(name: str) -> str:
    """'Cal Anderson Park, Seattle' -> 'cal anderson park'."""
    return _CORE_RE.sub("", str(name)).strip().lower()


def _prune_variants(df: pd.DataFrame, radius_m: float = 500.0) -> pd.DataFrame:
    """Drop 'Dolores Park, San Francisco' when 'Dolores Park' is 200 m away.

    The places feed carries many user-typed variants of the same name. A
    record is dropped when a better-supported kept name is a prefix of it
    (at a word boundary) and the two points are within ``radius_m``.
    """
    df = df.sort_values(["support", "conf"], ascending=False).reset_index(drop=True)
    kept_idx = []
    by_first: dict[str, list[int]] = {}
    lon = df["lon"].to_numpy(); lat = df["lat"].to_numpy()
    names = df["name"].tolist(); groups = df["group"].tolist()
    support = df["support"].to_numpy()
    for i, n in enumerate(names):
        first = n.split()[0].lower()
        dup = False
        for j in by_first.get(first, ()):
            k = names[j]
            if not (n.startswith(k) and len(n) > len(k) and n[len(k)] in " ,-/("):
                continue
            # a city suffix never marks a different place; anything else
            # (a branch, a sub-area) only when it is close by
            rest = n[len(k):].strip(" ,-/()").lower()
            if rest in _CITY_SUFFIXES:
                r = 6000.0                       # 'X, Seattle' anywhere
            elif groups[i] != groups[j]:
                continue                         # 'Dolores Park Cafe' is a cafe
            elif support[j] >= 3 and support[i] == 0:
                r = 6000.0                       # a same-kind variant of a well-known name
            else:
                r = radius_m
            if (abs(lon[i] - lon[j]) * LON_M_PER_DEG < r
                    and abs(lat[i] - lat[j]) * 111000 < r):
                dup = True
                break
        if not dup:
            kept_idx.append(i)
            by_first.setdefault(first, []).append(i)
    out = df.iloc[kept_idx].reset_index(drop=True)
    log.info("places: %d near-duplicate name variants pruned", len(df) - len(out))
    return out


def build_base() -> pd.DataFrame:
    """Named OpenStreetMap features from the Overture base theme."""
    import pyarrow.parquet as pq
    import shapely
    frames = []
    for typ, path in BASE_PARQUETS.items():
        if not path.exists():
            continue
        t = pq.read_table(path, columns=["names", "class", "geometry"]).to_pandas()
        name = t["names"].map(lambda n: (n or {}).get("primary") if isinstance(n, dict) else None)
        kind = [BASE_CLASSES.get((typ, c)) for c in t["class"]]
        keep = name.notna() & pd.Series(kind, index=t.index).notna()
        geom = shapely.from_wkb(t.loc[keep, "geometry"].values)
        pts = shapely.get_coordinates(shapely.point_on_surface(geom)) if len(geom) else np.zeros((0, 2))
        area = np.where(shapely.get_type_id(geom) >= 3, shapely.area(geom), 0.0) if len(geom) else []
        frames.append(pd.DataFrame({
            "name": name[keep].to_numpy(), "group": np.asarray(kind, dtype=object)[keep.to_numpy()],
            "lon": pts[:, 0], "lat": pts[:, 1], "area": area,
        }))
    if not frames:
        return pd.DataFrame(columns=["name", "group", "lon", "lat", "area"])
    df = pd.concat(frames, ignore_index=True)
    df = df[df["name"].str.len() >= 3]
    # a bridge is mapped once per carriageway and a park once per polygon
    # ring: keep the largest outline under each name
    df = (df.sort_values("area", ascending=False)
            .drop_duplicates(["name", "group"]).reset_index(drop=True))
    log.info("base features: %d named (%s)", len(df),
             ", ".join(f"{g} {n}" for g, n in df["group"].value_counts().head(8).items()))
    return df


def build_places() -> dict:
    """Compact place list: names, display kind, coordinates."""
    import pyarrow.parquet as pq
    base = build_base()
    t = pq.read_table(PLACES_PARQUET).to_pandas()
    names = t["names"].map(lambda n: (n or {}).get("primary") if isinstance(n, dict) else None)
    cats = t["categories"].map(lambda c: (c or {}).get("primary") if isinstance(c, dict) else None)
    conf = pd.to_numeric(t["confidence"], errors="coerce").fillna(0)
    import shapely
    geom = shapely.from_wkb(t["geometry"].values)
    lon = np.array([g.x for g in geom]); lat = np.array([g.y for g in geom])

    group = cats.map(lambda c: (_GROUP_OF.get(c) or ("landmark" if _KEEP_RE.search(c) else None)) if isinstance(c, str) else None)
    support = _support(names.fillna(""), lon, lat, names.fillna(""), lon, lat)
    keep = names.notna() & (names.str.len() >= 3) & group.notna() & (conf >= 0.6)
    # a famous place with no useful category still deserves a slot, and so
    # does anything the records around it keep mentioning (the Ferry
    # Building is filed under farming services, after its market)
    keep |= names.notna() & cats.isna() & (conf >= 0.9)
    keep |= names.notna() & (support >= 5) & (conf >= 0.6)
    df = pd.DataFrame({"name": names, "group": group.fillna("landmark"),
                       "conf": conf, "lon": lon, "lat": lat, "support": support})[keep]
    df = df[(df["lon"].between(CITY_BBOX[0], CITY_BBOX[1]))
            & (df["lat"].between(CITY_BBOX[2], CITY_BBOX[3]))]
    # one record per name: a search for 'Pike Place Market' should find the
    # market, not also a food stall of that name across downtown. The record
    # with the most corroborating neighbours wins.
    df = (df.sort_values(["support", "conf"], ascending=False)
            .drop_duplicates(["name"])
            .reset_index(drop=True))
    # the mapped feature wins over any POI record of the same name, or of a
    # trailing part of it ('Dolores Park' for 'Mission Dolores Park')
    mapped = set(_core(n) for n in base["name"])
    tails = set()
    for n in mapped:
        words = n.split()
        for k in range(1, len(words)):
            tail = " ".join(words[k:])
            if len(tail) >= 8:
                tails.add(tail)
    def superseded(n: str) -> bool:
        c = _core(n)
        head = re.split(r"\s*[,\-/(]\s*", c, maxsplit=1)[0]   # 'Ferry Building, Embarcadero'
        return c in mapped or c in tails or head in mapped or head in tails
    dup = df["name"].map(superseded)
    log.info("places: %d records superseded by mapped features", int(dup.sum()))
    df = _prune_variants(df[~dup].reset_index(drop=True))
    df = pd.concat([base[["name", "group", "lon", "lat"]], df[["name", "group", "lon", "lat"]]],
                   ignore_index=True)
    df = df[(df["lon"].between(CITY_BBOX[0], CITY_BBOX[1]))
            & (df["lat"].between(CITY_BBOX[2], CITY_BBOX[3]))]
    df = df.sort_values(["name"]).reset_index(drop=True)
    log.info("places: %d kept of %d POI records plus %d mapped features (%s)",
             len(df) - len(base), len(t), len(base),
             ", ".join(f"{g} {n}" for g, n in df["group"].value_counts().head(12).items()))
    groups = sorted(set(df["group"]))
    return {
        "names": df["name"].tolist(),
        "group": [groups.index(g) for g in df["group"]],
        "groups": groups,
        "lon": np.round(df["lon"].to_numpy(), 5).tolist(),
        "lat": np.round(df["lat"].to_numpy(), 5).tolist(),
    }


def build_addresses() -> dict:
    """One point per (street, number), with a street table."""
    import pyarrow.parquet as pq
    import shapely
    t = pq.read_table(ADDRESSES_PARQUET, columns=["number", "street", "geometry"]).to_pandas()
    t = t[t["street"].notna() & t["number"].notna()]
    num = pd.to_numeric(t["number"].astype(str).str.extract(r"^(\d+)")[0], errors="coerce")
    ok = num.notna() & (num < 65536)
    t = t[ok].copy(); t["num"] = num[ok].astype(int)
    geom = shapely.from_wkb(t["geometry"].values)
    t["lon"] = [g.x for g in geom]; t["lat"] = [g.y for g in geom]
    t["street_t"] = t["street"].map(_title_street)
    t = (t.sort_values(["street_t", "num"])
           .drop_duplicates(["street_t", "num"]).reset_index(drop=True))
    streets = sorted(t["street_t"].unique())
    sidx = {s: i for i, s in enumerate(streets)}
    log.info("addresses: %d unique street numbers on %d streets", len(t), len(streets))
    lon0, lat0 = CITY_BBOX[0], CITY_BBOX[2]
    return {
        "streets": streets,
        "street": t["street_t"].map(sidx).to_numpy().astype("<u2"),
        "number": t["num"].to_numpy().astype("<u2"),
        # 1e-5 degree offsets from the bbox corner fit in uint16 (~1 m)
        "lon": np.clip(np.round((t["lon"].to_numpy() - lon0) / 1e-5), 0, 65535).astype("<u2"),
        "lat": np.clip(np.round((t["lat"].to_numpy() - lat0) / 1e-5), 0, 65535).astype("<u2"),
        "origin": [lon0, lat0],
    }


def _water_mask(dem: np.ndarray, min_px: int = 400) -> np.ndarray:
    """Hydro-flattened water in a DEM: below the water floor, or dead flat.

    Lidar DEMs replace each water body with a single flat elevation (Puget
    Sound near 0 m, Lake Washington near 6 m). Land is never exactly flat
    over a few hundred metres, so a large connected patch with no variation
    at all is water. Where the DEM carries no water surface (San Francisco
    leaves it as no-data), this finds nothing and changes nothing.
    """
    from scipy import ndimage

    from .config import ELEVATION

    finite = np.isfinite(dem)
    z = np.where(finite, dem, 0.0)
    span = ndimage.maximum_filter(z, 3) - ndimage.minimum_filter(z, 3)
    flat = finite & (span < 0.02)
    floor = ELEVATION.water_below_m
    if floor is not None:
        flat |= finite & (dem < floor)
    labels, n = ndimage.label(flat)
    if n == 0:
        return flat
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    water = sizes[labels] >= min_px
    # close the one-pixel seams that averaging leaves along shorelines
    return ndimage.binary_opening(water, iterations=1)


def build_hillshade(width_px: int = 1600) -> dict:
    """Quiet shaded relief in WGS84, as a palette PNG data URI with bounds."""
    import rasterio
    from PIL import Image
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject

    from .elevation import DEM_MOSAIC
    from .viz_static import hillshade

    with step("rendering the hillshade base", log):
        with rasterio.open(DEM_MOSAIC) as src:
            transform, w, h = calculate_default_transform(
                src.crs, "EPSG:4326", src.width, src.height, *src.bounds)
            scale = width_px / w
            w2, h2 = int(w * scale), int(h * scale)
            transform = transform * transform.scale(w / w2, h / h2)
            dem = np.full((h2, w2), np.nan, dtype="float32")
            reproject(rasterio.band(src, 1), dem, dst_transform=transform,
                      dst_crs="EPSG:4326", dst_nodata=np.nan,
                      resampling=Resampling.average)
            nod = src.nodata
        dem = np.where(np.isfinite(dem) & (dem != nod) & (dem > -50), dem, np.nan)
        valid = np.isfinite(dem) & ~_water_mask(dem)
        filled = np.where(valid, dem, np.nanmedian(dem))
        px_m = abs(transform.a) * 111320 * np.cos(np.radians(CITY_LAT))
        hs = hillshade(filled, res=px_m, z_factor=1.8)
        shade = (0.72 + 0.28 * hs)[..., None]
        tint = np.clip(filled / 260, 0, 1)[..., None]
        base = np.array([243, 242, 238], float); dark = np.array([196, 194, 186], float)
        rgb = (base * (1 - tint * 0.45) + dark * (tint * 0.45)) * shade
        img = np.zeros((h2, w2, 4), np.uint8)
        img[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        img[..., 3] = np.where(valid, 255, 0)
        im = Image.fromarray(img, "RGBA").quantize(colors=96, method=Image.Quantize.FASTOCTREE)
        buf = io.BytesIO(); im.save(buf, "PNG", optimize=True)
        im.save(HILLSHADE_PNG)
        west, north = transform.c, transform.f
        east, south = west + transform.a * w2, north + transform.e * h2
        log.info("  hillshade %dx%d, %.0f KB", w2, h2, len(buf.getvalue()) / 1024)
    return {"bounds": [[south, west], [north, east]], "png": buf.getvalue(),
            "data_uri": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}
