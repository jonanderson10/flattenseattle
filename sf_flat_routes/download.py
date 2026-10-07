"""Data acquisition with local caching.

Everything is cached under ``data/raw`` and re-downloaded only when missing
(or when ``force=True``), so re-running the pipeline never re-fetches the
523 MB of lidar or re-scans the Overture release.

The Overture read is the interesting part: the transportation theme is ~64 GB
spread over 128 Parquet files.  Each file carries per-row-group statistics on
the ``bbox`` struct column, and Overture writes rows in spatial order, so we
read the 128 footers (cheap, a couple of range requests each), keep only the
row groups whose bounding box intersects San Francisco, and read just those.
In practice 7 row groups in a single file cover the city.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import json

import numpy as np
import requests

from . import sources
from .config import CITY_BBOX, CITY_SLUG, RAW_DIR
from .utils import configure_gdal_for_proxy, get_logger, human_bytes, progress, step

log = get_logger("sf_flat_routes.download")

_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

DEM_DIR = RAW_DIR / "dem"
SEGMENTS_PARQUET = RAW_DIR / f"overture_segments_{CITY_SLUG}.parquet"
CONNECTORS_PARQUET = RAW_DIR / f"overture_connectors_{CITY_SLUG}.parquet"
PLACES_PARQUET = RAW_DIR / f"overture_places_{CITY_SLUG}.parquet"
ADDRESSES_PARQUET = RAW_DIR / f"overture_addresses_{CITY_SLUG}.parquet"
#: Overture base theme (OpenStreetMap): mapped parks, schools, stations...
BASE_PARQUETS = {typ: RAW_DIR / f"overture_{typ}_{CITY_SLUG}.parquet"
                 for typ in ("land_use", "infrastructure", "land")}
NEIGHBORHOODS_GEOJSON = RAW_DIR / f"{CITY_SLUG}_neighborhoods.geojson"
DEM_13_TIF = RAW_DIR / "dem_13_n48w123.tif"
BIKEWAYS_GEOJSON = RAW_DIR / "sdot_bike_facilities.geojson"

#: Columns pulled from the Overture segment table. Everything unused is left
#: on the server -- the nested route/destination columns are large.
SEGMENT_COLUMNS = [
    "id", "names", "subtype", "class", "subclass", "connectors",
    "road_flags", "access_restrictions", "road_surface", "speed_limits",
    "level_rules", "geometry", "bbox", "sources",
]
CONNECTOR_COLUMNS = ["id", "geometry", "bbox"]
PLACE_COLUMNS = ["id", "names", "categories", "confidence", "geometry", "bbox"]
ADDRESS_COLUMNS = ["id", "number", "street", "unit", "postcode", "geometry", "bbox"]
BASE_COLUMNS = ["id", "names", "subtype", "class", "geometry", "bbox"]


# --------------------------------------------------------------------------
# generic helpers
# --------------------------------------------------------------------------
def _list_s3_keys(bucket: str, prefix: str, suffix: str = ".parquet") -> list[str]:
    """List keys under a public S3 prefix via the REST list-objects-v2 API."""
    keys: list[str] = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = requests.get(bucket + "/", params=params, timeout=120)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        for c in root.findall("s3:Contents", _S3_NS):
            key = c.find("s3:Key", _S3_NS).text
            if key.endswith(suffix):
                keys.append(key)
        if root.findtext("s3:IsTruncated", default="false", namespaces=_S3_NS) == "true":
            token = root.findtext("s3:NextContinuationToken", namespaces=_S3_NS)
        else:
            break
    return keys


def _download_file(url: str, dest: Path, force: bool = False,
                   retries: int = 5) -> Path:
    """Stream a URL to disk with retries and an atomic rename."""
    if dest.exists() and not force:
        log.info("cached %s (%s)", dest.name, human_bytes(dest.stat().st_size))
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=(30, 300)) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length") or 0)
                written = 0
                with open(tmp, "wb") as fh:
                    bar = progress(r.iter_content(chunk_size=1 << 20),
                                   desc=f"  {dest.name}",
                                   total=(total >> 20) + 1 if total else None,
                                   unit="MB")
                    for chunk in bar:
                        fh.write(chunk)
                        written += len(chunk)
            if total and written < total:
                raise IOError(f"short read: {written} of {total} bytes")
            tmp.replace(dest)
            log.info("downloaded %s (%s)", dest.name, human_bytes(dest.stat().st_size))
            return dest
        except Exception as exc:  # network flakiness is expected
            tmp.unlink(missing_ok=True)
            if attempt == retries:
                raise
            import time
            log.warning("attempt %d/%d for %s failed (%s); retrying in %.0fs",
                        attempt, retries, dest.name, exc, delay)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------
# Overture
# --------------------------------------------------------------------------
def _bbox_stat_columns(metadata) -> dict[str, int]:
    cols = {c["path_in_schema"]: i
            for i, c in enumerate(metadata.row_group(0).to_dict()["columns"])}
    return {k: cols[k] for k in ("bbox.xmin", "bbox.xmax", "bbox.ymin", "bbox.ymax")}


def _matching_row_groups(metadata, bbox) -> list[int]:
    lon_min, lon_max, lat_min, lat_max = bbox
    ix = _bbox_stat_columns(metadata)
    out = []
    for rg in range(metadata.num_row_groups):
        g = metadata.row_group(rg)
        xmax = g.column(ix["bbox.xmax"]).statistics.max
        xmin = g.column(ix["bbox.xmin"]).statistics.min
        ymax = g.column(ix["bbox.ymax"]).statistics.max
        ymin = g.column(ix["bbox.ymin"]).statistics.min
        if xmax >= lon_min and xmin <= lon_max and ymax >= lat_min and ymin <= lat_max:
            out.append(rg)
    return out


def _read_overture_type(overture_type: str, columns: list[str], dest: Path,
                        bbox=CITY_BBOX, force: bool = False,
                        prefix: str = sources.OVERTURE_PREFIX) -> Path:
    """Row-group-pruned read of one Overture type (any theme)."""
    import fsspec
    import pyarrow as pa
    import pyarrow.parquet as pq

    if dest.exists() and not force:
        log.info("cached %s (%s)", dest.name, human_bytes(dest.stat().st_size))
        return dest

    keys = _list_s3_keys(sources.OVERTURE_BUCKET, f"{prefix}/type={overture_type}/")
    log.info("overture %s: %d parquet files in release %s",
             overture_type, len(keys), sources.OVERTURE_RELEASE)

    fs = fsspec.filesystem("https")

    def scan(key: str):
        url = f"{sources.OVERTURE_BUCKET}/{key}"
        with fs.open(url, block_size=8 << 20) as fh:
            md = pq.ParquetFile(fh).metadata
            return key, _matching_row_groups(md, bbox)

    with step(f"scanning {len(keys)} {overture_type} footers for the SF bbox", log):
        with ThreadPoolExecutor(max_workers=16) as pool:
            scanned = list(pool.map(scan, keys))
    hits = [(k, rgs) for k, rgs in scanned if rgs]
    n_rg = sum(len(r) for _, r in hits)
    log.info("overture %s: %d row group(s) in %d file(s) intersect SF",
             overture_type, n_rg, len(hits))
    if not hits:
        raise RuntimeError(f"no Overture {overture_type} row groups intersect {bbox}")

    tables = []
    for key, rgs in progress(hits, desc=f"  reading {overture_type}", unit="file"):
        url = f"{sources.OVERTURE_BUCKET}/{key}"
        with fs.open(url, block_size=16 << 20) as fh:
            t = pq.ParquetFile(fh).read_row_groups(rgs, columns=columns)
        tables.append(t)
    table = pa.concat_tables(tables)

    # Row groups are coarse; clip precisely to the study bbox.
    bb = table.column("bbox").combine_chunks()
    xmin = np.asarray(bb.field("xmin")); xmax = np.asarray(bb.field("xmax"))
    ymin = np.asarray(bb.field("ymin")); ymax = np.asarray(bb.field("ymax"))
    lon_min, lon_max, lat_min, lat_max = bbox
    keep = ((xmax >= lon_min) & (xmin <= lon_max)
            & (ymax >= lat_min) & (ymin <= lat_max))
    table = table.filter(pa.array(keep))
    log.info("overture %s: %d features inside the study bbox", overture_type,
             table.num_rows)

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(dest)
    log.info("wrote %s (%s)", dest.name, human_bytes(dest.stat().st_size))
    return dest


def download_street_network(force: bool = False) -> tuple[Path, Path]:
    seg = _read_overture_type("segment", SEGMENT_COLUMNS, SEGMENTS_PARQUET, force=force)
    con = _read_overture_type("connector", CONNECTOR_COLUMNS, CONNECTORS_PARQUET,
                              force=force)
    return seg, con


def download_places(force: bool = False) -> tuple[Path, Path]:
    """Places and addresses for the route page's offline search.

    Neither is needed by the analysis; the route page degrades to
    intersection-only search when they are missing.
    """
    places = _read_overture_type("place", PLACE_COLUMNS, PLACES_PARQUET,
                                 force=force, prefix=sources.OVERTURE_PLACES_PREFIX)
    addrs = _read_overture_type("address", ADDRESS_COLUMNS, ADDRESSES_PARQUET,
                                force=force, prefix=sources.OVERTURE_ADDRESSES_PREFIX)
    for typ, dest in BASE_PARQUETS.items():
        _read_overture_type(typ, BASE_COLUMNS, dest, force=force,
                            prefix=sources.OVERTURE_BASE_PREFIX)
    return places, addrs


# --------------------------------------------------------------------------
# Elevation
# --------------------------------------------------------------------------
def download_dem(force: bool = False, include_seamless: bool = True) -> list[Path]:
    """Fetch the 1 m 3DEP tiles that cover the city."""
    paths = []
    for tile in sources.LIDAR_TILES:
        url = f"{sources.TNM_BUCKET}/{sources.LIDAR_PREFIX}/{tile}"
        paths.append(_download_file(url, DEM_DIR / tile, force=force))
    if include_seamless:
        try:
            _download_file(sources.SEAMLESS_DEM_URL, DEM_13_TIF, force=force)
        except Exception as exc:
            log.warning("optional 1/3 arc-second DEM unavailable (%s)", exc)
    return paths


# --------------------------------------------------------------------------
# Neighborhoods
# --------------------------------------------------------------------------
def download_neighborhoods(force: bool = False) -> Path:
    return _download_file(sources.NEIGHBORHOOD_URL, NEIGHBORHOODS_GEOJSON, force=force)


# --------------------------------------------------------------------------
# Bikeways
# --------------------------------------------------------------------------
#: SDOT facility statuses that are on the ground now (PLNRECON is an existing
#: facility with a rebuild planned); under-construction ones are left out.
_BIKE_STATUSES = ("INSVC", "PLNRECON")


def _arcgis_features(layer_url: str, where: str, fields: str) -> list[dict]:
    """Every feature of an ArcGIS feature layer as GeoJSON, paging past the
    server's record limit."""
    feats: list[dict] = []
    offset = 0
    while True:
        r = requests.get(f"{layer_url}/query", timeout=(30, 120), params={
            "where": where, "outFields": fields, "outSR": 4326, "f": "geojson",
            "orderByFields": "OBJECTID", "resultOffset": offset,
            "resultRecordCount": 1000})
        r.raise_for_status()
        page = r.json().get("features", [])
        feats += page
        if len(page) < 1000:
            return feats
        offset += len(page)


def download_bikeways(force: bool = False) -> Path:
    """SDOT on-street bike facilities and multi-use trails, as one GeoJSON
    whose features carry ``category`` (SDOT code, or ``TRAIL``) and ``street``."""
    if BIKEWAYS_GEOJSON.exists() and not force:
        log.info("cached %s", BIKEWAYS_GEOJSON.name)
        return BIKEWAYS_GEOJSON
    base = sources.BIKE_FACILITIES_LAYER
    statuses = ",".join(f"'{s}'" for s in _BIKE_STATUSES)
    out = []
    for f in _arcgis_features(f"{base}/2", f"CURRENT_STATUS IN ({statuses})",
                              "OBJECTID,CATEGORY,UNITDESC"):
        p = f.get("properties") or {}
        out.append({"type": "Feature", "geometry": f.get("geometry"),
                    "properties": {"category": p.get("CATEGORY") or "",
                                   "street": p.get("UNITDESC") or ""}})
    for f in _arcgis_features(f"{base}/1", "1=1", "OBJECTID,ORD_STNAME_CONCAT"):
        p = f.get("properties") or {}
        out.append({"type": "Feature", "geometry": f.get("geometry"),
                    "properties": {"category": "TRAIL",
                                   "street": p.get("ORD_STNAME_CONCAT") or ""}})
    BIKEWAYS_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    BIKEWAYS_GEOJSON.write_text(json.dumps({"type": "FeatureCollection", "features": out}))
    log.info("wrote %s: %d features", BIKEWAYS_GEOJSON.name, len(out))
    return BIKEWAYS_GEOJSON


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def download_all(force: bool = False) -> dict[str, object]:
    configure_gdal_for_proxy()
    out: dict[str, object] = {}
    with step("downloading street network (Overture)", log):
        out["segments"], out["connectors"] = download_street_network(force=force)
    with step("downloading neighborhood boundaries", log):
        out["neighborhoods"] = download_neighborhoods(force=force)
    with step("downloading SDOT bike facilities", log):
        try:
            out["bikeways"] = download_bikeways(force=force)
        except Exception as exc:  # optional: bike mode works without it
            log.warning("bike facilities unavailable (%s); 'prefer calm "
                        "streets' will use road class alone", exc)
    with step("downloading places and addresses (Overture)", log):
        try:
            out["places"], out["addresses"] = download_places(force=force)
        except Exception as exc:  # optional: the route page can do without
            log.warning("places/addresses unavailable (%s); the route page "
                        "will offer intersection search only", exc)
    with step("downloading USGS 3DEP elevation", log):
        out["dem"] = download_dem(force=force)
    return out
