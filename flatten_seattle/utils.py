"""Small shared helpers: logging, progress reporting and cache bookkeeping."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")

_LOG_FORMAT = "%(asctime)s  %(levelname)-7s %(name)-22s %(message)s"


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=_LOG_FORMAT,
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # rasterio / fiona are chatty at DEBUG
    for noisy in ("rasterio", "fiona", "pyogrio", "urllib3", "fsspec", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@contextmanager
def step(message: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """Log the start and wall-clock duration of an expensive stage."""
    log = logger or logging.getLogger("flatten_seattle")
    log.info("%s ...", message)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        log.info("%s done in %.1fs", message, time.perf_counter() - t0)


def progress(iterable: Iterable[T], desc: str = "", total: int | None = None,
             unit: str = "it") -> Iterable[T]:
    """tqdm progress bar that degrades gracefully when tqdm is absent."""
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        return iterable
    disable = not sys.stderr.isatty() and not os.environ.get("SFFR_FORCE_PROGRESS")
    return tqdm(iterable, desc=desc, total=total, unit=unit,
                disable=disable, mininterval=2.0, dynamic_ncols=True)


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def is_fresh(path: Path, *deps: Path) -> bool:
    """True if ``path`` exists and is newer than every dependency that exists."""
    if not path.exists():
        return False
    mtime = path.stat().st_mtime
    return all(not d.exists() or d.stat().st_mtime <= mtime for d in deps)


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def configure_gdal_for_proxy() -> None:
    """Make GDAL/rasterio work behind the sandbox HTTPS proxy.

    The environment routes outbound HTTPS through a CA-terminating proxy.
    GDAL's libcurl needs to be told about both the proxy and the CA bundle;
    without this ``/vsicurl/`` reads fail TLS verification.
    """
    ca = "/root/.ccr/ca-bundle.crt"
    if Path(ca).exists():
        os.environ.setdefault("CURL_CA_BUNDLE", ca)
        os.environ.setdefault("GDAL_HTTP_CAINFO", ca)
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:
        os.environ.setdefault("GDAL_HTTP_PROXY", proxy)
    # Cloud-optimised GeoTIFF friendly defaults
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("VSI_CACHE", "TRUE")
    os.environ.setdefault("VSI_CACHE_SIZE", "104857600")
    os.environ.setdefault("GDAL_CACHEMAX", "1024")
