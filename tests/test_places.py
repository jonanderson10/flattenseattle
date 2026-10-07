"""The route page's offline place index."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sf_flat_routes import places
from sf_flat_routes.config import CITY_BBOX
from sf_flat_routes.download import ADDRESSES_PARQUET, BASE_PARQUETS, PLACES_PARQUET


def test_street_names_are_title_cased_with_suffixes_kept_short():
    assert places._title_street("JOHN MUIR DR") == "John Muir Dr"
    assert places._title_street("24TH ST") == "24th St"
    assert places._title_street("VAN NESS AVE") == "Van Ness Ave"
    assert places._title_street("DR CARLTON B GOODLETT PL") == "Dr Carlton B Goodlett Pl"


def test_core_strips_a_trailing_city_name():
    assert places._core("Cal Anderson Park, Seattle") == "cal anderson park"
    assert places._core("Space Needle, Seattle WA") == "space needle"
    assert places._core("Gas Works Park - Seattle") == "gas works park"
    assert places._core("Green Lake") == "green lake"
    # a suffix that is part of the name is not a city suffix: 'Washington'
    # is a street, a state and a university here
    assert places._core("Cafe Washington") == "cafe washington"
    assert places._core("Washington Park Arboretum") == "washington park arboretum"


def test_support_counts_nearby_records_that_mention_the_name():
    names = pd.Series(["Dolores Park", "Dolores Park Cafe", "Dolores Park Tennis",
                       "Dolores Park", "Nowhere"])
    lon = np.array([-122.427, -122.4265, -122.4275, -122.414, -122.5])
    lat = np.array([37.7596, 37.7600, 37.7593, 37.784, 37.7])
    sup = places._support(names, lon, lat, names, lon, lat)
    assert sup[0] == 2          # the real park: two neighbours mention it
    assert sup[3] == 0          # the stray copy across town: none


def test_variant_pruning_keeps_different_kinds_and_distant_namesakes():
    df = pd.DataFrame({
        "name": ["Dolores Park", "Dolores Park, Seattle", "Dolores Park Cafe",
                 "Golden Gate Park", "Golden Gate Park - East", "Golden Gate Park Carousel"],
        "group": ["park", "park", "food", "park", "park", "landmark"],
        "lon": [-122.427, -122.421, -122.4259, -122.482, -122.458, -122.458],
        "lat": [37.7596, 37.7736, 37.7613, 37.7694, 37.7691, 37.7691],
        "conf": [0.97, 0.72, 0.99, 0.98, 0.9, 0.9],
        "support": [3, 0, 0, 5, 0, 0],
    })
    out = places._prune_variants(df)
    kept = set(out["name"])
    assert "Dolores Park" in kept
    assert "Dolores Park, Seattle" not in kept            # city suffix, any distance
    assert "Dolores Park Cafe" in kept                    # a different kind of place
    assert "Golden Gate Park" in kept
    assert "Golden Gate Park Carousel" in kept            # different kind
    assert "Golden Gate Park - East" not in kept          # same kind, unsupported variant


needs_places = pytest.mark.skipif(not PLACES_PARQUET.exists(),
                                  reason="Overture places not downloaded")
needs_base = pytest.mark.skipif(not all(p.exists() for p in BASE_PARQUETS.values()),
                                reason="Overture base theme not downloaded")
needs_addresses = pytest.mark.skipif(not ADDRESSES_PARQUET.exists(),
                                     reason="Overture addresses not downloaded")


@pytest.fixture(scope="module")
def index():
    return places.build_places()


@needs_places
@needs_base
def test_place_index_is_compact_and_inside_the_city(index):
    n = len(index["names"])
    assert 5000 < n < 20000
    assert len(index["group"]) == n == len(index["lon"]) == len(index["lat"])
    assert max(index["group"]) < len(index["groups"])
    assert min(index["lon"]) >= CITY_BBOX[0] and max(index["lon"]) <= CITY_BBOX[1]
    assert min(index["lat"]) >= CITY_BBOX[2] and max(index["lat"]) <= CITY_BBOX[3]
    assert len(set(index["names"])) == n or len(set(zip(index["names"], index["group"]))) == n


@needs_places
@needs_base
def test_famous_places_are_found_where_they_belong(index):
    """Landmarks and parks resolve to their real location, mapped parks are
    filed as parks, and a park name is not duplicated by stray POI copies."""
    hits: dict = {}
    for n, g, lo, la in zip(index["names"], index["group"], index["lon"], index["lat"]):
        hits.setdefault((n, index["groups"][g]), []).append((lo, la))
    for name, kind, lon, lat in [("Space Needle", "landmark", -122.3493, 47.6205),
                                 ("Gas Works Park", "park", -122.3344, 47.6456),
                                 ("Green Lake Park", "park", -122.3300, 47.6780),
                                 ("Alki Beach", "beach", -122.4050, 47.5810),
                                 ("Discovery Park", "park", -122.4200, 47.6610),
                                 ("Fremont Troll", "landmark", -122.3473, 47.6510)]:
        assert (name, kind) in hits, name
        assert len(hits[(name, kind)]) == 1, name
        lo, la = hits[(name, kind)][0]
        assert abs(lo - lon) < 0.004 and abs(la - lat) < 0.004, name


@needs_places
@needs_base
def test_pike_place_market_is_the_public_market_sign(index):
    """Hand-placed: the feed puts the market above Western Avenue, but people
    arrive at the sign at Pike Street and Pike Place. Exactly one entry."""
    hits = [(lo, la) for n, lo, la in zip(index["names"], index["lon"], index["lat"])
            if n == "Pike Place Market"]
    assert hits == [(-122.34, 47.60884)]


@needs_addresses
def test_addresses_pack_into_sorted_uint16_offsets():
    a = places.build_addresses()
    n = len(a["number"])
    assert n > 100_000
    assert a["street"].dtype == np.dtype("<u2") and a["number"].dtype == np.dtype("<u2")
    assert a["lon"].dtype == np.dtype("<u2") and a["lat"].dtype == np.dtype("<u2")
    # sorted by street then number, so the browser can binary-search
    key = a["street"].astype(np.int64) * 100_000 + a["number"]
    assert np.all(np.diff(key) > 0)
    lon = a["origin"][0] + a["lon"] * 1e-5
    lat = a["origin"][1] + a["lat"] * 1e-5
    assert lon.min() >= CITY_BBOX[0] and lon.max() <= CITY_BBOX[1] + 1e-4
    assert lat.min() >= CITY_BBOX[2] and lat.max() <= CITY_BBOX[3] + 1e-4
    assert "East Pike Street" in a["streets"]
