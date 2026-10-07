"""SFMTA bikeway conflation and the bike comfort multiplier."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from flatten_seattle import bikeways


def test_facility_codes_follow_sdot_categories():
    f = bikeways._facility
    assert f({"category": "TRAIL"}) == "path"
    assert f({"category": "BKF-OFFST"}) == "path"
    assert f({"category": "BKF-PBL"}) == "separated"
    assert f({"category": "BKF-BBL"}) == "buffered_lane"
    assert f({"category": "BKF-BL"}) == "lane"
    assert f({"category": "BKF-CLMB"}) == "lane"
    assert f({"category": "BKF-NGW"}) == "neighborway"
    assert f({"category": "BKF-SHW"}) == "route"
    assert f({"category": None}) == ""
    assert f({}) == ""


def test_stress_prefers_protected_lanes_and_penalises_bare_arterials():
    edges = pd.DataFrame({
        "cls": ["primary", "primary", "secondary", "residential", "residential",
                "trunk", "cycleway", "tertiary"],
    })
    fac = pd.Series(["", "separated", "lane", "", "route", "", "", "route"])
    m = bikeways.stress(edges, fac)
    assert m[0] == 1.6                       # Geary with nothing
    assert m[1] == 0.8                       # Geary with a protected lane
    assert m[2] == 1.0                       # a painted lane feels like a quiet street
    assert m[3] == 1.0                       # the baseline
    assert m[4] == pytest.approx(0.95)       # sharrows on a quiet street
    assert m[5] == 2.0                       # 19th Ave
    assert m[6] == 0.8
    # a class III route on a bigger street never counts worse than a tertiary
    assert m[7] == pytest.approx(1.2 * bikeways.ROUTE_SOFTEN)
    assert m.min() >= 0.8 and m.max() <= 2.0


def test_conflation_matches_parallel_nearby_segments_only():
    # three east-west blocks in UTM metres; a bikeway runs along the first
    # two, offset by 4 m (a centreline vs. a lane), and a cross street
    # passes through the third at right angles
    edges = gpd.GeoDataFrame({
        "cls": ["residential"] * 3, "bike_ok": [True] * 3,
        "geometry": [LineString([(0, 0), (100, 0)]), LineString([(100, 0), (200, 0)]),
                     LineString([(200, 0), (300, 0)])],
    }, crs="EPSG:26910")
    bw = gpd.GeoDataFrame({
        "facility": ["lane", "separated"], "street": ["A", "B"],
        "geometry": [LineString([(-10, 4), (205, 4)]), LineString([(250, -50), (250, 50)])],
    }, crs="EPSG:26910")
    fac = bikeways.conflate(edges, bw)
    assert list(fac) == ["lane", "lane", ""]


def test_conflation_takes_the_most_protected_overlapping_facility():
    edges = gpd.GeoDataFrame({
        "cls": ["primary"], "bike_ok": [True],
        "geometry": [LineString([(0, 0), (100, 0)])],
    }, crs="EPSG:26910")
    bw = gpd.GeoDataFrame({
        "facility": ["route", "separated"], "street": ["A", "A"],
        "geometry": [LineString([(0, 3), (100, 3)]), LineString([(0, -3), (100, -3)])],
    }, crs="EPSG:26910")
    assert list(bikeways.conflate(edges, bw)) == ["separated"]


needs_data = pytest.mark.skipif(not bikeways.BIKEWAYS_GEOJSON.exists(),
                                reason="SFMTA bikeway GeoJSON not downloaded")


@needs_data
def test_the_real_network_has_every_facility_class():
    bw = bikeways.load_bikeways()
    counts = bw["facility"].value_counts()
    assert counts["route"] > 800 and counts["lane"] > 700
    assert counts["neighborway"] > 600
    assert counts["separated"] > 500 and counts["path"] > 200
    assert bw.crs.to_epsg() == 4326
    assert np.all(np.isfinite(bw.to_crs(26910).geometry.length))
