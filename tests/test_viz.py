"""Tests for map-output helpers."""
import json

import numpy as np
import pytest

from flatten_seattle.viz_interactive import _round_geometry
from flatten_seattle.viz_static import hillshade

# Polyline encoding now lives in the browser payload packer; its tests are in
# tests/test_webgraph.py.


def test_round_geometry_shortens_coordinates():
    geo = {"type": "LineString",
           "coordinates": [(-122.419421234567, 37.774931234567),
                           (-122.425001234567, 37.778121234567)]}
    r = _round_geometry(geo, 5)
    assert r["coordinates"][0] == [-122.41942, 37.77493]
    assert len(json.dumps(r)) < len(json.dumps(geo))


def test_round_geometry_handles_multilinestring():
    geo = {"type": "MultiLineString",
           "coordinates": [[(-122.1234567, 37.1234567), (-122.2, 37.2)],
                           [(-122.3, 37.3)]]}
    r = _round_geometry(geo, 4)
    assert r["coordinates"][0][0] == [-122.1235, 37.1235]
    assert r["type"] == "MultiLineString"


def test_round_geometry_keeps_only_two_dimensions():
    geo = {"type": "LineString", "coordinates": [(-122.1, 37.1, 55.0)]}
    assert _round_geometry(geo, 5)["coordinates"][0] == [-122.1, 37.1]


def test_hillshade_is_bounded_and_flat_ground_is_uniform():
    flat = np.zeros((20, 20))
    hs = hillshade(flat, res=1.0)
    assert hs.shape == flat.shape
    assert np.all((hs >= 0) & (hs <= 1))
    assert np.ptp(hs) == pytest.approx(0.0, abs=1e-9)


def test_hillshade_distinguishes_slope_direction():
    ramp = np.tile(np.arange(20.0), (20, 1))          # rises to the east
    hs = hillshade(ramp, res=1.0)
    other = hillshade(-ramp, res=1.0)
    assert not np.allclose(hs, other)
