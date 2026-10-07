"""Integration checks against the real processed data.

These are skipped automatically when the pipeline has not been run, so the
unit suite stays runnable on a fresh clone.
"""
import numpy as np
import pandas as pd
import pytest

from sf_flat_routes.config import MIN_RELIABLE_GRADE_LENGTH_M, PROCESSED_DIR

EDGES = PROCESSED_DIR / "edges_metrics.parquet"
DIRECTED = PROCESSED_DIR / "edges_directed.parquet"
PAIRS = PROCESSED_DIR / "neighborhood_pairs.parquet"

pytestmark = pytest.mark.skipif(
    not (EDGES.exists() and DIRECTED.exists()),
    reason="processed data not built; run `python -m sf_flat_routes all`")


@pytest.fixture(scope="module")
def edges():
    import geopandas as gpd
    return gpd.read_parquet(EDGES)


@pytest.fixture(scope="module")
def directed():
    return pd.read_parquet(DIRECTED)


def test_every_edge_has_two_directions(edges, directed):
    assert len(directed) == 2 * len(edges)
    counts = directed.groupby("edge_id").size()
    assert counts.min() == 2 and counts.max() == 2


def test_gain_and_loss_mirror_between_directions(directed):
    piv = directed.pivot_table(index="edge_id", columns="direction",
                               values=["cum_gain", "cum_loss"])
    assert np.allclose(piv[("cum_gain", "fwd")], piv[("cum_loss", "rev")],
                       atol=1e-6)
    assert np.allclose(piv[("cum_loss", "fwd")], piv[("cum_gain", "rev")],
                       atol=1e-6)


def test_net_change_is_antisymmetric(directed):
    piv = directed.pivot_table(index="edge_id", columns="direction",
                               values="net_change")
    assert np.allclose(piv["fwd"], -piv["rev"], atol=1e-6)


def test_gain_minus_loss_equals_net_change(directed):
    """Exact per-edge identity, guaranteed by protecting edge boundaries
    from the dead-band pruning."""
    d = directed
    assert np.allclose(d["cum_gain"] - d["cum_loss"], d["net_change"], atol=1e-3)


def test_no_edge_exceeds_the_plausible_grade_clip(edges):
    from sf_flat_routes.config import ELEVATION
    assert edges["max_abs_grade"].max() <= ELEVATION.max_plausible_grade + 1e-9


def test_elevations_are_in_a_sane_range_for_san_francisco(edges):
    # Mount Davidson is 283 m; nothing should sit far below sea level
    assert edges["elev_max"].max() < 300
    assert edges["elev_min"].min() > -15


def test_distance_above_thresholds_are_nested_and_bounded(directed):
    ths = [3, 5, 8, 10, 15]
    for a, b in zip(ths[:-1], ths[1:]):
        assert (directed[f"d_above_{a}"] >= directed[f"d_above_{b}"] - 1e-6).all()
    # tolerance covers float accumulation: interval lengths are summed,
    # while length_m comes from the geometry
    assert (directed["d_above_3"] <= directed["length_m"] * (1 + 1e-6) + 1e-3).all()


def test_stairways_are_never_bicycle_traversable(directed):
    steps = directed[directed["cls"] == "steps"]
    assert len(steps) > 0
    assert not steps["bike_traversable"].any()


def test_known_flat_and_steep_streets_are_correctly_separated(edges):
    def gain_per_km(name):
        sub = edges[edges["name"] == name]
        km = sub["length_m"].sum() / 1000
        return sub["cum_gain_fwd"].sum() / km if km else np.nan

    flat = gain_per_km("Alki Avenue Southwest")
    steep = gain_per_km("East Roy Street")
    assert flat < 3.0, f"Alki Avenue should be level, got {flat:.1f} m/km"
    assert steep > 20.0, f"East Roy Street should be steep, got {steep:.1f} m/km"
    assert steep > 8 * flat


def test_queen_anne_counterbalance_is_steep_but_plausible(edges):
    """Queen Anne Avenue North up the Counterbalance is Seattle's best-known
    steep arterial. Not checked against a documented figure yet: this only
    guards against the hill vanishing (a flattened DEM) or turning into a
    cliff (a bad structure or water sample)."""
    sub = edges[(edges["name"] == "Queen Anne Avenue North")
                & (edges["length_m"] >= MIN_RELIABLE_GRADE_LENGTH_M)]
    assert 0.15 < sub["max_abs_grade"].max() < 0.30


@pytest.mark.skipif(not PAIRS.exists(), reason="pair analysis not run")
def test_no_objective_beats_the_shortest_path_on_distance():
    p = pd.read_parquet(PAIRS)
    assert (p["detour_ratio"] >= 0.999).all()


@pytest.mark.skipif(not PAIRS.exists(), reason="pair analysis not run")
def test_climb_averse_objectives_actually_reduce_climbing():
    p = pd.read_parquet(PAIRS)
    means = p.groupby("profile")["elev_gain_m"].mean()
    assert means["min_climb"] < means["shortest"]
    assert means["balanced"] < means["shortest"]


@pytest.mark.skipif(not PAIRS.exists(), reason="pair analysis not run")
def test_grade_averse_lowers_maximum_gradient_most(p_=None):
    p = pd.read_parquet(PAIRS)
    means = p.groupby("profile")["max_grade"].mean()
    assert means["grade_averse"] < means["min_climb"]
    assert means["grade_averse"] < means["shortest"]


@pytest.mark.skipif(not PAIRS.exists(), reason="pair analysis not run")
def test_route_metrics_are_internally_consistent():
    p = pd.read_parquet(PAIRS)
    assert (p["elev_gain_m"] >= 0).all() and (p["elev_loss_m"] >= 0).all()
    net = p["elev_gain_m"] - p["elev_loss_m"]
    endpoint_net = p["end_elev_m"] - p["start_elev_m"]
    assert np.allclose(net, endpoint_net, atol=0.05)
