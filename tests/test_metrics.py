"""Tests for the elevation metrics: grade, cumulative gain, percentiles."""
import numpy as np
import pytest

from flatten_seattle.config import ELEVATION, GRADE_THRESHOLDS
from flatten_seattle.metrics import (cumulative_gain_loss, deadband_filter,
                                    directional_metrics, distance_above,
                                    interval_grades, prune_reversal_indices,
                                    rectify_profile, weighted_percentile)

DB = ELEVATION.gain_deadband_m


# ---------------------------------------------------------------- grades
def test_grade_of_constant_slope():
    dist = np.arange(0, 101, 5.0)
    elev = dist * 0.08                       # a clean 8% ramp
    g, L = interval_grades(dist, elev)
    assert np.allclose(g, 0.08)
    assert L.sum() == pytest.approx(100.0)


def test_grade_sign_follows_direction_of_travel():
    dist = np.arange(0, 51, 10.0)               # 10 m stations
    elev = np.arange(0, 6) * 1.0                # 1 m per station => 10%
    g, _ = interval_grades(dist, elev)
    assert np.allclose(g, 0.10)
    g_rev, _ = interval_grades(dist, elev[::-1])
    assert np.allclose(g_rev, -0.10)


def test_grade_is_clipped_to_plausible_maximum():
    # a 1 m DEM artefact across a 0.5 m step would read as 200%
    dist = np.array([0.0, 0.5])
    elev = np.array([0.0, 1.0])
    g, _ = interval_grades(dist, elev)
    assert abs(g[0]) == pytest.approx(ELEVATION.max_plausible_grade)


def test_zero_length_intervals_do_not_divide_by_zero():
    dist = np.array([0.0, 0.0, 10.0])
    elev = np.array([5.0, 5.0, 6.0])
    g, L = interval_grades(dist, elev)
    assert np.all(np.isfinite(g))
    assert g[0] == 0.0


# ------------------------------------------------- cumulative gain / loss
def test_clean_climb_is_reported_in_full():
    dist = np.arange(0, 201, 5.0)
    elev = np.linspace(0, 100, dist.size)
    gain, loss = cumulative_gain_loss(elev, DB, dist)
    assert gain == pytest.approx(100.0, abs=1e-6)
    assert loss == pytest.approx(0.0, abs=1e-6)


def test_gain_is_additive_across_a_partition():
    """The property routing depends on: per-edge gains must sum correctly."""
    dist = np.arange(0, 401, 5.0)
    elev = np.linspace(0, 150, dist.size)
    whole, _ = cumulative_gain_loss(elev, DB, dist)
    rect = rectify_profile(dist, elev, DB)
    pieces = sum(max(0.0, rect[i + 1] - rect[i]) for i in range(rect.size - 1))
    assert pieces == pytest.approx(whole, abs=1e-6)


def test_gain_of_split_climb_equals_gain_of_whole_climb():
    """A climb cut into many edges must not lose height at each boundary.

    This is the regression test for the original backlash implementation,
    which charged one dead-band per edge and lost 17 m over Twin Peaks.
    """
    dist = np.arange(0, 501, 5.0)
    elev = np.linspace(0, 200, dist.size)
    rect = rectify_profile(dist, elev, DB)
    total = 0.0
    for k in range(0, rect.size - 1, 4):            # 4-sample "edges"
        seg = rect[k:k + 5]
        d = np.diff(seg)
        total += d[d > 0].sum()
    assert total == pytest.approx(200.0, abs=1e-6)


def test_small_oscillations_are_removed():
    z = np.array([10.0, 10.2, 9.9, 10.1, 10.0, 10.2, 9.95, 10.0])
    gain, loss = cumulative_gain_loss(z, DB)
    assert gain == pytest.approx(0.0, abs=1e-9)
    assert loss == pytest.approx(0.0, abs=1e-9)


def test_oscillation_larger_than_deadband_survives():
    z = np.array([0.0, 3.0, 0.0])
    gain, loss = cumulative_gain_loss(z, DB)
    assert gain == pytest.approx(3.0)
    assert loss == pytest.approx(3.0)


def test_net_identity_holds_exactly():
    rng = np.random.default_rng(7)
    dist = np.arange(0, 301, 5.0)
    z = np.linspace(0, 40, dist.size) + rng.normal(0, 0.3, dist.size)
    gain, loss = cumulative_gain_loss(z, DB, dist)
    assert gain - loss == pytest.approx(z[-1] - z[0], abs=1e-6)


def test_deadband_zero_returns_raw_total_variation():
    z = np.array([0.0, 1.0, 0.5, 2.0])
    gain, loss = cumulative_gain_loss(z, 0.0)
    assert gain == pytest.approx(1.0 + 1.5)
    assert loss == pytest.approx(0.5)


def test_endpoints_are_never_pruned():
    z = np.array([0.0, 0.1, 0.05, 0.12, 5.0])
    idx = prune_reversal_indices(z, DB)
    assert idx[0] == 0 and idx[-1] == z.size - 1


def test_backlash_filter_is_monotone_and_bounded():
    z = np.array([0.0, 0.2, 0.4, 0.3, 5.0])
    zf = deadband_filter(z, DB)
    assert zf[0] == z[0]
    assert np.all(np.abs(zf - z) <= DB + 1e-9)


# ----------------------------------------------------- distance above grade
def test_distance_above_threshold_is_directional():
    dist = np.array([0.0, 100.0, 200.0])
    elev = np.array([0.0, 10.0, 10.0])       # 10% then flat
    g, L = interval_grades(dist, elev)
    assert distance_above(g, L, 0.05) == pytest.approx(100.0)
    # travelling the other way it is a descent, so nothing is *climbed*
    g_rev, L_rev = interval_grades(dist, elev[::-1])
    assert distance_above(g_rev, L_rev, 0.05) == pytest.approx(0.0)
    # in absolute terms it is steep either way
    assert distance_above(g_rev, L_rev, 0.05, absolute=True) == pytest.approx(100.0)


def test_distance_above_thresholds_are_nested():
    dist = np.arange(0, 301, 100.0)
    elev = np.array([0.0, 4.0, 13.0, 33.0])   # 4%, 9%, 20%
    g, L = interval_grades(dist, elev)
    vals = [distance_above(g, L, t) for t in GRADE_THRESHOLDS]
    assert vals == sorted(vals, reverse=True)


# --------------------------------------------------------- percentiles
def test_weighted_percentile_respects_weights():
    # a short very steep piece must not dominate a long gentle one
    values = np.array([0.02, 0.30])
    weights = np.array([990.0, 10.0])
    assert weighted_percentile(values, weights, 95) == pytest.approx(0.02)
    assert weighted_percentile(values, weights, 99.9) == pytest.approx(0.30)


def test_weighted_percentile_handles_empty_input():
    assert weighted_percentile(np.array([]), np.array([]), 95) == 0.0


# ------------------------------------------------- directional edge metrics
def test_forward_gain_equals_reverse_loss_exactly():
    rng = np.random.default_rng(3)
    dist = np.arange(0, 201, 5.0)
    elev = np.linspace(0, 60, dist.size) + rng.normal(0, 0.4, dist.size)
    m = directional_metrics(dist, elev)
    assert m["fwd"]["cum_gain"] == pytest.approx(m["rev"]["cum_loss"], abs=1e-12)
    assert m["fwd"]["cum_loss"] == pytest.approx(m["rev"]["cum_gain"], abs=1e-12)


def test_directional_average_grade_flips_sign():
    dist = np.arange(0, 101, 5.0)
    elev = dist * 0.05
    m = directional_metrics(dist, elev)
    assert m["fwd"]["avg_grade"] == pytest.approx(0.05)
    assert m["rev"]["avg_grade"] == pytest.approx(-0.05)
    assert m["fwd"]["start_elev"] == pytest.approx(m["rev"]["end_elev"])


def test_max_grade_is_the_steepest_climb_in_that_direction():
    dist = np.array([0.0, 100.0, 200.0])
    elev = np.array([0.0, 12.0, 8.0])         # climb 12%, descend 4%
    m = directional_metrics(dist, elev)
    assert m["fwd"]["max_grade"] == pytest.approx(0.12)
    assert m["rev"]["max_grade"] == pytest.approx(0.04)


def test_net_change_ignores_the_shape_of_the_route():
    """A route that climbs 50 m and descends 50 m has zero net change.

    This is the analytical point of the whole project, so it is asserted.
    """
    dist = np.arange(0, 401, 5.0)
    up = np.linspace(0, 50, dist.size // 2 + 1)
    down = np.linspace(50, 0, dist.size - up.size)
    elev = np.concatenate([up, down])
    m = directional_metrics(dist, elev)
    assert m["fwd"]["net_change"] == pytest.approx(0.0, abs=1e-6)
    assert m["fwd"]["cum_gain"] == pytest.approx(50.0, abs=0.01)
    assert m["fwd"]["cum_loss"] == pytest.approx(50.0, abs=0.01)


def test_thresholds_present_for_both_directions():
    dist = np.arange(0, 101, 10.0)
    m = directional_metrics(dist, dist * 0.09)
    for d in ("fwd", "rev"):
        for t in (3, 5, 8, 10, 15):
            assert f"d_above_{t}" in m[d]
