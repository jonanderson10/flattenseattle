"""Tests for access-rule interpretation and graph construction."""
import numpy as np

from sf_flat_routes.network import _as_dict, _as_list, _flags, evaluate_access


def rule(access_type, mode=None, heading=None, during=None, between=None,
         recognized=None, using=None):
    return {"access_type": access_type, "between": between,
            "when": {"mode": mode, "heading": heading, "during": during,
                     "recognized": recognized, "using": using}}


def test_no_restrictions_means_allowed():
    r = evaluate_access(None, "foot")
    assert r["allowed"] and not r["oneway_forward_only"]


def test_one_way_is_a_backward_denial_for_all_modes():
    res = evaluate_access([rule("denied", heading="backward")], "bicycle")
    assert res["allowed"] and res["oneway_forward_only"]


def test_contraflow_bike_lane_cancels_the_one_way():
    rules = [rule("denied", heading="backward"),
             rule("allowed", mode=["bicycle"], heading="backward")]
    assert not evaluate_access(rules, "bicycle")["oneway_forward_only"]
    # ... but motor-vehicle-style one-way still stands for other modes
    assert evaluate_access([rule("denied", heading="backward")],
                           "foot")["oneway_forward_only"]


def test_mode_specific_denial_only_affects_that_mode():
    rules = [rule("denied", mode=["bicycle"])]
    assert not evaluate_access(rules, "bicycle")["allowed"]
    assert evaluate_access(rules, "foot")["allowed"]


def test_designated_counts_as_permitted():
    assert evaluate_access([rule("designated", mode=["bicycle"])],
                           "bicycle")["allowed"]


def test_private_access_is_not_routable_for_through_travel():
    r = evaluate_access([rule("allowed", recognized=["as_private"])], "foot")
    assert not r["allowed"] and r["restricted"]


def test_destination_only_access_is_not_routable():
    r = evaluate_access([rule("allowed", using=["at_destination"])], "foot")
    assert not r["allowed"]


def test_time_conditional_rules_are_ignored():
    rules = [rule("denied", mode=["bicycle"], during="Mo-Fr 07:00-09:00")]
    assert evaluate_access(rules, "bicycle")["allowed"]


def test_partial_segment_rules_do_not_veto_the_whole_edge():
    rules = [rule("denied", mode=["foot"], between=[0.0, 0.4])]
    r = evaluate_access(rules, "foot")
    assert r["allowed"] and r["partial_rules"]


def test_later_rule_overrides_an_earlier_general_one():
    rules = [rule("denied"), rule("allowed", mode=["foot"])]
    assert evaluate_access(rules, "foot")["allowed"]


def test_a_mode_specific_permit_outranks_a_later_general_restriction():
    """The Slow Street shape, exactly as Overture carries it for Cabrillo
    Street: an all-modes destination-only rule comes *after* the explicit
    foot and bicycle permits and must not override them."""
    rules = [rule("allowed", mode=["foot"]),
             rule("designated", mode=["bicycle"]),
             rule("allowed", mode=["motor_vehicle"], using=["at_destination"]),
             rule("allowed", using=["at_destination"])]
    for mode in ("foot", "bicycle"):
        r = evaluate_access(rules, mode)
        assert r["allowed"], mode
        assert not r["restricted"], mode


def test_a_general_restriction_still_applies_without_a_mode_permit():
    rules = [rule("allowed", mode=["motor_vehicle"], using=["at_destination"]),
             rule("allowed", using=["at_destination"])]
    assert not evaluate_access(rules, "foot")["allowed"]


def test_a_mode_specific_denial_outranks_an_earlier_general_permit():
    rules = [rule("allowed"), rule("denied", mode=["foot"])]
    assert not evaluate_access(rules, "foot")["allowed"]
    assert evaluate_access(rules, "bicycle")["allowed"]


def test_motor_vehicle_denial_does_not_block_walking_or_cycling():
    rules = [rule("allowed", mode=["foot", "bicycle"]),
             rule("denied", mode=["motor_vehicle"])]
    assert evaluate_access(rules, "foot")["allowed"]
    assert evaluate_access(rules, "bicycle")["allowed"]


# ------------------------------------------- Arrow / numpy coercion helpers
def test_nested_arrow_columns_arrive_as_ndarrays():
    """``to_pandas`` turns Arrow lists into object ndarrays, whose truthiness
    raises; every nested access must go through the coercion helpers."""
    arr = np.array([rule("denied", heading="backward")], dtype=object)
    res = evaluate_access(arr, "bicycle")
    assert res["oneway_forward_only"]


def test_as_list_handles_none_nan_scalars_and_arrays():
    assert _as_list(None) == []
    assert _as_list(float("nan")) == []
    assert _as_list([1, 2]) == [1, 2]
    assert _as_list(np.array([1, 2])) == [1, 2]


def test_as_dict_rejects_non_mappings():
    assert _as_dict(None) == {}
    assert _as_dict(float("nan")) == {}
    assert _as_dict({"a": 1}) == {"a": 1}


def test_flags_collects_values_from_nested_arrays():
    rf = np.array([{"values": np.array(["is_bridge"], dtype=object)},
                   {"values": ["is_link"]}], dtype=object)
    assert _flags(rf) == {"is_bridge", "is_link"}


def test_flags_of_empty_input():
    assert _flags(None) == set()


# ------------------------------------------------- the built graph (if present)
import pytest  # noqa: E402

from sf_flat_routes.config import PROCESSED_DIR  # noqa: E402


@pytest.mark.skipif(not (PROCESSED_DIR / "edges_metrics.parquet").exists(),
                    reason="processed network not built")
def test_residential_streets_are_walkable_and_bikeable():
    """Upstream, the access parser once read SF Slow Streets' all-modes
    destination rule as closing them to walking. Seattle's equivalent
    (Stay Healthy Streets) is not reliably named in OSM, so guard the
    general case: residential streets are essentially all open to both."""
    import pandas as pd
    e = pd.read_parquet(PROCESSED_DIR / "edges_metrics.parquet",
                        columns=["name", "cls", "walk_ok", "bike_ok", "length_m"])
    r = e[e["cls"] == "residential"]
    assert len(r) > 10000
    assert r["walk_ok"].mean() > 0.98, r["walk_ok"].mean()
    assert r["bike_ok"].mean() > 0.98, r["bike_ok"].mean()
