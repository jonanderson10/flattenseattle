"""Tests for the cost model and shortest-path machinery."""
import pandas as pd
import pytest

from flatten_seattle.config import (GRADE_THRESHOLDS, ROUTING_PROFILES,
                                   CostWeights, profile, with_alpha)
from flatten_seattle.routing import (build_route_graph, edge_costs, route,
                                    summarise_route)

_TH = [int(t * 100) for t in GRADE_THRESHOLDS]


def make_directed(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal directed-edge table with all required columns."""
    base = {
        "length_m": 100.0, "cum_gain": 0.0, "cum_loss": 0.0, "net_change": 0.0,
        "avg_grade": 0.0, "max_grade": 0.0, "p95_grade": 0.0,
        "mean_abs_grade": 0.0, "start_elev": 0.0, "end_elev": 0.0,
        "cls": "residential", "subclass": None, "name": "Test Street",
        "is_structure": False, "walk_ok": True, "walk_oneway": False,
        "bike_ok": True, "bike_oneway": False, "bike_facility": "",
        "low_stress": True, "walk_traversable": True, "bike_traversable": True,
        "direction": "fwd",
    }
    for t in _TH:
        base[f"d_above_{t}"] = 0.0
    out = []
    for i, r in enumerate(rows):
        d = dict(base)
        d.update(r)
        d.setdefault("edge_id", i)
        out.append(d)
    return pd.DataFrame(out)


# ------------------------------------------------------------ cost model
def test_shortest_cost_is_pure_distance():
    """The baseline objective must minimise distance only.

    Regression test: the bicycle comfort multipliers used to apply to the
    shortest objective too, which made the "shortest" bike route longer in
    real metres than the flat route and broke every distance comparison.
    """
    d = make_directed([{"length_m": 250.0, "cls": "trunk", "cum_gain": 40.0,
                        "d_above_8": 120.0}])
    c = edge_costs(d, profile("shortest"), mode="bike")
    assert c[0] == pytest.approx(250.0)
    c_walk = edge_costs(d, profile("shortest"), mode="walk")
    assert c_walk[0] == pytest.approx(250.0)


def test_climbing_weight_prices_gain_in_equivalent_metres():
    d = make_directed([{"length_m": 100.0, "cum_gain": 10.0}])
    w = CostWeights(alpha=8.0, beta=0.0, gamma=0.0,
                    threshold_penalties=(0, 0, 0, 0, 0), extreme_extra=0.0)
    assert edge_costs(d, w, "walk")[0] == pytest.approx(100.0 + 8.0 * 10.0)


def test_grade_penalty_is_cumulative_across_thresholds():
    """A 12% grade pays the 3, 5, 8 and 10% penalties simultaneously."""
    w = CostWeights(alpha=0.0, beta=1.0, gamma=0.0,
                    threshold_penalties=(0.25, 0.75, 2.0, 4.0, 10.0),
                    extreme_extra=0.0)
    gentle = make_directed([{"length_m": 100.0, "d_above_3": 100.0}])
    steep = make_directed([{"length_m": 100.0, "d_above_3": 100.0,
                            "d_above_5": 100.0, "d_above_8": 100.0,
                            "d_above_10": 100.0}])
    c_gentle = edge_costs(gentle, w, "walk")[0]
    c_steep = edge_costs(steep, w, "walk")[0]
    assert c_gentle == pytest.approx(100.0 + 25.0)
    assert c_steep == pytest.approx(100.0 + 100.0 * (0.25 + 0.75 + 2.0 + 4.0))
    # the marginal cost of steepness must rise, not stay linear
    assert (c_steep - 100.0) / (c_gentle - 100.0) > 4.0


def test_extreme_grade_incurs_the_gamma_term():
    w = CostWeights(alpha=0.0, beta=0.0, gamma=2.0,
                    threshold_penalties=(0, 0, 0, 0, 0), extreme_extra=10.0)
    d = make_directed([{"length_m": 50.0, f"d_above_{_TH[-1]}": 50.0}])
    assert edge_costs(d, w, "walk")[0] == pytest.approx(50.0 + 2.0 * 10.0 * 50.0)


def test_class_multiplier_applies_only_to_the_named_class():
    d = make_directed([{"cls": "steps"}, {"cls": "residential"}])
    w = profile("balanced")
    c = edge_costs(d, w, "walk")
    assert c[0] > c[1]                       # stairs cost more on foot
    assert c[1] == pytest.approx(100.0)


def test_costs_are_strictly_positive():
    d = make_directed([{"length_m": 0.0}])
    assert edge_costs(d, profile("balanced"), "walk")[0] > 0


def test_directional_costs_differ_for_the_same_street():
    """Uphill and downhill traversals of one edge must not cost the same."""
    up = make_directed([{"cum_gain": 20.0, "cum_loss": 0.0, "d_above_8": 80.0,
                         "direction": "fwd"}])
    down = make_directed([{"cum_gain": 0.0, "cum_loss": 20.0, "d_above_8": 0.0,
                           "direction": "rev"}])
    w = profile("balanced")
    assert edge_costs(up, w, "walk")[0] > edge_costs(down, w, "walk")[0]


def test_min_climb_prefers_climbing_over_distance_more_than_balanced():
    assert ROUTING_PROFILES["min_climb"].alpha > ROUTING_PROFILES["balanced"].alpha
    assert ROUTING_PROFILES["balanced"].alpha > ROUTING_PROFILES["shortest"].alpha


def test_with_alpha_leaves_other_weights_untouched():
    base = profile("balanced")
    w = with_alpha(base, 42.0)
    assert w.alpha == 42.0
    assert w.beta == base.beta
    assert w.threshold_penalties == base.threshold_penalties


# ------------------------------------------------------------ graph + paths
def _diamond():
    """Two routes from A to D: short+steep via B, long+flat via C."""
    rows = []
    def pair(eid, u, v, length, gain, steep):
        rows.append({"edge_id": eid, "from_node": u, "to_node": v,
                     "length_m": length, "cum_gain": gain, "cum_loss": 0.0,
                     "d_above_8": steep, "max_grade": 0.15 if steep else 0.01,
                     "mean_abs_grade": 0.15 if steep else 0.01,
                     "direction": "fwd"})
        rows.append({"edge_id": eid, "from_node": v, "to_node": u,
                     "length_m": length, "cum_gain": 0.0, "cum_loss": gain,
                     "d_above_8": 0.0, "max_grade": 0.01,
                     "mean_abs_grade": 0.15 if steep else 0.01,
                     "direction": "rev"})
    pair(0, "A", "B", 300.0, 45.0, 300.0)
    pair(1, "B", "D", 300.0, 0.0, 0.0)
    pair(2, "A", "C", 600.0, 3.0, 0.0)
    pair(3, "C", "D", 600.0, 0.0, 0.0)
    return make_directed(rows)


def test_shortest_takes_the_short_steep_route():
    g = build_route_graph(_diamond(), "walk")
    arcs, s = route(g, "A", "D", profile("shortest"))
    assert s["distance_m"] == pytest.approx(600.0)
    assert s["elev_gain_m"] == pytest.approx(45.0)


def test_flat_objective_takes_the_long_gentle_route():
    g = build_route_graph(_diamond(), "walk")
    arcs, s = route(g, "A", "D", profile("min_climb"))
    assert s["distance_m"] == pytest.approx(1200.0)
    assert s["elev_gain_m"] == pytest.approx(3.0)


def test_route_summary_sums_edge_metrics():
    g = build_route_graph(_diamond(), "walk")
    arcs, s = route(g, "A", "D", profile("min_climb"))
    t = g.table.iloc[arcs]
    assert s["distance_m"] == pytest.approx(t["length_m"].sum())
    assert s["elev_gain_m"] == pytest.approx(t["cum_gain"].sum())
    assert s["n_edges"] == len(arcs)


def test_one_way_blocks_bicycles_but_not_pedestrians():
    rows = [
        {"edge_id": 0, "from_node": "A", "to_node": "B", "direction": "fwd",
         "bike_oneway": True, "bike_traversable": True},
        {"edge_id": 0, "from_node": "B", "to_node": "A", "direction": "rev",
         "bike_oneway": True, "bike_traversable": False},
        {"edge_id": 1, "from_node": "B", "to_node": "A", "direction": "fwd"},
        {"edge_id": 1, "from_node": "A", "to_node": "B", "direction": "rev"},
    ]
    d = make_directed(rows)
    gb = build_route_graph(d, "bike")
    assert not any((gb.table["edge_id"] == 0) & (gb.table["direction"] == "rev"))
    gw = build_route_graph(d, "walk")
    assert any((gw.table["edge_id"] == 0) & (gw.table["direction"] == "rev"))


def test_parallel_arcs_take_the_cheaper_one_not_their_sum():
    """Two streets between the same intersections must not be summed.

    A COO->CSR conversion adds duplicate entries, which would invent a more
    expensive street than either real one.
    """
    rows = [
        {"edge_id": 0, "from_node": "A", "to_node": "B", "length_m": 100.0,
         "direction": "fwd"},
        {"edge_id": 1, "from_node": "A", "to_node": "B", "length_m": 400.0,
         "direction": "fwd"},
        {"edge_id": 0, "from_node": "B", "to_node": "A", "length_m": 100.0,
         "direction": "rev"},
        {"edge_id": 1, "from_node": "B", "to_node": "A", "length_m": 400.0,
         "direction": "rev"},
    ]
    g = build_route_graph(make_directed(rows), "walk")
    arcs, s = route(g, "A", "B", profile("shortest"))
    assert s["distance_m"] == pytest.approx(100.0)


def test_unreachable_target_raises():
    rows = [{"edge_id": 0, "from_node": "A", "to_node": "B", "direction": "fwd"},
            {"edge_id": 0, "from_node": "B", "to_node": "A", "direction": "rev"}]
    g = build_route_graph(make_directed(rows), "walk")
    with pytest.raises(KeyError):
        route(g, "A", "ZZZ", profile("shortest"))


def test_empty_route_summary_is_all_zero():
    g = build_route_graph(_diamond(), "walk")
    s = summarise_route(g, [])
    assert s["distance_m"] == 0.0 and s["elev_gain_m"] == 0.0


def test_with_scale_zero_is_pure_distance():
    from flatten_seattle.config import with_scale
    w = with_scale(profile("balanced"), 0.0)
    assert w.alpha == 0 and w.beta == 0 and w.gamma == 0
    assert w.use_class_multiplier is False
    d = make_directed([{"length_m": 250.0, "cls": "steps", "cum_gain": 40.0,
                        "d_above_8": 120.0}])
    assert edge_costs(d, w, "walk")[0] == pytest.approx(250.0)
    # ... and therefore coincides with the shortest objective exactly
    assert edge_costs(d, w, "walk")[0] == edge_costs(d, profile("shortest"), "walk")[0]


def test_with_scale_one_is_the_profile_itself():
    from flatten_seattle.config import with_scale
    base = profile("balanced")
    w = with_scale(base, 1.0)
    assert (w.alpha, w.beta, w.gamma) == (base.alpha, base.beta, base.gamma)
    assert w.use_class_multiplier == base.use_class_multiplier


def test_pareto_sweep_starts_at_zero_and_is_increasing():
    from flatten_seattle.config import PARETO_LAMBDA_SWEEP
    assert PARETO_LAMBDA_SWEEP[0] == 0.0
    assert list(PARETO_LAMBDA_SWEEP) == sorted(PARETO_LAMBDA_SWEEP)
