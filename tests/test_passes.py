"""Tests for the minimax (bottleneck) pass analysis."""
import pandas as pd
import pytest

import flatten_seattle.passes as P


def toy_edges():
    """Two low valleys joined by two ridges: a low col and a high one.

    A(0m) -- B(4m) -- C(40m) -- D(6m) -- E(2m)      via the 40 m col
    A      --------- F(95m) --------- E             via the 95 m ridge

    The minimax pass between A and E must be the 40 m col, never the ridge,
    and the responsible edge must be the one that reaches 40 m.
    """
    rows = [
        # (u, v, elev_max, elev_min, name)
        ("A", "B", 4.0, 0.0, "Valley Road"),
        ("B", "C", 40.0, 4.0, "Col Street"),
        ("C", "D", 40.0, 6.0, "Col Street"),
        ("D", "E", 6.0, 2.0, "Far Valley Road"),
        ("A", "F", 95.0, 0.0, "Ridge Drive"),
        ("F", "E", 95.0, 2.0, "Ridge Drive"),
    ]
    recs = []
    for i, (u, v, emax, emin, name) in enumerate(rows):
        recs.append({"edge_id": i, "u": u, "v": v, "elev_max": emax,
                     "elev_min": emin, "name": name, "cls": "residential",
                     "length_m": 100.0, "max_abs_grade": 0.1,
                     "walk_ok": True})
    return pd.DataFrame(recs)


def test_bottleneck_tree_finds_the_low_col_not_the_high_ridge():
    e = toy_edges()
    tree = P.build_bottleneck_tree(e, mode="walk")
    h, eid = P.pass_height(tree, "A", "E")
    assert h == pytest.approx(40.0)
    assert e.loc[e["edge_id"] == eid, "name"].iloc[0] == "Col Street"


def test_pass_height_is_symmetric():
    tree = P.build_bottleneck_tree(toy_edges(), mode="walk")
    assert P.pass_height(tree, "A", "E")[0] == pytest.approx(
        P.pass_height(tree, "E", "A")[0])


def test_pass_height_within_a_valley_is_low():
    tree = P.build_bottleneck_tree(toy_edges(), mode="walk")
    h, _ = P.pass_height(tree, "A", "B")
    assert h == pytest.approx(4.0)


def test_pass_height_is_at_least_the_endpoint_elevations():
    """The crossing can never be lower than where you start or finish."""
    e = toy_edges()
    tree = P.build_bottleneck_tree(e, mode="walk")
    for a, b in [("A", "E"), ("B", "D"), ("A", "F")]:
        h, _ = P.pass_height(tree, a, b)
        za = e[(e["u"] == a) | (e["v"] == a)]["elev_min"].min()
        zb = e[(e["u"] == b) | (e["v"] == b)]["elev_min"].min()
        assert h >= max(za, zb) - 1e-9


def test_unknown_node_returns_none():
    tree = P.build_bottleneck_tree(toy_edges(), mode="walk")
    assert P.pass_height(tree, "A", "nowhere") is None


def test_neighborhood_pass_matrix_covers_every_unordered_pair():
    e = toy_edges()
    tree = P.build_bottleneck_tree(e, mode="walk")
    pts = pd.DataFrame({"neighborhood": ["West", "East", "Middle"],
                        "node": ["A", "E", "C"]})
    m = P.neighborhood_pass_matrix(tree, pts)
    assert len(m) == 3                       # 3 choose 2
    ae = m[(m["neighborhood_a"] == "West") & (m["neighborhood_b"] == "East")]
    assert ae["pass_elev_m"].iloc[0] == pytest.approx(40.0)


def test_disjoint_union_find_merges_correctly():
    dsu = P._DSU(5)
    assert dsu.find(0) == 0
    assert dsu.union(0, 1) is not None
    assert dsu.find(0) == dsu.find(1)
    assert dsu.union(0, 1) is None           # already joined
    dsu.union(2, 3)
    assert dsu.find(2) == dsu.find(3)
    assert dsu.find(0) != dsu.find(2)


def test_node_elevations_are_the_median_of_incident_endpoints():
    directed = pd.DataFrame({
        "from_node": ["A", "A", "B"],
        "to_node": ["B", "B", "A"],
        "start_elev": [10.0, 12.0, 20.0],
        "end_elev": [20.0, 20.0, 11.0],
    })
    z = P.node_elevations(directed)
    assert z["A"] == pytest.approx(11.0)     # median of 10, 12, 11
    assert z["B"] == pytest.approx(20.0)


def test_basins_require_a_minimum_size():
    e = toy_edges()
    z = pd.Series({"A": 0.0, "B": 4.0, "C": 40.0, "D": 6.0, "E": 2.0, "F": 95.0})
    labels, basins = P.find_basins(e, z, threshold=15.0, min_km=10.0)
    assert basins == {}                      # nothing is 10 km long
    labels, basins = P.find_basins(e, z, threshold=15.0, min_km=0.05)
    assert len(basins) == 2                  # A-B and D-E, split by the col
