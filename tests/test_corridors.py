"""Tests for corridor scoring and merging."""
import pandas as pd

from flatten_seattle.config import ANALYSIS
from flatten_seattle.corridors import _canonical_name, _clean_names, score_edges


def edge_table(rows):
    import geopandas as gpd
    from shapely.geometry import LineString
    recs = []
    for i, r in enumerate(rows):
        d = {"edge_id": i, "name": "S", "cls": "residential", "length_m": 100.0,
             "max_abs_grade": 0.02, "avg_grade_fwd": 0.01, "cum_gain_fwd": 1.0,
             "elev_min": 0.0, "elev_max": 2.0, "low_stress": True,
             "bike_facility": "", "grade_reliable": True,
             "geometry": LineString([(i * 100, 0), (i * 100 + 100, 0)])}
        d.update(r)
        recs.append(d)
    return gpd.GeoDataFrame(recs, geometry="geometry", crs="EPSG:26910")


def usage_table(rows):
    recs = []
    for i, r in enumerate(rows):
        d = {"edge_id": i, "mode": "walk", "pair_count": 10,
             "neighborhood_span": 5, "climb_saved_m": 100.0,
             "detour_efficiency": 0.1, "pareto_count": 0}
        d.update(r)
        recs.append(d)
    return pd.DataFrame(recs)


def test_steep_edges_score_zero_however_heavily_used():
    """A busy steep street is a barrier, not a flat corridor."""
    e = edge_table([{"max_abs_grade": 0.02, "avg_grade_fwd": 0.01},
                    {"max_abs_grade": 0.22, "avg_grade_fwd": 0.19}])
    u = usage_table([{"pair_count": 300, "neighborhood_span": 30},
                     {"pair_count": 300, "neighborhood_span": 30}])
    s = score_edges(u, e)
    assert s.loc[s.edge_id == 0, "score"].iloc[0] > 0
    assert s.loc[s.edge_id == 1, "score"].iloc[0] == 0


def test_score_rises_with_usage_and_with_breadth():
    e = edge_table([{}, {}, {}])
    u = usage_table([
        {"pair_count": 5, "neighborhood_span": 3},
        {"pair_count": 50, "neighborhood_span": 3},
        {"pair_count": 50, "neighborhood_span": 30},
    ])
    s = score_edges(u, e).set_index("edge_id")["score"]
    assert s[0] < s[1] < s[2]


def test_a_short_stub_is_judged_on_average_grade_not_its_maximum():
    """A 5 m DEM artefact must not disqualify an otherwise flat block.

    Regression test: a 5 m connector at Market and 5th reported a 41%
    gradient from a 1.3 m artefact, which knocked flat blocks out of
    corridors.
    """
    e = edge_table([{"length_m": 5.0, "max_abs_grade": 0.41,
                     "avg_grade_fwd": 0.01, "grade_reliable": False}])
    s = score_edges(usage_table([{}]), e)
    assert bool(s["flat_enough"].iloc[0])
    assert s["score"].iloc[0] > 0


def test_a_long_steep_edge_is_still_excluded():
    e = edge_table([{"length_m": 90.0, "max_abs_grade": 0.41,
                     "avg_grade_fwd": 0.01, "grade_reliable": True}])
    s = score_edges(usage_table([{}]), e)
    assert not bool(s["flat_enough"].iloc[0])


def test_climb_saved_increases_the_score():
    e = edge_table([{}, {}])
    u = usage_table([{"climb_saved_m": 0.0}, {"climb_saved_m": 5000.0}])
    s = score_edges(u, e).set_index("edge_id")["score"]
    assert s[1] > s[0]


def test_average_grade_limit_comes_from_configuration():
    lim = ANALYSIS.corridor_max_avg_grade
    e = edge_table([{"avg_grade_fwd": lim - 0.001},
                    {"avg_grade_fwd": lim + 0.001}])
    s = score_edges(usage_table([{}, {}]), e).set_index("edge_id")
    assert bool(s.loc[0, "flat_enough"])
    assert not bool(s.loc[1, "flat_enough"])


def test_clean_names_drops_nan_and_blanks():
    assert _clean_names(["A", None, float("nan"), "", "  ", "B"]) == ["A", "B"]


def test_canonical_name_uses_the_most_common_streets():
    names = ["Valencia Street"] * 5 + ["Mission Street"] * 2 + [None]
    label = _canonical_name(names)
    assert label.startswith("Valencia Street")
    assert "Mission Street" in label


def test_canonical_name_of_all_unnamed():
    assert _canonical_name([None, float("nan")]) == "unnamed corridor"
