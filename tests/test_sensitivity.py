"""Tests for the parameter-override plumbing and the sensitivity harness."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _config_in_subprocess(env_extra: dict) -> dict:
    """Import the config in a fresh interpreter under the given environment."""
    code = (
        "import json; from flatten_seattle import config as c, elevation as e\n"
        "print(json.dumps({'spacing': c.ELEVATION.sample_spacing_m,"
        " 'sigma': c.ELEVATION.dem_sigma_m, 'sigma_mod': e.DEM_SMOOTH_SIGMA_M,"
        " 'rank': c.ANALYSIS.point_rank, 'processed': str(c.PROCESSED_DIR),"
        " 'outputs': str(c.OUTPUT_DIR), 'raw': str(c.RAW_DIR)}))"
    )
    env = dict(os.environ)
    env.pop("SFFR_OVERRIDES", None); env.pop("SFFR_RUN_DIR", None)
    env.update(env_extra)
    env["PYTHONPATH"] = str(ROOT)
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT,
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_defaults_without_overrides():
    c = _config_in_subprocess({})
    assert c["spacing"] == 5.0 and c["sigma"] == 3.0 and c["rank"] == 0
    assert c["sigma_mod"] == c["sigma"]
    assert c["processed"].endswith("data/processed")


def test_overrides_reach_every_module():
    c = _config_in_subprocess({"SFFR_OVERRIDES": json.dumps(
        {"sample_spacing_m": 10, "dem_sigma_m": 0, "point_rank": 2})})
    assert c["spacing"] == 10 and c["rank"] == 2
    # the elevation module reads sigma from the config, not a local constant
    assert c["sigma"] == 0 and c["sigma_mod"] == 0


def test_unknown_override_keys_are_ignored():
    c = _config_in_subprocess({"SFFR_OVERRIDES": json.dumps({"nonsense": 1})})
    assert c["spacing"] == 5.0


def test_malformed_overrides_fail_loudly():
    with pytest.raises(subprocess.CalledProcessError):
        _config_in_subprocess({"SFFR_OVERRIDES": "{not json"})


def test_run_dir_redirects_processed_and_outputs_but_not_raw(tmp_path):
    c = _config_in_subprocess({"SFFR_RUN_DIR": str(tmp_path)})
    assert c["processed"] == str(tmp_path / "processed")
    assert c["outputs"] == str(tmp_path / "outputs")
    # raw data is shared: an experiment must never re-download 725 MB
    assert c["raw"].endswith("data/raw")


# ------------------------------------------------------------- the writer
def _fake_row(tag, **over):
    r = {
        "tag": tag, "params": {},
        "shortest_detour_pct": 0.0, "shortest_gain_saved_pct": 0.0,
        "shortest_gain_ft": 480.0, "shortest_max_grade_pct": 27.0,
        "shortest_dist_mi": 3.9,
        "min_climb_detour_pct": 14.0, "min_climb_gain_saved_pct": 39.0,
        "min_climb_gain_ft": 280.0, "min_climb_max_grade_pct": 17.0,
        "min_climb_dist_mi": 4.5,
        "grade_averse_detour_pct": 48.0, "grade_averse_gain_saved_pct": 30.0,
        "grade_averse_gain_ft": 325.0, "grade_averse_max_grade_pct": 10.8,
        "grade_averse_dist_mi": 5.6,
        "balanced_detour_pct": 19.0, "balanced_gain_saved_pct": 33.0,
        "balanced_gain_ft": 309.0, "balanced_max_grade_pct": 12.8,
        "balanced_dist_mi": 4.6,
        "network_gain_per_km": 21.0,
        "grade_Filbert Street": 32.9, "grade_Jones Street": 31.1,
        "grade_22nd Street": 32.6, "grade_Bradford Street": 33.1,
        "gainkm_The Embarcadero": 1.1, "gainkm_Valencia Street": 6.0,
        "gainkm_Market Street": 14.0,
        "top_corridors": ["Valencia Street - X", "JFK - Y", "Mission Street"],
        "corridor_count": 50, "top_corridor_km": 6.9,
        "top_pass_ft": 255.0, "top_pass_pairs": 119, "top_pass_nbhd": "Golden Gate Park",
        "n_passes": 35, "wiggle_excess_flat_m": 5.9,
        "wiggle_excess_shortest_m": 19.2, "wiggle_discovered": True,
        "dem_rms_m": 0.68,
    }
    r.update(over)
    return r


def test_writer_produces_csv_and_markdown(tmp_path, monkeypatch):
    from flatten_seattle import sensitivity as S
    monkeypatch.setattr(S, "SENS_CSV", tmp_path / "s.csv")
    monkeypatch.setattr(S, "SENS_MD", tmp_path / "s.md")
    monkeypatch.setattr(S, "RUNS_DIR", tmp_path / "no-runs")   # no run dirs
    df = pd.DataFrame([
        _fake_row("baseline"),
        _fake_row("spacing_10m", min_climb_gain_saved_pct=36.0,
                  top_corridors=["Valencia Street - X", "Other", "Mission Street"]),
    ]).set_index("tag")
    S._write(df)
    md = (tmp_path / "s.md").read_text()
    assert "| baseline |" in md and "| spacing_10m |" in md
    assert "coarser DEM sampling" in md          # the reason column
    assert "| 2/12 |" in md                      # two lead streets kept
    assert "Other" in md                         # the street that entered
    csv = pd.read_csv(tmp_path / "s.csv", index_col="tag")
    assert "top_corridors" not in csv.columns    # list column dropped from CSV
    assert csv.loc["spacing_10m", "min_climb_gain_saved_pct"] == 36.0
    assert csv.loc["spacing_10m", "lead_streets_shared"] == 2
    assert pd.isna(csv.loc["spacing_10m", "edge_overlap_pct"])  # no run dirs


def test_jaccard():
    from flatten_seattle.sensitivity import _jaccard
    assert _jaccard(["a", "b"], ["a", "b"]) == 1.0
    assert _jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert _jaccard([], []) == 1.0


def test_edge_overlap_is_length_weighted():
    from flatten_seattle.sensitivity import _edge_overlap, _lead_streets
    a = {1: 100.0, 2: 300.0}
    b = {2: 300.0, 3: 100.0}
    # shared 300 of a 500 m union
    assert _edge_overlap(a, b) == pytest.approx(60.0)
    assert _edge_overlap(a, a) == 100.0
    assert _edge_overlap({}, {}) == 100.0
    import math
    assert math.isnan(_edge_overlap(None, b))
    assert _lead_streets(["Valencia Street - 16th Street", "Mission Street", ""]) == {
        "Valencia Street", "Mission Street"}


def test_grid_is_one_at_a_time_around_the_baseline():
    from flatten_seattle.sensitivity import BASELINE, GRID
    tags = [t for t, _, _ in GRID]
    assert tags[0] == "baseline" and len(set(tags)) == len(tags)
    for tag, over, why in GRID[1:]:
        assert len(over) == 1, f"{tag} perturbs more than one parameter"
        (k, v), = over.items()
        assert k in BASELINE and v != BASELINE[k]
        assert why
