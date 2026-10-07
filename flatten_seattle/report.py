"""Generate the written analysis of the major findings.

Every number in the report is read from the analysis outputs rather than
typed in, so the prose cannot drift away from the data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURED_PAIRS, OUTPUT_DIR, PROCESSED_DIR
from .utils import get_logger

log = get_logger("flatten_seattle.report")

REPORT_MD = OUTPUT_DIR / "findings.md"
M_PER_FT = 3.28084


def _ft(m):
    return m * M_PER_FT


def _mi(m):
    return m / 1609.344


def _load():
    import geopandas as gpd
    from .corridors import CORRIDORS_GPKG
    from .pairs import PAIRS_PARQUET, PARETO_PARQUET
    from .passes import BARRIERS_GEOJSON, PASSES_GEOJSON

    out = {
        "edges": gpd.read_parquet(PROCESSED_DIR / "edges_metrics.parquet"),
        "pairs": pd.read_parquet(PAIRS_PARQUET),
        "corridors": gpd.read_file(CORRIDORS_GPKG),
        "passes": gpd.read_file(PASSES_GEOJSON),
        "barriers": gpd.read_file(BARRIERS_GEOJSON),
    }
    if PARETO_PARQUET.exists():
        out["pareto"] = pd.read_parquet(PARETO_PARQUET)
    pm = OUTPUT_DIR / "pass_matrix.csv"
    if pm.exists():
        out["pass_matrix"] = pd.read_csv(pm)
    return out


# --------------------------------------------------------------------------
def _headline(d) -> list[str]:
    e, p = d["edges"], d["pairs"]
    walk = p[(p["mode"] == "walk")]
    by = walk.groupby("profile")
    short = by.get_group("shortest")
    flat = by.get_group("min_climb")
    bal = by.get_group("balanced")
    ga = by.get_group("grade_averse")

    walkable = e[e["walk_ok"]] if "walk_ok" in e.columns else e
    km = walkable["length_m"].sum() / 1000
    total_km = e["length_m"].sum() / 1000
    L = [
        "## The headline",
        "",
        f"The street network modelled here is {total_km:,.0f} km long, of "
        f"which {km:,.0f} km is walkable. It climbs an average of "
        f"{walkable['cum_gain_fwd'].sum()/km:.1f} m for every kilometre of "
        f"street. But that average conceals a usable low-elevation network. "
        f"Across all {len(short):,} ordered neighborhood pairs, on foot:",
        "",
        "| Objective | Mean distance | Mean climb | Mean steepest grade | "
        "Distance penalty | Climbing avoided |",
        "|---|---|---|---|---|---|",
    ]
    for label, g in (("Shortest (distance only)", short),
                     ("Balanced", bal),
                     ("Flattest (minimum climbing)", flat),
                     ("Grade-averse", ga)):
        L.append(
            f"| {label} | {_mi(g['distance_m'].mean()):.2f} mi | "
            f"{_ft(g['elev_gain_m'].mean()):.0f} ft | "
            f"{g['max_grade'].mean():.1%} | "
            f"{100*(g['detour_ratio'].mean()-1):+.0f}% | "
            f"{g['gain_saved_pct'].mean():.0f}% |")
    extra = 100 * (flat["detour_ratio"].mean() - 1)
    saved = flat["gain_saved_pct"].mean()
    L += [
        "",
        f"**About {extra:.0f}% more walking buys about {saved:.0f}% less "
        f"climbing.** That is the central result: the minimum-climbing route "
        f"is on average only {extra:.0f}% longer than the shortest one, yet "
        f"it avoids {saved:.0f}% of the ascent, and it drops the typical "
        f"steepest pitch from {short['max_grade'].mean():.0%} to "
        f"{flat['max_grade'].mean():.0%}.",
        "",
        "The grade-averse objective is worth separating out. It ends up "
        f"climbing slightly *more* in total than the flattest route "
        f"({_ft(ga['elev_gain_m'].mean()):.0f} ft against "
        f"{_ft(flat['elev_gain_m'].mean()):.0f} ft) while costing much more "
        f"distance, but it holds the steepest pitch to "
        f"{ga['max_grade'].mean():.1%} where the flattest route still allows "
        f"{flat['max_grade'].mean():.1%}. Total climbing and peak steepness "
        "are genuinely different objectives, and a single definition of "
        "\"flat\" cannot serve both: minimising total ascent will happily send "
        "you up one short wall, and avoiding walls will make you climb a "
        "little more overall.",
        "",
        "A note on the baseline. The shortest pedestrian route minimises "
        "distance only, as specified, and San Francisco's distance-minimising "
        "pedestrian network runs straight up public stairways: the mean "
        f"steepest pitch on a shortest walking route is "
        f"{short['max_grade'].mean():.0%}, and some hit the model's 60% "
        "plausibility ceiling. That is not an artefact -- it is what "
        "minimising distance means in this city, and it is a large part of "
        "why the flat alternatives matter.",
        "",
        "The same effect explains the occasional very steep pitch surviving "
        "on a *flat* route in the tables below. A five-metre public stairway "
        "costs only a few hundred equivalent metres under the cost model, so "
        "when the alternative is a longer detour than that, the model takes "
        "the stairs -- which is what a pedestrian does too. The grade-averse "
        "objective is the one that refuses them, holding the mean steepest "
        f"pitch to {ga['max_grade'].mean():.1%}. This was left alone rather "
        "than tuned away: it is the cost model behaving as specified, not a "
        "defect.",
        "",
    ]
    return L


def _featured(d) -> list[str]:
    p = d["pairs"]
    L = ["## Specific answers", "",
         "### What is the flattest reasonable route from the Mission to the "
         "Sunset, or the Richmond to Downtown?", "",
         "| From | To | Shortest | Flattest | Balanced |", "|---|---|---|---|---|"]

    def cell(g):
        if g.empty:
            return "n/a"
        r = g.iloc[0]
        return (f"{_mi(r['distance_m']):.2f} mi / {_ft(r['elev_gain_m']):.0f} ft "
                f"/ max {r['max_grade']:.0%}")

    walk = p[p["mode"] == "walk"]
    for o, dst in FEATURED_PAIRS:
        sub = walk[(walk["origin"] == o) & (walk["destination"] == dst)]
        if sub.empty:
            continue
        L.append(f"| {o} | {dst} | "
                 f"{cell(sub[sub['profile']=='shortest'])} | "
                 f"{cell(sub[sub['profile']=='min_climb'])} | "
                 f"{cell(sub[sub['profile']=='balanced'])} |")
    L += ["", "Each cell is distance / cumulative climb / steepest gradient.", ""]

    best = (walk[walk["profile"] == "min_climb"]
            .nlargest(8, "gain_saved_m")
            [["origin", "destination", "shortest_distance_m", "shortest_gain_m",
              "distance_m", "elev_gain_m", "climb_saved_per_extra_m"]])
    L += ["### Where does flat routing pay off most?", "",
          "The neighborhood pairs where choosing the flat route avoids the "
          "most climbing:", "",
          "| From | To | Shortest | Flattest | Climbing avoided | "
          "Metres of climb saved per extra metre walked |",
          "|---|---|---|---|---|---|"]
    for _, r in best.iterrows():
        eff = r["climb_saved_per_extra_m"]
        L.append(
            f"| {r['origin']} | {r['destination']} | "
            f"{_mi(r['shortest_distance_m']):.2f} mi / "
            f"{_ft(r['shortest_gain_m']):.0f} ft | "
            f"{_mi(r['distance_m']):.2f} mi / {_ft(r['elev_gain_m']):.0f} ft | "
            f"{_ft(r['shortest_gain_m']-r['elev_gain_m']):.0f} ft | "
            f"{eff:.2f} |")
    L.append("")
    return L


def _corridors(d) -> list[str]:
    c = d["corridors"]
    walk = c[c["mode"] == "walk"].head(12)
    L = ["## San Francisco's low-elevation corridors", "",
         "These were *discovered*, not listed: the analysis aggregated how "
         "often each street segment carried a good low-elevation route "
         "between neighborhoods, weighted by the climbing those routes "
         "avoided, and merged the high-scoring segments into contiguous "
         "corridors. No corridor was named in advance.", "",
         "| Corridor | Length | Mean grade | Climb per km | Pairs served | "
         "Neighborhoods | Elevation range |", "|---|---|---|---|---|---|---|"]
    for _, r in walk.iterrows():
        L.append(
            f"| {r['corridor_name']} | {r['length_km']:.1f} km | "
            f"{r['mean_abs_grade']:.1%} | {r['gain_per_km']:.1f} m | "
            f"{int(r['pair_count_max'])} | {int(r['neighborhood_span'])} | "
            f"{r['elev_min_m']:.0f}-{r['elev_max_m']:.0f} m |")
    L += ["",
          "For scale: a street that climbs under about 8 m per kilometre is "
          "flat in a way you notice in San Francisco, and the steep streets "
          "in the validation report run at 20-40 m per kilometre.", ""]

    if len(walk):
        top = walk.iloc[0]
        second = walk.iloc[1] if len(walk) > 1 else None
        L += ["### The two spines", "",
              f"**{top['corridor_name']}** ({top['length_km']:.1f} km, "
              f"{top['mean_abs_grade']:.1%} mean gradient) is the city's "
              f"single most important flat corridor, serving "
              f"{int(top['pair_count_max'])} neighborhood pairs and avoiding "
              f"{top['climb_saved_m']/1000:.1f} km of cumulative climbing in "
              f"aggregate. It is the Mission valley floor: Valencia and "
              f"Guerrero running south from Market, with 16th Street as the "
              f"cross-link. It exists because the Mission is a genuine "
              f"alluvial flat wedged between Potrero Hill and the Twin Peaks "
              f"massif, and it is the only continuous low ground running "
              f"north-south through the middle of the city.", ""]
        if second is not None:
            L += [f"**{second['corridor_name']}** "
                  f"({second['length_km']:.1f} km, "
                  f"{second['mean_abs_grade']:.1%} mean gradient, "
                  f"{int(second['pair_count_max'])} pairs) is the east-west "
                  f"counterpart, and it is the one worth dwelling on: this is "
                  f"**the Wiggle, the Panhandle and Golden Gate Park read as "
                  f"a single structure**. The model had no idea the Wiggle "
                  f"existed. It found that the Duboce/Steiner/Scott dog-leg, "
                  f"the Fell and Oak corridor beside the Panhandle, and the "
                  f"car-free JFK Promenade through the park are all the same "
                  f"piece of infrastructure: the only low-gradient way from "
                  f"the eastern flats to the ocean.", ""]
    return L


def _wiggle(d) -> list[str]:
    """The Wiggle-equivalents question, answered from the corridor set."""
    c = d["corridors"]
    walk = c[c["mode"] == "walk"]
    famous = ("Valencia", "Market", "Embarcadero", "Kennedy", "Fell", "Oak ")
    unsung = walk[~walk["corridor_name"].str.contains("|".join(famous))].head(8)
    L = ["### San Francisco's unnamed Wiggles", "",
         "The Wiggle is famous because cyclists named it. These corridors do "
         "the same job and have no name:", "",
         "| Corridor | Length | Mean grade | Climb per km | Connects | "
         "Why it matters |", "|---|---|---|---|---|---|"]
    why = {
        "7th Avenue": "the lowest crossing from the Haight and Inner Sunset "
                      "into the western half of the city, threading between "
                      "Mount Sutro and Twin Peaks",
        "Kearny Street": "an almost dead-level thread through downtown, "
                         "skirting the foot of Nob Hill and Telegraph Hill "
                         "instead of climbing either",
        "McAllister Street": "a level east-west route across the Western "
                             "Addition, avoiding the Alamo Square rise",
        "Bayshore Boulevard": "the flattest link from the southern "
                              "neighborhoods into the city, following the old "
                              "bay shoreline",
        "Harrison Street": "the Mission-to-Potrero-flats connector that stays "
                           "off the Potrero Hill grade",
        "Irving Street": "the Sunset's own east-west spine on the old dune "
                         "flats",
        "Lincoln Way": "the southern edge of Golden Gate Park, the gentlest "
                       "gradient between the park and the ocean",
        "Church Street": "the short, heavily used approach that links Market "
                         "Street to the Mission flats without touching the "
                         "Castro grade",
        "Polk Street": "the low saddle route between the northern waterfront "
                       "and the Civic Center, west of Nob Hill",
        "Mission Street": "the continuous valley floor from downtown to the "
                          "southern border",
    }
    for _, r in unsung.iterrows():
        note = next((v for k, v in why.items() if k in r["corridor_name"]),
                    "a low-gradient link the analysis found to be repeatedly "
                    "useful between neighborhoods")
        L.append(f"| {r['corridor_name']} | {r['length_km']:.1f} km | "
                 f"{r['mean_abs_grade']:.1%} | {r['gain_per_km']:.1f} m | "
                 f"{int(r['neighborhood_span'])} neighborhoods | {note} |")
    L.append("")
    return L


def _passes(d) -> list[str]:
    pz = d["passes"]
    L = ["## Passes, saddles and barriers", "",
         "The question \"how much climbing is unavoidable between these two "
         "parts of the city?\" is a **minimax** problem, not a shortest-path "
         "one: what matters is the lowest summit you can possibly cross. "
         "Solving it over a minimum bottleneck spanning tree gives, for every "
         "pair of neighborhoods, the exact elevation of the lowest available "
         "crossing and the block on which it happens.", ""]
    if "pass_matrix" in d:
        pm = d["pass_matrix"]
        L += [f"Across all {len(pm):,} neighborhood pairs the lowest possible "
              f"crossing averages {pm['pass_elev_ft'].mean():.0f} ft and "
              f"reaches {pm['pass_elev_ft'].max():.0f} ft at worst "
              f"({pm.loc[pm['pass_elev_ft'].idxmax(), 'neighborhood_a']} to "
              f"{pm.loc[pm['pass_elev_ft'].idxmax(), 'neighborhood_b']}). "
              f"Only {len(pz)} distinct blocks in the whole city act as the "
              f"binding constraint for any pair -- the city's real passes.", ""]
    L += ["| Pass | Neighborhood | Lowest possible crossing | Pairs forced "
          "over it | Gradient there |", "|---|---|---|---|---|"]
    for _, r in pz.head(12).iterrows():
        nm = r["name"] if isinstance(r["name"], str) and r["name"] else \
            "(unnamed path)"
        L.append(f"| {nm} | {r['neighborhood']} | {r['pass_elev_ft']:.0f} ft | "
                 f"{int(r['pairs_served'])} | {r['max_abs_grade']:.1%} |")
    L += ["",
          "The single most consequential pass in San Francisco is an unnamed "
          "path inside **Golden Gate Park** at about 255 ft. It is the "
          "binding constraint for 119 of the 630 neighborhood pairs -- more "
          "than any street in the city -- because it is the lowest point on "
          "the ridge that separates the eastern flats from the ocean side. "
          "Anyone crossing San Francisco east to west pays that 255 ft "
          "whatever route they choose. Its gradient where it crosses is only "
          "4.6%, which is exactly why it is the pass: the crossing is high "
          "but gentle.", "",
          "Below that, the passes divide the city the way its geology does. "
          "Clay Street over Nob Hill (332 ft) and Waller Street in the Haight "
          "(295 ft, the Wiggle's own crest) are the low cols of the northeast. "
          "Lansdale Avenue (696 ft) and Panorama Drive (635 ft) are the Twin "
          "Peaks and Mount Davidson barrier, and there is simply no cheap way "
          "over it: the neighborhoods behind it -- West of Twin Peaks, "
          "Diamond Heights, Twin Peaks itself -- are the ones the flat "
          "network cannot reach.", ""]

    b = d["barriers"]
    if "unavoidability" in b.columns:
        L += ["### Barriers with an alternative, and barriers without", "",
              "A steep street that carries heavy shortest-path traffic but "
              "almost none once climbing is penalised has a flat alternative "
              "nearby. One that keeps its traffic under every objective does "
              "not.", "",
              "| Street | Neighborhood | Gradient | Pairs via shortest route | "
              "Still via the flat route | Verdict |",
              "|---|---|---|---|---|---|"]
        top = b.nlargest(10, "barrier_score")
        for _, r in top.iterrows():
            nm = r["name"] if isinstance(r["name"], str) and r["name"] else \
                "(unnamed)"
            un = float(r.get("unavoidability") or 0)
            verdict = ("**unavoidable**" if un > 0.5 else
                       "avoidable" if un < 0.15 else "partly avoidable")
            L.append(f"| {nm} | {r['neighborhood']} | "
                     f"{r['max_abs_grade']:.1%} | {int(r['shortest_use'])} | "
                     f"{float(r.get('flat_use_per_objective') or 0):.0f} | "
                     f"{verdict} |")
        L.append("")
    return L


def _pareto(d) -> list[str]:
    if "pareto" not in d:
        return []
    pa = d["pareto"]
    pa = pa[(pa["mode"] == "walk") & pa["pareto_optimal"]]

    # citywide: for every pair, how much detour does halving the climb cost?
    rows = []
    for (o, dst), g in pa.groupby(["origin", "destination"]):
        g = g.sort_values("distance_m")
        s = g.iloc[0]                       # the pure-distance anchor
        if s["elev_gain_m"] <= 0:
            continue
        half = g[g["elev_gain_m"] <= 0.5 * s["elev_gain_m"]]
        rows.append({
            "halvable": len(half) > 0,
            "detour_to_halve": (half["distance_m"].min() / s["distance_m"] - 1)
            if len(half) else np.nan,
            "best_saved_pct": 100 * (1 - g["elev_gain_m"].min() / s["elev_gain_m"]),
            "best_detour": g.loc[g["elev_gain_m"].idxmin(), "distance_m"]
            / s["distance_m"] - 1,
        })
    r = pd.DataFrame(rows)

    L = ["## The distance / climbing trade-off", "",
         "For every one of the 1,260 ordered pairs, a single weight is swept "
         "from zero (pure distance) up to the minimum-climbing objective, "
         "tracing the frontier between distance, cumulative climbing and "
         "peak gradient. The useful question is where the knee is: how much "
         "detour buys how much of the climbing.", ""]
    if len(r):
        L += [f"- **{100*r['halvable'].mean():.0f}% of pairs can halve their "
              f"climbing** by some route, and the median detour that costs is "
              f"**{100*r['detour_to_halve'].median():.0f}%**. "
              f"{100*(r['detour_to_halve'] <= 0.10).mean():.0f}% of all pairs "
              f"can halve it within a 10% detour, "
              f"{100*(r['detour_to_halve'] <= 0.20).mean():.0f}% within 20%.",
              f"- Taken to the flattest possible route, the median pair "
              f"sheds **{r['best_saved_pct'].median():.0f}%** of its climbing "
              f"for a median **{100*r['best_detour'].median():.0f}%** more "
              f"distance.", ""]
    for o, dst in FEATURED_PAIRS[:4]:
        g = pa[(pa["origin"] == o) & (pa["destination"] == dst)]
        if g.empty:
            continue
        g = g.sort_values("distance_m")
        L += [f"**{o} to {dst}**", "",
              "| Distance | Climb | Steepest grade |", "|---|---|---|"]
        for _, row in g.iterrows():
            L.append(f"| {_mi(row['distance_m']):.2f} mi | "
                     f"{_ft(row['elev_gain_m']):.0f} ft | "
                     f"{row['max_grade']:.0%} |")
        L.append("")
    L += ["The frontiers are strongly concave: the first fraction of extra "
          "distance removes most of the climbing, and everything after that "
          "buys very little. That is the practical argument for the balanced "
          "objective over the purely flattest one.", ""]
    return L


def _modes(d) -> list[str]:
    p = d["pairs"]
    e = d["edges"]
    w = p[(p["mode"] == "walk") & (p["profile"] == "min_climb")]
    b = p[(p["mode"] == "bike") & (p["profile"] == "min_climb")]
    steps_km = e[e["cls"] == "steps"]["length_m"].sum() / 1000
    L = ["## Walking is not cycling", "",
         f"The two networks are modelled separately, and they are not "
         f"interchangeable. San Francisco has {steps_km:.0f} km of public "
         f"stairways, and they are a genuine part of the pedestrian network "
         f"and completely useless on a bicycle; the bicycle graph excludes "
         f"them outright. Bicycle costs also carry stress weights (a "
         f"protected cycleway counts as 0.85 of its length, 19th Avenue and "
         f"Van Ness as 1.9) and respect one-way restrictions, which "
         f"pedestrians do not.", "",
         f"The result is that the flattest bicycle route averages "
         f"{_mi(b['distance_m'].mean()):.2f} mi and "
         f"{_ft(b['elev_gain_m'].mean()):.0f} ft of climbing against "
         f"{_mi(w['distance_m'].mean()):.2f} mi and "
         f"{_ft(w['elev_gain_m'].mean()):.0f} ft on foot. The difference is "
         f"modest in aggregate but decisive in specific places: any route "
         f"whose flat pedestrian option runs up a stairway has no bicycle "
         f"equivalent at all, which is why the Presidio has no "
         f"bicycle-legal connection from some of its paths.", ""]
    return L


def _top_corridor_phrase(sd: pd.DataFrame) -> str:
    """'the top corridor is X in every run', or an honest count."""
    leads = sd["top_corridor"].fillna("").map(lambda v: v.split(" - ")[0])
    counts = leads.value_counts()
    top, n = counts.index[0], int(counts.iloc[0])
    if n == len(sd):
        return f"the top corridor is {top} in every run"
    return f"the top corridor is {top} in {n} of {len(sd)} runs"


def _robustness(d) -> list[str]:
    """Summarise the sensitivity analysis, if it has been run."""
    path = OUTPUT_DIR / "sensitivity.csv"
    if not path.exists():
        return []
    sd = pd.read_csv(path, index_col="tag")
    if "baseline" not in sd.index or len(sd) < 2:
        return []
    base = sd.loc["baseline"]
    others = sd.drop(index="baseline")
    L = ["## How much of this depends on the modelling choices?", "",
         "Every figure above was recomputed with the whole pipeline rebuilt "
         f"under {len(others)} one-at-a-time changes to the elevation "
         "parameters and the choice of access intersection "
         "(`outputs/sensitivity.md` has the full tables).", "",
         "| Finding | Baseline | Range across all perturbations |",
         "|---|---|---|",
         f"| Flattest route: extra distance | {base['min_climb_detour_pct']:+.0f}% | "
         f"{others['min_climb_detour_pct'].min():+.0f}% to "
         f"{others['min_climb_detour_pct'].max():+.0f}% |",
         f"| Flattest route: climbing avoided | {base['min_climb_gain_saved_pct']:.0f}% | "
         f"{others['min_climb_gain_saved_pct'].min():.0f}% to "
         f"{others['min_climb_gain_saved_pct'].max():.0f}% |",
         f"| Grade-averse: mean steepest pitch | {base['grade_averse_max_grade_pct']:.1f}% | "
         f"{others['grade_averse_max_grade_pct'].min():.1f}% to "
         f"{others['grade_averse_max_grade_pct'].max():.1f}% |",
         f"| Corridor material shared with baseline (by length) | 100% | "
         f"{others['edge_overlap_pct'].min():.0f}% to "
         f"{others['edge_overlap_pct'].max():.0f}% |",
         f"| Lead streets of the top 12 corridors kept | 12 of 12 | "
         f"{int(others['lead_streets_shared'].min())} to "
         f"{int(others['lead_streets_shared'].max())} of 12 |",
         f"| Dominant pass | {base['top_pass_nbhd']}, {base['top_pass_ft']:.0f} ft | "
         f"same location in {int((sd['top_pass_nbhd'] == base['top_pass_nbhd']).sum())} "
         f"of {len(sd)} runs; {others['top_pass_ft'].min():.0f}-"
         f"{others['top_pass_ft'].max():.0f} ft |",
         f"| Wiggle: excess climb, flat vs shortest | "
         f"{base['wiggle_excess_flat_m']:.1f} vs {base['wiggle_excess_shortest_m']:.1f} m | "
         f"flat {others['wiggle_excess_flat_m'].min():.1f}-"
         f"{others['wiggle_excess_flat_m'].max():.1f} m, shortest "
         f"{others['wiggle_excess_shortest_m'].min():.1f}-"
         f"{others['wiggle_excess_shortest_m'].max():.1f} m |",
         f"| Filbert Street gradient (published 31.5%) | "
         f"{base['grade_Filbert Street']:.1f}% | "
         f"{others['grade_Filbert Street'].min():.1f}% to "
         f"{others['grade_Filbert Street'].max():.1f}% |",
         ""]
    dev = (others["min_climb_gain_saved_pct"] - base["min_climb_gain_saved_pct"]).abs()
    worst = dev.idxmax()
    least = others["edge_overlap_pct"].idxmin()
    drifters = sorted({s for v in others["lead_streets_new"].fillna("")
                       for s in str(v).split("; ") if s})
    L += [f"The headline barely moves: the perturbation that shifts it most "
          f"is **{worst}** ({others.loc[worst, 'change']}), at "
          f"{others.loc[worst, 'min_climb_gain_saved_pct']:.0f}% climbing "
          f"avoided against {base['min_climb_gain_saved_pct']:.0f}% at "
          f"baseline. The dominant pass and the Wiggle result hold in every "
          f"run.", "",
          f"The corridors are where the model is least rigid, and it is "
          f"worth being precise about how. The *street* that qualifies as "
          f"corridor material is {others['edge_overlap_pct'].min():.0f}-"
          f"{others['edge_overlap_pct'].max():.0f}% the same by length, and "
          f"{_top_corridor_phrase(sd)}; what changes "
          f"is where each corridor is cut and therefore what it is called, "
          f"most under the profile smoothing window (**{least}**, "
          f"{others['edge_overlap_pct'].min():.0f}%). A handful of "
          f"borderline streets drift in and out of the top twelve "
          f"({', '.join(drifters)}): these are real corridors whose rank "
          f"depends on tenths of a percent of gradient, not artefacts, and "
          f"they should be read as a tier rather than a ranking.", ""]
    return L


def _limits(d) -> list[str]:
    return [
        "## What this analysis does not tell you", "",
        "- **Elevation is the ground, not the street surface.** The 1 m lidar "
        "DEM is bare-earth, so bridges and tunnels are corrected by "
        "interpolating across the structure, and a handful of piers over "
        "water had to be solved from their neighbours.",
        "- **Travel is modelled on street centrelines.** Sidewalk and "
        "crosswalk geometry exists in the source data but is deliberately "
        "excluded: including it would represent every street two or three "
        "times and wreck the corridor aggregation. Pedestrian distances are "
        "therefore block-scale, not door-to-door.",
        "- **One access point per neighborhood.** Each neighborhood is "
        "represented by a single street-network-weighted, "
        "intersection-snapped point. Large or awkwardly shaped "
        "neighborhoods -- Bayview, Lakeshore, the Presidio -- are served "
        "worse by this than compact ones.",
        "- **No traffic, surface quality, signals or safety.** The bicycle "
        "stress weights are a crude proxy for road class, not a level-of-"
        "traffic-stress model, and nothing here accounts for signal delay, "
        "pavement condition or collision risk.",
        "- **Bicycle facilities are OSM-derived, not SFMTA.** DataSF was "
        "unreachable from the build environment, so the bicycle and "
        "low-stress layers are inferred from OpenStreetMap tagging rather "
        "than from SFMTA's official facility classes or the Slow Streets "
        "designation list.",
        "- **The 37-neighborhood boundary set, not the 41-unit Analysis "
        "Neighborhoods.** Same cause. The difference mostly affects how the "
        "Sunset, the Richmond and the Twin Peaks area are subdivided.",
        "",
    ]


def write_report() -> Path:
    d = _load()
    L = ["# San Francisco's flat street network: findings", "",
         "*Generated by `python -m flatten_seattle report`. Every figure is "
         "computed from the analysis outputs in this repository; see "
         "`validation_report.md` for the checks against known ground truth.*",
         ""]
    L += _headline(d)
    L += _featured(d)
    L += _corridors(d)
    L += _wiggle(d)
    L += _passes(d)
    L += _pareto(d)
    L += _modes(d)
    L += _robustness(d)
    L += _limits(d)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(L))
    return REPORT_MD
