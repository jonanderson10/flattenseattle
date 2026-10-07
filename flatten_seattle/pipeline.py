"""Orchestration: assemble cached artefacts and run each analysis stage."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .utils import configure_gdal_for_proxy, get_logger, step

log = get_logger("flatten_seattle.pipeline")


@dataclass
class Context:
    """Everything the analysis stages need, loaded from cache where possible."""
    edges: object                  # undirected GeoDataFrame with metrics
    directed: pd.DataFrame         # directed edge table
    profiles: dict                 # edge_id -> (dist, elev)
    neighborhoods: object
    graphs: dict                   # mode -> RouteGraph
    points: dict                   # mode -> representative points GeoDataFrame


def build_context(modes=("walk", "bike"), force: bool = False) -> Context:
    """Load or build every prerequisite artefact."""
    from .download import SEGMENTS_PARQUET
    from .elevation import sample_edge_profiles
    from .metrics import compute_edge_metrics
    from .network import build_edges
    from .neighborhoods import choose_representative_points, load_neighborhoods
    from .routing import build_route_graph

    configure_gdal_for_proxy()
    with step("preparing street network", log):
        raw_edges = build_edges(SEGMENTS_PARQUET, force=force)
    with step("sampling elevation", log):
        prof = sample_edge_profiles(raw_edges, None, force=force)
    with step("computing edge metrics", log):
        edges, directed = compute_edge_metrics(raw_edges, prof, force=force)

    neighborhoods = load_neighborhoods()
    graphs, points = {}, {}
    for mode in modes:
        graphs[mode] = build_route_graph(directed, mode)
        points[mode] = choose_representative_points(
            edges, neighborhoods, mode=mode,
            valid_nodes=graphs[mode].node_ids, force=True)
    return Context(edges=edges, directed=directed, profiles=prof,
                   neighborhoods=neighborhoods, graphs=graphs, points=points)


def run_analysis(ctx: Context, force: bool = False):
    """Pair matrix, Pareto fronts, corridors, passes and barriers."""
    from .corridors import run_corridor_analysis
    from .pairs import run_pareto, run_pair_analysis
    from .passes import run_pass_analysis

    with step("neighborhood-pair routing", log):
        pairs, arc_store = run_pair_analysis(ctx.graphs, ctx.points, force=force)
    with step("Pareto frontier analysis", log):
        # every ordered pair: one Dijkstra per (weight, origin) serves all
        # destinations, so the full matrix costs no more than the sample did
        names = list(ctx.points["walk"]["neighborhood"])
        all_pairs = [(o, d) for o in names for d in names if o != d]
        pareto = run_pareto(ctx.graphs, ctx.points, all_pairs, force=force)
    with step("corridor detection", log):
        corridors, scores = run_corridor_analysis(
            ctx.graphs, arc_store, pairs, ctx.edges, ctx.neighborhoods,
            force=force)
    with step("pass and barrier analysis", log):
        passes, barriers, basins = run_pass_analysis(
            ctx.graphs, arc_store, ctx.edges, ctx.directed, ctx.neighborhoods,
            points=ctx.points["walk"], force=force)
    return {"pairs": pairs, "arc_store": arc_store, "pareto": pareto,
            "corridors": corridors, "edge_scores": scores, "passes": passes,
            "barriers": barriers, "basins": basins}
