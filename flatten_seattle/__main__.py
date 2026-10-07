"""Command-line interface.

    python -m flatten_seattle sources        # dataset provenance table
    python -m flatten_seattle download       # fetch and cache all source data
    python -m flatten_seattle build-network   # street graph + elevation + metrics
    python -m flatten_seattle analyze        # pairs, Pareto, corridors, passes
    python -m flatten_seattle validate       # checks against known ground truth
    python -m flatten_seattle map            # interactive + static maps
    python -m flatten_seattle site           # the route finder alone (no analysis)
    python -m flatten_seattle report         # written analysis of the findings
    python -m flatten_seattle all            # everything, in order
    python -m flatten_seattle route --from Ballard --to "Columbia City"

Every stage caches its output, so re-running is cheap; pass ``--force`` to
recompute a stage from scratch.
"""
from __future__ import annotations

import argparse
import sys

from . import sources
from .utils import get_logger, setup_logging, step

log = get_logger("flatten_seattle")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--force", action="store_true",
                   help="recompute instead of using cached artefacts")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m flatten_seattle",
        description="Flatten Seattle: the flattest routes across the "
                    "city, and the analysis behind them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, helptext in [
        ("sources", "print the dataset provenance table"),
        ("download", "download and cache every source dataset"),
        ("build-network", "build the street graph, sample elevation, "
                          "compute edge metrics"),
        ("analyze", "neighborhood pairs, Pareto fronts, corridors, passes"),
        ("validate", "validate the model against known ground truth"),
        ("map", "render the interactive and static maps"),
        ("site", "build only the route finder (needs build-network, not analyze)"),
        ("report", "write the analysis of major findings"),
        ("all", "run every stage in order"),
    ]:
        sp = sub.add_parser(name, help=helptext)
        _add_common(sp)

    sp = sub.add_parser("sensitivity",
                        help="rebuild the pipeline under perturbed parameters")
    _add_common(sp)
    sp.add_argument("--only", nargs="*", help="run only these configuration tags")

    sm = sub.add_parser("summarize", help=argparse.SUPPRESS)
    _add_common(sm)
    sm.add_argument("--tag", required=True)

    rp = sub.add_parser("route", help="route between two neighborhoods")
    _add_common(rp)
    rp.add_argument("--from", dest="origin", required=True)
    rp.add_argument("--to", dest="dest", required=True)
    rp.add_argument("--mode", default="walk", choices=["walk", "bike"])
    rp.add_argument("--profile", default=None,
                    help="one routing objective (default: compare all four)")
    return p


def cmd_sources(_args) -> int:
    print(sources.format_table())
    return 0


def cmd_download(args) -> int:
    from .download import download_all
    download_all(force=args.force)
    return 0


def cmd_build_network(args) -> int:
    from .pipeline import build_context
    ctx = build_context(force=args.force)
    log.info("network ready: %d edges, %d directed arcs; modes: %s",
             len(ctx.edges), len(ctx.directed),
             ", ".join(f"{m} ({g.n_nodes} nodes, {g.n_arcs} arcs)"
                       for m, g in ctx.graphs.items()))
    return 0


def cmd_analyze(args) -> int:
    from .pipeline import build_context, run_analysis
    ctx = build_context()
    res = run_analysis(ctx, force=args.force)
    log.info("analysis complete: %d routes, %d corridors, %d passes, "
             "%d barriers", len(res["pairs"]), len(res["corridors"]),
             len(res["passes"]), len(res["barriers"]))
    return 0


def cmd_validate(args) -> int:
    import geopandas as gpd
    from .corridors import CORRIDORS_GPKG
    from .pipeline import build_context
    from .validate import VALIDATION_MD, run_validation
    ctx = build_context()
    cor = gpd.read_file(CORRIDORS_GPKG) if CORRIDORS_GPKG.exists() else None
    run_validation(ctx, cor)
    print(VALIDATION_MD.read_text())
    return 0


def cmd_map(args) -> int:
    import geopandas as gpd
    from .corridors import CORRIDORS_GPKG
    from .passes import BARRIERS_GEOJSON, BASINS_GEOJSON, PASSES_GEOJSON
    from .pipeline import build_context
    from .viz_interactive import make_interactive_map
    from .viz_static import make_backbone_map, make_grade_map

    ctx = build_context()
    cor = gpd.read_file(CORRIDORS_GPKG)
    pz = gpd.read_file(PASSES_GEOJSON)
    ba = gpd.read_file(BARRIERS_GEOJSON)
    bs = gpd.read_file(BASINS_GEOJSON) if BASINS_GEOJSON.exists() else None

    with step("static maps", log):
        make_backbone_map(cor, ctx.edges, ctx.neighborhoods, pz)
        make_grade_map(ctx.edges, ctx.neighborhoods)
    with step("interactive map", log):
        make_interactive_map(ctx, cor, pz, ba, basins=bs)
    return 0


def cmd_site(args) -> int:
    from .pipeline import build_context
    from .viz_interactive import make_route_page

    make_route_page(build_context())
    return 0


def cmd_report(args) -> int:
    from .report import write_report
    path = write_report()
    log.info("wrote %s", path)
    return 0


def cmd_route(args) -> int:
    from .config import ROUTING_PROFILES
    from .pipeline import build_context
    from .routing import route

    ctx = build_context()
    graph = ctx.graphs[args.mode]
    pts = dict(zip(ctx.points[args.mode]["neighborhood"],
                   ctx.points[args.mode]["node"]))
    for name in (args.origin, args.dest):
        if name not in pts:
            print(f"unknown neighborhood {name!r}. Available:\n  "
                  + "\n  ".join(sorted(pts)), file=sys.stderr)
            return 2
    names = [args.profile] if args.profile else list(ROUTING_PROFILES)
    print(f"\n{args.origin}  ->  {args.dest}   [{args.mode}]")
    print(f"{'objective':14s} {'miles':>7s} {'climb ft':>9s} {'loss ft':>8s} "
          f"{'max %':>6s} {'>5% m':>7s} {'>8% m':>7s} {'vs shortest':>22s}")
    base = None
    for pname in names:
        arcs, s = route(graph, pts[args.origin], pts[args.dest],
                        ROUTING_PROFILES[pname])
        if base is None:
            base = s
        cmp_ = ""
        if pname != "shortest" and base:
            dd = 100 * (s["distance_m"] / base["distance_m"] - 1)
            dg = base["elev_gain_m"] - s["elev_gain_m"]
            cmp_ = f"{dd:+5.0f}% dist, {dg*3.28084:+6.0f} ft climb"
        print(f"{pname:14s} {s['distance_m']/1609.344:7.2f} "
              f"{s['elev_gain_m']*3.28084:9.0f} {s['elev_loss_m']*3.28084:8.0f} "
              f"{s['max_grade']*100:6.1f} {s['d_above_5']:7.0f} "
              f"{s['d_above_8']:7.0f} {cmp_:>22s}")
    print()
    return 0


def cmd_sensitivity(args) -> int:
    from .sensitivity import SENS_MD, run_sensitivity
    run_sensitivity(force=args.force, only=args.only)
    print(SENS_MD.read_text())
    return 0


def cmd_summarize(args) -> int:
    from .sensitivity import summarize_current_run
    summarize_current_run(args.tag)
    return 0


def cmd_all(args) -> int:
    for fn in (cmd_download, cmd_build_network, cmd_analyze, cmd_validate,
               cmd_map, cmd_report):
        rc = fn(args)
        if rc:
            return rc
    return 0


_DISPATCH = {
    "sources": cmd_sources, "download": cmd_download,
    "build-network": cmd_build_network, "analyze": cmd_analyze,
    "validate": cmd_validate, "map": cmd_map, "site": cmd_site,
    "report": cmd_report,
    "route": cmd_route, "all": cmd_all,
    "sensitivity": cmd_sensitivity, "summarize": cmd_summarize,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(getattr(args, "verbose", False))
    return _DISPATCH[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
