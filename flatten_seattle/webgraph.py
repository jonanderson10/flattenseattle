"""Pack the routable graph into a compact binary payload for the browser.

The interactive map used to ship ~10,000 precomputed routes, which meant it
could only answer questions about the 36 neighborhood access points.  It is
cheaper *and* far more useful to ship the graph itself: 168,000 directed arcs
and 70,000 nodes pack into a few megabytes, and a Dijkstra over that runs in
well under a second in JavaScript.  The map can then route between any two
points the viewer clicks, under any cost weights, with the cost model
evaluated in the browser exactly as Python evaluates it.

Packing
-------
Everything is a typed array.  All of them, plus the encoded geometry and the
vector overlays, are concatenated into a single buffer, gzipped, and
base64-encoded once; the browser inflates it with ``DecompressionStream`` and
takes ``TypedArray`` views straight onto the result.  That avoids parsing
megabytes of JSON numbers, and because the arrays are mostly small integers
with long runs of zeros, compression takes the payload down by roughly a
factor of three.  Quantisation is chosen so it is invisible at the scale the
metrics are reported:

* coordinates: ``int32`` micro-degrees (~0.1 m)
* lengths and steep distances: ``uint16`` in 5 cm units
* climbing: ``uint16`` centimetres
* gradients: ``int16`` hundredths of a percent

Geometry is one concatenated Google-encoded polyline string with per-edge
character offsets, which is already ASCII and so costs nothing to embed.

Parity with Python
------------------
Per-mode arc availability is computed here exactly as
``routing.build_route_graph`` computes it, *including* the restriction to the
largest strongly connected component, so a route found in the browser is the
same route Python finds. ``tests/test_webgraph.py`` asserts that on real data.
"""
from __future__ import annotations

import base64
import gzip

import numpy as np
import pandas as pd

from .bikeways import BIKEWAYS_GEOJSON, conflate, stress as bike_stress
from .config import GRADE_THRESHOLDS, MODES, ROUTING_PROFILES
from .network import _NON_STREET
from .utils import get_logger, step

log = get_logger("flatten_seattle.webgraph")

_TH = [int(t * 100) for t in GRADE_THRESHOLDS]

#: Quantisation scales. Chosen so that round-trip error is far below the
#: precision at which any metric is reported.
DM = 20.0          # 5 cm units per metre (max edge 1212 m fits uint16)
CM = 100.0         # centimetres per metre
GRADE_Q = 10000.0  # int16 units per unit gradient (0.01% resolution)
COORD_Q = 1e6      # micro-degrees
STRESS_Q = 100.0   # hundredths of a comfort multiplier


#: Arrays sent as successive differences: node coordinates (in Z-order) and
#: address fields (sorted by street, then number) change little step to step.
DELTA_ARRAYS = frozenset({"node_lon", "node_lat", "addr_lon", "addr_lat", "addr_number"})
#: Douglas-Peucker tolerance for drawn street geometry, metres.
GEOM_TOLERANCE_M = 1.0


def _morton(lon: np.ndarray, lat: np.ndarray, bits: int = 16) -> np.ndarray:
    """Z-order key interleaving quantised longitude and latitude."""
    def spread(v):
        v = v.astype(np.uint64)
        for shift, mask in ((8, 0x00FF00FF), (4, 0x0F0F0F0F),
                            (2, 0x33333333), (1, 0x55555555)):
            v = (v | (v << np.uint64(shift))) & np.uint64(mask)
        return v

    def quantise(v):
        span = max(float(v.max() - v.min()), 1e-12)
        return np.round((v - v.min()) / span * ((1 << bits) - 1))

    return spread(quantise(lon)) | (spread(quantise(lat)) << np.uint64(1))


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _u16(values, scale: float) -> np.ndarray:
    return np.clip(np.round(np.asarray(values, dtype="float64") * scale),
                   0, 65535).astype("<u2")


def _i16(values, scale: float) -> np.ndarray:
    return np.clip(np.round(np.asarray(values, dtype="float64") * scale),
                   -32768, 32767).astype("<i2")


def encode_polyline(coords, precision: int = 5) -> str:
    """Google encoded-polyline format for a sequence of (lon, lat)."""
    factor = 10 ** precision
    out: list[str] = []
    prev_lat = prev_lon = 0

    def enc(v: int) -> str:
        v = ~(v << 1) if v < 0 else (v << 1)
        chunks = []
        while v >= 0x20:
            chunks.append(chr((0x20 | (v & 0x1f)) + 63))
            v >>= 5
        chunks.append(chr(v + 63))
        return "".join(chunks)

    for lon, lat in coords:
        ilat = int(round(lat * factor))
        ilon = int(round(lon * factor))
        out.append(enc(ilat - prev_lat))
        out.append(enc(ilon - prev_lon))
        prev_lat, prev_lon = ilat, ilon
    return "".join(out)


# --------------------------------------------------------------------------
def _mode_availability(directed: pd.DataFrame, mode: str,
                       node_index: dict) -> np.ndarray:
    """Per-arc availability for a mode, matching ``build_route_graph``.

    Includes the restriction to the largest strongly connected component, so
    that anything the browser can route, Python can route identically.
    """
    import scipy.sparse as sp

    ok = directed[f"{mode}_traversable"].to_numpy()
    fi = directed["from_node"].map(node_index).to_numpy()
    ti = directed["to_node"].map(node_index).to_numpy()
    n = len(node_index)
    sub_f, sub_t = fi[ok], ti[ok]
    adj = sp.coo_matrix((np.ones(sub_f.size), (sub_f, sub_t)),
                        shape=(n, n)).tocsr()
    ncomp, labels = sp.csgraph.connected_components(adj, directed=True,
                                                    connection="strong")
    # the component containing the most *reachable* nodes, counting only
    # nodes that the mode actually touches
    touched = np.zeros(n, dtype=bool)
    touched[sub_f] = True
    touched[sub_t] = True
    counts = np.bincount(labels[touched], minlength=ncomp)
    biggest = int(counts.argmax())
    in_scc = (labels == biggest) & touched
    avail = ok & in_scc[fi] & in_scc[ti]
    log.info("  %s: %d of %d arcs available (largest SCC has %d nodes)",
             mode, int(avail.sum()), int(ok.sum()), int(in_scc.sum()))
    return avail, in_scc


def build_payload(edges, directed: pd.DataFrame) -> dict:
    """Pack nodes, arcs, geometry and display attributes for the browser."""
    with step("packing the routable graph for the browser", log):
        # ---- nodes -------------------------------------------------
        nodes = pd.unique(pd.concat([directed["from_node"],
                                     directed["to_node"]], ignore_index=True))
        node_index = {n: i for i, n in enumerate(nodes)}

        ll = edges.to_crs("EPSG:4326")
        lon = np.zeros(len(nodes)); lat = np.zeros(len(nodes))
        seen = np.zeros(len(nodes), dtype=bool)
        for u, v, geom in zip(edges["u"], edges["v"], ll.geometry):
            c = geom.coords
            for node, (x, y) in ((u, c[0]), (v, c[-1])):
                i = node_index.get(node)
                if i is not None and not seen[i]:
                    lon[i] = x; lat[i] = y; seen[i] = True

        # Number nodes along a Z-order curve, so that neighbours on the map
        # are neighbours in the arrays: coordinate deltas and arc heads then
        # come out small, and gzip packs them far tighter.
        order = np.argsort(_morton(lon, lat), kind="stable")
        nodes, lon, lat = nodes[order], lon[order], lat[order]
        node_index = {n: i for i, n in enumerate(nodes)}

        # ---- node elevations, for route elevation profiles ---------
        # Elevations are reconciled to one value per intersection upstream,
        # so a profile sampled at arc boundaries is exact at every corner.
        za = directed[["from_node", "start_elev"]].rename(
            columns={"from_node": "node", "start_elev": "z"})
        zb = directed[["to_node", "end_elev"]].rename(
            columns={"to_node": "node", "end_elev": "z"})
        zz = pd.concat([za, zb], ignore_index=True).groupby("node")["z"].median()
        node_z = zz.reindex(nodes).fillna(0.0).to_numpy()

        # ---- per-mode availability ---------------------------------
        flags = np.zeros(len(directed), dtype="<u1")
        node_flags = np.zeros(len(nodes), dtype="<u1")
        for bit, mode in enumerate(("walk", "bike")):
            avail, in_scc = _mode_availability(directed, mode, node_index)
            flags |= (avail.astype("<u1") << bit)
            node_flags |= (in_scc.astype("<u1") << bit)
        # bit 3 marks a street corner: a node on a street, not on a footway,
        # path or stair, and not underground. Searched places and clicks snap
        # to one, so a route to Pike Place Market ends on Pike Street rather
        # than down in the market's lower levels.
        tunnel = directed["edge_id"].map(edges.set_index("edge_id")["is_tunnel"]).fillna(False)
        on_street = (~directed["cls"].isin(_NON_STREET) & ~tunnel.astype(bool)).to_numpy()
        corner = np.zeros(len(nodes), dtype=bool)
        for col in ("from_node", "to_node"):
            idx = directed[col].map(node_index).to_numpy()
            corner[idx[on_street]] = True
        node_flags |= (corner.astype("<u1") << 3)
        keep = flags > 0
        d = directed[keep].reset_index(drop=True)
        flags = flags[keep]
        # bit 2 marks a reverse traversal, so the geometry is drawn backwards
        flags |= ((d["direction"].to_numpy() == "rev").astype("<u1") << 2)

        # ---- arcs, sorted into CSR order ---------------------------
        fi = d["from_node"].map(node_index).to_numpy()
        ti = d["to_node"].map(node_index).to_numpy()
        order = np.argsort(fi, kind="stable")
        fi, ti, d, flags = fi[order], ti[order], d.iloc[order], flags[order]
        indptr = np.zeros(len(nodes) + 1, dtype="<i4")
        np.add.at(indptr, fi + 1, 1)
        indptr = np.cumsum(indptr).astype("<i4")

        # arcs point at an edge by its row in ``edges`` (the order the
        # geometry and attributes are packed in), not by edge_id: ids have
        # gaps wherever metrics dropped an edge with no usable elevation
        edge_pos = pd.Index(edges["edge_id"]).get_indexer(d["edge_id"])
        assert (edge_pos >= 0).all(), "an arc refers to an edge that was not packed"

        classes = sorted(set(edges["cls"].dropna().unique())
                         | set(d["cls"].dropna().unique()))
        cls_idx = {c: i for i, c in enumerate(classes)}

        # Loss, maximum gradient and mean absolute gradient are carried
        # per arc rather than re-derived in the browser, so that the map
        # reports exactly the same figures as the analysis outputs. Deriving
        # them from the node elevation series would be close but not equal,
        # and a map that disagrees with its own CSV is worse than a slightly
        # larger file.
        arcs = {
            # relative to the tail node; the browser adds the tail back
            "head": (ti - fi).astype("<i4"),
            "edge": edge_pos.astype("<i4"),
            "len": _u16(d["length_m"], DM),
            "gain": _u16(d["cum_gain"], CM),
            "loss": _u16(d["cum_loss"], CM),
            "maxgrade": _i16(d["max_grade"], GRADE_Q),
            "meangrade": _u16(d["mean_abs_grade"], GRADE_Q),
            "cls": d["cls"].map(cls_idx).fillna(0).to_numpy().astype("<u1"),
            "flags": flags,
        }
        for t in _TH:
            arcs[f"th{t}"] = _u16(d[f"d_above_{t}"], DM)

        # ---- geometry: one polyline string with character offsets ---
        # Simplified to a metre: the line is only drawn, never measured (arc
        # lengths come from the full geometry), and the end points are kept.
        drawn = edges.geometry.simplify(GEOM_TOLERANCE_M).to_crs("EPSG:4326")
        geom_parts: list[str] = []
        offs = np.zeros(len(edges) + 1, dtype="<i4")
        pos = 0
        for i, geom in enumerate(drawn):
            s = encode_polyline(list(geom.coords))
            geom_parts.append(s)
            pos += len(s)
            offs[i + 1] = pos
        geom = "".join(geom_parts)

        # ---- per-edge display attributes ---------------------------
        if BIKEWAYS_GEOJSON.exists():
            facility = conflate(edges)
        else:
            log.warning("%s not found: bike stress falls back to road class only",
                        BIKEWAYS_GEOJSON)
            facility = None
        names = edges["name"].fillna("")
        name_values = sorted(set(names) - {""})
        name_idx = {n: i + 1 for i, n in enumerate(name_values)}
        buckets = np.digitize(edges["max_abs_grade"].fillna(0.0).to_numpy(),
                              [0.03, 0.05, 0.08, 0.10, 0.15]).astype("<u1")
        edge_attrs = {
            "name": names.map(lambda n: name_idx.get(n, 0)).to_numpy().astype("<u2"),
            "cls": edges["cls"].map(cls_idx).fillna(0).to_numpy().astype("<u1"),
            "bucket": buckets,
            "maxgrade": _u16(edges["max_abs_grade"].fillna(0.0), GRADE_Q),
            "avggrade": _i16(edges["avg_grade_fwd"].fillna(0.0), GRADE_Q),
            "len": _u16(edges["length_m"], DM),
            "gainkm": _u16((edges["cum_gain_fwd"]
                            / (edges["length_m"] / 1000.0).clip(lower=1e-6))
                           .clip(0, 6000), DM),
            "lowstress": edges["low_stress"].fillna(False).to_numpy().astype("<u1"),
            # bike comfort multiplier, hundredths (100 = an ordinary block),
            # from the SDOT bike facilities where it is on disk
            "stress": np.clip(np.round(bike_stress(edges, facility) * STRESS_Q),
                              1, 255).astype("<u1"),
        }

        arrays: dict[str, np.ndarray] = {
            "node_lon": np.round(lon * COORD_Q).astype("<i4"),
            "node_lat": np.round(lat * COORD_Q).astype("<i4"),
            "node_flags": node_flags,
            "node_elev": _i16(node_z, DM),
            "indptr": indptr,
            "geom_off": offs,
        }
        for k, v in arcs.items():
            arrays["arc_" + k] = v
        for k, v in edge_attrs.items():
            arrays["edge_" + k] = v

        meta = {
            "n_nodes": int(len(nodes)),
            "n_arcs": int(len(ti)),
            "n_edges": int(len(edges)),
            "classes": classes,
            "names": name_values,
            "thresholds": _TH,
            "scales": {"dm": DM, "cm": CM, "grade": GRADE_Q, "coord": COORD_Q,
                       "stress": STRESS_Q},
            "bikeways": facility is not None,
            "head_rel": True,
            "profiles": {
                name: {
                    "alpha": w.alpha, "beta": w.beta, "gamma": w.gamma,
                    "penalties": list(w.threshold_penalties),
                    "extreme": w.extreme_extra,
                    "use_class_multiplier": bool(w.use_class_multiplier),
                }
                for name, w in ROUTING_PROFILES.items()
            },
            "multipliers": {
                mode: [MODES[mode].class_multiplier.get(c, 1.0) for c in classes]
                for mode in MODES
            },
        }
        return {"arrays": arrays, "geom": geom, "meta": meta,
                "n_coords": sum(len(g.coords) for g in drawn)}


_DTYPE_TAG = {"int32": "i4", "uint32": "u4", "int16": "i2",
              "uint16": "u2", "uint8": "u1", "int8": "i1",
              "float32": "f4", "float64": "f8"}


def bundle(graph: dict, strings: dict[str, str]) -> dict:
    """Concatenate arrays and strings, gzip, and base64-encode once.

    Returns the manifest the browser needs to take views onto the inflated
    buffer, plus the encoded bundle itself.
    """
    blobs: list[bytes] = []
    manifest_arrays: dict[str, dict] = {}
    manifest_strings: dict[str, dict] = {}
    off = 0

    for name, arr in graph["arrays"].items():
        tag = _DTYPE_TAG.get(arr.dtype.name)
        if tag is None:
            raise TypeError(f"unsupported dtype {arr.dtype} for {name}")
        manifest_arrays[name] = {"t": tag, "o": off, "n": int(arr.size)}
        if name in DELTA_ARRAYS and arr.size:
            # differences wrap in the array's own integer type, exactly as a
            # running sum into a typed array of that type wraps back
            arr = np.diff(arr.astype(np.int64), prepend=0).astype(arr.dtype)
            manifest_arrays[name]["d"] = 1
        raw = np.ascontiguousarray(arr).tobytes()
        blobs.append(raw)
        off += len(raw)

    for name, text in strings.items():
        raw = text.encode("utf-8")
        manifest_strings[name] = {"o": off, "b": len(raw)}
        blobs.append(raw)
        off += len(raw)

    flat = b"".join(blobs)
    # mtime=0 keeps the output byte-reproducible; gzip otherwise stamps the
    # current time into the header and the map churns on every rebuild
    packed = gzip.compress(flat, compresslevel=9, mtime=0)
    log.info("  bundle: %.2f MB raw -> %.2f MB gzipped -> %.2f MB base64",
             len(flat) / 1e6, len(packed) / 1e6,
             len(packed) * 4 / 3 / 1e6)
    return {
        "manifest": {"arrays": manifest_arrays, "strings": manifest_strings,
                     "bytes": len(flat)},
        "b64": base64.b64encode(packed).decode("ascii"),
        "gz": packed,
        "meta": graph["meta"],
    }


