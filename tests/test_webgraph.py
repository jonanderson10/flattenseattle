"""Tests for the browser payload: quantisation, bundling and parity."""
import base64
import gzip

import numpy as np
import pytest

from flatten_seattle.webgraph import (CM, DM, GRADE_Q, _b64, _i16, _u16,
                                     bundle, encode_polyline)


def decode_polyline(s, precision=5):
    """Reference decoder, mirroring the JavaScript in the map."""
    factor = 10 ** precision
    index = lat = lon = 0
    out = []
    while index < len(s):
        for which in ("lat", "lon"):
            shift = result = 0
            while True:
                b = ord(s[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            d = ~(result >> 1) if (result & 1) else (result >> 1)
            if which == "lat":
                lat += d
            else:
                lon += d
        out.append((lon / factor, lat / factor))
    return out


# ------------------------------------------------------------- quantisation
def test_length_quantisation_covers_the_longest_edge():
    """The longest edge in the network is ~1212 m; uint16 must reach it."""
    assert 1212 * DM < 65535
    q = _u16([1211.9], DM)
    assert abs(q[0] / DM - 1211.9) <= 1.0 / DM


def test_length_quantisation_error_is_centimetre_scale():
    rng = np.random.default_rng(0)
    vals = rng.uniform(0, 1200, 5000)
    back = _u16(vals, DM) / DM
    assert np.max(np.abs(back - vals)) <= 0.5 / DM + 1e-9


def test_climb_quantisation_covers_the_steepest_edge():
    assert 70 * CM < 65535
    back = _u16([68.98], CM)[0] / CM
    assert abs(back - 68.98) < 0.01


def test_gradient_quantisation_is_signed_and_fine():
    for g in (-0.6, -0.315, 0.0, 0.0123, 0.315, 0.6):
        back = _i16([g], GRADE_Q)[0] / GRADE_Q
        assert abs(back - g) < 1e-4


def test_quantisation_clamps_instead_of_wrapping():
    """A wrapped uint16 would silently invent a tiny value for a huge one."""
    assert _u16([1e9], DM)[0] == 65535
    assert _u16([-5.0], DM)[0] == 0
    assert _i16([1e9], GRADE_Q)[0] == 32767
    assert _i16([-1e9], GRADE_Q)[0] == -32768


def test_b64_round_trips_a_typed_array():
    arr = np.arange(1000, dtype="<i4")
    raw = base64.b64decode(_b64(arr))
    assert np.array_equal(np.frombuffer(raw, dtype="<i4"), arr)


# ------------------------------------------------------------------ polyline
def test_polyline_round_trips_within_precision():
    coords = [(-122.41942, 37.77493), (-122.42500, 37.77812),
              (-122.43111, 37.78001), (-122.41000, 37.76000)]
    back = decode_polyline(encode_polyline(coords))
    assert len(back) == len(coords)
    for (x0, y0), (x1, y1) in zip(coords, back):
        assert abs(x0 - x1) < 2e-5 and abs(y0 - y1) < 2e-5


def test_polyline_handles_repeated_and_negative_deltas():
    coords = [(-122.4, 37.8), (-122.4, 37.8), (-122.5, 37.7)]
    back = decode_polyline(encode_polyline(coords))
    assert len(back) == 3
    assert back[1] == pytest.approx(back[0])


# -------------------------------------------------------------------- bundle
def test_bundle_round_trips_arrays_and_strings():
    arrays = {
        "a": np.arange(50, dtype="<i4"),
        "b": np.full(30, 7, dtype="<u2"),
        "c": np.array([-3, 0, 9], dtype="<i2"),
    }
    out = bundle({"arrays": arrays, "meta": {"hello": "world"}},
                 {"txt": "some text", "more": "ünïcode"})
    raw = gzip.decompress(base64.b64decode(out["b64"]))
    assert len(raw) == out["manifest"]["bytes"]
    for name, arr in arrays.items():
        m = out["manifest"]["arrays"][name]
        view = np.frombuffer(raw, dtype="<" + m["t"], count=m["n"],
                             offset=m["o"])
        assert np.array_equal(view, arr)
    for name, text in (("txt", "some text"), ("more", "ünïcode")):
        m = out["manifest"]["strings"][name]
        assert raw[m["o"]:m["o"] + m["b"]].decode("utf-8") == text
    assert out["meta"] == {"hello": "world"}


def test_bundle_offsets_are_contiguous_and_ordered():
    arrays = {f"a{i}": np.arange(10 + i, dtype="<i4") for i in range(5)}
    out = bundle({"arrays": arrays, "meta": {}}, {"s": "x" * 17})
    entries = sorted(
        [(m["o"], m["n"] * 4) for m in out["manifest"]["arrays"].values()]
        + [(m["o"], m["b"]) for m in out["manifest"]["strings"].values()])
    pos = 0
    for off, nbytes in entries:
        assert off == pos
        pos += nbytes
    assert pos == out["manifest"]["bytes"]


def test_bundle_rejects_an_unsupported_dtype():
    with pytest.raises(TypeError):
        bundle({"arrays": {"bad": np.array(["x"], dtype=object)}, "meta": {}}, {})


def test_bundle_actually_compresses():
    """The arrays are mostly small integers with long zero runs."""
    arrays = {"z": np.zeros(200_000, dtype="<u2")}
    out = bundle({"arrays": arrays, "meta": {}}, {})
    assert len(base64.b64decode(out["b64"])) < 0.05 * 400_000
