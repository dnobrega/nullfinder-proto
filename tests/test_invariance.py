"""Results must not depend on units, layout, axis direction or batching."""
import itertools

import numpy as np
import pytest

from conftest import match, offset_grid, pair_field, run, sample


@pytest.fixture(scope="module")
def pair():
    x, y, z = offset_grid()
    B = sample(pair_field, x, y, z)
    return B, (x, y, z), run(*B, x, y, z)


@pytest.mark.parametrize("c", [1e-12, 1e-6, 1e6, 1e12])
def test_field_scale_invariance(pair, c):
    B, xyz, ref = pair
    r = run(*(c * b for b in B), *xyz)
    assert len(r["pos"]) == len(ref["pos"])
    np.testing.assert_allclose(r["pos"], ref["pos"], atol=1e-12)
    np.testing.assert_array_equal(r["null_charge"], ref["null_charge"])


@pytest.mark.parametrize("L", [1e-8, 1e8])
def test_length_scale_invariance(pair, L):
    B, (x, y, z), ref = pair
    r = run(*B, L * x, L * y, L * z)
    np.testing.assert_allclose(r["pos"] / L, ref["pos"], atol=1e-12)
    np.testing.assert_array_equal(r["null_charge"], ref["null_charge"])
    np.testing.assert_allclose(r["nearest_opposite_distance_cells"],
                               ref["nearest_opposite_distance_cells"], rtol=1e-10)


@pytest.mark.parametrize("flips", [f for f in itertools.product([0, 1], repeat=3) if any(f)])
def test_decreasing_axes(pair, flips):
    B, xyz, ref = pair
    coords = [c[::-1].copy() if fl else c for c, fl in zip(xyz, flips)]
    sl = tuple(slice(None, None, -1) if fl else slice(None) for fl in flips)
    Bf = [np.asfortranarray(b[sl]) for b in B]
    r = run(*Bf, *coords)
    m, d = match(ref["pos"], r["pos"])
    assert d.max() < 1e-12
    np.testing.assert_array_equal(r["null_charge"], ref["null_charge"][m])
    np.testing.assert_array_equal(r["null_type"], ref["null_type"][m])
    # Poincare must follow the physical orientation.
    np.testing.assert_array_equal(r["poincare"], r["null_charge"])


def test_layouts_give_identical_results(random_field):
    B, x = random_field
    ref = run(*B, x, x, x)
    variants = {
        "C": [np.ascontiguousarray(b) for b in B],
        "strided": [np.asfortranarray(np.pad(b, ((0, 1), (0, 0), (0, 0))))[:-1] for b in B],
    }
    for name, Bv in variants.items():
        r = run(*Bv, x, x, x)
        np.testing.assert_array_equal(r["candidates"], ref["candidates"], err_msg=name)
        np.testing.assert_array_equal(r["pos"], ref["pos"], err_msg=name)


def test_float32_and_float64_agree(random_field):
    B, x = random_field
    r32 = run(*B, x, x, x)
    r64 = run(*(b.astype(np.float64) for b in B), x, x, x)
    # Same stored values, so the trilinear interpolants are identical.
    np.testing.assert_allclose(r32["pos"], r64["pos"], atol=1e-10)
    assert r32["settings"]["field_dtype"] == "float32"


def test_big_endian_input(pair):
    B, xyz, ref = pair
    r = run(*(b.astype(">f8") for b in B), *xyz)
    np.testing.assert_allclose(r["pos"], ref["pos"], atol=1e-14)


def test_chunking_and_threads_do_not_change_results(random_field):
    B, x = random_field
    ref = run(*B, x, x, x)
    for kw in ({"chunk_size": 7}, {"refine_parallel": False}, {"refine_parallel": True}):
        r = run(*B, x, x, x, **kw)
        np.testing.assert_array_equal(r["pos"], ref["pos"], err_msg=str(kw))


def test_exclusion_never_drops_roots(random_field):
    B, x = random_field
    ref = run(*B, x, x, x, exclusion_depth=0)
    assert ref["n_candidates_excluded"] == 0
    for depth in (1, 3, 5):
        r = run(*B, x, x, x, exclusion_depth=depth)
        assert r["n_candidates_excluded"] > 0
        np.testing.assert_array_equal(r["pos"], ref["pos"])
