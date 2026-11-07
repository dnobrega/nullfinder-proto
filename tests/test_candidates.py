"""The vectorised candidate scan against a direct NumPy reference."""

import numpy as np
import pytest

from nullfinder_proto.core import _get_candidate_cells


def reference_candidates(bx, by, bz):
    """Cells where every component has a vertex <= 0 and a vertex >= 0."""
    ok = None
    for a in (bx, by, bz):
        v = np.stack(
            [
                a[
                    i : a.shape[0] - 1 + i,
                    j : a.shape[1] - 1 + j,
                    k : a.shape[2] - 1 + k,
                ]
                for i in (0, 1)
                for j in (0, 1)
                for k in (0, 1)
            ]
        )
        c = np.isfinite(v).all(axis=0) & (v.min(axis=0) <= 0) & (v.max(axis=0) >= 0)
        ok = c if ok is None else ok & c
    i, j, k = np.nonzero(ok)
    order = np.lexsort((i, j, k))
    return np.stack([i, j, k], axis=1)[order]


@pytest.mark.parametrize("shape", [(17, 9, 13), (8, 8, 8), (2, 3, 5), (31, 2, 4)])
@pytest.mark.parametrize("layout", ["F", "C", "strided"])
def test_scan_matches_reference(shape, layout):
    rng = np.random.default_rng(sum(shape))
    # Small offsets so that many cells are candidates; some vertices exactly 0.
    B = [rng.normal(0.3, 1.0, shape).round(1) for _ in range(3)]
    if layout == "F":
        B = [np.asfortranarray(b) for b in B]
    elif layout == "C":
        B = [np.ascontiguousarray(b) for b in B]
    else:
        B = [np.asfortranarray(np.pad(b, ((0, 2), (0, 0), (0, 0))))[:-2] for b in B]
    ref = reference_candidates(*B)
    cand, counts = _get_candidate_cells(*B)
    assert len(ref) > 0
    np.testing.assert_array_equal(cand, ref)
    np.testing.assert_array_equal(
        counts, np.bincount(ref[:, 2], minlength=shape[2] - 1)
    )


def test_scan_float32_and_no_candidates():
    rng = np.random.default_rng(1)
    B = [
        np.asfortranarray(rng.normal(0, 1, (20, 20, 20)).astype(np.float32))
        for _ in range(3)
    ]
    np.testing.assert_array_equal(_get_candidate_cells(*B)[0], reference_candidates(*B))
    ones = np.ones((6, 6, 6))
    cand, counts = _get_candidate_cells(ones, ones, ones)
    assert cand.shape == (0, 3)
    assert counts.sum() == 0


@pytest.mark.parametrize("layout", ["F", "C", "strided"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_scan_excludes_nonfinite_vertex_cells(layout, bad):
    rng = np.random.default_rng(62)
    shape = (12, 7, 9)
    B = [rng.normal(size=shape) for _ in range(3)]
    # Each contaminated node touches several cells; no such cell may enter
    # the candidate list, regardless of the component containing the value.
    B[1][3, 4, 5] = bad
    B[2][7, 1, 2] = bad
    if layout == "F":
        B = [np.asfortranarray(b) for b in B]
    elif layout == "C":
        B = [np.ascontiguousarray(b) for b in B]
    else:
        B = [np.asfortranarray(np.pad(b, ((0, 2), (0, 0), (0, 0))))[:-2] for b in B]

    cand, counts = _get_candidate_cells(*B)
    ref = reference_candidates(*B)
    np.testing.assert_array_equal(cand, ref)
    np.testing.assert_array_equal(
        counts, np.bincount(ref[:, 2], minlength=shape[2] - 1)
    )


def test_scan_fields_with_mixed_contiguity():
    rng = np.random.default_rng(19)
    B = [rng.normal(size=(13, 14, 11)) for _ in range(3)]
    B[0] = np.asfortranarray(B[0])
    B[1] = np.ascontiguousarray(B[1])
    B[2] = np.asfortranarray(np.pad(B[2], ((0, 1), (0, 0), (0, 0))))[:-1]
    ref = reference_candidates(*B)
    cand, counts = _get_candidate_cells(*B)
    np.testing.assert_array_equal(cand, ref)
    np.testing.assert_array_equal(
        counts, np.bincount(ref[:, 2], minlength=B[0].shape[2] - 1)
    )


def test_candidate_scan_is_internal_only():
    import nullfinder_proto

    assert not hasattr(nullfinder_proto, "get_candidate_cells")
    assert "get_candidate_cells" not in nullfinder_proto.__all__
