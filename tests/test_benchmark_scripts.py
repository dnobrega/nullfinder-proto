"""Smoke tests of the benchmark scripts on small grids (not run in full in CI)."""

import warnings

import numpy as np
import pytest

from benchmarks import thread_scaling
from benchmarks.olshevsky2020_field import (
    EXACT_NULLS,
    PARAMS_A1,
    create_grid,
    olshevsky2020_field,
)


def test_field_vanishes_at_tabulated_nulls():
    # Table A.2 is printed with 4-5 digits, so |B| ~ 1e-4 there (vs |B| ~ 1 overall).
    b = [olshevsky2020_field(p[:1], p[1:2], p[2:3]) for p in EXACT_NULLS]
    bnorm = [np.sqrt(sum(c[0, 0, 0] ** 2 for c in bb)) for bb in b]
    assert max(bnorm) < 2e-3


def test_thread_scaling_script_runs(tmp_path):
    args = [*"--n 40 --threads 1 --repeat 1 --no-plot --out".split(), str(tmp_path)]
    rows, chiti, _ = thread_scaling.main(args)
    assert chiti is None
    assert rows[0]["matched"] == 14
    assert rows[0]["speedup"] == 1.0
    assert (tmp_path / "thread_scaling_N40.csv").exists()
    assert thread_scaling.numba.get_parallel_chunksize() == 0


def test_thread_scaling_rejects_negative_chunksize():
    with pytest.raises(SystemExit, match="2"):
        thread_scaling.main(["--n", "20", "--chunksize", "-1"])


def test_thread_scaling_reports_chunk_size(tmp_path):
    args = [
        "--n",
        "40",
        "--threads",
        "1",
        "--repeat",
        "1",
        "--no-plot",
        "--chunksize",
        "2",
        "--out",
        str(tmp_path),
    ]
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        rows, _, info = thread_scaling.main(args)
    if info["threading_layer"] != "tbb":
        assert any("dynamic scheduling" in str(w.message) for w in recorded)
    assert rows[0]["matched"] == 14
    assert info["parallel_chunksize"] == 2
    assert thread_scaling.numba.get_parallel_chunksize() == 0


def _reference_field(x, y, z):
    """Eq. (A.1) of Olshevsky et al. (2020) written directly, on full 3D grids."""
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    bx = -2 * np.sin(2 * Y) + np.sin(Z)
    by = 2 * np.sin(X) + np.sin(Z)
    bz = np.sin(X) + np.sin(Y)
    bx += 0.04 * (-2 * np.sin(2 * Y) + 2 * np.sin(X) * np.cos(Z) + np.sin(Y) + 0.1195)
    by += 0.04 * (2 * np.sin(X - Z) + np.sin(X + Z) + np.sqrt(30.0) / 7.0)
    bz += 0.04 * (
        -2 * np.cos(X) * np.sin(Z) + np.sin(Y) + 2 * np.sin(2 * Y) - 0.1378162
    )
    for r, (a, k, ell, X0, Y0, Z0) in enumerate(PARAMS_A1):
        if r < 2:
            phi = (
                a
                * k
                * np.exp(
                    -((X - X0) ** 2 + (Y - Y0) ** 2) / a**2 - (Z - Z0) ** 2 / ell**2
                )
            )
            bx += -2 * (Y - Y0) / a**2 * phi
            by += 2 * (X - X0) / a**2 * phi
        elif r < 5:
            phi = (
                a
                * k
                * np.exp(
                    -((X - X0) ** 2 + (Z - Z0) ** 2) / a**2 - (Y - Y0) ** 2 / ell**2
                )
            )
            bx += 2 * (Z - Z0) / a**2 * phi
            bz += -2 * (X - X0) / a**2 * phi
        else:
            phi = (
                a
                * k
                * np.exp(
                    -((Y - Y0) ** 2 + (Z - Z0) ** 2) / a**2 - (X - X0) ** 2 / ell**2
                )
            )
            by += -2 * (Z - Z0) / a**2 * phi
            bz += 2 * (Y - Y0) / a**2 * phi
    return bx, by, bz


def test_parallel_field_matches_reference_formula():
    c = create_grid(23)
    ref = _reference_field(c, c, c)
    out = olshevsky2020_field(c, c, c)
    for a, b in zip(ref, out, strict=True):
        np.testing.assert_allclose(b, a, atol=1e-12)
        assert b.flags.f_contiguous
