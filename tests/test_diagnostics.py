"""Poincare, resolution, periodicity, schema and input validation."""

import numpy as np
import pytest
from conftest import offset_grid, pair_field, random_solenoidal, run, sample

import null_finder as nf


def test_poincare_matches_charge_in_random_field(random_field):
    B, x = random_field
    r = run(*B, x, x, x)
    assert len(r["pos"]) > 100
    cells, counts = np.unique(r["cell"], axis=0, return_counts=True)
    single = {tuple(c) for c, n in zip(cells, counts, strict=True) if n == 1}
    sel = np.array([tuple(c) in single for c in r["cell"]])
    agree = np.mean(r["poincare"][sel] == r["null_charge"][sel])
    assert agree > 0.995


def bilinear_pair(c, kappa):
    """B = (x z - c, y, z - kappa x): nulls at x = +-sqrt(c/kappa), z = kappa x.

    Bilinear terms are reproduced exactly by trilinear interpolation, so the
    null positions of the interpolant are known exactly, also inside one cell.
    """
    return lambda X, Y, Z: (X * Z - c, Y, Z - kappa * X)


def test_poincare_sums_charges_of_close_pair():
    # Both nulls (+-0.05) in the cell [-0.09, 0.31]^3: its Poincare index is 0.
    x = np.linspace(-1, 1, 6) + 0.11
    r = run(*sample(bilinear_pair(0.0025, 1.0), x, x, x), x, x, x)
    assert len(r["pos"]) == 2
    np.testing.assert_allclose(np.sort(r["pos"][:, 0]), [-0.05, 0.05], atol=1e-12)
    assert len(np.unique(r["cell"], axis=0)) == 1
    assert sorted(r["null_charge"]) == [-1, 1]
    assert np.all(r["poincare"] == 0)
    assert r["snapshot_resolution"] == "unresolved"


def test_resolution_counts_cells_on_anisotropic_grid():
    xa = np.linspace(-1, 1, 41) + 0.0011  # dx = 0.05
    za = np.linspace(-0.2, 0.2, 65) + 0.0007  # dz = dx / 8
    kappa = 0.01
    s = 0.015  # nulls at x = +-s: 0.6 cells apart along x, 0.05 cells along z
    r = run(*sample(bilinear_pair(kappa * s**2, kappa), xa, xa, za), xa, xa, za)
    assert len(r["pos"]) == 2
    expected = np.hypot(2 * s / 0.05, 2 * kappa * s / 0.00625)
    np.testing.assert_allclose(
        r["nearest_opposite_distance_cells"], expected, rtol=1e-9
    )
    assert set(r["resolution_name"]) == {"marginal"}
    # Normalising by min(dx, dy, dz), as before, would have called it resolved.
    assert r["nearest_opposite_distance"][0] / r["hlocal_min"][0] > 4.0


def test_isolated_null():
    x, y, z = offset_grid(10)
    r = run(*sample(lambda X, Y, Z: (X, Y, -2 * Z), x, y, z), x, y, z)
    assert r["snapshot_resolution"] == "isolated"
    assert r["n_isolated"] == 1
    assert r["nearest_opposite_index"][0] == -1


def test_pad_periodic_closes_the_charge():
    B, x = random_solenoidal(n=32, seed=3)
    unpadded = run(*B, x, x, x)
    Bp = nf.pad_periodic(*B, x, x, x, axes=(0, 1, 2))
    r = run(*Bp)
    assert r["total_charge"] == 0
    assert len(r["pos"]) >= len(unpadded["pos"])
    assert Bp[3].size == 33
    assert Bp[3][-1] == 32.0


def test_pad_periodic_rejects_nonuniform_axis():
    x = np.array([0.0, 1.0, 3.0, 4.0])
    b = np.zeros((4, 4, 4))
    with pytest.raises(ValueError, match="not uniform"):
        nf.pad_periodic(b, b, b, x, x, x, axes=(0,))
    out = nf.pad_periodic(b, b, b, x, x, x, axes=(0,), periods={0: 5.0})
    assert out[3][-1] == 5.0


def _schema(r):
    return (
        set(r)
        | {f"settings.{k}" for k in r["settings"]}
        | {f"timing.{k}" for k in r["timing"]}
    )


@pytest.mark.parametrize("with_res", [True, False])
def test_schema_is_stable(with_res):
    xs = np.arange(6.0)
    no_candidates = [np.ones((6, 6, 6))] * 3
    # Candidates (all components change sign) but no common root.
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    no_roots = [X - 2.5, Y - 2.5, (X - 2.5) + 0.3 + 0.0 * Z]
    x, y, z = offset_grid(12)
    normal = sample(pair_field, x, y, z)
    results = [
        run(*no_candidates, xs, xs, xs, add_resolution_diagnostics=with_res),
        run(*no_roots, xs, xs, xs, add_resolution_diagnostics=with_res),
        run(*normal, x, y, z, add_resolution_diagnostics=with_res),
    ]
    schemas = [_schema(r) for r in results]
    assert schemas[0] == schemas[1] == schemas[2]
    assert ("snapshot_resolution" in results[0]) == with_res
    for r in results:
        assert r["cell"].dtype == np.int64
        n = len(r["pos"])
        for key in (
            "bnorm",
            "detJ",
            "null_charge",
            "poincare",
            "null_type",
            "div_residual",
        ):
            assert r[key].shape[0] == n
    assert results[0]["n_candidates"] == 0
    assert len(results[2]["pos"]) == 2


def test_topology_report_runs(capsys):
    x, y, z = offset_grid(12)
    r = run(*sample(pair_field, x, y, z), x, y, z)
    rep = nf.topology_report(
        r, expected_charge=0, show_null_table=True, show_pairs=True
    )
    assert rep["charge_ok"]
    assert rep["poincare_agreement"] == 1.0
    assert "pair resolution" in capsys.readouterr().out
    with pytest.raises(ValueError, match="use must be"):
        nf.topology_report(r, use="foo", verbose=False)


def test_rejects_nonfinite_and_bad_coordinates():
    x = np.arange(5.0)
    b = np.ones((5, 5, 5))
    bad = b.copy()
    bad[2, 2, 2] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        run(bad, b, b, x, x, x)
    with pytest.raises(ValueError, match="monotonic"):
        run(b, b, b, np.array([0, 1, 1, 2, 3.0]), x, x)
    with pytest.raises(ValueError, match="Coordinate sizes"):
        run(b, b, b, np.arange(6.0), x, x)
    with pytest.raises(ValueError, match="cannot both be 0"):
        run(b, b, b, x, x, x, b_tol=0.0, b_tol_rel=0.0)


def test_classify_false_keeps_divergence():
    x, y, z = offset_grid(12)
    r = run(*sample(pair_field, x, y, z), x, y, z, classify=False)
    assert np.all(r["null_type"] == nf.NULL_TYPE_UNKNOWN)
    assert np.all(np.isfinite(r["div_residual"]))
