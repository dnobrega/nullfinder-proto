"""Nulls of analytic fields: positions, charges, types, geometry."""

import numpy as np
import pytest
from conftest import offset_grid, pair_field, run, sample

import null_finder as nf


def test_pair_positions_charges_types():
    x, y, z = offset_grid()
    r = run(*sample(pair_field, x, y, z), x, y, z)
    assert len(r["pos"]) == 2
    order = np.argsort(r["pos"][:, 0])
    pos = r["pos"][order]
    # Trilinear interpolation of x^2 shifts the roots by O(h^2) only.
    np.testing.assert_allclose(pos[:, 0], [-0.3, 0.3], atol=5e-3)
    np.testing.assert_allclose(pos[:, 1:], 0.0, atol=1e-12)
    np.testing.assert_array_equal(r["null_charge"][order], [1, -1])
    np.testing.assert_array_equal(
        r["null_type"][order], [nf.NULL_TYPE_A, nf.NULL_TYPE_B]
    )
    np.testing.assert_array_equal(r["poincare"], r["null_charge"])
    assert r["total_charge"] == 0
    assert r["snapshot_resolution"] == "resolved"


LINEAR = {
    "A": (lambda X, Y, Z: (-X, -Y, 2 * Z), nf.NULL_TYPE_A, 1, (0, 0, 1)),
    "B": (lambda X, Y, Z: (X, Y, -2 * Z), nf.NULL_TYPE_B, -1, (0, 0, 1)),
    "As": (
        lambda X, Y, Z: (-X - 2 * Y, 2 * X - Y, 2 * Z),
        nf.NULL_TYPE_AS,
        1,
        (0, 0, 1),
    ),
    "Bs": (
        lambda X, Y, Z: (X - 2 * Y, 2 * X + Y, -2 * Z),
        nf.NULL_TYPE_BS,
        -1,
        (0, 0, 1),
    ),
    "A_spine_x": (lambda X, Y, Z: (2 * X, -Y, -Z), nf.NULL_TYPE_A, 1, (1, 0, 0)),
}


@pytest.mark.parametrize("name", list(LINEAR))
def test_linear_null_types(name):
    f, ntype, charge, spine = LINEAR[name]
    x, y, z = offset_grid(12)
    r = run(*sample(f, x, y, z), x, y, z)
    assert len(r["pos"]) == 1
    np.testing.assert_allclose(r["pos"][0], 0.0, atol=1e-12)
    assert r["null_type"][0] == ntype
    assert r["null_charge"][0] == charge
    assert r["poincare"][0] == charge
    # Axes have arbitrary sign.
    assert abs(np.dot(r["spine_vec"][0], spine)) == pytest.approx(1.0, abs=1e-10)
    assert abs(np.dot(r["fan_normal"][0], spine)) == pytest.approx(1.0, abs=1e-10)


def test_potential_and_spine_current():
    x, y, z = offset_grid(12)
    pot = run(*sample(LINEAR["B"][0], x, y, z), x, y, z)
    assert pot["is_potential"][0]
    assert pot["fan_is_perpendicular"][0]
    spiral = run(*sample(LINEAR["Bs"][0], x, y, z), x, y, z)
    # Spiral null with current purely along the spine.
    assert not spiral["is_potential"][0]
    assert spiral["current_spine_angle_deg"][0] == pytest.approx(0.0, abs=1e-6)
    assert spiral["fan_is_perpendicular"][0]


def test_null_on_grid_node_is_merged():
    xn = np.linspace(-0.5, 0.5, 21)  # contains 0 exactly

    def f(X, Y, Z):
        return X + X**2, Y, -2 * Z - 2 * X * Z

    r = run(*sample(f, xn, xn, xn), xn, xn, xn)
    assert r["n_roots_before_dedup"] == 8  # seen from the 8 cells sharing the node
    assert len(r["pos"]) == 1
    np.testing.assert_allclose(r["pos"][0], 0.0, atol=1e-14)
    assert r["poincare"][0] == nf.POINCARE_DEGENERATE
    assert r["null_charge"][0] == -1


def test_degenerate_line_null_is_not_reported():
    x, y, z = offset_grid(12)
    r = run(*sample(lambda X, Y, Z: (X, -Y, 0 * Z), x, y, z), x, y, z)
    assert len(r["pos"]) == 0
