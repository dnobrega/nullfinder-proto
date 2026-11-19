"""Check that multi-start Newton finds every root of the trilinear interpolant.

The finder proves most cells empty by Bernstein subdivision and looks for
roots with multi-start Newton in the cells that survive.  A Newton failure is
a root of the interpolant that no reported null accounts for.  The oracle
below finds such roots without sharing code with the package:

1. Subdivide every grid cell (not only the finder's candidates) into
   sub-boxes of 1/2**DEPTH cells, discarding each sub-box in which some
   component has corner values of one strict sign.  The corner values of a
   sub-box are its Bernstein coefficients, so a discarded sub-box provably
   holds no root, and every root lies in a surviving sub-box.
2. From the centre of every surviving sub-box, minimise |B| with a bounded
   Levenberg-Marquardt iteration on the continuous piecewise-trilinear field
   (over the whole grid, so a sub-box next to a null of the neighbouring cell
   ends in that null and not at the shared face).  A sub-box of 1e-3 cells
   that holds a root converges to it.  Every end point must be either
   a) a reported null (|B| / Bscale < ROOT_REL and within MATCH_CELLS), or
   b) a minimum of |B| that is not a null (|B| / Bscale > GAP_REL).
   A root that is not reported, or an end point between ROOT_REL and GAP_REL
   (undecided), fails the test.

The fields are smooth turbulence and white noise, the hardest case (up to
four roots per cell, close pairs).  noise-337 and noise-384 are fields where
multi-start Newton alone, as in version 0.1.0, missed nulls of close pairs;
the certification step finds them.
"""

import functools

import numpy as np
import pytest
from conftest import random_solenoidal, run
from scipy.spatial import cKDTree

DEPTH = 10  # sub-boxes of 1/1024 cell
ROOT_REL = 1e-9  # |B| / Bscale below this: a root of the interpolant
GAP_REL = 1e-7  # |B| / Bscale above this: a minimum that is not a null
MATCH_CELLS = 1e-6  # a root this close to a reported null is that null

# Corner (and child) offsets, x fastest.
_OCT = np.array([[a, b, c] for c in (0, 1) for b in (0, 1) for a in (0, 1)], float)


def _weights(q):
    """Trilinear weights of the 8 corners at local points q, shape (..., 8)."""
    return np.where(_OCT == 1, q[..., None, :], 1.0 - q[..., None, :]).prod(-1)


def _weight_grads(q):
    """d(weights)/dq, shape (..., 8, 3)."""
    f = np.where(_OCT == 1, q[..., None, :], 1.0 - q[..., None, :])
    df = np.where(_OCT == 1, 1.0, -1.0)
    out = np.empty(f.shape)
    for d in range(3):
        g = f.copy()
        g[..., d] = df[:, d]
        out[..., d] = g.prod(-1)
    return out


def _may_vanish(vals):
    """Every component has a corner value <= 0 and one >= 0."""
    return ((vals.min(axis=-1) <= 0) & (vals.max(axis=-1) >= 0)).all(axis=-1)


class Interpolant:
    """The finder's piecewise-trilinear field, in index coordinates."""

    def __init__(self, B):
        shape = tuple(s - 1 for s in B[0].shape)
        V = np.empty((*shape, 3, 8))
        for c, a in enumerate(B):
            a = np.asarray(a, dtype=float)
            for m, (dx, dy, dz) in enumerate(_OCT.astype(int)):
                V[..., c, m] = a[
                    dx : dx + shape[0], dy : dy + shape[1], dz : dz + shape[2]
                ]
        self.V = V.reshape(-1, 3, 8)  # cell vertex values
        self.shape = np.array(shape)
        self.cells = np.stack(np.unravel_index(np.arange(len(self.V)), shape), 1)
        self.bscale = np.linalg.norm(self.V, axis=1).max(axis=1)

    def locate(self, xi):
        c = np.clip(np.floor(xi).astype(int), 0, self.shape - 1)
        return np.ravel_multi_index(c.T, self.shape), xi - c

    def value_and_jac(self, xi):
        cid, q = self.locate(xi)
        V = self.V[cid]
        return (
            np.einsum("nck,nk->nc", V, _weights(q)),
            np.einsum("nck,nkd->ncd", V, _weight_grads(q)),
        )

    def surviving_subboxes(self, depth=DEPTH):
        """Centres of the sub-boxes that the Bernstein test cannot exclude."""
        ids = np.flatnonzero(_may_vanish(self.V))
        lo = np.zeros((len(ids), 3))
        h = 1.0
        for _ in range(depth):
            h *= 0.5
            ids = np.repeat(ids, 8)
            lo = (lo[:, None, :] + h * _OCT).reshape(-1, 3)
            vals = np.einsum(
                "nkm,ncm->nck", _weights(lo[:, None, :] + h * _OCT), self.V[ids]
            )
            keep = _may_vanish(vals)
            ids, lo = ids[keep], lo[keep]
        return self.cells[ids] + lo + 0.5 * h

    def minimise(self, xi, iters=100):
        """Bounded Levenberg-Marquardt on |B|, vectorised over start points.

        Returns end points and |B| / Bscale, with Bscale of the start cell.
        """
        xi = xi.copy()
        s = self.bscale[self.locate(xi)[0]][:, None]
        F, J = self.value_and_jac(xi)
        F, J = F / s, J / s[:, :, None]
        f = np.linalg.norm(F, axis=1)
        lam = np.full(len(xi), 1e-6)
        for _ in range(iters):
            H = np.einsum("nci,ncj->nij", J, J) + lam[:, None, None] * np.eye(3)
            step = np.linalg.solve(H, -np.einsum("nci,nc->ni", J, F)[..., None])[..., 0]
            new = np.clip(xi + step, 0.0, self.shape)
            Fn, Jn = self.value_and_jac(new)
            Fn, Jn = Fn / s, Jn / s[:, :, None]
            fn = np.linalg.norm(Fn, axis=1)
            ok = fn < f
            xi[ok], F[ok], J[ok], f[ok] = new[ok], Fn[ok], Jn[ok], fn[ok]
            lam = np.where(ok, np.maximum(lam / 10, 1e-12), np.minimum(lam * 10, 1e12))
        return xi, f


def white_noise(n, seed):
    rng = np.random.default_rng(seed)
    B = [np.asfortranarray(rng.standard_normal((n, n, n))) for _ in range(3)]
    return B, np.arange(n, dtype=float)


FIELDS = {
    "turbulent-0": lambda: random_solenoidal(32, 0, np.float64),
    "turbulent-1": lambda: random_solenoidal(32, 1, np.float64),
    "turbulent-f32": lambda: random_solenoidal(32, 2, np.float32),
    "noise-0": lambda: white_noise(16, 0),
    "noise-337": lambda: white_noise(16, 337),
    "noise-384": lambda: white_noise(16, 384),
}


@functools.cache
def oracle(name):
    """Field, and end points and |B| / Bscale of the minimisations."""
    B, x = FIELDS[name]()
    field = Interpolant(B)
    ends, rel = field.minimise(field.surviving_subboxes())
    return B, x, field, ends, rel


def check(name, **kw):
    """Run the finder; return its result, unreported roots and undecided points."""
    B, x, field, ends, rel = oracle(name)
    res = run(*B, x, x, x, classify=False, poincare_diag=False, **kw)
    found = res["cell"] + res["q"]  # index coordinates, since x = arange(n)
    is_root = rel < ROOT_REL
    dist = cKDTree(found).query(ends[is_root])[0]
    missed = np.unique(np.round(ends[is_root][dist > MATCH_CELLS], 6), axis=0)
    undecided = ends[(rel >= ROOT_REL) & (rel <= GAP_REL)]
    _, bnull = field.minimise(found, iters=0)
    return res, bnull, missed, undecided


@pytest.mark.parametrize("name", list(FIELDS))
def test_newton_misses_no_root(name):
    res, bnull, missed, undecided = check(name)

    # The check is only meaningful if many cells passed the sign-change test
    # and turned out to hold no null.
    assert len(res["pos"]) > 100
    assert res["n_candidates"] - res["n_candidates_with_success"] > 500

    # Every reported null is a root of the same interpolant, and every root of
    # the interpolant is a reported null.
    assert bnull.max() < ROOT_REL
    assert res["n_cells_uncertified"] == 0
    assert len(missed) == 0, f"roots not reported (index coordinates): {missed}"
    assert len(undecided) == 0, f"|B| minima too close to zero to decide: {undecided}"


def test_oracle_detects_missing_roots():
    """With only the 27 fixed starts, Newton misses roots, and the oracle sees it."""
    _, _, missed, undecided = check("noise-0", max_seed_starts=0, certify_depth=0)
    assert len(missed) > 0
    assert len(undecided) == 0


def test_certification_alone_finds_every_root():
    """Certification finds every root even when the start pattern is crippled."""
    res, bnull, missed, undecided = check("noise-337", max_seed_starts=0, max_iter=2)
    assert bnull.max() < ROOT_REL
    assert len(missed) == 0
    assert len(undecided) == 0
    assert res["n_cells_uncertified"] == 0
