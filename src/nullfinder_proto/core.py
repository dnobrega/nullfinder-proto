"""
Robust magnetic null finder for rectilinear 3D grids.

Pipeline
--------
1. Field components ``bx, by, bz`` with shape ``(nx, ny, nz)``, accessed as
   ``arr[i, j, k]``.  Any memory layout is accepted without copying; the
   candidate scan picks the loop order that matches the layout.
2. Sign-change candidate cells: all three components change sign (or vanish)
   over the 8 cell vertices.  This is a necessary condition for a root of the
   trilinear interpolant, because a multilinear function attains its extremes
   at the cell vertices.  The scan is a single, vectorised pass over the grid
   that records candidates in a bit mask (1 bit per cell).
3. Exact exclusion by Bernstein subdivision: the 8 vertex values of a
   trilinear cell are its Bernstein coefficients, so the field over any
   sub-box lies in the convex hull of the sub-box vertex values.  Candidate
   cells are recursively bisected and the sign-change test is re-applied; if
   no sub-box survives, the cell provably has no root and Newton is skipped.
4. Robust multi-start Newton refinement of the trilinear field inside each
   surviving cell, with a residual-decreasing line search.  Starts are a
   fixed 27-point pattern plus the centres of the sub-boxes that survive the
   exclusion test, improving recovery of roots near cell faces.  Several
   distinct roots per cell are retained up to ``max_roots_per_cell``; roots
   beyond that storage limit are discarded with a warning.
5. Same-charge duplicate roots (the same null seen from neighbouring cells or
   from different starts) are merged with a local grid-scale tolerance.
6. Diagnostics: topological charge ``sign(det J)``, an adaptive Poincare index
   per null cell, eigen-classification (A, B, As, Bs), spine/fan geometry,
   current and divergence diagnostics, and opposite-charge pair resolution
   measured in grid cells.

Conventions
-----------
* ``x, y, z`` are 1D, finite and strictly monotonic (increasing or decreasing).
* All three field components must be sampled at the same grid nodes
  ``(x[i], y[j], z[k])``; the caller is responsible for ensuring this.
* The Jacobian is ``J[a, b] = dB_a / dx_b`` of the trilinear interpolant of the
  containing cell, in field units per coordinate unit.  Newton, the Jacobian,
  the classification and the spine/fan geometry all use this same local
  representation, so they are mutually consistent.
* All numerical tolerances are relative by default, so results are invariant
  under ``B -> c B`` and under a change of length unit.

References
----------
* Lau, Y.-T. & Finn, J. M. 1990, ApJ 350, 672 (A/B null types).
* Parnell, C. E., Smith, J. M., Neukirch, T. & Priest, E. R. 1996,
  Phys. Plasmas 3, 759 (linear null classification).
* Haynes, M. & Parnell, C. E. 2007, Phys. Plasmas 14, 082107 (trilinear null
  finding).
"""

import math
import time
import warnings

import numba
import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree

__all__ = [
    "NULL_TYPE_A",
    "NULL_TYPE_AS",
    "NULL_TYPE_B",
    "NULL_TYPE_BS",
    "NULL_TYPE_UNKNOWN",
    "POINCARE_DEGENERATE",
    "POINCARE_NOT_COMPUTED",
    "RESOLUTION_DEGENERATE",
    "RESOLUTION_ISOLATED",
    "RESOLUTION_MARGINAL",
    "RESOLUTION_RESOLVED",
    "RESOLUTION_UNRESOLVED",
    "find_magnetic_nulls",
    "null_type_name",
    "pad_periodic",
    "resolution_flag_name",
    "topology_report",
]

# ============================================================
# Public constants
# ============================================================

# Null types (Lau & Finn 1990).  A/As: fan eigenvalues with negative real part
# (field lines approach the null in the fan, leave along the spine);
# B/Bs: the opposite.  With det J = lambda_spine * |lambda_fan|^2 this gives
# charge +1 for A/As and -1 for B/Bs.
NULL_TYPE_UNKNOWN = 0
NULL_TYPE_A = 1  # radial/improper; fan converging, spine diverging
NULL_TYPE_B = 2  # radial/improper; fan diverging, spine converging
NULL_TYPE_AS = 3  # spiral; fan converging
NULL_TYPE_BS = 4  # spiral; fan diverging

# Poincare diagnostic: -1, 0, +1 (or larger in magnitude for several nulls in
# one cell) are computed indices; 99 means |B| = 0 exactly at a sampled
# surface point; -99 means the diagnostic was not computed.
POINCARE_DEGENERATE = 99
POINCARE_NOT_COMPUTED = -99

# Opposite-charge pair resolution flags.
RESOLUTION_RESOLVED = 0
RESOLUTION_MARGINAL = 1
RESOLUTION_UNRESOLVED = 2
RESOLUTION_ISOLATED = 3  # no opposite-charge null in the detected set
RESOLUTION_DEGENERATE = 4  # null charge is 0 (singular Jacobian)

WORK_DTYPE = np.float64
INDEX_DTYPE = np.int64
NUMBA_CACHE = True
FASTMATH = False

# Number of Newton starting points tried per candidate cell (see _start_q).
NEWTON_STARTS_PER_CANDIDATE = 27

_NULL_TYPE_NAMES = {
    NULL_TYPE_UNKNOWN: "unknown",
    NULL_TYPE_A: "A",
    NULL_TYPE_B: "B",
    NULL_TYPE_AS: "As",
    NULL_TYPE_BS: "Bs",
}

_RESOLUTION_NAMES = {
    RESOLUTION_RESOLVED: "resolved",
    RESOLUTION_MARGINAL: "marginal",
    RESOLUTION_UNRESOLVED: "unresolved",
    RESOLUTION_ISOLATED: "isolated",
    RESOLUTION_DEGENERATE: "degenerate",
}


def null_type_name(code):
    """Readable name of a ``NULL_TYPE_*`` code."""
    return _NULL_TYPE_NAMES.get(int(code), "unknown")


def resolution_flag_name(code):
    """Readable name of a ``RESOLUTION_*`` code."""
    return _RESOLUTION_NAMES.get(int(code), "unknown")


# ============================================================
# Input preparation
# ============================================================


def _parse_bool(value, name):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be True or False.")


def _resolve_field_dtype(dtypes, field_dtype):
    """Common floating dtype for the three components (native byte order)."""
    if field_dtype is not None:
        dt = np.dtype(field_dtype)
        if not np.issubdtype(dt, np.floating):
            raise ValueError("field_dtype must be a floating dtype.")
    else:
        dt = np.result_type(*dtypes)
        if not np.issubdtype(dt, np.floating):
            dt = np.dtype(np.float64)
        elif dt == np.dtype(np.float16):
            dt = np.dtype(np.float32)
        if len({np.dtype(d).newbyteorder("=") for d in dtypes}) > 1:
            warnings.warn(
                "bx, by, bz have mixed dtypes "
                f"{tuple(str(d) for d in dtypes)}; converting all three to {dt}. "
                "Pass field_dtype explicitly to control this conversion.",
                RuntimeWarning,
                stacklevel=3,
            )
    return np.dtype(dt).newbyteorder("=")


def _prepare_field_array(a, name, dtype):
    """Cast to ``dtype`` (native byte order) only if needed; keep the layout."""
    a = np.asarray(a)
    if a.ndim != 3:
        raise ValueError(f"{name} must be a 3D array with shape (nx, ny, nz).")
    if a.dtype != dtype:
        a = a.astype(dtype, order="K")
    return a


def _prepare_coord_array(c, name):
    c = np.asarray(c)
    if c.ndim != 1:
        raise ValueError(f"{name} must be a 1D coordinate array.")
    c = np.ascontiguousarray(c, dtype=WORK_DTYPE)
    if c.size < 2:
        raise ValueError(f"{name} must contain at least two nodes.")
    d = np.diff(c)
    if (not np.all(np.isfinite(c))) or not (np.all(d > 0.0) or np.all(d < 0.0)):
        raise ValueError(f"{name} must be finite and strictly monotonic.")
    return c


def _is_c_layout(a):
    return bool(a.flags["C_CONTIGUOUS"] and not a.flags["F_CONTIGUOUS"])


# ============================================================
# 1. Low-level utilities
# ============================================================


@njit(cache=NUMBA_CACHE, fastmath=FASTMATH)
def _solve_3x3_scalars(A, b, det_tol_abs, det_tol_rel):
    """Solve A x = b for a 3x3 A by Cramer's rule. Returns ok, x0, x1, x2, det.

    The determinant threshold is ``max(det_tol_abs, det_tol_rel*||A||_F^3)``;
    the relative part is invariant under rescaling of A.
    """
    a00, a01, a02 = A[0, 0], A[0, 1], A[0, 2]
    a10, a11, a12 = A[1, 0], A[1, 1], A[1, 2]
    a20, a21, a22 = A[2, 0], A[2, 1], A[2, 2]

    det = (
        a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
    )

    anorm = np.sqrt(
        a00 * a00
        + a01 * a01
        + a02 * a02
        + a10 * a10
        + a11 * a11
        + a12 * a12
        + a20 * a20
        + a21 * a21
        + a22 * a22
    )
    det_tol_eff = max(det_tol_abs, det_tol_rel * anorm * anorm * anorm)

    if not (np.abs(det) > det_tol_eff):
        return False, 0.0, 0.0, 0.0, det

    b0, b1, b2 = b[0], b[1], b[2]

    detx = (
        b0 * (a11 * a22 - a12 * a21)
        - a01 * (b1 * a22 - a12 * b2)
        + a02 * (b1 * a21 - a11 * b2)
    )
    dety = (
        a00 * (b1 * a22 - a12 * b2)
        - b0 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * b2 - b1 * a20)
    )
    detz = (
        a00 * (a11 * b2 - b1 * a21)
        - a01 * (a10 * b2 - b1 * a20)
        + b0 * (a10 * a21 - a11 * a20)
    )

    return True, detx / det, dety / det, detz / det, det


@njit(cache=NUMBA_CACHE, fastmath=FASTMATH)
def _start_q(start_idx):
    """Normalised starting point (q0, q1, q2) for a start index in [0, 27).

    0      : cell centre
    1..8   : points q in {0.25, 0.75}^3
    9..26  : remaining points of the 3x3x3 grid q in {0.25, 0.5, 0.75}^3
    """
    if start_idx == 0:
        return 0.5, 0.5, 0.5

    if start_idx <= 8:
        m = start_idx - 1
        qx = 0.25 if (m & 1) == 0 else 0.75
        qy = 0.25 if (m & 2) == 0 else 0.75
        qz = 0.25 if (m & 4) == 0 else 0.75
        return qx, qy, qz

    target = start_idx - 9
    count = 0
    for iz in range(3):
        for iy in range(3):
            for ix in range(3):
                if ix == 1 and iy == 1 and iz == 1:
                    continue  # centre (index 0)
                if ix != 1 and iy != 1 and iz != 1:
                    continue  # corners (indices 1..8)
                if count == target:
                    return 0.25 + 0.25 * ix, 0.25 + 0.25 * iy, 0.25 + 0.25 * iz
                count += 1

    return 0.5, 0.5, 0.5  # not reached for start_idx < 27


# ============================================================
# 2. Trilinear interpolation
# ============================================================


@njit(cache=NUMBA_CACHE, fastmath=FASTMATH, inline="always")
def _trilinear_scalar(a, i, j, k, qx, qy, qz):
    """Trilinear value of one component at normalised cell coordinates q."""
    x0 = 1.0 - qx
    y0 = 1.0 - qy
    z0 = 1.0 - qz
    lower = (a[i, j, k] * x0 + a[i + 1, j, k] * qx) * y0 + (
        a[i, j + 1, k] * x0 + a[i + 1, j + 1, k] * qx
    ) * qy
    upper = (a[i, j, k + 1] * x0 + a[i + 1, j, k + 1] * qx) * y0 + (
        a[i, j + 1, k + 1] * x0 + a[i + 1, j + 1, k + 1] * qx
    ) * qy
    return lower * z0 + upper * qz


@njit(cache=NUMBA_CACHE, fastmath=FASTMATH)
def _trilinear_value_and_jac_cell(bx, by, bz, i, j, k, qx, qy, qz, B, Jq):
    """Trilinear B and its Jacobian dB/dq at normalised cell coordinates q.

    B  : output (3,)
    Jq : output (3, 3), Jq[a, b] = dB_a / dq_b

    Vertex names are c<dz><dy><dx>: c001 is the vertex (i+1, j, k),
    c010 is (i, j+1, k) and c100 is (i, j, k+1).
    """
    x0, x1 = 1.0 - qx, qx
    y0, y1 = 1.0 - qy, qy
    z0, z1 = 1.0 - qz, qz

    for comp in range(3):
        if comp == 0:
            arr = bx
        elif comp == 1:
            arr = by
        else:
            arr = bz

        c000 = arr[i, j, k]
        c001 = arr[i + 1, j, k]
        c010 = arr[i, j + 1, k]
        c011 = arr[i + 1, j + 1, k]
        c100 = arr[i, j, k + 1]
        c101 = arr[i + 1, j, k + 1]
        c110 = arr[i, j + 1, k + 1]
        c111 = arr[i + 1, j + 1, k + 1]

        B[comp] = (
            c000 * x0 * y0 * z0
            + c001 * x1 * y0 * z0
            + c010 * x0 * y1 * z0
            + c011 * x1 * y1 * z0
            + c100 * x0 * y0 * z1
            + c101 * x1 * y0 * z1
            + c110 * x0 * y1 * z1
            + c111 * x1 * y1 * z1
        )

        Jq[comp, 0] = (
            -c000 * y0 * z0
            + c001 * y0 * z0
            - c010 * y1 * z0
            + c011 * y1 * z0
            - c100 * y0 * z1
            + c101 * y0 * z1
            - c110 * y1 * z1
            + c111 * y1 * z1
        )
        Jq[comp, 1] = (
            -c000 * x0 * z0
            - c001 * x1 * z0
            + c010 * x0 * z0
            + c011 * x1 * z0
            - c100 * x0 * z1
            - c101 * x1 * z1
            + c110 * x0 * z1
            + c111 * x1 * z1
        )
        Jq[comp, 2] = (
            -c000 * x0 * y0
            - c001 * x1 * y0
            - c010 * x0 * y1
            - c011 * x1 * y1
            + c100 * x0 * y0
            + c101 * x1 * y0
            + c110 * x0 * y1
            + c111 * x1 * y1
        )


@njit(cache=NUMBA_CACHE, fastmath=FASTMATH)
def _cell_bscale(bx, by, bz, i, j, k):
    """Maximum |B| over the eight cell vertices."""
    bmax = 0.0
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                b0 = bx[i + dx, j + dy, k + dz]
                b1 = by[i + dx, j + dy, k + dz]
                b2 = bz[i + dx, j + dy, k + dz]
                bmax = max(bmax, np.sqrt(b0 * b0 + b1 * b1 + b2 * b2))
    return bmax


@njit(cache=NUMBA_CACHE, fastmath=FASTMATH)
def _cell_is_finite(bx, by, bz, i, j, k):
    """True if all 24 vertex values of the cell are finite."""
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                if not (
                    np.isfinite(bx[i + dx, j + dy, k + dz])
                    and np.isfinite(by[i + dx, j + dy, k + dz])
                    and np.isfinite(bz[i + dx, j + dy, k + dz])
                ):
                    return False
    return True


# ============================================================
# 3. Candidate cells
# ============================================================


# The scan is a single pass over the grid that sets one bit per candidate cell
# in a mask (1 bit per cell, e.g. 134 MB for 1024^3), followed by a cheap pass
# over the mask that lists the candidates.  Along the contiguous axis, the
# min/max of each component over the 2x2 block of nodes in the other two
# directions is computed once per node and shared by the two cells that use it;
# these contiguous row loops allow the compiler to optimise memory access.
# by and bz are only read for rows in which some cell passes the bx test.
# Cells touching non-finite vertices are not included in the candidate list.

_POPCOUNT8 = np.array([bin(v).count("1") for v in range(256)], dtype=np.int64)


@njit(cache=NUMBA_CACHE, fastmath=FASTMATH, inline="always")
def _min4(a, b, c, d):
    return min(min(a, b), min(c, d))


@njit(cache=NUMBA_CACHE, fastmath=FASTMATH, inline="always")
def _max4(a, b, c, d):
    return max(max(a, b), max(c, d))


@njit(parallel=True, cache=NUMBA_CACHE, fastmath=FASTMATH)
def _mark_candidates_f(bx, by, bz, mask):
    """Fortran layout: parallel over z layers k, rows along x.

    Sets bit ``j*(nx-1) + i`` of ``mask[k]`` for every candidate cell (i, j, k).
    """
    nx, ny, nz = bx.shape
    for k in prange(nz - 1):
        mn = np.empty((3, nx))
        mx = np.empty((3, nx))
        okx = np.empty(nx - 1, dtype=np.bool_)
        for j in range(ny - 1):
            for i in range(nx):
                a, b = bx[i, j, k], bx[i, j + 1, k]
                c, d = bx[i, j, k + 1], bx[i, j + 1, k + 1]
                mn[0, i] = _min4(a, b, c, d)
                mx[0, i] = _max4(a, b, c, d)
            anyx = False
            for i in range(nx - 1):
                okx[i] = (
                    min(mn[0, i], mn[0, i + 1]) <= 0.0 <= max(mx[0, i], mx[0, i + 1])
                )
                anyx |= okx[i]
            if not anyx:
                continue
            for i in range(nx):
                a, b = by[i, j, k], by[i, j + 1, k]
                c, d = by[i, j, k + 1], by[i, j + 1, k + 1]
                mn[1, i] = _min4(a, b, c, d)
                mx[1, i] = _max4(a, b, c, d)
                a, b = bz[i, j, k], bz[i, j + 1, k]
                c, d = bz[i, j, k + 1], bz[i, j + 1, k + 1]
                mn[2, i] = _min4(a, b, c, d)
                mx[2, i] = _max4(a, b, c, d)
            base = j * (nx - 1)
            for i in range(nx - 1):
                if (
                    okx[i]
                    and min(mn[1, i], mn[1, i + 1])
                    <= 0.0
                    <= max(mx[1, i], mx[1, i + 1])
                    and min(mn[2, i], mn[2, i + 1])
                    <= 0.0
                    <= max(mx[2, i], mx[2, i + 1])
                    and _cell_is_finite(bx, by, bz, i, j, k)
                ):
                    bit = base + i
                    mask[k, bit >> 3] |= np.uint8(1 << (bit & 7))


@njit(parallel=True, cache=NUMBA_CACHE, fastmath=FASTMATH)
def _mark_candidates_c(bx, by, bz, mask):
    """C layout: parallel over x layers i, rows along z.

    Sets bit ``j*(nz-1) + k`` of ``mask[i]`` for every candidate cell (i, j, k).
    """
    nx, ny, nz = bx.shape
    for i in prange(nx - 1):
        mn = np.empty((3, nz))
        mx = np.empty((3, nz))
        okx = np.empty(nz - 1, dtype=np.bool_)
        for j in range(ny - 1):
            for k in range(nz):
                a, b = bx[i, j, k], bx[i + 1, j, k]
                c, d = bx[i, j + 1, k], bx[i + 1, j + 1, k]
                mn[0, k] = _min4(a, b, c, d)
                mx[0, k] = _max4(a, b, c, d)
            anyx = False
            for k in range(nz - 1):
                okx[k] = (
                    min(mn[0, k], mn[0, k + 1]) <= 0.0 <= max(mx[0, k], mx[0, k + 1])
                )
                anyx |= okx[k]
            if not anyx:
                continue
            for k in range(nz):
                a, b = by[i, j, k], by[i + 1, j, k]
                c, d = by[i, j + 1, k], by[i + 1, j + 1, k]
                mn[1, k] = _min4(a, b, c, d)
                mx[1, k] = _max4(a, b, c, d)
                a, b = bz[i, j, k], bz[i + 1, j, k]
                c, d = bz[i, j + 1, k], bz[i + 1, j + 1, k]
                mn[2, k] = _min4(a, b, c, d)
                mx[2, k] = _max4(a, b, c, d)
            base = j * (nz - 1)
            for k in range(nz - 1):
                if (
                    okx[k]
                    and min(mn[1, k], mn[1, k + 1])
                    <= 0.0
                    <= max(mx[1, k], mx[1, k + 1])
                    and min(mn[2, k], mn[2, k + 1])
                    <= 0.0
                    <= max(mx[2, k], mx[2, k + 1])
                    and _cell_is_finite(bx, by, bz, i, j, k)
                ):
                    bit = base + k
                    mask[i, bit >> 3] |= np.uint8(1 << (bit & 7))


@njit(parallel=True, cache=NUMBA_CACHE)
def _mask_counts(mask):
    counts = np.zeros(mask.shape[0], dtype=np.int64)
    for o in prange(mask.shape[0]):
        c = 0
        for w in range(mask.shape[1]):
            c += _POPCOUNT8[mask[o, w]]
        counts[o] = c
    return counts


@njit(parallel=True, cache=NUMBA_CACHE)
def _mask_to_cells(mask, n_inner, offsets, out, c_layout):
    """List the cells marked in ``mask`` (see _mark_candidates_f/_c)."""
    for o in prange(mask.shape[0]):
        pos = offsets[o]
        for w in range(mask.shape[1]):
            v = mask[o, w]
            if v == 0:
                continue
            for bit in range(8):
                if (v >> bit) & 1:
                    b = w * 8 + bit
                    inner = b % n_inner
                    j = b // n_inner
                    if c_layout:
                        out[pos, 0] = o
                        out[pos, 2] = inner
                    else:
                        out[pos, 0] = inner
                        out[pos, 2] = o
                    out[pos, 1] = j
                    pos += 1


def _get_candidate_cells(bx, by, bz):
    """Return sign-change candidate cells and their count per z layer.

    A cell is a candidate when all 24 field values at its 8 vertices are
    finite and each component changes sign (or vanishes) over the vertices.
    The mask uses approximately one bit per cell; output order is independent
    of the arrays' memory layout.

    Returns
    -------
    candidates : ndarray, shape (Nc, 3), int64
        Lower-corner indices (i, j, k), ordered by k, then j, then i,
        independently of the memory layout of the input.
    counts : ndarray, shape (nz-1,), int64
        Number of candidates per z layer.
    """
    nx, ny, nz = bx.shape
    c_layout = _is_c_layout(bx)
    if c_layout:
        n_outer, n_inner, kernel = nx - 1, nz - 1, _mark_candidates_c
    else:
        n_outer, n_inner, kernel = nz - 1, nx - 1, _mark_candidates_f
    mask = np.zeros((n_outer, (n_inner * (ny - 1) + 7) // 8), dtype=np.uint8)
    kernel(bx, by, bz, mask)
    per_outer = _mask_counts(mask)
    offsets = np.zeros_like(per_outer)
    if per_outer.size > 1:
        offsets[1:] = np.cumsum(per_outer[:-1])
    candidates = np.empty((int(per_outer.sum()), 3), dtype=INDEX_DTYPE)
    _mask_to_cells(mask, n_inner, offsets, candidates, c_layout)
    if c_layout:
        order = np.lexsort((candidates[:, 0], candidates[:, 1], candidates[:, 2]))
        candidates = candidates[order]
        counts = np.bincount(candidates[:, 2], minlength=nz - 1).astype(np.int64)
    else:
        counts = per_outer
    return candidates, counts


# ============================================================
# 4. Exact exclusion by Bernstein subdivision
# ============================================================


@njit(cache=NUMBA_CACHE, fastmath=FASTMATH)
def _subbox_component_may_vanish(a, i, j, k, q0, q1, q2, h, eps):
    lo = np.inf
    hi = -np.inf
    for c in range(8):
        v = _trilinear_scalar(
            a,
            i,
            j,
            k,
            q0 + h * (c & 1),
            q1 + h * ((c >> 1) & 1),
            q2 + h * ((c >> 2) & 1),
        )
        lo = min(lo, v)
        hi = max(hi, v)
    return lo <= eps and hi >= -eps


@njit(cache=NUMBA_CACHE, fastmath=FASTMATH)
def _cell_may_contain_root(bx, by, bz, i, j, k, max_depth, eps):
    """False only if no point of the cell can have |B_c| <= eps for every c.

    Depth-first bisection with an explicit stack (Numba caching does not
    support recursive functions reliably).  Each stack entry is a sub-box
    (q0, q1, q2, h, depth) with lower corner q and edge length h.
    """
    cap = 7 * max_depth + 8
    stack = np.empty((cap, 5), dtype=WORK_DTYPE)
    stack[0, 0] = 0.0
    stack[0, 1] = 0.0
    stack[0, 2] = 0.0
    stack[0, 3] = 1.0
    stack[0, 4] = 0.0
    top = 1
    while top > 0:
        top -= 1
        q0 = stack[top, 0]
        q1 = stack[top, 1]
        q2 = stack[top, 2]
        h = stack[top, 3]
        depth = int(stack[top, 4])
        if not _subbox_component_may_vanish(bx, i, j, k, q0, q1, q2, h, eps):
            continue
        if not _subbox_component_may_vanish(by, i, j, k, q0, q1, q2, h, eps):
            continue
        if not _subbox_component_may_vanish(bz, i, j, k, q0, q1, q2, h, eps):
            continue
        if depth >= max_depth:
            return True
        hh = 0.5 * h
        for c in range(8):
            stack[top, 0] = q0 + hh * (c & 1)
            stack[top, 1] = q1 + hh * ((c >> 1) & 1)
            stack[top, 2] = q2 + hh * ((c >> 2) & 1)
            stack[top, 3] = hh
            stack[top, 4] = depth + 1
            top += 1
    return False


@njit(parallel=True, cache=NUMBA_CACHE, fastmath=FASTMATH)
def _exclusion_filter(bx, by, bz, candidates, depth, b_tol_abs, b_tol_rel):
    """Boolean mask of candidates that may contain a root.

    The margin eps equals the Newton acceptance tolerance of the cell, so a
    cell is excluded only if no point in it could be accepted as a null.
    """
    n = candidates.shape[0]
    keep = np.zeros(n, dtype=np.bool_)
    for m in prange(n):
        i = candidates[m, 0]
        j = candidates[m, 1]
        k = candidates[m, 2]
        eps = b_tol_abs + b_tol_rel * _cell_bscale(bx, by, bz, i, j, k)
        keep[m] = _cell_may_contain_root(bx, by, bz, i, j, k, depth, eps)
    return keep


# ============================================================
# 5. Newton refinement
# ============================================================


@njit(cache=NUMBA_CACHE, fastmath=FASTMATH)
def _refine_null_in_cell_from_start(
    bx,
    by,
    bz,
    x,
    y,
    z,
    i,
    j,
    k,
    q0_start,
    q1_start,
    q2_start,
    max_iter,
    b_tol_eff,
    q_tol,
    det_tol_abs,
    det_tol_rel,
    B,
    Jq,
    Jphys,
    Btrial,
    Jtrial,
):
    """Newton refinement from one start with a residual-decreasing line search.

    Returns ok, x, y, z, qx, qy, qz, |B|.  On success Jphys holds dB/dx.
    """
    q0 = q0_start
    q1 = q1_start
    q2 = q2_start
    success = False
    bnorm = 1.0e99
    rhs = np.empty(3, dtype=WORK_DTYPE)

    for it in range(max_iter + 1):
        _trilinear_value_and_jac_cell(bx, by, bz, i, j, k, q0, q1, q2, B, Jq)
        bnorm = np.sqrt(B[0] * B[0] + B[1] * B[1] + B[2] * B[2])

        if bnorm <= b_tol_eff:
            success = True
            break
        if it == max_iter:
            break

        rhs[0] = -B[0]
        rhs[1] = -B[1]
        rhs[2] = -B[2]
        ok, dq0, dq1, dq2, _det = _solve_3x3_scalars(Jq, rhs, det_tol_abs, det_tol_rel)
        if not ok:
            break

        # Damped step: accept only if it stays in the (tolerated) cell box and
        # strictly reduces |B|.  The Newton direction is a descent direction
        # of |B|^2, so a small enough damping normally succeeds.
        damp = 1.0
        accepted = False
        for _d in range(10):
            qnew0 = q0 + damp * dq0
            qnew1 = q1 + damp * dq1
            qnew2 = q2 + damp * dq2
            if (
                -q_tol <= qnew0 <= 1.0 + q_tol
                and -q_tol <= qnew1 <= 1.0 + q_tol
                and -q_tol <= qnew2 <= 1.0 + q_tol
            ):
                _trilinear_value_and_jac_cell(
                    bx, by, bz, i, j, k, qnew0, qnew1, qnew2, Btrial, Jtrial
                )
                btrial = np.sqrt(
                    Btrial[0] * Btrial[0]
                    + Btrial[1] * Btrial[1]
                    + Btrial[2] * Btrial[2]
                )
                if btrial < bnorm:
                    q0, q1, q2 = qnew0, qnew1, qnew2
                    accepted = True
                    break
            damp *= 0.5

        if not accepted:
            break

    if not success:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, bnorm

    # Report the root inside the closed cell and re-check it there.
    qx = min(1.0, max(0.0, q0))
    qy = min(1.0, max(0.0, q1))
    qz = min(1.0, max(0.0, q2))
    _trilinear_value_and_jac_cell(bx, by, bz, i, j, k, qx, qy, qz, B, Jq)
    bnorm = np.sqrt(B[0] * B[0] + B[1] * B[1] + B[2] * B[2])
    if bnorm > b_tol_eff:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, bnorm

    dx = x[i + 1] - x[i]
    dy = y[j + 1] - y[j]
    dz = z[k + 1] - z[k]
    for comp in range(3):
        Jphys[comp, 0] = Jq[comp, 0] / dx
        Jphys[comp, 1] = Jq[comp, 1] / dy
        Jphys[comp, 2] = Jq[comp, 2] / dz

    return True, x[i] + qx * dx, y[j] + qy * dy, z[k] + qz * dz, qx, qy, qz, bnorm


@njit(cache=NUMBA_CACHE, fastmath=FASTMATH)
def _root_is_new_in_cell(base, nroots, qx, qy, qz, null_q, tol):
    for r in range(nroots):
        idx = base + r
        dq0 = qx - null_q[idx, 0]
        dq1 = qy - null_q[idx, 1]
        dq2 = qz - null_q[idx, 2]
        if np.sqrt(dq0 * dq0 + dq1 * dq1 + dq2 * dq2) <= tol:
            return False
    return True


@njit(cache=NUMBA_CACHE, fastmath=FASTMATH)
def _surviving_leaf_centres(bx, by, bz, i, j, k, max_depth, eps, out):
    """Centres of the depth-``max_depth`` sub-boxes that pass the exclusion test.

    Every root of the cell lies in one of these leaves, so they are good extra
    Newton starts, in particular for roots close to a cell face that the fixed
    27-point pattern may not reach.  Writes at most ``out.shape[0]`` centres
    into ``out`` and returns how many were written.
    """
    cap = 7 * max_depth + 8
    stack = np.empty((cap, 5), dtype=WORK_DTYPE)
    stack[0, 0] = 0.0
    stack[0, 1] = 0.0
    stack[0, 2] = 0.0
    stack[0, 3] = 1.0
    stack[0, 4] = 0.0
    top = 1
    nout = 0
    while top > 0 and nout < out.shape[0]:
        top -= 1
        q0 = stack[top, 0]
        q1 = stack[top, 1]
        q2 = stack[top, 2]
        h = stack[top, 3]
        depth = int(stack[top, 4])
        if not _subbox_component_may_vanish(bx, i, j, k, q0, q1, q2, h, eps):
            continue
        if not _subbox_component_may_vanish(by, i, j, k, q0, q1, q2, h, eps):
            continue
        if not _subbox_component_may_vanish(bz, i, j, k, q0, q1, q2, h, eps):
            continue
        if depth >= max_depth:
            out[nout, 0] = q0 + 0.5 * h
            out[nout, 1] = q1 + 0.5 * h
            out[nout, 2] = q2 + 0.5 * h
            nout += 1
            continue
        hh = 0.5 * h
        for c in range(8):
            stack[top, 0] = q0 + hh * (c & 1)
            stack[top, 1] = q1 + hh * ((c >> 1) & 1)
            stack[top, 2] = q2 + hh * ((c >> 2) & 1)
            stack[top, 3] = hh
            stack[top, 4] = depth + 1
            top += 1
    return nout


@njit(parallel=True, cache=NUMBA_CACHE, fastmath=FASTMATH)
def _refine_candidates(
    bx,
    by,
    bz,
    x,
    y,
    z,
    candidates,
    max_iter,
    b_tol_abs,
    b_tol_rel,
    q_tol,
    det_tol_abs,
    det_tol_rel,
    seed_depth,
    max_seeds,
    max_roots_per_cell,
    root_duplicate_q_tol,
):
    """Multi-start Newton in every candidate cell.

    Starts: the fixed 27-point pattern, then (if ``seed_depth > 0``) the
    centres of up to ``max_seeds`` surviving sub-boxes of the exclusion test.

    Output storage is n_candidates * max_roots_per_cell slots; slot
    ``m*max_roots_per_cell + r`` holds root r of candidate m.
    """
    n = candidates.shape[0]
    nslots = n * max_roots_per_cell

    found_mask = np.zeros(nslots, dtype=np.bool_)
    null_pos = np.full((nslots, 3), np.nan, dtype=WORK_DTYPE)
    null_q = np.full((nslots, 3), np.nan, dtype=WORK_DTYPE)
    bnorms = np.full(nslots, np.nan, dtype=WORK_DTYPE)
    jacs = np.full((nslots, 3, 3), np.nan, dtype=WORK_DTYPE)
    cand_nroots = np.zeros(n, dtype=np.int32)
    cand_attempts = np.zeros(n, dtype=np.int32)
    cand_overflow = np.zeros(n, dtype=np.bool_)

    for m in prange(n):
        i = candidates[m, 0]
        j = candidates[m, 1]
        k = candidates[m, 2]
        base = m * max_roots_per_cell
        # Cells with a NaN/Inf vertex are skipped: an infinite vertex would make
        # the relative tolerance infinite and accept any point.  The field is not
        # validated globally, so this check only costs 24 reads per refined cell.
        if not _cell_is_finite(bx, by, bz, i, j, k):
            continue
        b_tol_eff = b_tol_abs + b_tol_rel * _cell_bscale(bx, by, bz, i, j, k)

        B = np.empty(3, dtype=WORK_DTYPE)
        Jq = np.empty((3, 3), dtype=WORK_DTYPE)
        Jp = np.empty((3, 3), dtype=WORK_DTYPE)
        Bt = np.empty(3, dtype=WORK_DTYPE)
        Jt = np.empty((3, 3), dtype=WORK_DTYPE)

        starts = np.empty(
            (NEWTON_STARTS_PER_CANDIDATE + max_seeds, 3), dtype=WORK_DTYPE
        )
        for s in range(NEWTON_STARTS_PER_CANDIDATE):
            q0, q1, q2 = _start_q(s)
            starts[s, 0] = q0
            starts[s, 1] = q1
            starts[s, 2] = q2
        nstarts = NEWTON_STARTS_PER_CANDIDATE
        if seed_depth > 0 and max_seeds > 0:
            nstarts += _surviving_leaf_centres(
                bx,
                by,
                bz,
                i,
                j,
                k,
                seed_depth,
                b_tol_eff,
                starts[NEWTON_STARTS_PER_CANDIDATE:],
            )
        cand_attempts[m] = nstarts

        nroots = 0
        for s in range(nstarts):
            ok, xn, yn, zn, qx, qy, qz, bn = _refine_null_in_cell_from_start(
                bx,
                by,
                bz,
                x,
                y,
                z,
                i,
                j,
                k,
                starts[s, 0],
                starts[s, 1],
                starts[s, 2],
                max_iter,
                b_tol_eff,
                q_tol,
                det_tol_abs,
                det_tol_rel,
                B,
                Jq,
                Jp,
                Bt,
                Jt,
            )
            if not ok:
                continue
            if not _root_is_new_in_cell(
                base, nroots, qx, qy, qz, null_q, root_duplicate_q_tol
            ):
                continue
            if nroots >= max_roots_per_cell:
                cand_overflow[m] = True
                continue
            slot = base + nroots
            found_mask[slot] = True
            null_pos[slot, 0] = xn
            null_pos[slot, 1] = yn
            null_pos[slot, 2] = zn
            null_q[slot, 0] = qx
            null_q[slot, 1] = qy
            null_q[slot, 2] = qz
            bnorms[slot] = bn
            for a in range(3):
                for b in range(3):
                    jacs[slot, a, b] = Jp[a, b]
            nroots += 1
        cand_nroots[m] = nroots

    return (
        found_mask,
        null_pos,
        null_q,
        bnorms,
        jacs,
        cand_nroots,
        cand_attempts,
        cand_overflow,
    )


def _choose_parallel(refine_parallel, n_tasks, threshold):
    if isinstance(refine_parallel, str):
        m = refine_parallel.lower().strip()
        if m == "auto":
            return n_tasks >= threshold
        if m in ("true", "parallel", "yes", "1"):
            return True
        if m in ("false", "serial", "no", "0"):
            return False
        raise ValueError('refine_parallel must be True, False, or "auto".')
    return bool(refine_parallel)


def _refine_in_chunks(
    bx, by, bz, x, y, z, candidates, chunk_size, parallel, kernel_args
):
    """Run the Newton kernel chunk by chunk and keep only the found roots.

    Peak temporary memory is about ``1.2 kB * chunk_size * max_roots_per_cell/8``
    instead of growing with the total number of candidates.
    """
    max_roots_per_cell = kernel_args[-2]
    n = len(candidates)
    pos, q, bnorm, jac, cell = [], [], [], [], []
    n_with_root = 0
    n_attempts = 0
    n_overflow = 0
    nthreads = numba.get_num_threads()
    try:
        if not parallel:
            numba.set_num_threads(1)
        for start in range(0, n, chunk_size):
            chunk = np.ascontiguousarray(candidates[start : start + chunk_size])
            fm, p, qq, bn, jj, nroots, attempts, over = _refine_candidates(
                bx, by, bz, x, y, z, chunk, *kernel_args
            )
            pos.append(p[fm])
            q.append(qq[fm])
            bnorm.append(bn[fm])
            jac.append(jj[fm])
            cell.append(np.repeat(chunk, max_roots_per_cell, axis=0)[fm])
            n_with_root += int(np.count_nonzero(nroots))
            n_attempts += int(attempts.sum())
            n_overflow += int(np.count_nonzero(over))
    finally:
        numba.set_num_threads(nthreads)

    if pos:
        return (
            np.concatenate(pos),
            np.concatenate(cell).astype(INDEX_DTYPE, copy=False),
            np.concatenate(q),
            np.concatenate(bnorm),
            np.concatenate(jac),
            n_with_root,
            n_attempts,
            n_overflow,
        )
    return (
        np.empty((0, 3)),
        np.empty((0, 3), dtype=INDEX_DTYPE),
        np.empty((0, 3)),
        np.empty(0),
        np.empty((0, 3, 3)),
        0,
        0,
        0,
    )


# ============================================================
# 6. Duplicate removal
# ============================================================


def _local_h_for_cells(cell, x, y, z):
    """min(|dx|, |dy|, |dz|) of each cell given by its lower-corner indices."""
    cell = np.asarray(cell)
    if len(cell) == 0:
        return np.empty(0, dtype=float)
    dx = np.abs(np.diff(x))
    dy = np.abs(np.diff(y))
    dz = np.abs(np.diff(z))
    return np.minimum(np.minimum(dx[cell[:, 0]], dy[cell[:, 1]]), dz[cell[:, 2]])


def _det3_batch(jac):
    """det(J) for a batch with shape (N, 3, 3)."""
    jac = np.asarray(jac, dtype=float)
    if jac.ndim != 3 or jac.shape[0] == 0:
        return np.empty(0, dtype=float)
    return np.linalg.det(jac)


def _remove_duplicate_nulls(
    pos, cell, q, bnorm, jac, x, y, z, duplicate_tol, duplicate_tol_factor
):
    """Merge same-charge roots closer than a local tolerance.

    Two roots are linked when their charges match and their distance is at
    most ``duplicate_tol_factor * min(h_a, h_b)`` (or ``duplicate_tol`` if
    given).  Opposite-charge roots are never merged.  In each connected
    component the root with the smallest residual is kept.
    """
    n = len(pos)
    if n <= 1:
        return pos, cell, q, bnorm, jac, 0

    charge = np.sign(_det3_batch(jac)).astype(int)
    hlocal = _local_h_for_cells(cell, x, y, z)

    if duplicate_tol is None:
        search_radius = float(duplicate_tol_factor) * float(np.max(hlocal))
    else:
        search_radius = float(duplicate_tol)
    if not np.isfinite(search_radius) or search_radius <= 0.0:
        return pos, cell, q, bnorm, jac, 0

    pairs = cKDTree(pos).query_pairs(search_radius, output_type="ndarray")
    if len(pairs) == 0:
        return pos, cell, q, bnorm, jac, 0

    a, b = pairs[:, 0], pairs[:, 1]
    d = np.linalg.norm(pos[a] - pos[b], axis=1)
    if duplicate_tol is None:
        tol = float(duplicate_tol_factor) * np.minimum(hlocal[a], hlocal[b])
    else:
        tol = np.full(len(a), float(duplicate_tol))
    link = (charge[a] == charge[b]) & (d <= tol)
    if not np.any(link):
        return pos, cell, q, bnorm, jac, 0

    parent = np.arange(n)

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for u, v in zip(a[link], b[link], strict=True):
        ru, rv = find(int(u)), find(int(v))
        if ru != rv:
            parent[rv] = ru

    roots = np.array([find(u) for u in range(n)])
    keep = []
    for r in np.unique(roots):
        members = np.where(roots == r)[0]
        keep.append(members[np.argmin(bnorm[members])])
    keep = np.sort(np.asarray(keep, dtype=int))
    return pos[keep], cell[keep], q[keep], bnorm[keep], jac[keep], int(n - len(keep))


# ============================================================
# 7. Poincare index (adaptive)
# ============================================================


@njit(cache=NUMBA_CACHE, fastmath=FASTMATH)
def _solid_angle_triangle(v0, v1, v2):
    """Signed solid angle of a unit-vector triangle seen from the origin
    (Van Oosterom & Strackee 1983).  Positive for counter-clockwise order seen
    from outside the sphere."""
    cx = v1[1] * v2[2] - v1[2] * v2[1]
    cy = v1[2] * v2[0] - v1[0] * v2[2]
    cz = v1[0] * v2[1] - v1[1] * v2[0]
    t = v0[0] * cx + v0[1] * cy + v0[2] * cz
    denom = (
        1.0
        + v0[0] * v1[0]
        + v0[1] * v1[1]
        + v0[2] * v1[2]
        + v1[0] * v2[0]
        + v1[1] * v2[1]
        + v1[2] * v2[2]
        + v2[0] * v0[0]
        + v2[1] * v0[1]
        + v2[2] * v0[2]
    )
    if denom == 0.0:
        if t > 0.0:
            return np.pi
        if t < 0.0:
            return -np.pi
        return 0.0
    return 2.0 * np.arctan2(t, denom)


@njit(cache=NUMBA_CACHE, fastmath=FASTMATH)
def _dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(cache=NUMBA_CACHE, fastmath=FASTMATH)
def _face_unit_b(bx, by, bz, i, j, k, ax, side, u, v):
    """B/|B| at face point (u, v) of the face normal to axis ``ax``.

    Face parametrisation (u, v): ax=0 -> (y, z), ax=1 -> (z, x), ax=2 -> (x, y),
    so that (u, v) counter-clockwise corresponds to the +ax normal.
    Returns the unit vector and a flag that is True if |B| == 0.
    """
    s = float(side)
    if ax == 0:
        qx, qy, qz = s, u, v
    elif ax == 1:
        qx, qy, qz = v, s, u
    else:
        qx, qy, qz = u, v, s
    b0 = _trilinear_scalar(bx, i, j, k, qx, qy, qz)
    b1 = _trilinear_scalar(by, i, j, k, qx, qy, qz)
    b2 = _trilinear_scalar(bz, i, j, k, qx, qy, qz)
    nrm = np.sqrt(b0 * b0 + b1 * b1 + b2 * b2)
    if nrm == 0.0:
        return (0.0, 0.0, 0.0), True
    return (b0 / nrm, b1 / nrm, b2 / nrm), False


@njit(cache=NUMBA_CACHE, fastmath=FASTMATH)
def _push_quad(stack, top, u0, v0, u1, v1, A, Bc, C, D, depth):
    stack[top, 0] = u0
    stack[top, 1] = v0
    stack[top, 2] = u1
    stack[top, 3] = v1
    for c in range(3):
        stack[top, 4 + c] = A[c]
        stack[top, 7 + c] = Bc[c]
        stack[top, 10 + c] = C[c]
        stack[top, 13 + c] = D[c]
    stack[top, 16] = depth
    return top + 1


@njit(cache=NUMBA_CACHE, fastmath=FASTMATH)
def _face_solid_angle(bx, by, bz, i, j, k, ax, side, max_depth, cos_tol):
    """Solid angle swept by B/|B| over one cell face, oriented by the +ax normal.

    The face is split into quads with corners A(u0,v0), B(u1,v0), C(u1,v1),
    D(u0,v1); a quad is bisected while any two corner directions differ by
    more than acos(cos_tol), down to ``max_depth``.  Explicit stack instead of
    recursion (Numba caching does not support recursive functions reliably).
    Returns (omega, degenerate), degenerate meaning |B| == 0 at a sample.
    """
    A, dA = _face_unit_b(bx, by, bz, i, j, k, ax, side, 0.0, 0.0)
    Bc, dB = _face_unit_b(bx, by, bz, i, j, k, ax, side, 1.0, 0.0)
    C, dC = _face_unit_b(bx, by, bz, i, j, k, ax, side, 1.0, 1.0)
    D, dD = _face_unit_b(bx, by, bz, i, j, k, ax, side, 0.0, 1.0)
    if dA or dB or dC or dD:
        return 0.0, True

    stack = np.empty((3 * max_depth + 5, 17), dtype=WORK_DTYPE)
    top = _push_quad(stack, 0, 0.0, 0.0, 1.0, 1.0, A, Bc, C, D, 0.0)
    omega = 0.0
    while top > 0:
        top -= 1
        u0 = stack[top, 0]
        v0 = stack[top, 1]
        u1 = stack[top, 2]
        v1 = stack[top, 3]
        A = (stack[top, 4], stack[top, 5], stack[top, 6])
        Bc = (stack[top, 7], stack[top, 8], stack[top, 9])
        C = (stack[top, 10], stack[top, 11], stack[top, 12])
        D = (stack[top, 13], stack[top, 14], stack[top, 15])
        depth = int(stack[top, 16])

        dmin = min(
            min(min(_dot3(A, Bc), _dot3(Bc, C)), min(_dot3(C, D), _dot3(D, A))),
            min(_dot3(A, C), _dot3(Bc, D)),
        )
        if depth >= max_depth or dmin >= cos_tol:
            omega += _solid_angle_triangle(A, Bc, C) + _solid_angle_triangle(A, C, D)
            continue

        um = 0.5 * (u0 + u1)
        vm = 0.5 * (v0 + v1)
        E, dE = _face_unit_b(bx, by, bz, i, j, k, ax, side, um, v0)
        F, dF = _face_unit_b(bx, by, bz, i, j, k, ax, side, u1, vm)
        G, dG = _face_unit_b(bx, by, bz, i, j, k, ax, side, um, v1)
        H, dH = _face_unit_b(bx, by, bz, i, j, k, ax, side, u0, vm)
        M, dM = _face_unit_b(bx, by, bz, i, j, k, ax, side, um, vm)
        if dE or dF or dG or dH or dM:
            return 0.0, True
        d1 = depth + 1.0
        top = _push_quad(stack, top, u0, v0, um, vm, A, E, M, H, d1)
        top = _push_quad(stack, top, um, v0, u1, vm, E, Bc, F, M, d1)
        top = _push_quad(stack, top, um, vm, u1, v1, M, F, C, G, d1)
        top = _push_quad(stack, top, u0, vm, um, v1, H, M, G, D, d1)
    return omega, False


@njit(cache=NUMBA_CACHE, fastmath=FASTMATH)
def _poincare_index_cell(bx, by, bz, x, y, z, i, j, k, max_depth, cos_tol):
    """Degree of B/|B| over the cell surface, in physical orientation.

    Each face is refined adaptively until neighbouring sample directions
    differ by less than acos(cos_tol) (or ``max_depth`` is reached).  The
    result is multiplied by sign(dx*dy*dz) so that it equals sign(det J) of
    the physical Jacobian also for decreasing coordinate axes.
    """
    omega = 0.0
    for ax in range(3):
        for side in range(2):
            w, dg = _face_solid_angle(bx, by, bz, i, j, k, ax, side, max_depth, cos_tol)
            if dg:
                return POINCARE_DEGENERATE
            if side == 1:
                omega += w
            else:
                omega -= w

    orient = (x[i + 1] - x[i]) * (y[j + 1] - y[j]) * (z[k + 1] - z[k])
    index = int(np.round(omega / (4.0 * np.pi)))
    if orient < 0.0:
        index = -index
    return index


@njit(parallel=True, cache=NUMBA_CACHE, fastmath=FASTMATH)
def _compute_poincare_for_cells(bx, by, bz, x, y, z, cells, max_depth, cos_tol):
    n = cells.shape[0]
    pidx = np.zeros(n, dtype=np.int32)
    for m in prange(n):
        pidx[m] = _poincare_index_cell(
            bx,
            by,
            bz,
            x,
            y,
            z,
            cells[m, 0],
            cells[m, 1],
            cells[m, 2],
            max_depth,
            cos_tol,
        )
    return pidx


# ============================================================
# 8. Classification and current/divergence diagnostics
# ============================================================


def _unit_rows(v):
    n = np.linalg.norm(v, axis=1)
    out = np.full_like(v, np.nan)
    ok = np.isfinite(n) & (n > 0.0)
    out[ok] = v[ok] / n[ok, None]
    return out


def _empty_classification(N):
    return {
        "null_type": np.full(N, NULL_TYPE_UNKNOWN, dtype=np.int8),
        "eigenvalues": np.full((N, 3), np.nan + 0.0j, dtype=complex),
        "spine_vec": np.full((N, 3), np.nan),
        "fan_normal": np.full((N, 3), np.nan),
        "div_residual": np.full(N, np.nan),
        "div_warning": np.zeros(N, dtype=bool),
        "current_ratio": np.full(N, np.nan),
        "curlB_vec": np.full((N, 3), np.nan),
        "j_parallel": np.full(N, np.nan),
        "j_perp": np.full(N, np.nan),
        "j_parallel_ratio": np.full(N, np.nan),
        "j_perp_ratio": np.full(N, np.nan),
        "current_spine_angle_deg": np.full(N, np.nan),
        "is_potential": np.zeros(N, dtype=bool),
        "fan_is_perpendicular": np.zeros(N, dtype=bool),
    }


def _classify_nulls(
    jacs, current_tol=1.0e-8, div_tol_rel=0.1, classify_traceless=False
):
    """Classify nulls from the physical Jacobian J = dB_i/dx_j (vectorised).

    A null is classified when its eigenvalues have a clear 2+1 structure: two
    fan eigenvalues (a real pair, or a complex-conjugate pair for spiral
    nulls) whose real parts share one sign, and a real spine eigenvalue of the
    opposite sign.  Anything else (e.g. all real parts of one sign because of
    a large divergence, or a zero eigenvalue) is NULL_TYPE_UNKNOWN.

    Eigenvectors have arbitrary sign, so ``spine_vec`` and ``fan_normal``
    define axes, not oriented directions.  Non-finite Jacobians stay unknown.
    """
    J = np.asarray(jacs, dtype=float)
    N = len(J)
    out = _empty_classification(N)
    if N == 0:
        return out

    fin = np.where(np.isfinite(J).all(axis=(1, 2)))[0]
    if fin.size == 0:
        return out
    Jf = J[fin]

    frob = np.linalg.norm(Jf, axis=(1, 2))
    trace = np.trace(Jf, axis1=1, axis2=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["div_residual"][fin] = np.where(frob > 0.0, np.abs(trace) / frob, np.nan)
    dr = out["div_residual"]
    out["div_warning"][:] = np.isfinite(dr) & (dr > div_tol_rel)

    curl = np.stack(
        [
            Jf[:, 2, 1] - Jf[:, 1, 2],
            Jf[:, 0, 2] - Jf[:, 2, 0],
            Jf[:, 1, 0] - Jf[:, 0, 1],
        ],
        axis=1,
    )
    cn = np.linalg.norm(curl, axis=1)
    out["curlB_vec"][fin] = curl
    out["current_ratio"][fin] = cn / np.maximum(frob, 1.0e-300)
    cr = out["current_ratio"]
    out["is_potential"][:] = np.isfinite(cr) & (cr < current_tol)

    Jc = Jf - (trace[:, None, None] / 3.0) * np.eye(3) if classify_traceless else Jf

    # Eigen-decomposition (batched; per-matrix fallback if LAPACK fails).
    nf_ = len(fin)
    ev = np.full((nf_, 3), np.nan + 0.0j)
    vec = np.full((nf_, 3, 3), np.nan + 0.0j)
    lev = np.full((nf_, 3), np.nan + 0.0j)
    lvec = np.full((nf_, 3, 3), np.nan + 0.0j)
    ok = np.zeros(nf_, dtype=bool)
    try:
        ev[:], vec[:] = np.linalg.eig(Jc)
        lev[:], lvec[:] = np.linalg.eig(np.swapaxes(Jc, 1, 2))
        ok[:] = True
    except np.linalg.LinAlgError:
        for m in range(nf_):
            try:
                ev[m], vec[m] = np.linalg.eig(Jc[m])
                lev[m], lvec[m] = np.linalg.eig(Jc[m].T)
                ok[m] = True
            except np.linalg.LinAlgError:
                pass
    out["eigenvalues"][fin] = ev

    re = ev.real
    im = ev.imag
    scale = np.max(np.abs(ev), axis=1)
    tol = 1.0e-10 * np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0e-300)
    cmask = np.abs(im) > tol[:, None]
    ncomplex = cmask.sum(axis=1)

    spine_idx = np.full(nf_, -1)
    fan_idx = np.full(nf_, -1)
    spiral = ok & (ncomplex == 2)
    spine_idx[spiral] = np.argmin(cmask[spiral], axis=1)
    fan_idx[spiral] = np.argmax(cmask[spiral], axis=1)

    signs = np.where(re > tol[:, None], 1, np.where(re < -tol[:, None], -1, 0))
    ssum = signs.sum(axis=1)
    real_ok = ok & (ncomplex == 0) & np.all(signs != 0, axis=1) & (np.abs(ssum) == 1)
    spine_idx[real_ok] = np.argmax(signs[real_ok] == -ssum[real_ok, None], axis=1)
    fan_idx[real_ok] = np.argmax(signs[real_ok] == ssum[real_ok, None], axis=1)

    rows = np.where(spine_idx >= 0)[0]
    fan_re = re[rows, fan_idx[rows]]
    spine_re = re[rows, spine_idx[rows]]
    t = tol[rows]
    converging = (fan_re < -t) & (spine_re > t)
    diverging = (fan_re > t) & (spine_re < -t)
    good = converging | diverging
    rows, converging, diverging = rows[good], converging[good], diverging[good]
    sp = spiral[rows]

    ntype = np.full(nf_, NULL_TYPE_UNKNOWN, dtype=np.int8)
    ntype[rows[converging & ~sp]] = NULL_TYPE_A
    ntype[rows[converging & sp]] = NULL_TYPE_AS
    ntype[rows[diverging & ~sp]] = NULL_TYPE_B
    ntype[rows[diverging & sp]] = NULL_TYPE_BS
    out["null_type"][fin] = ntype

    if rows.size:
        sidx = spine_idx[rows]
        spine = _unit_rows(np.real(vec[rows, :, sidx]))
        lam = ev[rows, sidx]
        match = np.argmin(np.abs(lev[rows] - lam[:, None]), axis=1)
        fan_n = _unit_rows(np.real(lvec[rows, :, match]))
        g = fin[rows]
        out["spine_vec"][g] = spine
        out["fan_normal"][g] = fan_n

        c = curl[rows]
        c_n = cn[rows]
        jp = np.einsum("ij,ij->i", c, spine)
        jperp = np.linalg.norm(c - jp[:, None] * spine, axis=1)
        has_j = c_n > 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            out["j_parallel"][g] = jp
            out["j_perp"][g] = jperp
            out["j_parallel_ratio"][g] = np.where(has_j, np.abs(jp) / c_n, np.nan)
            out["j_perp_ratio"][g] = np.where(has_j, jperp / c_n, np.nan)
            out["current_spine_angle_deg"][g] = np.where(
                has_j, np.degrees(np.arctan2(jperp, np.abs(jp))), np.nan
            )
        # Potential nulls have a symmetric J, so the fan is perpendicular to
        # the spine; otherwise the fan is perpendicular when the current is
        # aligned with the spine.
        perp = out["is_potential"][g] | (has_j & (out["j_perp_ratio"][g] < current_tol))
        out["fan_is_perpendicular"][g] = perp

    return out


# ============================================================
# 9. Opposite-charge resolution diagnostics
# ============================================================


def _add_resolution_diagnostics(
    res, x, y, z, resolved_threshold=1.0, marginal_threshold=0.5
):
    """Nearest opposite-charge null of each null, with distances in grid cells.

    Distances are measured in fractional index space, ``cell + q``, so a
    separation of 1.0 means one grid cell regardless of how anisotropic or
    stretched the grid is.  Physical distances are reported as well.

    Flags (per null and per pair):
      resolved   : d_cells >= resolved_threshold
      marginal   : marginal_threshold <= d_cells < resolved_threshold
      unresolved : d_cells < marginal_threshold
      isolated   : no opposite-charge null in the detected set
      degenerate : the null itself has charge 0 (singular Jacobian)
    """
    pos = np.asarray(res["pos"], dtype=float).reshape(-1, 3)
    cell = np.asarray(res["cell"]).reshape(-1, 3)
    q = np.asarray(res["q"], dtype=float).reshape(-1, 3)
    N = len(pos)
    charge = np.asarray(res["null_charge"], dtype=int)

    frac = cell.astype(float) + q
    nearest = np.full(N, -1, dtype=INDEX_DTYPE)
    d_cells = np.full(N, np.inf)
    for c in (1, -1):
        src = np.where(charge == c)[0]
        tgt = np.where(charge == -c)[0]
        if src.size and tgt.size:
            d, m = cKDTree(frac[tgt]).query(frac[src], k=1)
            nearest[src] = tgt[m]
            d_cells[src] = d
    has = nearest >= 0
    d_phys = np.full(N, np.inf)
    d_phys[has] = np.linalg.norm(pos[has] - pos[nearest[has]], axis=1)

    def flags_for(d):
        return np.where(
            d < marginal_threshold,
            RESOLUTION_UNRESOLVED,
            np.where(d < resolved_threshold, RESOLUTION_MARGINAL, RESOLUTION_RESOLVED),
        )

    flag = np.full(N, RESOLUTION_ISOLATED, dtype=np.int8)
    flag[has] = flags_for(d_cells[has])
    flag[charge == 0] = RESOLUTION_DEGENERATE

    # Unique nearest-opposite pairs (not a one-to-one matching).
    if np.any(has):
        idx = np.where(has)[0]
        pr = np.unique(np.sort(np.stack([idx, nearest[idx]], axis=1), axis=1), axis=0)
    else:
        pr = np.empty((0, 2), dtype=INDEX_DTYPE)
    pr = pr.astype(INDEX_DTYPE)
    pd_cells = (
        np.linalg.norm(frac[pr[:, 0]] - frac[pr[:, 1]], axis=1)
        if len(pr)
        else np.empty(0)
    )
    pd_phys = (
        np.linalg.norm(pos[pr[:, 0]] - pos[pr[:, 1]], axis=1)
        if len(pr)
        else np.empty(0)
    )
    pflag = (
        flags_for(pd_cells).astype(np.int8) if len(pr) else np.empty(0, dtype=np.int8)
    )

    pairs = [
        {
            "i": int(a),
            "j": int(b),
            "charge_i": int(charge[a]),
            "charge_j": int(charge[b]),
            "distance": float(dp),
            "distance_cells": float(dc),
            "resolution_flag": int(f),
            "resolution_name": resolution_flag_name(f),
        }
        for (a, b), dp, dc, f in zip(pr, pd_phys, pd_cells, pflag, strict=True)
    ]

    n_pair_unres = int(np.sum(pflag == RESOLUTION_UNRESOLVED))
    n_pair_marg = int(np.sum(pflag == RESOLUTION_MARGINAL))
    if N == 0:
        snapshot = "no_nulls"
    elif len(pr) == 0:
        snapshot = "isolated"
    elif n_pair_unres:
        snapshot = "unresolved"
    elif n_pair_marg:
        snapshot = "marginal"
    else:
        snapshot = "resolved"

    finite = d_cells[np.isfinite(d_cells)]
    res.update(
        {
            "hlocal_min": _local_h_for_cells(cell, x, y, z),
            "nearest_opposite_index": nearest,
            "nearest_opposite_distance": d_phys,
            "nearest_opposite_distance_cells": d_cells,
            "resolution_flag": flag,
            "resolution_name": np.array(
                [resolution_flag_name(f) for f in flag], dtype="<U10"
            ),
            "opposite_pairs": pairs,
            "opposite_pair_index": pr,
            "opposite_pair_distance_cells": pd_cells,
            "n_resolved": int(np.sum(flag == RESOLUTION_RESOLVED)),
            "n_marginal": int(np.sum(flag == RESOLUTION_MARGINAL)),
            "n_unresolved": int(np.sum(flag == RESOLUTION_UNRESOLVED)),
            "n_isolated": int(np.sum(flag == RESOLUTION_ISOLATED)),
            "n_degenerate": int(np.sum(flag == RESOLUTION_DEGENERATE)),
            "n_opposite_pairs": len(pr),
            "n_pair_resolved": int(np.sum(pflag == RESOLUTION_RESOLVED)),
            "n_pair_marginal": n_pair_marg,
            "n_pair_unresolved": n_pair_unres,
            "min_opposite_distance_cells": float(finite.min())
            if finite.size
            else np.inf,
            "snapshot_resolution": snapshot,
            "resolution_thresholds": {
                "resolved_threshold": float(resolved_threshold),
                "marginal_threshold": float(marginal_threshold),
            },
        }
    )
    return res


# ============================================================
# 10. Reports and helpers
# ============================================================


def _count_values(arr):
    arr = np.asarray(arr)
    if arr.size == 0:
        return {}
    vals, counts = np.unique(arr, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts, strict=True)}


def topology_report(
    res,
    expected_charge=None,
    use="detJ",
    verbose=True,
    show_resolution=True,
    show_null_table=False,
    max_rows=50,
    show_pairs=False,
    max_pairs=50,
):
    """Return (and optionally print) a compact topology/resolution summary.

    Parameters
    ----------
    res : dict
        Output of :func:`find_magnetic_nulls`.
    expected_charge : int, optional
        If given, compare it with the measured total charge.
    use : {"detJ", "poincare"}
        Which charge measure ``expected_charge`` is compared with.  The
        Poincare sum is a cell-level diagnostic, not an exact verification.
    """
    use_l = str(use).lower()
    if use_l not in ("detj", "poincare"):
        raise ValueError('use must be "detJ" or "poincare".')

    n = len(res.get("pos", []))
    detJ = np.asarray(res.get("detJ", np.empty(0)), dtype=float)
    if detJ.shape != (n,):
        detJ = _det3_batch(res["jacobian"])
    sign_detJ = np.sign(detJ).astype(int)
    p = np.asarray(res.get("poincare", np.full(n, POINCARE_NOT_COMPUTED)), dtype=int)
    if p.shape != (n,):
        p = np.full(n, POINCARE_NOT_COMPUTED, dtype=int)
    poincare_computed = bool(p.size) and not np.all(p == POINCARE_NOT_COMPUTED)
    preg = p[(p != POINCARE_DEGENERATE) & (p != POINCARE_NOT_COMPUTED)]

    out = {
        "n_nulls": n,
        "detJ": detJ,
        "sign_detJ": sign_detJ,
        "sign_detJ_counts": _count_values(sign_detJ),
        "sum_sign_detJ": int(sign_detJ.sum()),
        "poincare": p,
        "poincare_counts": _count_values(p),
        "poincare_computed": poincare_computed,
        "sum_poincare_regular": int(preg.sum()),
        "null_type_counts": {
            null_type_name(k): v
            for k, v in _count_values(res.get("null_type", [])).items()
        },
    }
    if poincare_computed:
        nonzero = (p == 1) | (p == -1)
        out["poincare_agreement"] = (
            float(np.mean(p[nonzero] == sign_detJ[nonzero]))
            if nonzero.any()
            else np.nan
        )

    if expected_charge is not None:
        out["expected_charge"] = int(expected_charge)
        if use_l == "poincare":
            out["charge_source_note"] = (
                "Poincare sum is a cell-level diagnostic, not an exact verification."
            )
            out["charge_ok"] = (
                (out["sum_poincare_regular"] == int(expected_charge))
                if poincare_computed
                else None
            )
        else:
            out["charge_ok"] = out["sum_sign_detJ"] == int(expected_charge)

    res_keys = (
        "snapshot_resolution",
        "min_opposite_distance_cells",
        "n_resolved",
        "n_marginal",
        "n_unresolved",
        "n_isolated",
        "n_degenerate",
        "n_opposite_pairs",
        "n_pair_resolved",
        "n_pair_marginal",
        "n_pair_unresolved",
    )
    has_res = "snapshot_resolution" in res
    if has_res:
        out.update({key: res[key] for key in res_keys})

    if verbose:
        print(f"N nulls: {n}")
        print("topology:")
        print(f"  sign(detJ) counts: {out['sign_detJ_counts']}")
        print(f"  total charge:      {out['sum_sign_detJ']}")
        print(f"  null types:        {out['null_type_counts']}")
        if poincare_computed:
            print(f"  Poincare counts:   {out['poincare_counts']}")
            print(f"  Poincare sum:      {out['sum_poincare_regular']} (regular)")
            n_nonzero = int(np.count_nonzero((p == 1) | (p == -1)))
            if n_nonzero:
                print(
                    "  Poincare agreement: "
                    f"{out['poincare_agreement']:.1%} "
                    f"({n_nonzero} nonzero indices)"
                )
            else:
                print("  Poincare agreement: n/a (no nonzero indices)")
        else:
            print("  Poincare: not computed")
        if expected_charge is not None:
            print(f"  expected charge:  {expected_charge} (OK: {out['charge_ok']})")
        if show_resolution and has_res:
            print("pair resolution:")
            print(f"  status:             {res['snapshot_resolution']}")
            print(
                "  min opposite d:     "
                f"{res['min_opposite_distance_cells']:.5f} grid cells"
            )
            print(
                f"  nearest-opposite links: {res['n_opposite_pairs']} unique "
                f"(resolved={res['n_pair_resolved']}, "
                f"marginal={res['n_pair_marginal']}, "
                f"unresolved={res['n_pair_unresolved']})"
            )
            print(
                f"  per-null counts: resolved={res['n_resolved']}, "
                f"marginal={res['n_marginal']}, "
                f"unresolved={res['n_unresolved']}, "
                f"isolated={res['n_isolated']}, "
                f"degenerate={res['n_degenerate']}"
            )
        if show_null_table and n > 0 and has_res:
            nshow = min(int(max_rows), n)
            print(f"\nnull table: showing {nshow}/{n}")
            print("   n | charge | type    | nearest opp | d [cells] | resolution")
            for i in range(nshow):
                m = int(res["nearest_opposite_index"][i])
                d = res["nearest_opposite_distance_cells"][i]
                tname = null_type_name(res["null_type"][i])
                mstr = "-" if m < 0 else str(m)
                dstr = "inf" if not np.isfinite(d) else f"{d:9.3f}"
                print(
                    f"{i:4d} | {sign_detJ[i]:+6d} | {tname:7s} | {mstr:>11s} | "
                    f"{dstr:>9s} | {res['resolution_name'][i]}"
                )
            if nshow < n:
                print(f"... {n - nshow} additional nulls not shown")
        pairs = res.get("opposite_pairs", [])
        if show_pairs and pairs:
            nshow = min(int(max_pairs), len(pairs))
            print(f"\nnearest opposite-charge pairs: showing {nshow}/{len(pairs)}")
            for pr in pairs[:nshow]:
                print(
                    f"  {pr['i']}-{pr['j']}: "
                    f"charges {pr['charge_i']:+d}/{pr['charge_j']:+d}, "
                    f"d = {pr['distance_cells']:.3f} cells, {pr['resolution_name']}"
                )
            if nshow < len(pairs):
                print(f"... {len(pairs) - nshow} additional pairs not shown")
    return out


def pad_periodic(bx, by, bz, x, y, z, axes=(0, 1), periods=None):
    """Append one periodic image layer along each axis in ``axes``.

    The null finder only scans cells between existing nodes, so the seam cells
    joining the last and the first node of a periodic axis are skipped.  Pad
    with this helper before calling :func:`find_magnetic_nulls` to include them.

    Parameters
    ----------
    bx, by, bz : ndarray, shape (nx, ny, nz)
    x, y, z : 1D ndarray
    axes : iterable of int
        Periodic axes (0 = x, 1 = y, 2 = z).  Default: x and y.
    periods : dict, optional
        ``{axis: L}`` period lengths.  If an axis is missing, the grid along it
        must be uniform and ``L = n * (c[1] - c[0])`` is used.

    Returns
    -------
    bx, by, bz, x, y, z : padded copies.

    Notes
    -----
    Nulls found in seam cells have coordinates between ``c[-1]`` and
    ``c[0] + L``, i.e. a valid periodic image.  Wrap them with
    ``(pos - c[0]) % L + c[0]`` if needed.  A null lying exactly on the node
    plane ``c[0]`` could be reported twice (at ``c[0]`` and ``c[0] + L``).
    """
    comps = [np.asarray(bx), np.asarray(by), np.asarray(bz)]
    coords = [
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(z, dtype=float),
    ]
    periods = dict(periods or {})
    f_order = not _is_c_layout(comps[0])
    for ax in sorted({int(a) for a in axes}):
        if ax not in (0, 1, 2):
            raise ValueError("axes must contain only 0, 1 or 2.")
        c = coords[ax]
        if ax in periods:
            # Same sign as the axis direction, so the padded axis stays monotonic.
            L = abs(float(periods[ax])) * np.sign(c[1] - c[0])
        else:
            step = c[1] - c[0]
            if not np.allclose(np.diff(c), step, rtol=1e-6, atol=0.0):
                raise ValueError(
                    f"axis {ax} is not uniform; pass its period length in `periods`."
                )
            L = c.size * step
        coords[ax] = np.append(c, c[0] + L)
        comps = [np.concatenate([a, a.take([0], axis=ax)], axis=ax) for a in comps]
    if f_order:
        comps = [np.asfortranarray(a) for a in comps]
    return comps[0], comps[1], comps[2], coords[0], coords[1], coords[2]


# ============================================================
# 11. Public API
# ============================================================


def find_magnetic_nulls(
    bx,
    by,
    bz,
    x,
    y,
    z,
    *,
    max_iter=30,
    b_tol=0.0,
    b_tol_rel=1.0e-12,
    q_tol=1.0e-6,
    det_tol=0.0,
    det_tol_rel=1.0e-14,
    exclusion_depth=3,
    max_seed_starts=64,
    poincare_diag=True,
    poincare_max_depth=6,
    poincare_max_angle_deg=45.0,
    remove_duplicates=True,
    duplicate_tol=None,
    duplicate_tol_factor=1.0e-3,
    classify=True,
    current_tol=1.0e-8,
    div_tol_rel=0.1,
    classify_traceless=False,
    field_dtype=None,
    max_roots_per_cell=8,
    root_duplicate_q_tol=1.0e-6,
    refine_parallel="auto",
    parallel_threshold=500,
    chunk_size=200_000,
    add_resolution_diagnostics=True,
    resolution_resolved_threshold=1.0,
    resolution_marginal_threshold=0.5,
    verbose=True,
):
    """
    Locate and classify the magnetic nulls of a field sampled on a rectilinear grid.

    Parameters
    ----------
    bx, by, bz : ndarray, shape (nx, ny, nz)
        Field components sampled at the same grid nodes. Any memory layout
        and floating dtype is accepted without copying; components are only cast if
        their dtypes differ or ``field_dtype`` asks for it.
    x, y, z : 1D ndarray
        Node coordinates, sizes nx, ny, nz; finite and strictly monotonic
        (increasing or decreasing).
    max_iter : int
        Maximum Newton iterations per starting point.
    b_tol, b_tol_rel : float
        Newton acceptance: ``|B| <= b_tol + b_tol_rel * Bscale``, where Bscale
        is the largest vertex |B| of the cell.  ``b_tol`` is absolute (field
        units) and 0 by default, which keeps the result scale invariant.
    q_tol : float
        Margin around the normalised cell box [0, 1]^3 allowed during Newton.
    det_tol, det_tol_rel : float
        Singular-Jacobian thresholds for the Newton step:
        ``|det| > max(det_tol, det_tol_rel * ||Jq||_F^3)``.  ``det_tol`` is
        absolute and 0 by default.
    exclusion_depth : int
        Bisection depth of the exact Bernstein exclusion test applied to the
        sign-change candidates before Newton (0 disables it).  Excluded cells
        provably contain no point with |B| below the acceptance tolerance.
        The same bisection also seeds Newton (see ``max_seed_starts``).
    max_seed_starts : int
        Besides the fixed 27-point start pattern, Newton is also started from
        the centres of up to this many sub-boxes of depth ``exclusion_depth``
        that survive the exclusion test.  Every root lies in one of them, so
        this recovers roots near cell faces that the fixed pattern can miss
        (e.g. the second root of a close pair).  0 disables it.
    poincare_diag : bool
        Compute the Poincare index of each null cell as a diagnostic.
    poincare_max_depth : int
        Maximum adaptive refinement depth per cell face (4**depth quads).
    poincare_max_angle_deg : float
        A face quad is refined while any two of its corner field directions
        differ by more than this angle.
    remove_duplicates : bool
        Merge same-charge roots of the same null (from neighbouring cells or
        different starts).
    duplicate_tol : float or None
        Absolute merge distance (coordinate units).  If None, use
        ``duplicate_tol_factor * min(h_a, h_b)``, with h the smallest local
        grid spacing of each root's cell.
    duplicate_tol_factor : float
        Local merge-distance factor used when ``duplicate_tol`` is None.
    classify : bool
        Compute eigenvalues, null type, spine/fan geometry and current
        diagnostics.  Divergence diagnostics are computed in any case.
    current_tol : float
        Relative threshold for ``is_potential`` and ``fan_is_perpendicular``.
    div_tol_rel : float
        ``div_warning`` threshold on ``|tr J| / ||J||_F``.  Note that the
        trilinear interpolant is not pointwise solenoidal even for discretely
        solenoidal data; values of a few 1e-2 are normal in resolved
        turbulence-like fields.
    classify_traceless : bool
        Classify with ``J - tr(J) I / 3``.  Charge and current diagnostics
        always use the raw J.
    field_dtype : numpy dtype or None
        Force the working field dtype.  None keeps the common input dtype
        (float32 stays float32).  Newton always works in float64.
    max_roots_per_cell : int
        Maximum number of distinct roots retained in one candidate cell
        (default 8). This is a storage limit, not a physical assumption.
        If extra distinct roots are detected, they are discarded, a
        ``RuntimeWarning`` is issued, and ``n_cells_root_overflow`` counts
        the affected cells. Reaching a user-set cap does not by itself imply
        a degenerate magnetic null.
    root_duplicate_q_tol : float
        Distance in normalised cell coordinates below which roots found from
        different starts in one cell are the same root.
    refine_parallel : bool or "auto"
        Multithreaded Newton refinement.  "auto" uses threads when
        ``n_newton_start_slots >= parallel_threshold``.
    parallel_threshold : int
        Threshold for "auto".
    chunk_size : int
        Candidates refined per batch.  Temporary memory is about
        ``153 bytes * chunk_size * max_roots_per_cell``.
    add_resolution_diagnostics : bool
        Add nearest opposite-charge diagnostics (see Returns).
    resolution_resolved_threshold, resolution_marginal_threshold : float
        Thresholds in grid cells (see Returns).
    verbose : bool
        Print a compact summary.

    Returns
    -------
    dict
        Per-null arrays have length N (unique nulls).  The key set depends
        only on ``add_resolution_diagnostics``, never on whether candidates or
        nulls were found.

        Location
            ``pos`` (N, 3) float: null positions (x, y, z).
            ``cell`` (N, 3) int64: lower-corner indices (i, j, k) of the cell.
            ``q`` (N, 3) float: normalised position in the cell, in [0, 1]^3;
            ``pos[:, 0] = x[i] + q[:, 0] * (x[i+1] - x[i])`` etc.
            ``bnorm`` (N,) float: |B| of the interpolant at ``pos``.

        Jacobian and charge
            ``jacobian`` (N, 3, 3): ``J[a, b] = dB_a/dx_b`` of the cell
            interpolant.  For a null exactly on a cell face, edge or node the
            interpolant is only continuous, so J is the one-sided Jacobian of
            the cell that was kept.
            ``detJ`` (N,), ``null_charge`` (N,) int: sign(det J), 0 only for a
            numerically singular J.
            ``total_charge`` int: sum of ``null_charge``.  For a fully
            periodic domain with all isolated nulls included, the expected
            sum is zero.  For an open volume the sum need not vanish; when B
            does not vanish on the boundary, it equals the boundary degree.
            ``poincare`` (N,) int32: numerical approximation to the
            topological degree on the containing cell surface, in physical
            orientation.  It should agree with ``null_charge`` for a single
            isolated non-degenerate null, provided the surface integral is
            adequately resolved and B is nonzero on the cell boundary.  A
            close opposite-charge pair may have net index zero;
            ``POINCARE_DEGENERATE`` (99) indicates |B| = 0 at a sampled
            surface point; ``POINCARE_NOT_COMPUTED`` (-99) if disabled.

        Classification (from J, or J - tr(J)I/3 if ``classify_traceless``)
            ``null_type`` (N,) int8: ``NULL_TYPE_*`` (0 unknown, 1 A, 2 B,
            3 As, 4 Bs); see :func:`null_type_name`.  A/As: converging fan,
            charge +1; B/Bs: diverging fan, charge -1.
            ``eigenvalues`` (N, 3) complex, LAPACK order.
            ``spine_vec``, ``fan_normal`` (N, 3): unit axes (arbitrary sign),
            NaN when unclassified.  ``fan_normal`` is the left eigenvector of
            the spine eigenvalue.

        Current and divergence (raw J; curl B, not divided by mu0)
            ``div_residual`` = |tr J| / ||J||_F; ``div_warning`` =
            ``div_residual > div_tol_rel``.
            ``curlB_vec`` (N, 3); ``current_ratio`` = |curl B| / ||J||_F;
            ``is_potential`` = ``current_ratio < current_tol``.
            ``j_parallel`` (arbitrary sign; use abs), ``j_perp``: components
            of curl B along and across the spine; ``j_parallel_ratio``,
            ``j_perp_ratio``: same divided by |curl B|;
            ``current_spine_angle_deg``: angle between curl B and the spine
            axis, 0 (spine current) to 90 (fan current) degrees.
            ``fan_is_perpendicular``: True for potential nulls or when
            ``j_perp_ratio < current_tol``; False when unclassified.

        Resolution (if ``add_resolution_diagnostics``)
            ``nearest_opposite_index`` (N,) int (-1 if none),
            ``nearest_opposite_distance`` (N,) physical distance,
            ``nearest_opposite_distance_cells`` (N,): distance in fractional
            index space (cell + q), i.e. in grid cells along each axis.
            ``resolution_flag`` (N,) int8 and ``resolution_name`` (N,) str:
            resolved (>= resolved_threshold cells), marginal
            (>= marginal_threshold), unresolved, isolated (no opposite charge
            present) or degenerate (charge 0).
            ``opposite_pairs``: list of dicts for the unique nearest-opposite
            pairs (not a one-to-one matching); also as arrays
            ``opposite_pair_index`` (P, 2) and ``opposite_pair_distance_cells``.
            Counts ``n_resolved``, ``n_marginal``, ``n_unresolved``,
            ``n_isolated``, ``n_degenerate`` (per null) and
            ``n_opposite_pairs``, ``n_pair_resolved``, ``n_pair_marginal``,
            ``n_pair_unresolved``; ``min_opposite_distance_cells``;
            ``snapshot_resolution`` (worst pair: unresolved > marginal >
            resolved; "isolated" without pairs; "no_nulls" if N == 0);
            ``hlocal_min`` (N,): min(dx, dy, dz) of each null cell;
            ``resolution_thresholds``.

        Bookkeeping
            ``candidates`` (Nc, 3) int64 sign-change cells; ``counts`` per z
            layer; ``n_candidates``; ``n_candidates_excluded`` (removed by the
            exclusion test); ``n_candidates_refined``;
            ``n_candidates_with_success``, ``n_candidates_failed`` (refined
            cells with and without roots); ``n_newton_attempts``;
            ``n_roots_before_dedup``; ``n_duplicates_removed``;
            ``n_cells_root_overflow``; ``timing`` (seconds per stage);
            ``settings`` (parameters actually used).

    Notes
    -----
    * Degenerate nulls (line or sheet nulls, det J = 0, e.g. 2.5D fields) are
      not reported: Newton cannot converge to a non-isolated root.
    * Periodic seam cells are not scanned; use :func:`pad_periodic`.
    * Field values are not validated, so that the routine never pays for an
      extra pass over the data.  Candidate cells with a NaN or infinite
      vertex are skipped before Newton, so non-finite values never produce a
      null, but nulls in the affected cells are silently missed; check the
      input beforehand if it may contain such values.
    """
    t_start = time.perf_counter()
    poincare_diag = _parse_bool(poincare_diag, "poincare_diag")
    max_roots_per_cell = int(max_roots_per_cell)
    if max_roots_per_cell < 1:
        raise ValueError("max_roots_per_cell must be >= 1.")
    exclusion_depth = int(exclusion_depth)
    if exclusion_depth < 0:
        raise ValueError("exclusion_depth must be >= 0.")
    max_seed_starts = int(max_seed_starts)
    if max_seed_starts < 0:
        raise ValueError("max_seed_starts must be >= 0.")
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1.")
    poincare_max_depth = int(poincare_max_depth)
    if poincare_max_depth < 0:
        raise ValueError("poincare_max_depth must be >= 0.")
    for name, val in (
        ("b_tol", b_tol),
        ("b_tol_rel", b_tol_rel),
        ("det_tol", det_tol),
        ("det_tol_rel", det_tol_rel),
        ("q_tol", q_tol),
    ):
        if not (np.isfinite(val) and val >= 0.0):
            raise ValueError(f"{name} must be finite and >= 0.")
    if b_tol == 0.0 and b_tol_rel == 0.0:
        raise ValueError("b_tol and b_tol_rel cannot both be 0.")

    raw = [np.asarray(a) for a in (bx, by, bz)]
    input_dtypes = tuple(a.dtype for a in raw)
    dtype = _resolve_field_dtype(input_dtypes, field_dtype)
    bx, by, bz = (
        _prepare_field_array(a, n, dtype)
        for a, n in zip(raw, ("bx", "by", "bz"), strict=True)
    )
    x = _prepare_coord_array(x, "x")
    y = _prepare_coord_array(y, "y")
    z = _prepare_coord_array(z, "z")
    if bx.shape != by.shape or bx.shape != bz.shape:
        raise ValueError("bx, by, bz must have the same shape (nx, ny, nz).")
    nx, ny, nz = bx.shape
    if (x.size, y.size, z.size) != (nx, ny, nz):
        raise ValueError(
            f"Coordinate sizes must match bx.shape={bx.shape}: "
            f"x.size={x.size}, y.size={y.size}, z.size={z.size}."
        )
    t_prep = time.perf_counter() - t_start

    # --- candidates and exclusion ------------------------------------------
    t0 = time.perf_counter()
    candidates, counts = _get_candidate_cells(bx, by, bz)
    t_candidates = time.perf_counter() - t0

    t0 = time.perf_counter()
    if exclusion_depth > 0 and len(candidates):
        keep = _exclusion_filter(
            bx, by, bz, candidates, exclusion_depth, float(b_tol), float(b_tol_rel)
        )
        refine_cells = np.ascontiguousarray(candidates[keep])
    else:
        refine_cells = candidates
    t_exclusion = time.perf_counter() - t0

    # --- Newton ------------------------------------------------------------
    n_start_slots = len(refine_cells) * NEWTON_STARTS_PER_CANDIDATE
    do_parallel = _choose_parallel(
        refine_parallel, n_start_slots, int(parallel_threshold)
    )
    kernel_args = (
        int(max_iter),
        float(b_tol),
        float(b_tol_rel),
        float(q_tol),
        float(det_tol),
        float(det_tol_rel),
        exclusion_depth,
        int(max_seed_starts),
        max_roots_per_cell,
        float(root_duplicate_q_tol),
    )
    t0 = time.perf_counter()
    pos, cell, q, bnorm, jac, n_with_root, n_attempts, n_overflow = _refine_in_chunks(
        bx, by, bz, x, y, z, refine_cells, chunk_size, do_parallel, kernel_args
    )
    t_refine = time.perf_counter() - t0
    n_roots_before_dedup = len(pos)
    if n_overflow:
        warnings.warn(
            f"max_roots_per_cell={max_roots_per_cell} was reached in "
            f"{n_overflow} cell(s). Additional distinct roots were "
            "discarded. Increase max_roots_per_cell to avoid losing roots.",
            RuntimeWarning,
            stacklevel=2,
        )

    # --- duplicates --------------------------------------------------------
    t0 = time.perf_counter()
    n_dup = 0
    if remove_duplicates and len(pos) > 1:
        pos, cell, q, bnorm, jac, n_dup = _remove_duplicate_nulls(
            pos,
            cell,
            q,
            bnorm,
            jac,
            x,
            y,
            z,
            duplicate_tol,
            float(duplicate_tol_factor),
        )
    t_dedup = time.perf_counter() - t0

    detJ = _det3_batch(jac)
    null_charge = np.sign(detJ).astype(int)

    # --- Poincare ----------------------------------------------------------
    t0 = time.perf_counter()
    if poincare_diag and len(cell):
        cos_tol = math.cos(math.radians(float(poincare_max_angle_deg)))
        pidx = _compute_poincare_for_cells(
            bx, by, bz, x, y, z, np.ascontiguousarray(cell), poincare_max_depth, cos_tol
        )
    else:
        pidx = np.full(len(cell), POINCARE_NOT_COMPUTED, dtype=np.int32)
    t_poincare = time.perf_counter() - t0

    # --- classification ----------------------------------------------------
    t0 = time.perf_counter()
    cdict = _classify_nulls(
        jac,
        current_tol=current_tol,
        div_tol_rel=div_tol_rel,
        classify_traceless=classify_traceless,
    )
    if not classify:
        keep_div = {k: cdict[k] for k in ("div_residual", "div_warning")}
        cdict = _empty_classification(len(pos))
        cdict.update(keep_div)
    t_classify = time.perf_counter() - t0

    result = {
        "pos": pos,
        "cell": cell,
        "q": q,
        "bnorm": bnorm,
        "jacobian": jac,
        "detJ": detJ,
        "null_charge": null_charge,
        "total_charge": int(null_charge.sum()),
        "poincare": pidx,
        **cdict,
        "candidates": candidates,
        "counts": counts,
        "n_candidates": len(candidates),
        "n_candidates_excluded": int(len(candidates) - len(refine_cells)),
        "n_candidates_refined": len(refine_cells),
        "n_candidates_with_success": int(n_with_root),
        "n_candidates_failed": int(len(refine_cells) - n_with_root),
        "n_newton_attempts": int(n_attempts),
        "n_roots_before_dedup": int(n_roots_before_dedup),
        "n_duplicates_removed": int(n_dup),
        "n_cells_root_overflow": int(n_overflow),
    }

    t0 = time.perf_counter()
    if add_resolution_diagnostics:
        _add_resolution_diagnostics(
            result,
            x,
            y,
            z,
            resolved_threshold=float(resolution_resolved_threshold),
            marginal_threshold=float(resolution_marginal_threshold),
        )
    t_resolution = time.perf_counter() - t0

    result["timing"] = {
        "prepare_s": t_prep,
        "candidate_s": t_candidates,
        "exclusion_s": t_exclusion,
        "refine_s": t_refine,
        "dedup_s": t_dedup,
        "poincare_s": t_poincare,
        "classify_s": t_classify,
        "resolution_s": t_resolution,
        "total_s": time.perf_counter() - t_start,
    }
    result["settings"] = {
        "max_iter": int(max_iter),
        "b_tol": float(b_tol),
        "b_tol_rel": float(b_tol_rel),
        "q_tol": float(q_tol),
        "det_tol": float(det_tol),
        "det_tol_rel": float(det_tol_rel),
        "exclusion_depth": exclusion_depth,
        "max_seed_starts": max_seed_starts,
        "poincare_diag": poincare_diag,
        "poincare_max_depth": poincare_max_depth,
        "poincare_max_angle_deg": float(poincare_max_angle_deg),
        "remove_duplicates": bool(remove_duplicates),
        "duplicate_tol": None if duplicate_tol is None else float(duplicate_tol),
        "duplicate_tol_factor": float(duplicate_tol_factor),
        "classify": bool(classify),
        "current_tol": float(current_tol),
        "div_tol_rel": float(div_tol_rel),
        "classify_traceless": bool(classify_traceless),
        "nstarts": NEWTON_STARTS_PER_CANDIDATE,
        "max_roots_per_cell": max_roots_per_cell,
        "root_duplicate_q_tol": float(root_duplicate_q_tol),
        "field_dtype": str(bx.dtype),
        "field_input_dtypes": tuple(str(d) for d in input_dtypes),
        "field_layout": "C"
        if _is_c_layout(bx)
        else ("F" if bx.flags["F_CONTIGUOUS"] else "strided"),
        "refine_parallel": bool(do_parallel),
        "num_threads": int(numba.get_num_threads()) if do_parallel else 1,
        "chunk_size": chunk_size,
        "n_newton_start_slots": int(n_start_slots),
    }

    if verbose:
        tm = result["timing"]
        print(
            f"[null finder] candidates: {result['n_candidates']} "
            f"(excluded {result['n_candidates_excluded']}, "
            f"refined {result['n_candidates_refined']})"
        )
        print(
            f"[null finder] refine: {'parallel' if do_parallel else 'serial'}, "
            f"cells with roots {n_with_root}, roots {n_roots_before_dedup}, "
            f"duplicates removed {n_dup}"
        )
        if n_overflow:
            print(
                "[null finder] WARNING: max_roots_per_cell reached in "
                f"{n_overflow} cell(s)"
            )
        print(
            f"[null finder] unique nulls: {len(pos)}  "
            f"(total charge {result['total_charge']})"
        )
        print("[null finder] timing [s]:")
        for key, value in tm.items():
            print(f"    {key.removesuffix('_s'):<11} {value:.5f}")
        if add_resolution_diagnostics:
            print(
                f"[null finder] pair resolution: {result['snapshot_resolution']} "
                f"(min d_opp = {result['min_opposite_distance_cells']:.5f} cells, "
                f"pairs resolved/marginal/unresolved = {result['n_pair_resolved']}/"
                f"{result['n_pair_marginal']}/{result['n_pair_unresolved']})"
            )

    return result
