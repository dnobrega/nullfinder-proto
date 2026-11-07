"""Analytic benchmark field of Olshevsky et al. (2020), Appendix A.

The field (their Eq. A.1) is a triply periodic background, a small analytic
perturbation and the curls of seven localized Gaussian vector potentials. Its 14
nulls are listed in their Table A.2 (``EXACT_NULLS``).

Two corrections with respect to the printed text:

* Y_2 = 0.117 instead of 0.0117: with the printed value the field does not vanish
  at the tabulated nulls 10 and 12.
* The constant in B_y is sqrt(30)/7, not sqrt(30/7).

:func:`olshevsky2020_field` evaluates the field in parallel with Numba, directly
into Fortran-ordered arrays of the requested dtype, so the peak memory is the
output arrays only. This makes 1024^3 grids practical: about 13 GB in float32.
"""

import numpy as np
from numba import njit, prange

# Table A.1: a_i, k_i, l_i, X_i, Y_i, Z_i (Y_2 corrected, see above).
PARAMS_A1 = np.array(
    [
        [-0.2, 50.0, 0.2, 2.8455, 0.0267, -0.147],
        [-0.3, 4.0, 0.2, np.pi + 0.191, 0.117, np.pi],
        [1.0, 0.8, 1.0, 0.1, np.pi - 0.037, 0.2],
        [-0.22, 1.0, 0.9, np.pi + 0.0771, np.pi + 0.087, np.pi],
        [-0.22, 2.0, 0.5, np.pi / 2, np.pi / 2, np.pi / 2],
        [1.0, 1.0, 1.0, 0.073, 0.0198, -0.07],
        [-0.22, 2.0, 0.5, np.pi / 2, np.pi / 2, np.pi / 2],
    ]
)

# Table A.2: exact null positions and types.
EXACT_NULLS = np.array(
    [
        [0.2952, -0.2505, -1.4803],
        [0.04581, -0.0005391, -0.01064],
        [2.8657, -0.1878, -0.5961],
        [2.9737, -0.1500, 0.2246],
        [3.2286, 0.06090, 0.2457],
        [-0.2124, 3.3212, 0.4444],
        [-0.2728, 0.2386, 1.2466],
        [3.1391, 3.1344, -0.03510],
        [0.01951, -0.01637, 3.2123],
        [3.3225, 0.1453, 2.9898],
        [-0.0017705, 3.1360, 3.1707],
        [3.3119, 0.1539, 3.2509],
        [2.9915, -0.09684, 3.5047],
        [3.0203, 3.0182, 3.3935],
    ]
)
EXACT_TYPES = [
    "Bs",
    "As",
    "A",
    "Bs",
    "As",
    "Bs",
    "Bs",
    "A",
    "As",
    "Bs",
    "As",
    "As",
    "Bs",
    "Bs",
]

DOMAIN = (-np.pi / 2, 3 * np.pi / 2)


def create_grid(n):
    """Uniform 1D coordinates with ``n`` nodes on the benchmark domain."""
    return np.linspace(*DOMAIN, n)


@njit(parallel=True, cache=True)
def _fill(bx, by, bz, x, y, z, gx, gy, gz, fx, fy, fz, c):
    """Evaluate Eq. (A.1) at every node, in parallel over z planes.

    The background is written with products of 1D sines and cosines, and each
    Gaussian ring r as products of the 1D factors g*[r] = exp(-(s - S_r)^2 / w^2)
    and f*[r] = (s - S_r) g*[r], so the inner loop has no transcendental calls.
    """
    nx, ny, nz = x.size, y.size, z.size
    nr = c.size
    sq = np.sqrt(30.0) / 7.0
    for k in prange(nz):
        sz = np.sin(z[k])
        cz = np.cos(z[k])
        for j in range(ny):
            sy = np.sin(y[j])
            s2y = np.sin(2.0 * y[j])
            for i in range(nx):
                sx = np.sin(x[i])
                cx = np.cos(x[i])
                vx = -2.0 * s2y + sz + 0.04 * (-2.0 * s2y + 2.0 * sx * cz + sy + 0.1195)
                # sin(x - z) + ... expanded: 2 sin(x-z) + sin(x+z)
                vy = 2.0 * sx + sz + 0.04 * (3.0 * sx * cz - cx * sz + sq)
                vz = sx + sy + 0.04 * (-2.0 * cx * sz + sy + 2.0 * s2y - 0.1378162)
                for r in range(nr):
                    if r < 2:  # curl(phi e_z)
                        vx += -c[r] * gx[r, i] * fy[r, j] * gz[r, k]
                        vy += c[r] * fx[r, i] * gy[r, j] * gz[r, k]
                    elif r < 5:  # curl(phi e_y)
                        vx += c[r] * gx[r, i] * gy[r, j] * fz[r, k]
                        vz += -c[r] * fx[r, i] * gy[r, j] * gz[r, k]
                    else:  # curl(phi e_x)
                        vy += -c[r] * gx[r, i] * gy[r, j] * fz[r, k]
                        vz += c[r] * gx[r, i] * fy[r, j] * gz[r, k]
                bx[i, j, k] = vx
                by[i, j, k] = vy
                bz[i, j, k] = vz


def olshevsky2020_field(x, y, z, dtype=np.float64, *, out=None):
    """Evaluate the benchmark field on the Cartesian product of x, y and z.

    By default, allocate three Fortran-ordered output arrays using ``dtype``.
    If ``out=(bx, by, bz)`` is supplied, write directly into the given writable
    float32/float64 NumPy arrays without allocating the output fields. In this
    case the arrays' dtype takes precedence over ``dtype``. In particular,
    non-owning Fortran-order views of preallocated MLX buffers are supported.

    Intermediate one-dimensional Gaussian factors are allocated in float64.
    """
    x, y, z = (np.ascontiguousarray(c, dtype=float) for c in (x, y, z))
    a, kappa, ell, X0, Y0, Z0 = (np.ascontiguousarray(p) for p in PARAMS_A1.T)
    idx = np.arange(len(a))
    # Width along each axis: rings 1-2 are elongated (ell) along z, 3-5 along y,
    # 6-7 along x; the other two widths are |a|.
    wx = np.where(idx < 5, a, ell)
    wy = np.where((idx < 2) | (idx >= 5), a, ell)
    wz = np.where(idx < 2, ell, a)
    dx, dy, dz = x[None] - X0[:, None], y[None] - Y0[:, None], z[None] - Z0[:, None]
    gx = np.exp(-(dx**2) / wx[:, None] ** 2)
    gy = np.exp(-(dy**2) / wy[:, None] ** 2)
    gz = np.exp(-(dz**2) / wz[:, None] ** 2)
    c = 2.0 * a * kappa / a**2
    shape = (x.size, y.size, z.size)
    if out is None:
        result = tuple(np.empty(shape, dtype=dtype, order="F") for _ in range(3))
    else:
        if not isinstance(out, (tuple, list)) or len(out) != 3:
            raise ValueError("out must contain exactly three NumPy arrays")
        result = tuple(out)
        for name, field in zip(("bx", "by", "bz"), result, strict=True):
            if not isinstance(field, np.ndarray):
                raise TypeError(f"out[{name}] must be a NumPy ndarray")
            if field.shape != shape:
                raise ValueError(
                    f"out[{name}] has shape {field.shape}; expected {shape}"
                )
            if field.dtype not in (np.dtype("float32"), np.dtype("float64")):
                raise TypeError(f"out[{name}] must have dtype float32 or float64")
            if not field.flags.writeable:
                raise ValueError(f"out[{name}] must be writable")
        if len({field.dtype for field in result}) != 1:
            raise TypeError("out must contain three arrays with the same dtype")
    _fill(*result, x, y, z, gx, gy, gz, dx * gx, dy * gy, dz * gz, c)
    return result


def match_exact(pos, types=None, tol=None, h=None):
    """One-to-one match of detected nulls to Table A.2 (minimum total distance).

    Returns (n_matched, n_type_ok, distances of matched nulls). A match counts if
    it lies within ``tol`` (default 0.5 h).
    """
    from scipy.optimize import linear_sum_assignment

    pos = np.asarray(pos, dtype=float).reshape(-1, 3)
    if len(pos) == 0:
        return 0, 0, np.empty(0)
    tol = 0.5 * h if tol is None else tol
    D = np.linalg.norm(EXACT_NULLS[:, None] - pos[None], axis=-1)
    r, c = linear_sum_assignment(D)
    ok = D[r, c] <= tol
    n_type = 0
    if types is not None:
        n_type = sum(
            types[j] == EXACT_TYPES[i] for i, j in zip(r[ok], c[ok], strict=True)
        )
    return int(ok.sum()), int(n_type), D[r, c][ok]
