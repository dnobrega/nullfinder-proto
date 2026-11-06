import numpy as np
import pytest


def sample(f, x, y, z, dtype=np.float64, order="F"):
    """Sample a vector field f(X, Y, Z) -> (bx, by, bz) on the node grid."""
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return [np.asarray(c, dtype=dtype, order=order) for c in f(X, Y, Z)]


def offset_grid(n=24, lo=-1.0, hi=1.0, offset=0.0137):
    """Uniform grid shifted so that analytic nulls do not sit on nodes."""
    c = np.linspace(lo, hi, n) + offset
    return c, c.copy(), c.copy()


def pair_field(X, Y, Z, a=0.3):
    """Solenoidal field with an A null at (-a,0,0) and a B null at (+a,0,0)."""
    return X**2 - a**2, Y, -Z - 2.0 * X * Z


def random_solenoidal(n=40, seed=0, dtype=np.float32):
    """Periodic random field B = curl A (spectrally exact div B = 0)."""
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(n) * n
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1.0
    amp = K2 ** (-11 / 12) * np.exp(-K2 / (n / 6) ** 2)
    A = [np.fft.fftn(rng.standard_normal((n, n, n))) * amp for _ in range(3)]
    bx = np.real(np.fft.ifftn(1j * (KY * A[2] - KZ * A[1])))
    by = np.real(np.fft.ifftn(1j * (KZ * A[0] - KX * A[2])))
    bz = np.real(np.fft.ifftn(1j * (KX * A[1] - KY * A[0])))
    x = np.arange(n, dtype=float)
    return [np.asfortranarray(b.astype(dtype)) for b in (bx, by, bz)], x


@pytest.fixture(scope="session")
def random_field():
    return random_solenoidal()


def run(bx, by, bz, x, y, z, **kw):
    from nullfinder_proto import find_magnetic_nulls

    kw.setdefault("verbose", False)
    return find_magnetic_nulls(bx, by, bz, x, y, z, **kw)


def match(pos_a, pos_b):
    """Index into pos_a of the nearest point for each row of pos_b, and distances."""
    from scipy.spatial import cKDTree

    d, m = cKDTree(pos_a).query(pos_b)
    return m, d
