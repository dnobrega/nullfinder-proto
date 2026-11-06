"""Find and classify the nulls of a random solenoidal field.

Run:  python examples/random_field_demo.py [n]
"""
import sys

import numpy as np

from null_finder import find_magnetic_nulls, null_type_name, pad_periodic, topology_report


def random_solenoidal(n, seed=1):
    """Periodic random field B = curl A with a Kolmogorov-like spectrum."""
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
    return [np.asfortranarray(b.astype(np.float32)) for b in (bx, by, bz)]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    bx, by, bz = random_solenoidal(n)
    x = y = z = np.arange(n, dtype=float)

    # The field is periodic in all three directions: pad to scan the seam cells.
    res = find_magnetic_nulls(*pad_periodic(bx, by, bz, x, y, z, axes=(0, 1, 2)))

    print()
    rep = topology_report(res, expected_charge=0)  # closed domain -> total charge 0
    print("Poincare agreement with sign(det J):", rep["poincare_agreement"])

    print("\nFirst nulls:")
    for i in range(min(5, len(res["pos"]))):
        print(f"  {np.round(res['pos'][i], 3)}  {null_type_name(res['null_type'][i]):3s}"
              f"  charge {res['null_charge'][i]:+d}"
              f"  angle(j, spine) = {res['current_spine_angle_deg'][i]:5.1f} deg"
              f"  nearest opposite = {res['nearest_opposite_distance_cells'][i]:.2f} cells")
