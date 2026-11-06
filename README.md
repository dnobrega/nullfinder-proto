# nullfinder-proto

A parallel Python toolkit to find and classify magnetic null points in MHD simulations on Cartesian (uniform or stretched) grids.

The field is represented by its trilinear interpolant in every grid cell.
Null points are the roots of that interpolant, located with a multi-start
Newton method, and everything reported about them (Jacobian, eigenvalues,
null type, spine and fan, currents) comes from the same local representation,
so the results are mutually consistent.

**Author:** Daniel Nóbrega-Siverio (Université Paris-Saclay, CEA, AIM; Rosseland
Centre for Solar Physics, University of Oslo).

**Contributors:**

- Fernando Moreno-Insertis (Instituto de Astrofísica de Canarias; Universidad de
  La Laguna).
- Claude Opus 5.5 (Anthropic), AI assistant: code review, bug fixes, testing,
  packaging and documentation.
- ChatGPT v5.6 (OpenAI), AI assistant: optimization and documentation.

## Features

- **Complete search.** Sign-change candidates, an exact exclusion test
  (Bernstein subdivision) and multi-start Newton seeded from the surviving
  sub-boxes, so several roots per cell and roots close to cell faces are found.
- **Scale invariant.** All tolerances are relative by default: results do not
  depend on the field or length units.
- **Any grid orientation and memory layout.** Increasing or decreasing axes,
  C, Fortran or strided arrays, float32 or float64, without copying.
- **Classification.** A, B, As, Bs types (Lau & Finn 1990), spine axis, fan
  normal, current parallel/perpendicular to the spine and the angle between
  them, potential test, divergence residual.
- **Topology checks.** Topological charge `sign(det J)`, an adaptive Poincaré
  index per null cell, and the distance (in grid cells) from each null to the
  nearest null of opposite charge, flagged as resolved / marginal / unresolved.
- **Fast and memory-bounded.** Numba-parallel kernels, chunked refinement.

## Installation

Install directly from GitHub:
```bash
pip install git+https://github.com/dnobrega/nullfinder-proto.git
```
Or clone the repository for development:
```bash
git clone https://github.com/dnobrega/nullfinder-proto.git
cd nullfinder-proto
pip install -e .
```

Requires Python ≥ 3.10, NumPy, SciPy and Numba. The first call compiles the
kernels (a few seconds); compiled code is cached on disk afterwards. If the
package directory is read-only, set `NUMBA_CACHE_DIR` to a writable folder.

## Quick start

```python
import numpy as np
from nullfinder_proto import find_magnetic_nulls, topology_report

# bx, by, bz: arrays of shape (nx, ny, nz), sampled at the nodes (x[i], y[j], z[k])
res = find_magnetic_nulls(bx, by, bz, x, y, z)

res["pos"]  # (N, 3) null positions
res["null_type"]  # 1 = A, 2 = B, 3 = As, 4 = Bs, 0 = unknown
res["null_charge"]  # +1 / -1
res["spine_vec"], res["fan_normal"]
res["nearest_opposite_distance_cells"], res["resolution_name"]

topology_report(res)
```

See the docstring of `find_magnetic_nulls` for every parameter and every key
of the result dictionary, and `examples/random_field_demo.py` for a complete,
self-contained run.

## Conventions (please read)

**Co-located data.** All three components must be sampled at the same points,
the grid nodes `(x[i], y[j], z[k])`. Many codes store `B` on a staggered mesh;
in Bifrost, for instance, the components live on cell faces. Passing
face-centred components together with cell-centred coordinates shifts each
component by half a cell. Near a null that error is comparable to `B` itself
and can create or destroy null pairs. Interpolate to a common location first,
ideally with the code's own staggered operators. A minimal second-order
version, assuming `bx[i]` is stored on the face at `x[i] - dx/2` and the
domain is periodic in x (check the convention of your code):

```python
bx_c = 0.5 * (bx + np.roll(bx, -1, axis=0))  # value at x[i]
by_c = 0.5 * (by + np.roll(by, -1, axis=1))
bz_c = 0.5 * (bz + np.roll(bz, -1, axis=2))  # non-periodic z: handle the last layer
```

**Indexing.** Arrays have shape `(nx, ny, nz)` and are accessed as
`arr[i, j, k]`. Coordinates are 1D, finite and strictly monotonic.

**Jacobian and charge.** `J[a, b] = dB_a/dx_b` of the cell interpolant.
`null_charge = sign(det J)`: A and As nulls have charge +1, B and Bs nulls −1.
For a null lying exactly on a cell face, edge or node the interpolant is only
continuous there, so `J` is the one-sided Jacobian of the cell that was kept.

**Axes, not directions.** `spine_vec` and `fan_normal` are eigenvectors and
have arbitrary sign; so does `j_parallel` (use its absolute value).

**Periodic domains.** Only cells between existing nodes are scanned, so the
seam cells of a periodic axis are skipped. Use `pad_periodic` first:

```python
from nullfinder_proto import pad_periodic

res = find_magnetic_nulls(*pad_periodic(bx, by, bz, x, y, z, axes=(0, 1)))
```

## Diagnostics and how to read them

| Key | Meaning |
|---|---|
| `poincare` | Degree of B/‖B‖ over the cell surface, refined adaptively. Equals `null_charge` for one simple null per cell; it is the sum of charges if a cell holds a close pair (then usually 0); 99 if B = 0 exactly at a sample point. |
| `nearest_opposite_distance_cells` | Distance to the nearest opposite-charge null in fractional index space (`cell + q`), i.e. in grid cells along each axis, also on stretched or anisotropic grids. |
| `resolution_name` | `resolved` (≥ 1 cell), `marginal` (0.5–1), `unresolved` (< 0.5), `isolated` (no opposite charge present), `degenerate` (charge 0). Thresholds are parameters. |
| `div_residual` | ‖tr J‖ / ‖J‖_F. The trilinear interpolant is not pointwise solenoidal even for discretely solenoidal data; values of a few × 1e-2 are normal for turbulence-like fields. `div_warning` uses `div_tol_rel = 0.1`. |
| `current_spine_angle_deg` | Angle between `curl B` and the spine axis: 0° for a pure spine current (spiral nulls), 90° for a pure fan current. |
| `total_charge` | Sum of charges. Zero is expected only for closed domains (e.g. fully periodic and padded). |

## Limitations

- Degenerate nulls (line or sheet nulls, `det J = 0`, e.g. 2.5D fields) are
  not reported.
- Nulls are those of the trilinear interpolant. Structures below the grid
  scale are not represented, and close pairs (`unresolved`) should be treated
  with care.
- Field values are not validated (no extra pass over the data). Cells with a
  NaN or infinite vertex value are skipped, so such values never produce a
  null, but nulls in those cells are missed.

## Performance

Random solenoidal field, 128³ float32, 2 threads, 8403 nulls among 70 854
sign-change candidates:

| Stage | Time |
|---|---|
| candidate scan | 0.045 s |
| exclusion test (depth 3, removes 77 % of candidates) | 0.03 s |
| Newton refinement | 0.27 s |
| Poincaré, classification, resolution | 0.18 s |
| **total** | **0.53 s** |

The previous version (single file, no exclusion test) took 1.29 s on the same
field. Temporary memory during refinement is bounded by `chunk_size`.

## Benchmark

`benchmarks/olshevsky2020_null_benchmark.ipynb` applies the finder to the
analytic field of Olshevsky et al. (2020), with 14 known nulls, at several
resolutions, and compares it with other implementations, such as the public
finder of [Chiti (2020)](https://github.com/FedericaChiti/Null_Finder). That
code is not distributed here; download it and set its path in the notebook.

Note that Table A.1 of the paper prints Y₂ = 0.0117, but the tabulated nulls
10 and 12 are only reproduced with Y₂ = 0.117, which the notebook uses.

With a matching tolerance of half a grid spacing:

| N³ | Found | Matched | ⟨d⟩ | d_max | Types correct |
|---|---|---|---|---|---|
| 20³ | 12 | 11 | 3.07×10⁻² | 1.21×10⁻¹ | 8 |
| 80³ | 14 | 14 | 4.35×10⁻³ | 1.18×10⁻² | 14 |
| 512³ | 14 | 14 | 1.06×10⁻⁴ | 3.29×10⁻⁴ | 14 |

## Development

```bash
pip install -e ".[dev]"
pytest                 # test suite
ruff check .           # lint (rules in pyproject.toml)
ruff format --check .  # formatting
```

CI runs all three on every push and pull request.

The suite covers analytic nulls of every type, close pairs inside one cell,
nulls on grid nodes, invariance under field and length rescaling, axis
reversal, memory layout, dtype and chunking, the exclusion test, Poincaré
accuracy, resolution on anisotropic grids, periodic padding and the stability
of the result schema.

## References

- Lau, Y.-T. & Finn, J. M. 1990, ApJ, 350, 672 — A/B null types.
- Parnell, C. E., Smith, J. M., Neukirch, T. & Priest, E. R. 1996, Phys.
  Plasmas, 3, 759 — classification of linear nulls.
- Haynes, A. L. & Parnell, C. E. 2007, Phys. Plasmas, 14, 082107 — trilinear
  null finding.
- Olshevsky, V., Pontin, D. I., Williams, B., et al. 2020, A&A, 644, A150 —
  comparison of null-finding methods and analytic benchmark field.
- Van Oosterom, A. & Strackee, J. 1983, IEEE Trans. Biomed. Eng., 30, 125 —
  solid angle of a triangle (Poincaré index).

## How to cite

If you use this code, please cite the accompanying paper (Nóbrega-Siverio &
Moreno-Insertis, in preparation) and the software itself; see
[`CITATION.cff`](CITATION.cff), or the "Cite this repository" button on GitHub.

## Acknowledgements

This research has been supported by the European Research Council through the
Synergy Grant number 810218 ("The Whole Sun", ERC-2018-SyG); by the Centre
National d'Etudes Spatiales (CNES) Solar Orbiter project; by the Research
Council of Norway through its Centres of Excellence scheme, project number
262622; and by grant CEX2025-001609-S, awarded to the Instituto de Astrofísica
de Canarias under the Severo Ochoa Centre of Excellence program and funded by
MICIU/AEI/10.13039/501100011033.

## License

BSD 3-Clause License; see [`LICENSE`](LICENSE).
