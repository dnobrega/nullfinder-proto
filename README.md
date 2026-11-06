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
- ChatGPT (OpenAI), AI assistant: algorithm development, numerical optimization,
  code review and documentation.

## Features

- **Robust multi-root search.** Sign-change candidates, a rigorous exclusion
  test (Bernstein subdivision), and multi-start Newton seeded from surviving
  sub-boxes. Multiple roots per cell can be found, including roots near cell
  faces. The exclusion test can certify that a cell has no roots; Newton is
  not a mathematical guarantee of finding every root in a surviving cell.
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

**This is only an illustrative stencil, not a general Bifrost conversion.**
Verify the actual stagger locations and boundary conditions for each component.
In particular, `np.roll` wraps the boundary: do not use it unchanged on a
non-periodic axis. Prefer the simulation code's staggered-grid operators.

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
| `poincare` | Numerical approximation to the topological degree of B/‖B‖ over the cell surface, refined adaptively. For an isolated, non-degenerate null it should agree with `null_charge` if the surface integral is adequately resolved and B does not vanish on the cell boundary. Opposite-charge pairs in a cell can cancel (net degree 0). The sentinel 99 means B = 0 at a sampled surface point. |
| `nearest_opposite_distance_cells` | Distance to the nearest opposite-charge null in fractional index space (`cell + q`), i.e. in grid cells along each axis, also on stretched or anisotropic grids. |
| `resolution_name` | `resolved` (≥ 1 cell), `marginal` (0.5–1), `unresolved` (< 0.5), `isolated` (no opposite charge present), `degenerate` (charge 0). Thresholds are parameters. |
| `div_residual` | ‖tr J‖ / ‖J‖_F. The trilinear interpolant is not pointwise solenoidal even for discretely solenoidal data; values of a few × 1e-2 are normal for turbulence-like fields. `div_warning` uses `div_tol_rel = 0.1`. |
| `current_spine_angle_deg` | Angle between `curl B` and the spine axis: 0° for a pure spine current (spiral nulls), 90° for a pure fan current. |
| `total_charge` | Sum of detected null charges. For a fully periodic domain with all isolated nulls accounted for, the net charge is expected to be zero. In an open volume, the total charge need not vanish: when B is nonzero on the boundary, it equals the degree of the field-direction map on that boundary. |

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

**Strong scaling (1024³).** Timings of the Olshevsky et al. (2020) analytic
field with 14 known nulls, comparing the previous two-pass candidate scan
with the current single-pass bit-mask scan. Both use `float32`, Numba/OpenMP
on macOS ARM64 (18 logical CPUs reported), and the same grid and field.
Values are **median search times**, after a full-grid warm-up, excluding
field generation and JIT compilation; the older version used 5 timed runs,
the current version 3. Both versions found and correctly classified all 14
nulls, with the same maximum positional error (1.08×10⁻⁴).

| Numba threads | Previous (s) | Current (s) | Faster by | Current parallel efficiency |
|---:|---:|---:|---:|---:|
| 1 | 8.327 | 0.834 | **10.0×** | 100.0% |
| 2 | 4.248 | 0.438 | **9.7×** | 95.2% |
| 4 | 2.328 | 0.224 | **10.4×** | 93.0% |
| 8 | 1.316 | 0.133 | **9.9×** | 78.6% |
| 16 | — | 0.074 | — | 70.6% |

At one thread, the candidate scan itself falls from **8.325 s to 0.833 s**
(about 10×); the exclusion and Newton stages change little. With 16 threads,
the current implementation takes **0.074 s**, an **11.3× strong-scaling
speedup** over its own one-thread time (70.6% parallel efficiency). The
previous version was not measured with 16 threads. These results describe the
null search on a field already in memory, not the time to generate the field.

The current scan performs a single pass using a compact bit mask
(approximately one bit per cell), reusing 2×2 vertex-block extrema along the
contiguous memory axis. The benefit depends on the field, memory layout and
thread count; in particular, this 14-null analytic field is not representative
of a turbulent simulation with many candidates. Run
`benchmarks/thread_scaling.py` to measure your own workload. Temporary memory
during refinement is bounded by `chunk_size`; the candidate mask adds about
134 MB for a 1024³ grid.

## Benchmark

`benchmarks/olshevsky2020_null_benchmark.ipynb` applies the finder to the
analytic field of Olshevsky et al. (2020), with 14 known nulls, at several
resolutions, and compares it with other implementations, such as the public
finder of [Chiti (2020)](https://github.com/FedericaChiti/Null_Finder). That
code is not distributed here; download it and set `FEDERICA_PATH` in the
notebook. **If `FEDERICA_PATH = None`, Federica's method is not run**: the
notebook then benchmarks only the installed finder (and any optionally supplied
local implementation). Since Chiti's routine returns positions but not null
types, the notebook calculates its classification separately, using the
Jacobian of Chiti's own trilinear interpolation and the same classification
rules as this package.

Note that Table A.1 of the paper prints Y₂ = 0.0117, but the tabulated nulls
10 and 12 are only reproduced with Y₂ = 0.117, which the notebook uses.

With a one-to-one matching tolerance of **half a grid spacing**
(`MATCH_TOL_CELLS = 0.5` in the notebook), the following values refer to
**our finder**; classification counts do not come from Chiti's original
function. Re-run the notebook with the same tolerance to reproduce them:

| N³ | Found | Matched | ⟨d⟩ | d_max | Types correct |
|---|---|---|---|---|---|
| 20³ | 12 | 11 | 3.07×10⁻² | 1.21×10⁻¹ | 8 |
| 80³ | 14 | 14 | 4.35×10⁻³ | 1.18×10⁻² | 14 |
| 512³ | 14 | 14 | 1.06×10⁻⁴ | 3.29×10⁻⁴ | 14 |

`benchmarks/thread_scaling.py` measures how the run time scales with the number
of Numba threads on the same field at large resolution (1024³ by default, about
13 GB in float32), checks that the 14 nulls are recovered, and can time Chiti's
finder on the same data:

```bash
python benchmarks/thread_scaling.py --n 1024 --threads 1,2,4,8,16
python benchmarks/thread_scaling.py --n 512 --dtype float64 --chiti path/to/null_finder.py
```

It writes a CSV table and a figure with run times per stage, measured and
ideal speedup, and parallel efficiency into the git-ignored
`benchmarks/results/` directory by default. The candidate scan is parallel
over z planes for Fortran-ordered fields and over x planes for C-ordered fields;
its relative cost depends on the number of candidate cells and memory layout.

`--threading-layer {default,omp,tbb,workqueue}` selects Numba's parallel backend
before its parallel runtime starts. `--chunksize N` requests a positive chunk
size; **Numba's dynamic scheduling requires the TBB threading layer**. The
script warns when a positive chunk size is requested with another backend.
Changing the active backend requires a fresh Python process. For example:

```bash
python benchmarks/thread_scaling.py --n 512 --threads 1,2,4,8 \
  --threading-layer tbb --chunksize 4
```

If the OpenMP runtime prints `omp_set_nested routine deprecated`, that warning
originates in OpenMP, not in the null-finding algorithm. For the Intel OpenMP
runtime, `KMP_WARNINGS=false` may suppress informational messages; alternatively
use a supported Numba backend such as TBB if it is installed.

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
