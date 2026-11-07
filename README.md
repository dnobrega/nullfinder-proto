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
Or clone the repository:
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
from nullfinder_proto import find_magnetic_nulls, topology_report

# bx, by, bz: arrays of shape (nx, ny, nz), sampled at the nodes (x[i], y[j], z[k])
res = find_magnetic_nulls(bx, by, bz, x, y, z)

res["pos"]  # (N, 3) null positions
res["null_type"]  # 1 = A, 2 = B, 3 = As, 4 = Bs, 0 = unknown
res["null_charge"]  # +1 / -1
res["spine_vec"], res["fan_normal"]
res["nearest_opposite_distance_cells"], res["resolution_name"]

report = topology_report(res)
```

See the docstring of `find_magnetic_nulls` for every parameter and every key
of the result dictionary, and `examples/random_field_demo.py` for a complete,
self-contained run.

## Conventions (please read)

**Co-located data.** The three magnetic-field components `bx`, `by`, and `bz`
must be sampled at the **same grid nodes** `(x[i], y[j], z[k])`.
The caller is responsible for supplying co-located field components and
coordinates consistent with their locations.

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

## Performance and benchmarks

### Strong scaling

The current `nullfinder-proto` implementation was benchmarked on the
14-null analytic magnetic field of Olshevsky et al. (2020), sampled on a
`1024³` grid (`float32`, approximately 12.9 GB of field data). Measurements
were made on Apple Silicon using Numba with **TBB**, dynamic scheduling
(`chunksize=4`), and seven timed repetitions after warm-up. Times are medians
for the **complete null search** (not just the candidate scan), excluding
field generation and JIT compilation. All runs recovered and correctly
classified the 14 nulls.

| Numba threads | Time (s) | Speedup | Parallel efficiency |
|---:|---:|---:|---:|
| 1 | 0.812 | 1.00× | 100% |
| 2 | 0.408 | 1.99× | 100% |
| 4 | 0.213 | 3.82× | 95% |
| 8 | 0.122 | 6.66× | 83% |
| 16 | **0.069** | **11.81×** | **74%** |

The scan uses a compact bit mask (approximately one bit per cell); at this
resolution its mask requires about 134 MB. These timings are specific to the
analytic field and hardware. Real MHD fields can have different candidate
densities and performance characteristics.

### Numerical validation

[`benchmarks/olshevsky2020_null_benchmark.ipynb`](benchmarks/olshevsky2020_null_benchmark.ipynb)
compares the detected nulls with the 14 known positions and optionally with the
[Chiti (2020) null finder](https://github.com/FedericaChiti/Null_Finder).
Chiti's code is not included: set `FEDERICA_PATH` in the notebook to run that
comparison. Its null classifications are calculated separately in the
notebook, rather than by the Chiti finder itself.

Table A.1 of Olshevsky et al. (2020) lists `Y₂ = 0.0117`, whereas its
reported null positions 10 and 12 are reproduced using `Y₂ = 0.117`;
the benchmark uses the latter value. With one-to-one matching tolerance
`MATCH_TOL_CELLS = 0.5`, our finder gives:

| N³ | Found | Matched (of 14) | Mean error | Max error | Types correct |
|---:|---:|---:|---:|---:|---:|
| 20³ | 12 | 11 | 3.07×10⁻² | 1.21×10⁻¹ | 8 |
| 80³ | 14 | 14 | 4.35×10⁻³ | 1.18×10⁻² | 14 |
| 512³ | 14 | 14 | 1.06×10⁻⁴ | 3.29×10⁻⁴ | 14 |

### Reproduce the benchmarks

The thread-scaling script records the time per stage, speedup and parallel
efficiency. It writes CSV and PNG outputs to the Git-ignored
`benchmarks/results/` directory by default.

```bash
# Reproduce the 1024³ strong-scaling configuration (TBB must be installed).
python benchmarks/thread_scaling.py --n 1024 --threads 1,2,4,8,16 \
  --repeat 7 --threading-layer tbb --chunksize 4

# Optional comparison with Chiti at 512³.
python benchmarks/thread_scaling.py --n 512 --dtype float64 \
  --chiti path/to/null_finder.py
```

OpenMP and other Numba backends remain supported; TBB is **optional**.
Positive `--chunksize` values enable dynamic scheduling only with TBB.
To switch threading backends, start a new Python process.

## Experimental Apple Metal benchmark

An optional, **completed feasibility experiment** in
[`benchmarks/metal_candidate_scan.py`](benchmarks/metal_candidate_scan.py)
compares three GPU candidate-scan strategies (`mask`, `compact`, `column`) with
Numba on Apple Silicon. It is **not** a GPU implementation of the complete null
finder, and MLX is **not required** to install or use `nullfinder-proto`.

See [the experiment report](benchmarks/METAL_EXPERIMENT.md) for measured results,
reproduction commands and the distinction between MLX-backed shared buffers and
NumPy inputs. The experiment is retained for reference, not recommended as the
default backend for analyzing stored MHD snapshots.


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
