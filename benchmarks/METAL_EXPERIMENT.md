# Apple Metal candidate-scan experiment (completed)

**Status:** feasibility study completed, November 2026. This experiment is
preserved for reproducibility; it is **not** part of the supported
`nullfinder-proto` API and does **not** implement Bernstein exclusion, Newton
refinement, null classification or other topological diagnostics on the GPU.
MLX remains an optional benchmark-only dependency.

## Question and methods

We tested whether Apple Metal/MLX can outperform the optimized CPU
`get_candidate_cells()` scan when searching for candidate magnetic-null cells.
Every GPU implementation applies the same component-wise sign-change and
finite-vertex conditions and was checked against the **exact sorted cell-index
list** produced by the CPU implementation.

The experiment compares three approaches:

- **`mask` (v2):** scan cells and return a dense 1-bit-per-cell mask decoded on CPU.
- **`compact` (v3):** append candidate cell indices directly on GPU using an
  atomic counter; output is bounded by `--compact-capacity` (default 65,536 per
  launch), and an overflow raises an error rather than silently dropping cells.
- **`column` (v4):** scan along the z direction, reuse adjacent-plane extrema,
  test Bx before By/Bz, and use SIMD reduction plus a summary of nonempty rows.

Two **different input scenarios** are timed:

- **`--input numpy`:** the three field components already exist as NumPy arrays;
  every scan includes NumPy-to-MLX copies, kernel execution and reconstruction
  of the candidate indices. This is the more relevant comparison for existing
  NumPy-based post-processing workflows.
- **`--input unified`:** the fields are generated directly into MLX-allocated,
  CPU/GPU-shared buffers through NumPy views. The scan incurs no additional
  NumPy-to-MLX copy. This is a best-case scenario for a pipeline explicitly
  designed around shared buffers, **not** evidence that existing Bifrost
  snapshots are loaded from disk without a copy or processing overhead.

## Measured results

Olshevsky et al. (2020) analytic field, `float32`, 512³ grid points,
**97 identical candidate cells** in each method. The CPU baseline used
16 OpenMP threads with `chunksize=0`; GPU measurements used MLX 0.32.4 on
Apple Silicon. Each row is a median of seven timed repetitions after warm-up.
All times exclude analytic field generation, disk I/O and initial kernel
compilation. These are **candidate-scan** timings only.

| Input mode | Implementation | Median (ms) | Relative to matching CPU |
| --- | --- | ---: | ---: |
| Shared MLX buffers | CPU Numba/OpenMP | 10.15 | 1.00× |
| Shared MLX buffers | GPU `column` | 3.06 | 3.32× faster |
| Shared MLX buffers | GPU `compact` | **1.95** | **5.20× faster** |
| Existing NumPy arrays | CPU Numba/OpenMP | **10.50** | 1.00× |
| Existing NumPy arrays | GPU `column` | 29.72 | 2.83× slower |
| Existing NumPy arrays | GPU `compact` | 27.97 | 2.66× slower |

At 128³ with shared buffers (five repetitions), CPU/`column`/`mask`/`compact`
medians were **0.52 / 0.28 / 0.30 / 0.27 ms**, respectively, with **100
identical candidate cells**. The v4 self-tests also passed on the user's Mac
(C/F layouts, NaN/Inf, zeros, partial groups, slabs and buffer overflow).

The `compact` GPU method was fastest on the *sparse-candidate analytic field*;
this does not establish the fastest method for all MHD fields or candidate
densities. The apparent GPU advantage with shared buffers concerns the
**scan after the data are already resident**, not an end-to-end simulation
analysis including loading or generating those data.

The NumPy-input `compact` benchmark spent approximately **22.78 ms** of its
27.97 ms median total in the per-scan upload stage (median of that stage,
so stage medians need not sum exactly to the total). That copy dominates the
performance difference in the tested workload.

## Reproduce

Run from the repository root on an Apple Silicon Mac with an isolated
MLX-enabled environment. Install MLX only in that environment (e.g.
`python -m pip install mlx`). The main Python package does **not** depend on MLX.

```bash
python benchmarks/metal_candidate_scan.py \
    --n 128 --repeat 5 --cpu-threads 16 \
    --input unified --modes column,mask,compact

python benchmarks/metal_candidate_scan.py \
    --n 512 --repeat 7 --cpu-threads 16 \
    --input unified --modes column,compact

python benchmarks/metal_candidate_scan.py \
    --n 512 --repeat 7 --cpu-threads 16 \
    --input numpy --modes column,compact --slab-cells 511
```

Results are saved as JSON under `benchmarks/results/metal/`, which should remain
ignored by Git. They include per-stage timings and validation status.

## Decision

**Do not integrate Metal into the stable null finder at present.** For the
measured workflow starting with NumPy arrays, the existing CPU scan is faster
and avoids an optional GPU dependency. The Metal experiment is valuable when
inputs are already MLX-backed and as a possible foundation for GPU-resident
simulation or extrapolation pipelines. A future re-evaluation would need to
benchmark actual snapshot reading into shared buffers, or process a field
already produced on GPU, including all required preparation and synchronization.

The results here are **engineering measurements**, not results proposed for
the method comparison with Chiti in the scientific paper.
