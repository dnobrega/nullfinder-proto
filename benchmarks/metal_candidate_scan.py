#!/usr/bin/env python3AA
"""Experimental Metal candidate scan v4 for nullfinder-proto (Apple Silicon).

This is an experiment, NOT a GPU null finder. It tests only the necessary
component-wise sign-change filter and the finite-vertex requirement, and checks
that every GPU variant returns exactly the same sorted (i, j, k) candidate list
as the Numba CPU scan ``_get_candidate_cells``.

GPU variants (``--modes``):

* ``mask`` (v2): one thread per 32 consecutive cells, 8 vertex loads per cell,
  dense 1-bit/cell output decoded on the CPU.
* ``compact`` (v3): as v2, but each candidate gets a slot from one global
  atomic counter; bounded output, unsorted, overflow raises an error.
* ``column`` (v4): one thread per (i, j) column and z chunk, marching along k.
  Each thread loads only the 2x2 block of the new plane (4 loads per cell instead of
  8) and reuses the previous plane's min/max; neighbouring lanes read
  neighbouring addresses (coalesced). by/bz are read only where bx passes.
  The 32 results of a SIMD group are OR-reduced into one 32-bit word, written
  only if non-zero; spare lanes are guarded. A row summary (1 bit per
  (j, k) row, set with ``atomic_fetch_or`` only for non-empty words) lets
  the CPU decode just the active rows.
  No capacity limit, deterministic sorted output.

Input (``--input``):

* ``unified`` (default): generate the field directly in MLX-allocated memory
  through writable NumPy views. CPU and GPU use the same arrays without a
  NumPy-to-MLX copy. This is a best-case shared-buffer experiment, NOT proof
  that an existing Bifrost snapshot can be loaded into MLX without conversion.
  Each GPU variant runs one full-field launch.
* ``numpy``: v3 behaviour. NumPy fields are copied slab by slab into MLX on
  every run, and the copies are part of the timing.

Usage from the repository root, with science_metal activated:

    python benchmarks/metal_candidate_scan.py --n 128 --repeat 5
    python benchmarks/metal_candidate_scan.py --n 1024 --repeat 7
    python benchmarks/metal_candidate_scan.py --n 512 --input numpy --modes column,mask
    NUMBA_THREADING_LAYER=tbb python benchmarks/metal_candidate_scan.py \\
        --n 1024 --cpu-chunksize 4

Requires MLX >= 0.32, numpy, numba, scipy and nullfinder-proto. Saves JSON to
benchmarks/results/metal/ (ignored by Git).

EXPERIMENT STATUS (November 2026): feasibility study completed on Apple
Silicon with MLX 0.32.4. All three kernels matched the CPU candidate list
at N=128; ``column`` and ``compact`` were additionally compared at N=512.
At N=512 (float32; 97 candidates), ``compact`` was fastest with the field
already in shared MLX buffers (1.95 ms versus
10.15 ms for the 16-thread OpenMP CPU baseline). With fields initially in
NumPy, ``compact`` required 27.97 ms including copies versus 10.50 ms on CPU.
These are median timings for this field, NOT general GPU performance claims.
Field generation, file I/O and the remaining null-finder stages were excluded.
See benchmarks/METAL_EXPERIMENT.md for the complete results and limitations.
This script is retained for reproducibility; MLX is not a package dependency.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx
except ImportError as exc:
    raise SystemExit("Requires MLX: activate science_metal and install mlx.") from exc

ROOT = Path(__file__).resolve().parent.parent
if (ROOT / "src" / "nullfinder_proto").is_dir():
    sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmarks.olshevsky2020_field import (  # noqa: E402
    create_grid,
    olshevsky2020_field,
)
from nullfinder_proto.core import _get_candidate_cells  # noqa: E402

ALL_MODES = ("column", "mask", "compact")

METAL_HEADER = r"""
inline bool spans_zero(const device float* v, uint base,
                       uint sx, uint sy, uint sz) {
    float a = v[base];
    float b = v[base + sx];
    float c = v[base + sy];
    float d = v[base + sx + sy];
    float e = v[base + sz];
    float f = v[base + sx + sz];
    float g = v[base + sy + sz];
    float h = v[base + sx + sy + sz];
    if (!(metal::isfinite(a) && metal::isfinite(b) &&
          metal::isfinite(c) && metal::isfinite(d) &&
          metal::isfinite(e) && metal::isfinite(f) &&
          metal::isfinite(g) && metal::isfinite(h))) return false;

    float lo = metal::min(metal::min(a, b), metal::min(c, d));
    lo = metal::min(lo, metal::min(metal::min(e, f), metal::min(g, h)));
    if (lo > 0.0f) return false;
    float hi = metal::max(metal::max(a, b), metal::max(c, d));
    hi = metal::max(hi, metal::max(metal::max(e, f), metal::max(g, h)));
    return hi >= 0.0f;
}

// min, max and finiteness of the 2x2 block of nodes (i..i+1, j..j+1) of one
// z plane, starting at flat index b (Fortran order, row length sy = NX).
inline void plane_minmax(const device float* v, uint b, uint sy,
                         thread float& lo, thread float& hi, thread bool& fin) {
    float a = v[b];
    float c = v[b + 1u];
    float d = v[b + sy];
    float e = v[b + sy + 1u];
    fin = metal::isfinite(a) && metal::isfinite(c) &&
          metal::isfinite(d) && metal::isfinite(e);
    lo = metal::min(metal::min(a, c), metal::min(d, e));
    hi = metal::max(metal::max(a, c), metal::max(d, e));
}
"""


def _check_indexing(nx, ny, nz_nodes):
    if max(nx * ny * nz_nodes, (nx - 1) * (ny - 1) * (nz_nodes - 1)) >= 2**32:
        raise ValueError(
            "These Metal kernels use 32-bit indices; use --input numpy "
            "with smaller --slab-cells."
        )


# ---------------------------------------------------------------------------
# v2: dense bit mask, one thread per 32 cells
# ---------------------------------------------------------------------------


def make_kernel(nx: int, ny: int, nz_nodes: int):
    """v2 kernel for one slab shape (compiled on first use)."""
    _check_indexing(nx, ny, nz_nodes)
    source = f"""
        const uint NX = {nx}u;
        const uint NY = {ny}u;
        const uint NZ = {nz_nodes}u;
        const uint CX = NX - 1u;
        const uint CY = NY - 1u;
        const uint NCELLS = CX * CY * (NZ - 1u);
        const uint sx = 1u, sy = NX, sz = NX * NY;

        uint wid = thread_position_in_grid.x;
        uint start = wid * 32u;
        if (start >= NCELLS) return;
        uint bits = 0u;
        for (uint bit = 0u; bit < 32u; ++bit) {{
            uint cell = start + bit;
            if (cell >= NCELLS) break;
            uint i = cell % CX;
            uint tmp = cell / CX;
            uint j = tmp % CY;
            uint k = tmp / CY;
            uint base = i + NX * (j + NY * k); // Fortran order
            if (spans_zero(bx, base, sx, sy, sz) &&
                spans_zero(by, base, sx, sy, sz) &&
                spans_zero(bz, base, sx, sy, sz)) {{
                bits |= (1u << bit);
            }}
        }}
        mask[wid] = bits;
    """
    return mx.fast.metal_kernel(
        name=f"nf_sign_mask_{nx}_{ny}_{nz_nodes}",
        input_names=["bx", "by", "bz"],
        output_names=["mask"],
        source=source,
        header=METAL_HEADER,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


DECODE_MAX_ACTIVE_WORDS = 262_144


def mask_to_candidates(words, nx, ny, z_cells, z0):
    """Sorted (i, j, k) from the v2 32-bit packed mask (linear cell order)."""
    words = np.asarray(words)
    if words.dtype != np.dtype("uint32") or words.ndim != 1:
        raise TypeError("Mask must be a 1-D uint32 array")
    active = np.flatnonzero(words)
    if not active.size:
        return np.empty((0, 3), dtype=np.int64)
    chunks = []
    for left in range(0, active.size, DECODE_MAX_ACTIVE_WORDS):
        pos = active[left : left + DECODE_MAX_ACTIVE_WORDS]
        values = np.ascontiguousarray(words[pos].astype("<u4", copy=False))
        bit_positions = np.flatnonzero(
            np.unpackbits(values.view(np.uint8), bitorder="little")
        )
        if bit_positions.size:
            ids = pos[bit_positions // 32].astype(np.int64) * 32
            ids += bit_positions % 32
            chunks.append(ids)
    if not chunks:
        return np.empty((0, 3), dtype=np.int64)
    ids = np.concatenate(chunks)
    if np.any(ids >= (nx - 1) * (ny - 1) * z_cells):
        raise RuntimeError("Metal set mask bits beyond the slab's final cell.")
    cx, cy = nx - 1, ny - 1
    return np.column_stack((ids % cx, (ids // cx) % cy, ids // (cx * cy) + z0))


# ---------------------------------------------------------------------------
# v3: sparse atomic compaction
# ---------------------------------------------------------------------------


def make_compact_kernel(nx: int, ny: int, nz_nodes: int, capacity: int):
    """v3 kernel: sparse atomic scatter of uint32 cell ids (unsorted)."""
    _check_indexing(nx, ny, nz_nodes)
    if not 1 <= capacity < 2**32:
        raise ValueError("--compact-capacity must be within 1..(2**32-1).")
    source = f"""
        const uint NX = {nx}u;
        const uint NY = {ny}u;
        const uint NZ = {nz_nodes}u;
        const uint CX = NX - 1u;
        const uint CY = NY - 1u;
        const uint NCELLS = CX * CY * (NZ - 1u);
        const uint MAX_CANDIDATES = {capacity}u;
        const uint sx = 1u, sy = NX, sz = NX * NY;

        uint wid = thread_position_in_grid.x;
        uint start = wid * 32u;
        if (start >= NCELLS) return;
        for (uint bit = 0u; bit < 32u; ++bit) {{
            uint cell = start + bit;
            if (cell >= NCELLS) break;
            uint i = cell % CX;
            uint tmp = cell / CX;
            uint j = tmp % CY;
            uint k = tmp / CY;
            uint base = i + NX * (j + NY * k);
            if (spans_zero(bx, base, sx, sy, sz) &&
                spans_zero(by, base, sx, sy, sz) &&
                spans_zero(bz, base, sx, sy, sz)) {{
                uint slot = atomic_fetch_add_explicit(
                    &count[0], 1u, memory_order_relaxed);
                if (slot < MAX_CANDIDATES) {{
                    atomic_store_explicit(
                        &indices[slot], cell, memory_order_relaxed);
                }}
            }}
        }}
    """
    return mx.fast.metal_kernel(
        name=f"nf_atomic_compact_{nx}_{ny}_{nz_nodes}_{capacity}",
        input_names=["bx", "by", "bz"],
        output_names=["indices", "count"],
        source=source,
        header=METAL_HEADER,
        ensure_row_contiguous=True,
        atomic_outputs=True,
        compile_options={"math_mode": "safe"},
    )


def compact_to_candidates(indices, count, nx, ny, z_cells, z0, capacity):
    """Sorted (i, j, k) from the unordered v3 scatter."""
    n = int(np.asarray(count).reshape(-1)[0])
    if n > capacity:
        raise RuntimeError(
            f"Metal compact scan overflow: {n:,} candidates in slab beginning "
            f"at z={z0}, capacity={capacity:,}. Retry with a larger "
            "--compact-capacity or another mode."
        )
    raw = np.asarray(indices)
    if raw.dtype != np.dtype("uint32") or raw.ndim != 1 or raw.size < capacity:
        raise TypeError("GPU compact buffer must be uint32 with sufficient capacity")
    if n == 0:
        return np.empty((0, 3), dtype=np.int64)
    ids = np.sort(raw[:n].astype(np.int64))
    if ids[-1] >= (nx - 1) * (ny - 1) * z_cells:
        raise RuntimeError("Compact buffer contains out-of-range cell indices")
    if np.any(ids[1:] == ids[:-1]):
        raise RuntimeError("Compact buffer contains duplicate cell indices")
    cx, cy = nx - 1, ny - 1
    return np.column_stack((ids % cx, (ids // cx) % cy, ids // (cx * cy) + z0))


# ---------------------------------------------------------------------------
# v4: column marching, SIMD OR reduction, row summary
# ---------------------------------------------------------------------------


def column_layout(nx, ny, nz_nodes):
    """Sizes of the v4 outputs: words per row, rows, mask words, summary words."""
    cx, cy, cz = nx - 1, ny - 1, nz_nodes - 1
    wpr = (cx + 31) // 32
    nrows = cy * cz
    return wpr, nrows, nrows * wpr, max(1, (nrows + 31) // 32)


def make_column_kernel(nx: int, ny: int, nz_nodes: int, kchunk: int):
    """GPU kernel with plane reuse and SIMD row packing.

    Launch x is rounded up to a multiple of 32 and spare lanes are guarded.
    This keeps SIMD groups aligned with 32-bit output words even when nx-1
    is not a multiple of 32 (e.g., 127 for a 128-node grid).
    """
    _check_indexing(nx, ny, nz_nodes)
    if kchunk < 1:
        raise ValueError("kchunk must be positive")
    wpr, _, _, _ = column_layout(nx, ny, nz_nodes)
    source = f"""
        const uint NX = {nx}u;
        const uint CX = NX - 1u;
        const uint CY = {ny - 1}u;
        const uint CZ = {nz_nodes - 1}u;
        const uint KCHUNK = {kchunk}u;
        const uint WPR = {wpr}u;
        const uint SZ = NX * {ny}u;

        uint i = thread_position_in_grid.x;
        uint j = thread_position_in_grid.y;
        uint k0 = thread_position_in_grid.z * KCHUNK;
        uint k1 = metal::min(CZ, k0 + KCHUNK);
        bool valid = (i < CX) && (j < CY) && (k0 < CZ);
        uint col = i + NX * j;

        float lox0 = 0.0f, hix0 = 0.0f;
        bool fx0 = false;
        if (valid) {{
            plane_minmax(bx, col + k0 * SZ, NX, lox0, hix0, fx0);
        }}
        float loy0 = 0.0f, hiy0 = 0.0f, loz0 = 0.0f, hiz0 = 0.0f;
        bool fy0 = false, fz0 = false;
        uint cached = 0xffffffffu;

        for (uint k = k0; k < k1; ++k) {{
            float lox1 = 0.0f, hix1 = 0.0f;
            bool fx1 = false;
            bool pass = false;
            uint b1 = col + (k + 1u) * SZ;
            if (valid) {{
                plane_minmax(bx, b1, NX, lox1, hix1, fx1);
                pass = fx0 && fx1 &&
                       metal::min(lox0, lox1) <= 0.0f &&
                       metal::max(hix0, hix1) >= 0.0f;
                if (pass) {{
                    if (cached != k) {{
                        plane_minmax(by, col + k * SZ, NX, loy0, hiy0, fy0);
                        plane_minmax(bz, col + k * SZ, NX, loz0, hiz0, fz0);
                    }}
                    float loy1, hiy1, loz1, hiz1;
                    bool fy1, fz1;
                    plane_minmax(by, b1, NX, loy1, hiy1, fy1);
                    plane_minmax(bz, b1, NX, loz1, hiz1, fz1);
                    pass = fy0 && fy1 &&
                           metal::min(loy0, loy1) <= 0.0f &&
                           metal::max(hiy0, hiy1) >= 0.0f &&
                           fz0 && fz1 &&
                           metal::min(loz0, loz1) <= 0.0f &&
                           metal::max(hiz0, hiz1) >= 0.0f;
                    loy0 = loy1; hiy0 = hiy1; fy0 = fy1;
                    loz0 = loz1; hiz0 = hiz1; fz0 = fz1;
                    cached = k + 1u;
                }}
            }}
            // No early return: all SIMD lanes, including padded lanes, vote.
            uint bits = simd_or(pass ? (1u << (i & 31u)) : 0u);
            if (valid && (i & 31u) == 0u && bits != 0u) {{
                uint row = k * CY + j;
                atomic_store_explicit(&mask[row * WPR + (i >> 5)], bits,
                                      memory_order_relaxed);
                atomic_fetch_or_explicit(&rows[row >> 5], 1u << (row & 31u),
                                         memory_order_relaxed);
            }}
            lox0 = lox1; hix0 = hix1; fx0 = fx1;
        }}
    """
    return mx.fast.metal_kernel(
        name=f"nf_column_safe_{nx}_{ny}_{nz_nodes}_{kchunk}",
        input_names=["bx", "by", "bz"],
        output_names=["mask", "rows"],
        source=source,
        header=METAL_HEADER,
        ensure_row_contiguous=True,
        atomic_outputs=True,
        compile_options={"math_mode": "safe"},
    )


DECODE_MAX_ACTIVE_ROWS = 65_536


def column_to_candidates(mask, rows, nx, ny, z_cells, z0):
    """Sorted (i, j, k) from the v4 row-padded mask, reading only active rows."""
    cx, cy = nx - 1, ny - 1
    wpr, nrows, nwords, _ = column_layout(nx, ny, z_cells + 1)
    mask = np.asarray(mask)
    rows = np.asarray(rows)
    expected_summary = max(1, (nrows + 31) // 32)
    if mask.dtype != np.dtype("uint32") or mask.size != nwords:
        raise TypeError("v4 mask must be uint32 with CY*CZ*WPR words")
    if rows.dtype != np.dtype("uint32") or rows.size != expected_summary:
        raise TypeError("v4 row summary must be uint32 with ceil(nrows/32) words")
    summary = np.ascontiguousarray(rows.astype("<u4", copy=False)).view(np.uint8)
    active = np.flatnonzero(np.unpackbits(summary, bitorder="little"))
    if active.size and active[-1] >= nrows:
        raise RuntimeError("Row summary has bits beyond the last row")
    if not active.size:
        return np.empty((0, 3), dtype=np.int64)
    by_row = mask.reshape(nrows, wpr)
    if np.any(~np.any(by_row[active], axis=1)):
        raise RuntimeError("Row summary marks a row without any candidate bits")
    pieces = []
    for left in range(0, active.size, DECODE_MAX_ACTIVE_ROWS):
        r = active[left : left + DECODE_MAX_ACTIVE_ROWS]
        words = np.ascontiguousarray(by_row[r].astype("<u4", copy=False))
        bits = np.unpackbits(words.view(np.uint8), bitorder="little")
        r_idx, i = np.nonzero(bits.reshape(len(r), wpr * 32))  # row-major: sorted
        if i.size and i.max() >= cx:
            raise RuntimeError("v4 mask has bits beyond the last cell of a row")
        row = r[r_idx].astype(np.int64)
        pieces.append(np.column_stack((i, row % cy, row // cy + z0)))
    out = np.concatenate(pieces).astype(np.int64, copy=False)
    if not len(out):
        raise RuntimeError("Row summary marks rows whose mask words are empty")
    return out


# ---------------------------------------------------------------------------
# Common driver
# ---------------------------------------------------------------------------


class Mode:
    """One GPU variant: kernel factory, launch and decode."""

    def __init__(self, name, threadgroup, capacity, kchunk):
        if name not in ALL_MODES:
            raise ValueError(name)
        self.name = name
        self.tg = threadgroup
        self.capacity = capacity
        self.kchunk = kchunk
        self.label = {
            "mask": "v2 mask",
            "compact": "v3 compact",
            "column": "v4 column",
        }[name]

    def build(self, nx, ny, nodes):
        if self.name == "mask":
            return make_kernel(nx, ny, nodes)
        if self.name == "compact":
            return make_compact_kernel(nx, ny, nodes, self.capacity)
        return make_column_kernel(nx, ny, nodes, self.kchunk)

    def launch(self, kernel, arrays, nx, ny, nodes):
        cells = (nx - 1) * (ny - 1) * (nodes - 1)
        nwords = (cells + 31) // 32
        if self.name == "mask":
            return kernel(
                inputs=list(arrays),
                grid=(nwords, 1, 1),
                threadgroup=(self.tg, 1, 1),
                output_shapes=[(nwords,)],
                output_dtypes=[mx.uint32],
            )
        if self.name == "compact":
            return kernel(
                inputs=list(arrays),
                grid=(nwords, 1, 1),
                threadgroup=(self.tg, 1, 1),
                output_shapes=[(self.capacity,), (1,)],
                output_dtypes=[mx.uint32, mx.uint32],
                init_value=0,
            )
        _, _, nmask, nsum = column_layout(nx, ny, nodes)
        nchunks = (nodes - 1 + self.kchunk - 1) // self.kchunk
        padded_x = ((nx - 1 + 31) // 32) * 32
        return kernel(
            inputs=list(arrays),
            grid=(padded_x, ny - 1, nchunks),
            threadgroup=(min(self.tg, padded_x), 1, 1),
            output_shapes=[(nmask,), (nsum,)],
            output_dtypes=[mx.uint32, mx.uint32],
            init_value=0,
        )

    def decode(self, host, nx, ny, nodes, z0):
        if self.name == "mask":
            return mask_to_candidates(host[0], nx, ny, nodes - 1, z0)
        if self.name == "compact":
            return compact_to_candidates(
                host[0], host[1], nx, ny, nodes - 1, z0, self.capacity
            )
        return column_to_candidates(host[0], host[1], nx, ny, nodes - 1, z0)


def gpu_to_numpy(output, readback):
    """Zero-copy view of an evaluated MLX array, or an explicit copy."""
    if readback == "view":
        return np.array(output, copy=False)
    if readback == "copy":
        return np.array(output, copy=True)
    raise ValueError("readback must be 'view' or 'copy'")


def slab_specs(shape, slab_cells):
    nz = shape[2]
    if min(shape) < 2:
        raise ValueError("All dimensions must have at least two nodes.")
    if slab_cells < 1:
        raise ValueError("--slab-cells must be positive.")
    for z0 in range(0, nz - 1, slab_cells):
        yield z0, min(nz - 1, z0 + slab_cells)  # nodes z0..z1, cells z0..z1-1


def flat_numpy_slab(field, z0, z1):
    return np.ravel(field[:, :, z0 : z1 + 1], order="F")


def gpu_scan(mode, kernels, source, slabs, nx, ny, readback, batch_slabs):
    """One full GPU scan -> sorted candidates, with synchronised stage timings.

    ``source`` is either a tuple of three NumPy fields (copied slab by slab into
    MLX, ``upload_s`` > 0) or a tuple of three flat MLX arrays holding the whole
    field (unified input; ``slabs`` must then be a single whole-field slab).
    """
    unified = isinstance(source[0], mx.array)
    times = dict.fromkeys(("upload_s", "kernel_s", "readback_s", "decode_s"), 0.0)
    start = time.perf_counter()
    pieces = []
    for left in range(0, len(slabs), batch_slabs):
        group = slabs[left : left + batch_slabs]
        t0 = time.perf_counter()
        if unified:
            inputs = [source for _ in group]
        else:
            inputs = [
                tuple(mx.array(flat_numpy_slab(f, z0, z1)) for f in source)
                for z0, z1 in group
            ]
            mx.eval(*(a for inp in inputs for a in inp))
        times["upload_s"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        outs = [
            mode.launch(kernels[z1 - z0 + 1], inp, nx, ny, z1 - z0 + 1)
            for (z0, z1), inp in zip(group, inputs, strict=True)
        ]
        mx.eval(*(o for t in outs for o in t))
        times["kernel_s"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        host = [[gpu_to_numpy(o, readback) for o in t] for t in outs]
        times["readback_s"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        for (z0, z1), h in zip(group, host, strict=True):
            pieces.append(mode.decode(h, nx, ny, z1 - z0 + 1, z0))
        times["decode_s"] += time.perf_counter() - t0
        del host, outs, inputs
    result = np.concatenate(pieces, axis=0) if pieces else np.empty((0, 3), np.int64)
    times["total_s"] = time.perf_counter() - start
    return result, times


def build_kernels(mode, nx, ny, slabs):
    return {z1 - z0 + 1: mode.build(nx, ny, z1 - z0 + 1) for z0, z1 in slabs}


# ---------------------------------------------------------------------------
# Unified-memory input
# ---------------------------------------------------------------------------


def unified_fields(shape):
    """Three MLX float32 buffers and writable Fortran-order NumPy views of them.

    The views alias the MLX memory (checked here), so anything written through
    them is read by GPU kernels without a copy. Writing behind MLX's back is
    outside MLX's array semantics; it is safe here because the buffers are
    evaluated before any write and are only read by kernels afterwards.
    """
    n = int(np.prod(shape))
    flat = [mx.zeros((n,), dtype=mx.float32) for _ in range(3)]
    mx.eval(*flat)
    views = []
    for a in flat:
        v = np.array(a, copy=False)
        if v.strides != (4,) or v.size != n or not v.flags.writeable:
            raise RuntimeError("MLX buffer cannot be viewed as a writable NumPy array")
        v[0], v[-1] = 1.5, -2.5
        if a[0].item() != 1.5 or a[-1].item() != -2.5:
            raise RuntimeError("NumPy view does not alias MLX memory on this system")
        views.append(v.reshape(shape, order="F"))
    return tuple(flat), tuple(views)


def to_unified(fields):
    """Copy existing NumPy fields into unified buffers (used by the self-test)."""
    flat, views = unified_fields(fields[0].shape)
    for v, f in zip(views, fields, strict=True):
        v[...] = f
    return flat, views


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_same(cpu, gpu, label=""):
    if np.array_equal(cpu, gpu):
        return
    n = min(len(cpu), len(gpu))
    diff = np.flatnonzero(np.any(cpu[:n] != gpu[:n], axis=1))
    first = int(diff[0]) if diff.size else n
    a = cpu[first] if first < len(cpu) else None
    b = gpu[first] if first < len(gpu) else None
    raise RuntimeError(
        f"CPU/GPU candidate mismatch {label}: CPU={len(cpu)}, GPU={len(gpu)}; "
        f"first mismatch at {first}: CPU={a}, GPU={b}"
    )


def self_test(modes, threadgroup, capacity, kchunk):
    """Every mode vs the CPU scan: C/F layouts, NaN/Inf, exact zeros, slab
    boundaries, partial SIMD groups and rows wider than 32 cells, dense and
    empty fields, unified input, v3 overflow detection."""
    rng = np.random.default_rng(2026)
    cases = []
    for shape in ((9, 8, 7), (45, 6, 5), (70, 3, 9)):
        f = tuple(
            np.asfortranarray(rng.normal(size=shape).astype(np.float32))
            for _ in range(3)
        )
        f[0][2, 1, 1] = 0.0
        f[1][4, 1, 3] = np.nan
        f[2][5, 2, 2] = np.inf
        cases.append(f)
    dense = tuple(np.zeros((40, 5, 4), dtype=np.float32, order="F") for _ in range(3))
    const = tuple(np.ones((9, 8, 7), dtype=np.float32, order="F") for _ in range(3))
    cases += [dense, const]

    for name in modes:
        mode = Mode(name, threadgroup, capacity, kchunk)
        for fields in cases:
            shape = fields[0].shape
            cpu, _ = _get_candidate_cells(*fields)
            if name == "compact" and len(cpu) > capacity:
                continue  # overflow is tested separately below
            nx, ny, _ = shape
            for variant in (fields, tuple(np.ascontiguousarray(f) for f in fields)):
                slabs = list(slab_specs(shape, 2))
                kernels = build_kernels(mode, nx, ny, slabs)
                for readback in ("view", "copy"):
                    gpu, _ = gpu_scan(
                        mode, kernels, variant, slabs, nx, ny, readback, batch_slabs=2
                    )
                    validate_same(cpu, gpu, f"[{name}, numpy {shape}]")
            flat, _ = to_unified(fields)
            whole = [(0, shape[2] - 1)]
            gpu, _ = gpu_scan(
                mode, build_kernels(mode, nx, ny, whole), flat, whole, nx, ny, "view", 1
            )
            validate_same(cpu, gpu, f"[{name}, unified {shape}]")

    if "compact" in modes:
        mode = Mode("compact", threadgroup, 1, kchunk)
        shape = dense[0].shape
        slabs = list(slab_specs(shape, 2))
        overflowed = False
        try:
            gpu_scan(
                mode,
                build_kernels(mode, shape[0], shape[1], slabs),
                dense,
                slabs,
                shape[0],
                shape[1],
                "view",
                2,
            )
        except RuntimeError as err:
            overflowed = "overflow" in str(err)
        if not overflowed:
            raise AssertionError("Expected GPU compaction capacity overflow")
    print(
        f"Self-test PASS ({', '.join(modes)}; C/F, NaN/Inf, zeros, slabs, "
        "rows > 32 cells, dense, empty, unified input)."
    )


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def stats(samples):
    return {
        "median_s": float(statistics.median(samples)),
        "min_s": float(min(samples)),
        "sd_s": float(statistics.stdev(samples)) if len(samples) > 1 else 0.0,
        "repeats": len(samples),
    }


def time_cpu(fields, repeat, threads, chunksize):
    import numba

    if threads > numba.config.NUMBA_NUM_THREADS:
        raise ValueError(
            f"CPU threads {threads} exceeds Numba max {numba.config.NUMBA_NUM_THREADS}"
        )
    before_threads = numba.get_num_threads()
    before_chunk = numba.get_parallel_chunksize()
    try:
        numba.set_num_threads(threads)
        numba.set_parallel_chunksize(chunksize)
        q = np.ones((4, 4, 4), dtype=np.float32, order="F")
        _get_candidate_cells(q, q, q)  # compilation before timing
        layer = numba.threading_layer()
        if chunksize > 0 and layer != "tbb":
            warnings.warn(
                f"CPU chunksize={chunksize} only schedules dynamically with TBB "
                f"(active layer: {layer}).",
                RuntimeWarning,
                stacklevel=2,
            )
        _get_candidate_cells(*fields)  # warm-up on the real data
        measured, cpu = [], None
        for _ in range(repeat):
            t0 = time.perf_counter()
            cpu, _ = _get_candidate_cells(*fields)
            measured.append(time.perf_counter() - t0)
    finally:
        numba.set_parallel_chunksize(before_chunk)
        numba.set_num_threads(before_threads)
    return cpu, measured, layer


def main(argv=None):
    import numba

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n", type=int, default=128, help="grid nodes per direction")
    p.add_argument("--repeat", type=int, default=5, help="timed repetitions")
    p.add_argument(
        "--modes",
        default=",".join(ALL_MODES),
        help="comma-separated GPU variants: column, mask, compact",
    )
    p.add_argument(
        "--input",
        choices=("unified", "numpy"),
        default="unified",
        help="unified: zero-copy MLX buffers; numpy: copy slabs each run",
    )
    p.add_argument(
        "--slab-cells", type=int, default=128, help="z cells per slab (--input numpy)"
    )
    p.add_argument(
        "--batch-slabs",
        type=int,
        default=1,
        help="slabs copied and launched together (--input numpy)",
    )
    p.add_argument(
        "--threadgroup",
        type=int,
        default=256,
        help="GPU threads per threadgroup (multiple of 32)",
    )
    p.add_argument(
        "--kchunk",
        type=int,
        default=64,
        help="z planes per thread in the column kernel",
    )
    p.add_argument("--readback", choices=("view", "copy"), default="view")
    p.add_argument(
        "--compact-capacity",
        type=int,
        default=65_536,
        help="v3: maximum candidates per launch; overflow raises",
    )
    p.add_argument("--cpu-threads", type=int, default=numba.config.NUMBA_NUM_THREADS)
    p.add_argument(
        "--cpu-chunksize",
        type=int,
        default=0,
        help="Numba chunk size (dynamic scheduling needs TBB)",
    )
    p.add_argument("--skip-cpu", action="store_true")
    p.add_argument("--no-self-test", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "metal",
    )
    a = p.parse_args(argv)
    modes = tuple(m.strip() for m in a.modes.split(",") if m.strip())
    if not modes or any(m not in ALL_MODES for m in modes):
        p.error(f"--modes must be a comma-separated subset of {ALL_MODES}")
    if a.n < 2 or a.repeat < 1 or a.slab_cells < 1 or a.batch_slabs < 1:
        p.error("Require n>=2, repeat>=1, slab-cells>=1, batch-slabs>=1")
    if not 32 <= a.threadgroup <= 1024 or a.threadgroup % 32:
        p.error("--threadgroup must be a multiple of 32 in 32..1024")
    if a.kchunk < 1 or a.cpu_threads < 1 or a.cpu_chunksize < 0:
        p.error("Require kchunk>=1, cpu-threads>=1 and cpu-chunksize>=0")
    if not 1 <= a.compact_capacity < 2**32:
        p.error("Require 1 <= compact-capacity < 2**32")

    print(f"MLX {getattr(mx, '__version__', 'unknown')} | device {mx.default_device()}")
    print("PROTOTYPE: candidate scan only (no Bernstein, Newton or topology).")
    if a.input == "unified":
        print(
            "Input mode: MLX-backed shared buffers; generation and disk I/O "
            "are excluded from scan timings."
        )
    else:
        print("Input mode: NumPy arrays; each scan includes NumPy-to-MLX copies.")
    if not a.no_self_test:
        self_test(modes, a.threadgroup, a.compact_capacity, a.kchunk)

    c = create_grid(a.n)
    shape = (a.n, a.n, a.n)
    t0 = time.perf_counter()
    if a.input == "unified":
        flat, fields = unified_fields(shape)
        olshevsky2020_field(c, c, c, out=fields)
        source, slabs = flat, [(0, a.n - 1)]
    else:
        fields = olshevsky2020_field(c, c, c, dtype=np.float32)
        source, slabs = fields, list(slab_specs(shape, a.slab_cells))
    field_build = time.perf_counter() - t0
    print(
        f"Field {a.n}^3 float32 built in {field_build:.2f} s "
        f"({sum(f.nbytes for f in fields) / 1e9:.2f} GB, input={a.input}, "
        f"{len(slabs)} GPU launch(es) per scan)"
    )

    results = {
        "n": a.n,
        "dtype": "float32",
        "input": a.input,
        "mlx": getattr(mx, "__version__", "unknown"),
        "platform": platform.platform(),
        "slab_cells": a.slab_cells,
        "batch_slabs": a.batch_slabs,
        "threadgroup": a.threadgroup,
        "kchunk": a.kchunk,
        "readback": a.readback,
        "modes": list(modes),
        "compact_capacity": a.compact_capacity,
        "field_build_s": field_build,
        "note": "candidate scan only; field generation and Metal JIT excluded",
    }

    cpu_ref = None
    if not a.skip_cpu:
        cpu_ref, ct, layer = time_cpu(fields, a.repeat, a.cpu_threads, a.cpu_chunksize)
        results["cpu"] = {
            **stats(ct),
            "threads": a.cpu_threads,
            "chunksize": a.cpu_chunksize,
            "threading_layer": layer,
            "n_candidates": len(cpu_ref),
        }
        print(
            f"CPU ({layer}, {a.cpu_threads} threads): "
            f"median {results['cpu']['median_s'] * 1e3:.2f} ms, "
            f"{len(cpu_ref):,} candidates"
        )
    else:
        print("WARNING: CPU validation skipped; GPU results UNVALIDATED.")

    stage_keys = ("upload_s", "kernel_s", "readback_s", "decode_s")
    for name in modes:
        mode = Mode(name, a.threadgroup, a.compact_capacity, a.kchunk)
        kernels = build_kernels(mode, a.n, a.n, slabs)
        ref, _ = gpu_scan(
            mode, kernels, source, slabs, a.n, a.n, a.readback, a.batch_slabs
        )  # JIT warm-up and validation
        if cpu_ref is not None:
            validate_same(cpu_ref, ref, f"[{name}]")
        totals, stages = [], {k: [] for k in stage_keys}
        for _ in range(a.repeat):
            got, tm = gpu_scan(
                mode, kernels, source, slabs, a.n, a.n, a.readback, a.batch_slabs
            )
            if not np.array_equal(ref, got):
                raise RuntimeError(f"GPU {name} result changed between runs")
            totals.append(tm["total_s"])
            for k in stage_keys:
                stages[k].append(tm[k])
        results[name] = {
            **stats(totals),
            "n_candidates": len(ref),
            "validated": cpu_ref is not None,
            "stages": {k: stats(v) for k, v in stages.items()},
        }
        med = {k: results[name]["stages"][k]["median_s"] * 1e3 for k in stage_keys}
        print(
            f"GPU {mode.label:<11}: median {results[name]['median_s'] * 1e3:8.2f} ms"
            f"  (upload {med['upload_s']:.2f}, kernel {med['kernel_s']:.2f}, "
            f"readback {med['readback_s']:.2f}, decode {med['decode_s']:.2f} ms)"
            + ("  validated" if cpu_ref is not None else "")
        )

    if cpu_ref is not None:
        cpu_med = results["cpu"]["median_s"]
        print("\nCPU time / GPU time (>1: GPU faster):")
        for name in modes:
            kmed = results[name]["stages"]["kernel_s"]["median_s"]
            label = Mode(name, 32, 1, 1).label
            total = cpu_med / results[name]["median_s"]
            print(
                f"  {label:<11} total {total:6.2f}x   "
                f"kernel only {cpu_med / kmed:6.2f}x"
            )

    if not a.no_save:
        a.out.mkdir(parents=True, exist_ok=True)
        mode_tag = "-".join(modes)
        base = (
            f"metal_candidate_scan_N{a.n}_{a.input}_{mode_tag}"
            f"_tg{a.threadgroup}_k{a.kchunk}_z{a.slab_cells}_v4"
        )
        dest = a.out / f"{base}.json"
        suffix = 1
        while dest.exists():
            dest = a.out / f"{base}_{suffix:02d}.json"
            suffix += 1
        dest.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
