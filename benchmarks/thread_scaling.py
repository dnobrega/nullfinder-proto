"""Thread scaling of nullfinder_proto on a large Olshevsky et al. (2020) field.

The analytic field with 14 known nulls is sampled on an N^3 grid, and
``find_magnetic_nulls`` is timed with different numbers of Numba threads. For each
thread count, the script runs an untimed warm-up on the full grid and reports
median wall time and standard deviation over ``--repeat`` runs. It also reports
stage timings from the run closest to the median, measured and ideal speedup,
and parallel efficiency relative to the smallest thread count. It checks the
14 known nulls. Optionally, Chiti's (2020) single-threaded finder is timed
repeatedly on the same field for comparison.

Examples
--------
    python benchmarks/thread_scaling.py --n 1024 --threads 1,2,4,8,16
    python benchmarks/thread_scaling.py --n 512 --chiti Null_Finder/null_finder.py

Memory: the input field alone takes 3 * N^3 * itemsize bytes (about 12.9 GB
for 1024^3 in float32, the default dtype). Field generation, intermediate
arrays, and Chiti's finder require additional memory; the check is only a
rough baseline estimate, not a peak-memory guarantee. For comparisons with
Chiti, prefer --dtype float64: with float32 input it can miss nulls.
The maximum number of threads is set by Numba (environment variable
NUMBA_NUM_THREADS, by default the number of cores).

``--threading-layer`` selects ``omp``, ``tbb``, or ``workqueue`` before Numba's
parallel runtime starts. ``--chunksize`` configures parallel chunk size; Numba
supports dynamic scheduling only on the TBB threading layer. Other backends
produce a warning when a positive chunk size is requested.

Outputs ``thread_scaling_N<N>.csv`` and ``thread_scaling_N<N>.png`` in
``benchmarks/results/`` by default (git-ignored). Set ``--out`` to override.
"""

import argparse
import csv
import importlib.util
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import numba
import numpy as np

import nullfinder_proto as nf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.olshevsky2020_field import (
    create_grid,
    match_exact,
    olshevsky2020_field,
)

STAGES = ("candidate_s", "exclusion_s", "refine_s")


def physical_memory_bytes():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def parse_threads(spec, max_threads):
    if spec == "auto":
        t, out = 1, []
        while t < max_threads:
            out.append(t)
            t *= 2
        return [*out, max_threads]
    threads = sorted({int(s) for s in spec.split(",")})
    bad = [t for t in threads if t < 1 or t > max_threads]
    if bad:
        raise SystemExit(
            f"Thread counts {bad} are outside 1..{max_threads}. Numba's maximum is "
            "set by the environment variable NUMBA_NUM_THREADS (default: cores)."
        )
    return threads


def check_memory(n, dtype, with_chiti, force):
    field_bytes = 3 * n**3 * np.dtype(dtype).itemsize
    # Heuristic only: intermediate allocations can greatly exceed this estimate.
    baseline_bytes = field_bytes * (3 if with_chiti else 1)
    total = physical_memory_bytes()
    msg = (
        f"Field storage: {field_bytes / 1e9:.1f} GB; "
        f"rough baseline estimate: {baseline_bytes / 1e9:.1f} GB "
        "(not peak memory; field-generation temporaries are excluded)"
    )
    if total:
        msg += f"; physical memory: {total / 1e9:.1f} GB"
    print(msg)
    if total and baseline_bytes > 0.85 * total and not force:
        raise SystemExit(
            "Memory estimate is too close to physical RAM. Use a smaller --n, "
            "--dtype float32, drop --chiti, or pass --force to try anyway."
        )


def load_chiti(path):
    spec = importlib.util.spec_from_file_location("chiti_null_finder", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["chiti_null_finder"] = mod
    spec.loader.exec_module(mod)
    return mod


def run_nullfinder(bx, by, bz, c):
    t0 = time.perf_counter()
    res = nf.find_magnetic_nulls(bx, by, bz, c, c, c, verbose=False)
    return time.perf_counter() - t0, res


def timing_stats(times):
    """Return median, mean, sample SD (or 0 for one run), and minimum."""
    values = np.asarray(times, dtype=float)
    return {
        "total_s": float(np.median(values)),
        "total_mean_s": float(np.mean(values)),
        "total_std_s": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "total_min_s": float(np.min(values)),
        "n_repeats": len(values),
    }


def scaling(n, dtype, threads, repeat, chiti_path=None):
    """Run the benchmark. Returns (rows, chiti_row or None, info dict)."""
    c = create_grid(n)
    h = c[1] - c[0]

    t0 = time.perf_counter()
    bx, by, bz = olshevsky2020_field(c, c, c, dtype=dtype)
    t_field = time.perf_counter() - t0
    print(f"Field {n}^3 ({np.dtype(dtype).name}) built in {t_field:.1f} s, h = {h:.5f}")
    active_layer = numba.threading_layer()
    active_chunksize = numba.get_parallel_chunksize()
    print(
        f"Numba threading layer: {active_layer}; "
        f"parallel chunk size: {active_chunksize} "
        "(0 = static scheduling)"
    )
    if active_chunksize > 0 and active_layer != "tbb":
        warnings.warn(
            f"--chunksize={active_chunksize} requested with Numba's "
            f"'{active_layer}' threading layer; dynamic scheduling "
            "is supported only with TBB. No dynamic scheduling improvement "
            "should be assumed.",
            RuntimeWarning,
            stacklevel=2,
        )

    rows = []
    max_threads = numba.config.NUMBA_NUM_THREADS
    try:
        for t in threads:
            numba.set_num_threads(t)
            # The first call on the REAL grid is deliberately untimed. It
            # includes JIT compilation on the first iteration, and warms caches
            # and the thread pool under the same workload used for measurement.
            run_nullfinder(bx, by, bz, c)
            trials = [run_nullfinder(bx, by, bz, c) for _ in range(repeat)]
            stats = timing_stats([wall for wall, _ in trials])
            # Use a single actual run for all stage timings, so that their
            # breakdown is consistent with its measured total wall time.
            idx = min(
                range(len(trials)),
                key=lambda i: abs(trials[i][0] - stats["total_s"]),
            )
            representative_wall, res = trials[idx]
            names = [nf.null_type_name(k) for k in res["null_type"]]
            matched, types_ok, d = match_exact(res["pos"], names, h=h)
            tm = res["timing"]
            row = {
                "threads": t,
                **stats,
                **{k: tm[k] for k in STAGES},
                "other_s": representative_wall - sum(tm[k] for k in STAGES),
                "found": len(res["pos"]),
                "matched": matched,
                "types_ok": types_ok,
                "max_err": float(d.max()) if len(d) else np.nan,
            }
            rows.append(row)
            print(
                f"  {t:3d} threads: median {stats['total_s']:8.3f} s "
                f"(SD {stats['total_std_s']:.3f}, min {stats['total_min_s']:.3f}; "
                f"scan {tm['candidate_s']:.3f}, exclusion {tm['exclusion_s']:.3f}, "
                f"Newton {tm['refine_s']:.3f})  "
                f"nulls {row['found']}, matched {matched}/14, types ok {types_ok}"
            )
    finally:
        numba.set_num_threads(max_threads)

    base = rows[0]
    for r in rows:
        r["speedup"] = base["total_s"] / r["total_s"]
        r["ideal_speedup"] = r["threads"] / base["threads"]
        r["parallel_efficiency"] = r["speedup"] / r["ideal_speedup"]
        r["speedup_scan"] = (
            base["candidate_s"] / r["candidate_s"] if r["candidate_s"] > 0 else np.nan
        )

    chiti_row = None
    if chiti_path is not None:
        if np.dtype(dtype) != np.float64:
            print(
                "  Note: with float32 input Chiti's finder can miss nulls and return "
                "spurious ones\n  (seen at 512^3: null 14 lost); use --dtype float64 "
                "for a like-for-like comparison."
            )
        mod = load_chiti(chiti_path)

        def call_chiti():
            start = time.perf_counter()
            nulls = mod.find_nulls(n, n, n, c, c, c, bx, by, bz)
            return time.perf_counter() - start, nulls

        # Same policy as for nullfinder_proto: warm up, then time repeats.
        call_chiti()
        chiti_trials = [call_chiti() for _ in range(repeat)]
        chiti_stats = timing_stats([wall for wall, _ in chiti_trials])
        idx = min(
            range(len(chiti_trials)),
            key=lambda i: abs(chiti_trials[i][0] - chiti_stats["total_s"]),
        )
        nulls = chiti_trials[idx][1]
        pos = np.array([[q["x"], q["y"], q["z"]] for q in nulls]).reshape(-1, 3)
        matched, _, _ = match_exact(pos, h=h)
        chiti_row = {**chiti_stats, "found": len(pos), "matched": matched}
        print(
            f"  Chiti (single thread): median {chiti_row['total_s']:8.3f} s "
            f"(SD {chiti_row['total_std_s']:.3f}), "
            f"nulls {len(pos)}, matched {matched}/14"
        )

    info = {
        "n": n,
        "dtype": np.dtype(dtype).name,
        "field_build_s": t_field,
        "timing_statistic": "median wall time (excluding full-grid warm-up)",
        "stage_timing": "one run closest to median wall time",
        "cpu_count": os.cpu_count(),
        "numba_max_threads": max_threads,
        "threading_layer": active_layer,
        "parallel_chunksize": active_chunksize,
        "machine": f"{platform.system()} {platform.machine()} {platform.processor()}",
    }
    return rows, chiti_row, info


def write_csv(rows, chiti_row, info, path):
    with open(path, "w", newline="") as f:
        for k, v in info.items():
            f.write(f"# {k}: {v}\n")
        if chiti_row:
            for k, v in chiti_row.items():
                f.write(f"# chiti_{k}: {v}\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot(rows, chiti_row, info, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    th = np.array([r["threads"] for r in rows])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax1.loglog(
        th, [r["total_s"] for r in rows], "o-", label="nullfinder_proto (median)"
    )
    ax1.loglog(th, [r["candidate_s"] for r in rows], "s--", label="candidate scan")
    newton = [r["exclusion_s"] + r["refine_s"] for r in rows]
    ax1.loglog(th, newton, "^:", label="exclusion + Newton")
    if chiti_row:
        ax1.axhline(
            chiti_row["total_s"], color="k", ls="-.", label="Chiti (median, 1 thread)"
        )
    ax1.set(xlabel="Numba threads", ylabel="time [s]")
    ax1.set_title(f"{info['n']}$^3$ grid, {info['dtype']}")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.plot(th, [r["speedup"] for r in rows], "o-", label="measured speedup")
    ax2.plot(th, [r["speedup_scan"] for r in rows], "s--", label="candidate scan")
    ax2.plot(th, [r["ideal_speedup"] for r in rows], "k:", label="ideal speedup")
    ax2.set(xlabel="Numba threads", ylabel=f"speedup vs {th[0]} thread(s)")
    ax2.set_title("Strong scaling")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    from matplotlib.ticker import NullFormatter

    for ax in (ax1, ax2):
        ax.set_xticks(th, [str(t) for t in th])
        ax.xaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n", type=int, default=1024, help="grid points per direction")
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--threads", default="auto", help='e.g. "1,2,4,8" (default: auto)')
    p.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="timed repetitions per thread count (plus one untimed warm-up)",
    )
    p.add_argument("--chiti", type=Path, help="path to Chiti's null_finder.py")
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="output folder (default: benchmarks/results/, ignored by Git)",
    )
    p.add_argument("--no-plot", action="store_true", help="skip the PNG figure")
    p.add_argument("--force", action="store_true", help="skip the memory check")
    p.add_argument(
        "--threading-layer",
        choices=["default", "omp", "tbb", "workqueue"],
        default="default",
        help="Numba threading layer (TBB requires a compatible TBB installation)",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=0,
        help="parallel chunk size (0: static scheduling; dynamic scheduling needs TBB)",
    )
    a = p.parse_args(argv)
    if a.repeat < 1:
        p.error("--repeat must be >= 1")
    if a.n < 2:
        p.error("--n must be >= 2")
    if a.chunksize < 0:
        p.error("--chunksize must be >= 0")

    # Numba's threading layer cannot be changed once its parallel runtime
    # has launched. This also prevents misleading results in an interactive
    # Python session that has already run other parallel Numba kernels.
    if a.threading_layer != "default":
        try:
            active_layer = numba.threading_layer()
        except ValueError:  # parallel runtime has not been initialized
            numba.config.THREADING_LAYER = a.threading_layer
        else:
            if active_layer != a.threading_layer:
                p.error(
                    f"Numba already initialized with '{active_layer}'; "
                    f"cannot switch to '{a.threading_layer}' in this process. "
                    "Run this benchmark in a fresh Python process."
                )

    threads = parse_threads(a.threads, numba.config.NUMBA_NUM_THREADS)
    check_memory(a.n, a.dtype, a.chiti is not None, a.force)
    previous_chunksize = numba.get_parallel_chunksize()
    try:
        numba.set_parallel_chunksize(a.chunksize)
        rows, chiti_row, info = scaling(a.n, a.dtype, threads, a.repeat, a.chiti)
    finally:
        numba.set_parallel_chunksize(previous_chunksize)

    b = rows[0]["threads"]
    sp = f"speedup (T{b}/Tn)"
    print(
        f"\n{'threads':>7}  {'median [s]':>11}  {'SD [s]':>9}  "
        f"{sp:>16}  {'ideal':>8}  {'parallel eff.':>13}"
    )
    for r in rows:
        print(
            f"{r['threads']:7d}  {r['total_s']:11.3f}  "
            f"{r['total_std_s']:9.3f}  {r['speedup']:16.2f}  "
            f"{r['ideal_speedup']:8.2f}  {r['parallel_efficiency']:13.0%}"
        )
    print(
        f"speedup = T({b}) / T(n); ideal speedup = n/{b}; "
        "parallel efficiency = measured / ideal (100% is ideal)."
    )
    if chiti_row:
        fastest = min(r["total_s"] for r in rows)
        print(
            f"Chiti / fastest nullfinder_proto: {chiti_row['total_s'] / fastest:.1f}x"
        )

    a.out.mkdir(parents=True, exist_ok=True)
    csv_path = a.out / f"thread_scaling_N{a.n}.csv"
    write_csv(rows, chiti_row, info, csv_path)
    print(f"\nWrote {csv_path}")
    if not a.no_plot:
        png = a.out / f"thread_scaling_N{a.n}.png"
        plot(rows, chiti_row, info, png)
        print(f"Wrote {png}")
    return rows, chiti_row, info


if __name__ == "__main__":
    main()
