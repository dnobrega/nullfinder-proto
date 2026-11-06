"""Thread scaling of nullfinder_proto on a large Olshevsky et al. (2020) field.

The analytic field with 14 known nulls is sampled on an N^3 grid, and
``find_magnetic_nulls`` is timed with different numbers of Numba threads. For each
thread count the script reports the best of ``--repeat`` runs, split by stage, the
speedup and parallel efficiency with respect to the smallest thread count, and
checks that the 14 nulls are recovered. Optionally, Chiti's (2020) single-threaded
finder is timed on the same field for comparison.

Examples
--------
    python benchmarks/thread_scaling.py --n 1024 --threads 1,2,4,8,16
    python benchmarks/thread_scaling.py --n 512 --chiti Null_Finder/null_finder.py

Memory: the field takes 3 * N^3 * itemsize bytes (about 13 GB for 1024^3 in
float32, the default dtype). Chiti's finder needs roughly twice as much again,
and should be compared with --dtype float64: with float32 input it can miss
nulls.
The maximum number of threads is set by Numba (environment variable
NUMBA_NUM_THREADS, by default the number of cores).

Outputs ``thread_scaling_N<N>.csv`` and ``thread_scaling_N<N>.png`` in ``--out``.
"""

import argparse
import csv
import importlib.util
import os
import platform
import sys
import time
from pathlib import Path

import numba
import numpy as np

import nullfinder_proto as nf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from olshevsky2020_field import create_grid, match_exact, olshevsky2020_field

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
    need = 3 * n**3 * np.dtype(dtype).itemsize * (3 if with_chiti else 1)
    total = physical_memory_bytes()
    msg = f"Estimated memory: {need / 1e9:.1f} GB"
    if total:
        msg += f" of {total / 1e9:.1f} GB physical"
    print(msg)
    if total and need > 0.85 * total and not force:
        raise SystemExit(
            "Not enough memory for this grid. Use a smaller --n, --dtype float32, "
            "drop --chiti, or pass --force to try anyway."
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


def scaling(n, dtype, threads, repeat, chiti_path=None):
    """Run the benchmark. Returns (rows, chiti_row or None, info dict)."""
    c = create_grid(n)
    h = c[1] - c[0]

    # Compile (and start the thread pool) on a small grid of the same dtype.
    cw = create_grid(16)
    run_nullfinder(*olshevsky2020_field(cw, cw, cw, dtype=dtype), cw)

    t0 = time.perf_counter()
    bx, by, bz = olshevsky2020_field(c, c, c, dtype=dtype)
    t_field = time.perf_counter() - t0
    print(f"Field {n}^3 ({np.dtype(dtype).name}) built in {t_field:.1f} s, h = {h:.5f}")

    rows = []
    max_threads = numba.config.NUMBA_NUM_THREADS
    try:
        for t in threads:
            numba.set_num_threads(t)
            best = None
            for _ in range(repeat):
                wall, res = run_nullfinder(bx, by, bz, c)
                if best is None or wall < best[0]:
                    best = (wall, res)
            wall, res = best
            names = [nf.null_type_name(k) for k in res["null_type"]]
            matched, types_ok, d = match_exact(res["pos"], names, h=h)
            tm = res["timing"]
            row = {
                "threads": t,
                "total_s": wall,
                **{k: tm[k] for k in STAGES},
                "other_s": wall - sum(tm[k] for k in STAGES),
                "found": len(res["pos"]),
                "matched": matched,
                "types_ok": types_ok,
                "max_err": float(d.max()) if len(d) else np.nan,
            }
            rows.append(row)
            print(
                f"  {t:3d} threads: {wall:8.3f} s  (scan {tm['candidate_s']:.3f}, "
                f"exclusion {tm['exclusion_s']:.3f}, Newton {tm['refine_s']:.3f})  "
                f"nulls {row['found']}, matched {matched}/14, types ok {types_ok}"
            )
    finally:
        numba.set_num_threads(max_threads)

    base = rows[0]
    for r in rows:
        r["speedup"] = base["total_s"] / r["total_s"]
        r["efficiency"] = r["speedup"] * base["threads"] / r["threads"]
        r["speedup_scan"] = base["candidate_s"] / r["candidate_s"]

    chiti_row = None
    if chiti_path is not None:
        if np.dtype(dtype) != np.float64:
            print(
                "  Note: with float32 input Chiti's finder can miss nulls and return "
                "spurious ones\n  (seen at 512^3: null 14 lost); use --dtype float64 "
                "for a like-for-like comparison."
            )
        mod = load_chiti(chiti_path)
        t0 = time.perf_counter()
        nulls = mod.find_nulls(n, n, n, c, c, c, bx, by, bz)
        wall = time.perf_counter() - t0
        pos = np.array([[q["x"], q["y"], q["z"]] for q in nulls]).reshape(-1, 3)
        matched, _, _ = match_exact(pos, h=h)
        chiti_row = {"total_s": wall, "found": len(pos), "matched": matched}
        print(
            f"  Chiti (single thread): {wall:8.3f} s  "
            f"nulls {len(pos)}, matched {matched}/14"
        )

    info = {
        "n": n,
        "dtype": np.dtype(dtype).name,
        "field_build_s": t_field,
        "cpu_count": os.cpu_count(),
        "numba_max_threads": max_threads,
        "threading_layer": numba.threading_layer(),
        "machine": f"{platform.system()} {platform.machine()} {platform.processor()}",
    }
    return rows, chiti_row, info


def write_csv(rows, chiti_row, info, path):
    with open(path, "w", newline="") as f:
        for k, v in info.items():
            f.write(f"# {k}: {v}\n")
        if chiti_row:
            f.write(f"# chiti_total_s: {chiti_row['total_s']:.6g}\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot(rows, chiti_row, info, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    th = np.array([r["threads"] for r in rows])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax1.loglog(th, [r["total_s"] for r in rows], "o-", label="nullfinder_proto (total)")
    ax1.loglog(th, [r["candidate_s"] for r in rows], "s--", label="candidate scan")
    newton = [r["exclusion_s"] + r["refine_s"] for r in rows]
    ax1.loglog(th, newton, "^:", label="exclusion + Newton")
    if chiti_row:
        ax1.axhline(chiti_row["total_s"], color="k", ls="-.", label="Chiti (1 thread)")
    ax1.set(xlabel="Numba threads", ylabel="time [s]")
    ax1.set_title(f"{info['n']}$^3$ grid, {info['dtype']}")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.plot(th, [r["speedup"] for r in rows], "o-", label="total")
    ax2.plot(th, [r["speedup_scan"] for r in rows], "s--", label="candidate scan")
    ax2.plot(th, th / th[0], "k:", label="ideal")
    ax2.set(xlabel="Numba threads", ylabel=f"speedup vs {th[0]} thread(s)")
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
    p.add_argument("--repeat", type=int, default=3, help="runs per thread count")
    p.add_argument("--chiti", type=Path, help="path to Chiti's null_finder.py")
    p.add_argument("--out", type=Path, default=Path("."), help="output folder")
    p.add_argument("--no-plot", action="store_true", help="skip the PNG figure")
    p.add_argument("--force", action="store_true", help="skip the memory check")
    a = p.parse_args(argv)

    threads = parse_threads(a.threads, numba.config.NUMBA_NUM_THREADS)
    check_memory(a.n, a.dtype, a.chiti is not None, a.force)
    rows, chiti_row, info = scaling(a.n, a.dtype, threads, a.repeat, a.chiti)

    print(f"\n{'threads':>7} {'time [s]':>9} {'speedup':>8} {'efficiency':>10}")
    for r in rows:
        print(
            f"{r['threads']:7d} {r['total_s']:9.3f} "
            f"{r['speedup']:8.2f} {r['efficiency']:10.0%}"
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
