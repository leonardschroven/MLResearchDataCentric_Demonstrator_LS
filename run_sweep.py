"""
Headless, resumable runner for the Data-Selective Training parameter sweep.

Run this in your OWN terminal (not through Streamlit) so a long sweep is not tied
to a browser session:

    cd "C:\\Users\\leona\\OneDrive\\Desktop\\MS_Research_Demonstrator\\Application"
    .\\.venv\\Scripts\\python.exe run_sweep.py                 # sequential
    .\\.venv\\Scripts\\python.exe run_sweep.py --workers 6     # parallel (6 cores)
    .\\.venv\\Scripts\\python.exe run_sweep.py --dry-run       # just show what's left

It is safe to stop (Ctrl+C) and restart at any time: completed configurations are
already on disk and are skipped on the next run. The sweep now spans all three
selection methods; historical rows (method 'Weakpoint distance') are reused.

The FIXED fields below must match the values used for the existing CSV rows, or the
resume key won't match and those rows will re-run. They mirror the page's sidebar
defaults; override any of them with --key value if you swept different fixed values.
"""
from __future__ import annotations

import argparse
import time

from scripts.dataselect import sweep as SW

# Held-fixed fields — mirror the Data-Selective Training page sidebar defaults.
FIXED_DEFAULTS = dict(
    model_name="MLP Neural Network", complexity=0.5,
    center_x=0.5, center_y=0.5,
    n_eval=1000, grid_res=35, extract_q=0.85,
    sel_mode="Sample ∝ weight",
    shift_center_x=0.30, shift_center_y=0.30, shift_spread=0.15,
    seed=42,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel worker processes (joblib/loky). 1 = sequential.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only report done / remaining / ETA, then exit.")
    # allow overriding any fixed field, e.g. --seed 7 --complexity 0.3
    for k, v in FIXED_DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    args = ap.parse_args()

    fixed = {k: getattr(args, k) for k in FIXED_DEFAULTS}
    total = len(SW.parameter_grid())
    remaining = SW.remaining_combos(fixed, SW.CSV_PATH)
    done_here = total - len(remaining)
    eta_min = len(remaining) * SW.SECS_PER_RUN / 60

    print(f"CSV:        {SW.CSV_PATH}")
    print(f"Grid:       {total:,} configs  ({', '.join(f'{k}x{len(v)}' for k, v in SW.SWEEP_GRID.items())})")
    print(f"Done (this fixed set): {done_here:,}")
    print(f"Remaining:  {len(remaining):,}  ->  ~{eta_min:,.0f} min at {SW.SECS_PER_RUN:.0f}s/config"
          + (f"  (~{eta_min/args.workers:,.0f} min with {args.workers} workers)" if args.workers > 1 else ""))
    if done_here == 0:
        print("WARNING: 0 done for this fixed set — your fixed values likely differ "
              "from the existing rows, so nothing will be reused.")
    if args.dry_run or not remaining:
        print("Nothing to run." if not remaining else "Dry run — exiting.")
        return

    t0 = time.time()
    n = len(remaining)

    def tick(done: int) -> None:
        el = time.time() - t0
        eta = (el / max(done, 1)) * (n - done) / 60
        print(f"  {done:,}/{n:,}  elapsed {el/60:5.1f} min  ETA {eta:6.1f} min", flush=True)

    if args.workers > 1:
        from joblib import Parallel, delayed, parallel_backend
        i, w, done = 0, args.workers, 0
        while i < n:
            batch = max(w * 4, 1)
            try:
                with parallel_backend("loky", inner_max_num_threads=1):
                    with Parallel(n_jobs=w) as par:
                        while i < n:
                            chunk = remaining[i:i + batch]
                            for rows in par(delayed(SW.safe_run_one)(p) for p in chunk):
                                if rows:
                                    SW.append_rows(SW.CSV_PATH, rows)
                            i += len(chunk); done += len(chunk); tick(done)
                break
            except Exception as e:
                w = max(1, w // 2)
                print(f"  workers terminated ({type(e).__name__}); reducing to {w} and continuing", flush=True)
                if w == 1:
                    for p in remaining[i:]:
                        rows = SW.safe_run_one(p)
                        if rows:
                            SW.append_rows(SW.CSV_PATH, rows)
                        i += 1; done += 1; tick(done)
                    break
    else:
        for i, p in enumerate(remaining, 1):
            try:
                SW.append_rows(SW.CSV_PATH, SW.run_one(p))
            except Exception as e:
                print(f"  config {i} failed: {e}", flush=True)
            if i % 10 == 0 or i == n:
                tick(i)

    print(f"DONE in {(time.time()-t0)/60:.1f} min. Rows appended to {SW.CSV_PATH}")


if __name__ == "__main__":
    main()
