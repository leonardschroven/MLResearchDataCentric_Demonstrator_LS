r"""
Offline runner for the Data-Selective Training parameter sweep.

Runs the exact same computation as the Streamlit page's "Run Sweep" button, but
head-less from a terminal — so it works over SSH on an EC2 instance and is not
tied to a browser session. It writes one row per (configuration x detector) to a
CSV and is fully resumable: stop with Ctrl+C and rerun to continue.

--------------------------------------------------------------------------------
Quick start (Linux / EC2)
--------------------------------------------------------------------------------
    git clone <repo> && cd MLResearchDataCentric_Demonstrator_LS   # the repo root
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

    # resume/append the default results file (default sidebar settings):
    python run_sweep.py --workers $(nproc)

    # OR start a clean, self-consistent run in a NEW file (keeps the old one):
    python run_sweep.py --fresh --workers $(nproc)

    # keep it running after you log out:
    nohup python run_sweep.py --fresh --workers $(nproc) > sweep.log 2>&1 &
    tail -f sweep.log

--------------------------------------------------------------------------------
Windows (PowerShell, from the Application folder)
--------------------------------------------------------------------------------
    .\.venv\Scripts\python.exe run_sweep.py --fresh --workers 6

--------------------------------------------------------------------------------
Output selection
--------------------------------------------------------------------------------
* default        -> append/resume  data/experiment_results/data_selective_training_sweep.csv
* --fresh        -> a NEW timestamped file  ...sweep_fresh_YYYYmmdd_HHMMSS.csv
                    (the existing results file is left completely untouched, so a
                     fresh run is consistent: one code version, one file, full grid)
* --output PATH  -> write/resume an explicit path (overrides --fresh)

It is always safe to stop and restart: completed configurations already on disk
are skipped. The sweep spans all three selection methods; when appending to a
pre-existing file, historical 'Weakpoint distance' rows are reused.

The FIXED fields below mirror the page's sidebar defaults and must match the
values used for any file you resume, or those rows won't match and will re-run.
Override any of them with --<field> <value> (e.g. --seed 7 --complexity 0.3).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import time
from pathlib import Path

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


def resolve_output(args) -> Path:
    """Pick the CSV to write to: explicit --output, else a fresh timestamped file, else default."""
    if args.output:
        return Path(args.output)
    if args.fresh:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        return SW.CSV_PATH.with_name(f"{SW.CSV_PATH.stem}_fresh_{ts}.csv")
    return SW.CSV_PATH


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel worker processes (joblib/loky). 1 = sequential.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only report target file, done / remaining / ETA, then exit.")
    ap.add_argument("--fresh", action="store_true",
                    help="Write to a NEW timestamped CSV, leaving existing results "
                         "untouched (full grid runs from scratch — a consistent, "
                         "single-code-version experiment).")
    ap.add_argument("--output", type=str, default=None,
                    help="Explicit CSV path to write/resume. Overrides --fresh.")
    # allow overriding any fixed field, e.g. --seed 7 --complexity 0.3
    for k, v in FIXED_DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    args = ap.parse_args()

    out = resolve_output(args)
    fixed = {k: getattr(args, k) for k in FIXED_DEFAULTS}
    total = len(SW.parameter_grid())
    remaining = SW.remaining_combos(fixed, out)
    done_here = total - len(remaining)
    eta_min = len(remaining) * SW.SECS_PER_RUN / 60
    existed = out.exists()

    print(f"Target CSV: {out}"
          + ("  (resuming existing file)" if existed else "  (new file)"))
    if out != SW.CSV_PATH:
        print(f"Default results file left untouched: {SW.CSV_PATH}")
    print(f"Grid:       {total:,} configs  "
          f"({', '.join(f'{k}x{len(v)}' for k, v in SW.SWEEP_GRID.items())})")
    print(f"Done (in target, this fixed set): {done_here:,}")
    print(f"Remaining:  {len(remaining):,}  ->  ~{eta_min:,.0f} min at "
          f"{SW.SECS_PER_RUN:.0f}s/config"
          + (f"  (~{eta_min/args.workers:,.0f} min with {args.workers} workers)"
             if args.workers > 1 else ""))
    if existed and done_here == 0:
        print("WARNING: 0 done for this fixed set in an existing file — your fixed "
              "values likely differ from its rows, so nothing will be reused.")
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
                                    SW.append_rows(out, rows)
                            i += len(chunk); done += len(chunk); tick(done)
                break
            except Exception as e:
                w = max(1, w // 2)
                print(f"  workers terminated ({type(e).__name__}); reducing to {w} "
                      f"and continuing", flush=True)
                if w == 1:
                    for p in remaining[i:]:
                        rows = SW.safe_run_one(p)
                        if rows:
                            SW.append_rows(out, rows)
                        i += 1; done += 1; tick(done)
                    break
    else:
        for i, p in enumerate(remaining, 1):
            try:
                SW.append_rows(out, SW.run_one(p))
            except Exception as e:
                print(f"  config {i} failed: {e}", flush=True)
            if i % 10 == 0 or i == n:
                tick(i)

    print(f"DONE in {(time.time()-t0)/60:.1f} min. Rows appended to {out}")


if __name__ == "__main__":
    main()
