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

    # dry-run first — prints the target file, grid size and ETA, then exits:
    python run_sweep.py --config alpha_boundary --dry-run

    # run it (default config is alpha_boundary; --workers $(nproc) uses all cores):
    python run_sweep.py --config alpha_boundary --workers $(nproc)

    # keep it running after you log out (recommended on EC2):
    nohup python run_sweep.py --config alpha_boundary --workers $(nproc) > sweep.log 2>&1 &
    tail -f sweep.log

--------------------------------------------------------------------------------
Windows (PowerShell, from the Application folder)
--------------------------------------------------------------------------------
    .\.venv\Scripts\python.exe run_sweep.py --fresh --workers 6

--------------------------------------------------------------------------------
Output selection
--------------------------------------------------------------------------------
* --config NAME  -> pick the sweep config (grid + detectors) from
                    scripts/dataselect/sweep_configs/NAME.json  (default: alpha_boundary)
* default        -> append/resume the active config's file
                    data/experiment_results/sweep__<config>.csv
* --fresh        -> a NEW timestamped file  sweep__<config>_fresh_YYYYmmdd_HHMMSS.csv
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

import os
import warnings

# Quiet the expected sklearn ConvergenceWarnings (MLP hits max_iter by design —
# training time is a swept parameter) so they don't bury the progress bar. Setting
# PYTHONWARNINGS before joblib spawns workers propagates the filter to them too.
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")

import argparse
import datetime as _dt
import time
from pathlib import Path

from scripts.dataselect import sweep as SW

# Held-fixed fields — mirror the Data-Selective Training page sidebar defaults.
# (``seed`` is NOT here: it is a swept axis in the config grid now.)
FIXED_DEFAULTS = dict(
    model_name="MLP Neural Network", complexity=0.5,
    center_x=0.5, center_y=0.5,
    n_eval=1000, grid_res=35, extract_q=0.85,
    sel_mode="Sample ∝ weight",
    shift_center_x=0.30, shift_center_y=0.30, shift_spread=0.15,
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
    ap.add_argument("--config", type=str, default=None,
                    help="Sweep config name from scripts/dataselect/sweep_configs/ "
                         "(e.g. alpha_boundary, isolate_alpha_x_radius). "
                         "Default: the module's active config (alpha_boundary).")
    ap.add_argument("--no-early-stopping", dest="early_stopping", action="store_false",
                    help="Disable MLP early stopping for the whole sweep (default: on).")
    ap.set_defaults(early_stopping=True)
    # allow overriding any fixed field, e.g. --complexity 0.3
    for k, v in FIXED_DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    args = ap.parse_args()

    if args.config:                       # switch grid + detectors + output path
        SW.set_active_config(args.config)

    out = resolve_output(args)
    fixed = {k: getattr(args, k) for k in FIXED_DEFAULTS}
    fixed["early_stopping"] = bool(args.early_stopping)
    total = len(SW.parameter_grid())
    remaining = SW.remaining_combos(fixed, out)
    done_here = total - len(remaining)
    eta_min = len(remaining) * SW.SECS_PER_RUN / 60
    existed = out.exists()

    print(f"Config:     {SW.SWEEP_CONFIG_NAME}  (early_stopping={fixed['early_stopping']})")
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

    # Live progress bar (tqdm). Falls back to periodic prints if tqdm is absent so
    # the runner never hard-depends on it. tqdm draws a single updating line with a
    # percentage, count, rate and ETA, and works both on a terminal and when the
    # output is redirected to a log file.
    try:
        from tqdm import tqdm
        bar = tqdm(total=n, unit="cfg", dynamic_ncols=True, smoothing=0.05, desc="sweep")
    except ImportError:
        bar = None
        print("(tqdm not installed — periodic text progress instead; "
              "`pip install tqdm` for a live bar)", flush=True)

    def advance(k: int, done: int, force: bool = False) -> None:
        if bar is not None:
            bar.update(k)
            return
        if force or done % 10 == 0 or done == n:
            el = time.time() - t0
            eta = (el / max(done, 1)) * (n - done) / 60
            print(f"  {done:,}/{n:,}  elapsed {el/60:5.1f} min  ETA {eta:6.1f} min",
                  flush=True)

    def note(msg: str) -> None:
        bar.write(msg) if bar is not None else print(f"  {msg}", flush=True)

    if args.workers > 1:
        from joblib import Parallel, delayed, parallel_backend
        w, done = args.workers, 0
        while done < n:
            try:
                with parallel_backend("loky", inner_max_num_threads=1):
                    # Stream results as each config finishes (ordered), so we append
                    # its rows and tick the bar per config — not per batch. Ordered
                    # streaming means only yielded configs are appended, so a mid-run
                    # worker crash never double-writes: we simply resume at `done`.
                    gen = Parallel(n_jobs=w, return_as="generator")(
                        delayed(SW.safe_run_one)(p) for p in remaining[done:])
                    for rows in gen:
                        if rows:
                            SW.append_rows(out, rows)
                        done += 1
                        advance(1, done)
                break
            except Exception as e:
                if w == 1:                       # already minimal — fall back in-process
                    note(f"parallel failed at 1 worker ({type(e).__name__}); running sequentially")
                    for p in remaining[done:]:
                        rows = SW.safe_run_one(p)
                        if rows:
                            SW.append_rows(out, rows)
                        done += 1; advance(1, done)
                    break
                w = max(1, w // 2)
                note(f"workers terminated ({type(e).__name__}, usually OOM); "
                     f"reducing to {w} and continuing")
    else:
        for i, p in enumerate(remaining, 1):
            try:
                SW.append_rows(out, SW.run_one(p))
            except Exception as e:
                note(f"config {i} failed: {e}")
            advance(1, i)

    if bar is not None:
        bar.close()
    print(f"DONE in {(time.time()-t0)/60:.1f} min. Rows appended to {out}")


if __name__ == "__main__":
    main()
