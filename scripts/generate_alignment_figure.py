"""
Generate the ground-truth alignment figure for the paper
(``figures/weakspot_alignment.png``, referenced from Problem Setup in
``main.tex``).

The figure varies the induced-weakspot location across the same 3x3 centre
grid used by the parameter sweep, trains the regression model exactly as the
experiment does, and shows that the model's prediction error concentrates
inside the induced data gap at every location. This is the visual evidence
that the induced gap is a faithful ground-truth weakspot.

All computation is delegated to ``scripts/weakspot/validation.py`` so this
script and the Streamlit app share one implementation and cannot diverge.

Usage
-----
    python -m scripts.generate_alignment_figure
    python -m scripts.generate_alignment_figure --radius 0.10 --n-bumps 7
    python -m scripts.generate_alignment_figure --out some/other/path.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running both as a module (-m) and as a bare script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.weakspot.validation import (
    run_alignment_grid, render_alignment_figure,
    run_baseline_error_surface, render_alignment_delta_figure,
)

_FIG_DIR = (
    "Documents/Paper_Advanced EnsembleMethods/"
    "Paper_Advanced-Ensembles-for-Weakspot-Identification/figures/"
)
DEFAULT_OUT = Path(_FIG_DIR + "weakspot_alignment.png")
DEFAULT_OUT_DELTA = Path(_FIG_DIR + "weakspot_alignment_delta.png")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"alignment PNG path (default: {DEFAULT_OUT})")
    p.add_argument("--out-delta", type=Path, default=DEFAULT_OUT_DELTA,
                   help=f"baseline-subtracted PNG path (default: {DEFAULT_OUT_DELTA})")
    p.add_argument("--radius", type=float, default=0.15,
                   help="exclusion radius of the induced weakspot (default: 0.15)")
    p.add_argument("--n-bumps", type=int, default=5,
                   help="function complexity, number of Gaussian bumps (default: 5)")
    p.add_argument("--noise-std", type=float, default=0.10,
                   help="additive noise sigma on the underlying function (default: 0.10)")
    p.add_argument("--model", default="MLP Neural Network",
                   help="regression model name (default: 'MLP Neural Network')")
    p.add_argument("--complexity", type=float, default=0.5,
                   help="model-capacity knob in [0,1] (default: 0.5)")
    p.add_argument("--iterations", type=int, default=100,
                   help="training iterations / estimators (default: 100)")
    p.add_argument("--n-samples", type=int, default=1500,
                   help="dataset size before exclusion (default: 1500)")
    p.add_argument("--grid-res", type=int, default=60,
                   help="error-surface grid resolution (default: 60)")
    p.add_argument("--dpi", type=int, default=300, help="output DPI (default: 300)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    print(f"Training {args.model} at 9 induced-weakspot locations "
          f"(radius={args.radius}, n_bumps={args.n_bumps}, noise={args.noise_std}) ...")
    panels = run_alignment_grid(
        radius=args.radius, n_bumps=args.n_bumps, noise_std=args.noise_std,
        model_name=args.model, complexity=args.complexity,
        iterations=args.iterations, n_samples=args.n_samples,
        grid_res=args.grid_res,
    )

    ratios = [p.ratio for p in panels if p.ratio == p.ratio]  # drop NaN
    if ratios:
        print(f"  inside/outside error ratio: min {min(ratios):.1f}x, "
              f"mean {sum(ratios)/len(ratios):.1f}x, max {max(ratios):.1f}x "
              f"(higher means the gap is clearly the model's weak region)")

    fig = render_alignment_figure(
        panels, model_name=args.model, n_bumps=args.n_bumps, noise_std=args.noise_std,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"  wrote {args.out}")

    # Baseline-subtracted view: isolate the error the withheld data caused by
    # subtracting a model trained on the complete data (no weakspot).
    print("Training baseline model on the complete data (no weakspot) ...")
    _xx, _yy, baseline_raw = run_baseline_error_surface(
        n_bumps=args.n_bumps, noise_std=args.noise_std, model_name=args.model,
        complexity=args.complexity, iterations=args.iterations,
        n_samples=args.n_samples, grid_res=args.grid_res,
    )
    fig_delta = render_alignment_delta_figure(
        panels, baseline_raw, model_name=args.model,
        n_bumps=args.n_bumps, noise_std=args.noise_std,
    )
    args.out_delta.parent.mkdir(parents=True, exist_ok=True)
    fig_delta.savefig(args.out_delta, dpi=args.dpi, bbox_inches="tight")
    print(f"  wrote {args.out_delta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
