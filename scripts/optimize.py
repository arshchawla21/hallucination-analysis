"""Optimise a signal over a search space and save a runnable config.

    python scripts/optimize.py configs/spaces/spike.json [--trials 500]

Selection uses ONLY the train split; test metrics are computed once for the
winner and reported. Writes configs/<signal>_<stamp>.json (runnable config +
metrics + provenance) and configs/<signal>_<stamp>_trials.csv (every trial).
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.engine import (format_metrics, load_examples, split_examples)
from backtest.optimize import (load_config, run_search, run_with_params,
                               save_config)
from signals import load_signal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("space", help="configs/spaces/<signal>.json")
    ap.add_argument("--trials", type=int, default=None,
                    help="random-subsample the grid to this many trials")
    ap.add_argument("--objective", default=None)
    ap.add_argument("--data", default="data/results.json")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    spec = load_config(args.space)
    name = spec["signal"]
    objective = args.objective or spec.get("objective", "auroc")
    mod = load_signal(name)

    examples = load_examples(args.data)
    train, test = split_examples(examples, args.test_frac, args.seed)
    print(f"optimising {name} on {len(train)} train responses "
          f"({len(test)} held out), objective={objective}")

    best_params, best_train, trials = run_search(
        mod, spec["space"], train, objective,
        n_trials=args.trials, seed=args.seed, verbose=True)
    test_m = run_with_params(mod, best_params, test)

    print(f"\nbest params: {best_params}")
    print(format_metrics(best_train, "train (in)"))
    print(format_metrics(test_m, "test (out)"))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("configs") / f"{name}_{stamp}.json"
    save_config(
        out, signal=name, params=best_params, objective=objective,
        split={"test_frac": args.test_frac, "seed": args.seed},
        search={"space": spec["space"], "n_trials": args.trials,
                "seed": args.seed},
        in_sample=best_train, out_sample=test_m,
        data=args.data, stamp=stamp)
    print(f"\nwrote {out}")

    if trials:
        trials_path = out.with_name(out.stem + "_trials.csv")
        with open(trials_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for t in trials
                                                     for k in t}))
            w.writeheader()
            w.writerows(trials)
        print(f"wrote {trials_path}")


if __name__ == "__main__":
    main()
