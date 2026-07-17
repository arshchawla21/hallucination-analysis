"""Backtest a signal on the train/test split.

    python scripts/run_backtest.py spike                    # DEFAULTS params
    python scripts/run_backtest.py configs/spike_<stamp>.json
    python scripts/run_backtest.py bollinger --set ENTRY_Z=2.5 WINDOW=8
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.engine import (format_metrics, load_examples, run_backtest,
                             split_examples)
from backtest.optimize import load_config
from signals import load_signal


def _parse_set(pairs):
    params = {}
    for pair in pairs or []:
        k, v = pair.split("=", 1)
        try:
            params[k] = json.loads(v)
        except json.JSONDecodeError:
            params[k] = v
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="signal name or a configs/*.json path")
    ap.add_argument("--set", nargs="*", metavar="K=V",
                    help="override params (JSON-typed values)")
    ap.add_argument("--data", default="data/results.json")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.target.endswith(".json"):
        cfg = load_config(args.target)
        name, params = cfg["signal"], cfg["params"]
        split = cfg.get("split", {})
        args.test_frac = split.get("test_frac", args.test_frac)
        args.seed = split.get("seed", args.seed)
    else:
        name, params = args.target, {}
    params.update(_parse_set(args.set))

    mod = load_signal(name)
    full_params = {**mod.DEFAULTS, **params}
    examples = load_examples(args.data)
    train, test = split_examples(examples, args.test_frac, args.seed)

    print(f"signal: {name}  params: {full_params}")
    print(format_metrics(run_backtest(mod, params, train), "train (in)"))
    print(format_metrics(run_backtest(mod, params, test), "test (out)"))


if __name__ == "__main__":
    main()
