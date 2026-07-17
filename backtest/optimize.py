"""Parameter search over a signal's space -- the optimiser.

Space files (configs/spaces/<signal>.json) map each param to either an
explicit values list or {low, high, step}. The search is a full grid when
small enough, otherwise a seeded random subsample of n_trials points.

Selection happens ONLY on the train split; the held-out test split is
evaluated once for the chosen params and merely reported -- same discipline
as the trading repo's in/out-sample convention.
"""

import itertools
import json
from pathlib import Path

import numpy as np

from backtest.engine import run_backtest
from signals import load_signal


def load_config(path):
    with open(path) as f:
        return json.load(f)


def _param_values(spec):
    if isinstance(spec, list):
        return spec
    if isinstance(spec, dict):
        vals = np.arange(spec["low"], spec["high"] + 1e-12, spec["step"])
        return [round(float(v), 10) for v in vals]
    return [spec]  # fixed scalar


def iter_space(space):
    """All param combinations of a space, as dicts."""
    keys = sorted(space)
    grids = [_param_values(space[k]) for k in keys]
    for combo in itertools.product(*grids):
        yield dict(zip(keys, combo))


def run_search(mod, space, examples, objective="auroc",
               n_trials=None, seed=0, verbose=False):
    """Grid / random search. Returns (best_params, best_metrics, trials)."""
    combos = list(iter_space(space))
    if n_trials is not None and n_trials < len(combos):
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in rng.choice(len(combos), n_trials,
                                                replace=False)]
    best_params, best_m, trials = None, None, []
    for params in combos:
        try:
            m = run_backtest(mod, params, examples)
        except ValueError:      # invalid combo (e.g. EXIT >= ENTRY analogs)
            continue
        trials.append({**params, **{k: v for k, v in m.items()
                                    if isinstance(v, float)}})
        if best_m is None or m[objective] > best_m[objective]:
            best_params, best_m = params, m
            if verbose:
                print(f"  new best {objective}={m[objective]:.4f}  {params}")
    if best_m is None:
        raise RuntimeError("no valid parameter combination in the space")
    return best_params, best_m, trials


def run_with_params(mod, params, examples, **kw):
    return run_backtest(mod, params, examples, **kw)


def save_config(path, *, signal, params, objective, split, search,
                in_sample, out_sample, data, stamp):
    """Runnable config + metrics + provenance, like configs/<strat>_<stamp>.json."""
    cfg = {
        "signal": signal,
        "params": params,
        "objective": objective,
        "split": split,
        "search": search,
        "in_sample": {k: v for k, v in in_sample.items()
                      if isinstance(v, (int, float))},
        "out_sample": {k: v for k, v in out_sample.items()
                       if isinstance(v, (int, float))},
        "data": data,
        "created": stamp,
    }
    Path(path).write_text(json.dumps(cfg, indent=2) + "\n")
    return cfg
