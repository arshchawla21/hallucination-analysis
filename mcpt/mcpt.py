"""Monte Carlo permutation tests for hallucination signals.

Trading MCPT shuffles daily returns in time: the return distribution
survives, the temporal patterns die. The analog here shuffles TOKEN ORDER
within each response (surprisal and entropy jointly, so per-token pairing
survives). Every marginal statistic of a response -- mean, max, total
surprisal, its label, its length -- is untouched; only the temporal shape
(spikes after calm stretches, drift into uncertainty, regimes) is
destroyed.

So the null hypothesis is sharp and interesting:

    "this signal's edge comes only from the marginal distribution of
     uncertainty, not from WHEN the uncertainty happens."

An order-free signal (mean surprisal) scores identically on every
permutation -> p = 1.0 by construction. A genuinely temporal signal should
lose accuracy when the order is scrambled -> small p. This is the test of
whether the time-series/trading framing buys anything over bag-of-token
"raw ML" features.

Two modes, driven by a saved optimiser config (configs/*.json):

  in-sample  ("in"):  permute the train responses and RE-OPTIMISE the whole
                      search space per permutation. Null: "an optimiser this
                      flexible finds equally good params on order-free data".
  out-sample ("out"): permute only the held-out test responses, re-run the
                      already-fitted params (no refit). Null: "the fitted
                      signal does just as well on order-free future data".

p-value = (1 + #{permuted >= real}) / (n_perms + 1).

[1] https://github.com/neurotrader888/mcpt
"""

import json
from pathlib import Path

import numpy as np

from backtest.engine import (load_examples, roc_curve, run_backtest,
                             split_examples)
from backtest.optimize import load_config, run_search, run_with_params
from signals import load_signal


def permute_examples(examples, rng=None):
    """Copy of `examples` with token order shuffled inside each response
    (one permutation per response, applied to surprisal AND entropy)."""
    rng = np.random.default_rng() if rng is None else rng
    out = []
    for ex in examples:
        p = rng.permutation(len(ex["surprisal"]))
        out.append({**ex,
                    "surprisal": ex["surprisal"][p],
                    "entropy": ex["entropy"][p]})
    return out


def run_mcpt(config, sample="in", n_perms=100, seed=0, data_path=None,
             verbose=True):
    """Permutation test for a saved optimiser config. Returns a result dict."""
    cfg = config if isinstance(config, dict) else load_config(config)
    if sample not in ("in", "out"):
        raise ValueError("sample must be 'in' or 'out'")

    mod = load_signal(cfg["signal"])
    objective = cfg["objective"]
    sp = cfg["search"]

    examples = load_examples(data_path or cfg.get("data", "data/results.json"))
    train, test = split_examples(examples,
                                 test_frac=cfg["split"]["test_frac"],
                                 seed=cfg["split"]["seed"])
    rng = np.random.default_rng(seed)

    if sample == "in":
        exs = train

        def real_run():
            best_params, best_m, _ = run_search(
                mod, sp["space"], exs, objective,
                n_trials=sp.get("n_trials"), seed=sp["seed"])
            return best_params, best_m

        def perm_run(exs_perm):
            best_params, best_m, _ = run_search(
                mod, sp["space"], exs_perm, objective,
                n_trials=sp.get("n_trials"), seed=sp["seed"])
            return best_params, best_m
    else:
        exs = test

        def real_run():
            return cfg["params"], run_with_params(mod, cfg["params"], exs)

        def perm_run(exs_perm):
            return cfg["params"], run_with_params(mod, cfg["params"], exs_perm)

    real_params, real_m = real_run()
    real_val = real_m[objective]
    stored = cfg["in_sample" if sample == "in" else "out_sample"][objective]
    if verbose and abs(real_val - stored) > 1e-6:
        print(f"note: recomputed real {objective} ({real_val:.3f}) differs "
              f"from the saved config ({stored:.3f}) -- code or data changed "
              "since it was optimised")
    real_roc = roc_curve(*_scores_labels(mod, real_params, exs))

    perm_vals, perm_rocs = [], []
    exceed = 0
    for i in range(n_perms):
        exs_perm = permute_examples(exs, rng=rng)
        best_params, best_m = perm_run(exs_perm)
        perm_vals.append(best_m[objective])
        perm_rocs.append(roc_curve(*_scores_labels(mod, best_params, exs_perm)))
        exceed += best_m[objective] >= real_val
        if verbose:
            print(f"[{i + 1:>4}/{n_perms}] permuted {objective}: "
                  f"{best_m[objective]:>7.4f}  (real {real_val:.4f}, "
                  f"running p={(1 + exceed) / (i + 2):.3f})")

    p_value = (1 + exceed) / (n_perms + 1)
    return {
        "config": str(config) if not isinstance(config, dict) else None,
        "signal": cfg["signal"],
        "params": cfg["params"] if sample == "out" else None,
        "sample": sample,
        "objective": objective,
        "n": len(exs),
        "n_perms": n_perms,
        "seed": seed,
        "real_value": float(real_val),
        "perm_values": [float(v) for v in perm_vals],
        "p_value": float(p_value),
        "real_roc": real_roc,
        "perm_rocs": perm_rocs,
    }


def _scores_labels(mod, params, examples):
    m = run_backtest(mod, params, examples, return_scores=True)
    return m["scores"], m["labels"]


# ---------------------------------------------------------------------- plot

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
ACTUAL = "#2a78d6"      # highlighted real result
PERMUTED = "#b9b7b0"    # the noise ensemble


def plot_mcpt(result, save_path, show_max_curves=200):
    """Two panels: ROC of every permutation vs the actual run, and the
    permuted objective distribution with the real value marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample = result["sample"]
    obj = result["objective"]
    p = result["p_value"]
    perm_vals = np.asarray(result["perm_values"])
    real_val = result["real_value"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 5), width_ratios=[1.2, 1], dpi=150)
    fig.patch.set_facecolor(SURFACE)

    title = ("In-sample MCPT (re-optimised on every token permutation)"
             if sample == "in" else
             "Out-of-sample MCPT (fixed params, token-shuffled held-out responses)")
    verdict = (f"p = {p:.3f}: only {p:.1%} of order-free permutations match "
               "the real result -- the signal exploits real temporal structure"
               if p <= 0.05 else
               f"p = {p:.3f}: {p:.1%} of order-free permutations do this "
               "well -- the edge lives in the marginal distribution, not in time")
    fig.suptitle(title, x=0.07, ha="left", fontsize=13,
                 fontweight="bold", color=INK)
    fig.text(0.07, 0.925, verdict, ha="left", fontsize=10, color=INK_2)

    # -- left: ROC curves
    ax1.set_facecolor(SURFACE)
    for fpr, tpr in result["perm_rocs"][:show_max_curves]:
        ax1.plot(fpr, tpr, color=PERMUTED, lw=0.8, alpha=0.45, zorder=1)
    fpr, tpr = result["real_roc"]
    ax1.plot(fpr, tpr, color=ACTUAL, lw=2.0, zorder=3, solid_capstyle="round")
    ax1.plot([0, 1], [0, 1], color=BASELINE, lw=1, ls="--", zorder=2)
    ax1.annotate("Actual", (fpr[len(fpr) // 3], tpr[len(tpr) // 3]),
                 xytext=(8, -4), textcoords="offset points",
                 color=ACTUAL, fontsize=10, fontweight="bold")
    ax1.annotate(f"{len(result['perm_rocs'])} permutations",
                 (0.62, 0.05), xycoords="axes fraction",
                 color=MUTED, fontsize=9)
    ax1.set_xlabel("false positive rate", fontsize=9, color=INK_2)
    ax1.set_ylabel("true positive rate", fontsize=9, color=INK_2)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)

    # -- right: objective distribution
    ax2.set_facecolor(SURFACE)
    lo = min(perm_vals.min(), real_val)
    hi = max(perm_vals.max(), real_val)
    pad = 0.05 * (hi - lo or 1.0)
    ax2.hist(perm_vals, bins=max(10, len(perm_vals) // 8),
             range=(lo - pad, hi + pad), color=PERMUTED, edgecolor=SURFACE,
             linewidth=1.0, zorder=2)
    ax2.axvline(real_val, color=ACTUAL, lw=2, zorder=3)
    ax2.annotate(f"Actual = {real_val:.3f}", (real_val, 1.0),
                 xycoords=("data", "axes fraction"), xytext=(5, -12),
                 textcoords="offset points", color=ACTUAL,
                 fontsize=10, fontweight="bold")
    ax2.annotate(f"p = {p:.3f}", (0.97, 0.90), xycoords="axes fraction",
                 ha="right", color=INK, fontsize=12, fontweight="bold")
    ax2.set_xlabel(f"permuted best {obj}" if sample == "in"
                   else f"permuted {obj}", fontsize=9, color=INK_2)
    ax2.set_ylabel("count", fontsize=9, color=INK_2)

    for ax in (ax1, ax2):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)

    fig.tight_layout(rect=(0.02, 0, 1, 0.88))
    fig.savefig(save_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_mcpt(result, path):
    """Persist everything except the bulky curves."""
    slim = {k: v for k, v in result.items()
            if k not in ("real_roc", "perm_rocs")}
    Path(path).write_text(json.dumps(slim, indent=2) + "\n")
    return path
