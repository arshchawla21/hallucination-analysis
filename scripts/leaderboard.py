"""Leaderboard: every saved config, re-evaluated, ranked by validation AUROC.

    python scripts/leaderboard.py [--configs configs] [--data data/results.json]

Re-runs each config's params on its own split (so stored metrics can't go
stale) and prints in-sample vs out-of-sample side by side. The number to
watch is the in/out GAP: a strategy that only performs where it was tuned
is overfit, exactly like a trading strategy that dies out of sample.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from backtest.engine import best_f1, load_examples, run_backtest, split_examples
from backtest.optimize import load_config
from signals import load_signal


def _balanced_accuracy(scores, labels, theta):
    """(TPR + TNR) / 2 -- accuracy on a class-balanced population, so a
    flag-nothing (or flag-everything) rule scores exactly 0.5."""
    pred = scores >= theta
    pos, neg = labels == 1, labels == 0
    tpr = float(pred[pos].mean()) if pos.any() else 0.0
    tnr = float((~pred[neg]).mean()) if neg.any() else 0.0
    return (tpr + tnr) / 2


def threshold_transfer(m_train, m_test):
    """Tune the decision threshold on train (balanced-accuracy-optimal),
    apply it frozen to the test scores.
    Returns (theta, balanced_acc, precision, recall)."""
    tr_s, tr_l = m_train["scores"], m_train["labels"]
    cands = np.unique(tr_s)
    theta = max(cands, key=lambda th: _balanced_accuracy(tr_s, tr_l, th))
    bal = _balanced_accuracy(m_test["scores"], m_test["labels"], theta)
    pred = m_test["scores"] >= theta
    lb = m_test["labels"]
    tp = int(((pred == 1) & (lb == 1)).sum())
    fp = int(((pred == 1) & (lb == 0)).sum())
    fn = int(((pred == 0) & (lb == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return float(theta), bal, prec, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--data", default=None,
                    help="override the data file recorded in each config")
    args = ap.parse_args()

    paths = sorted(Path(args.configs).glob("*.json"))
    if not paths:
        sys.exit(f"no configs found in {args.configs}/")

    rows = []
    cache = {}
    for path in paths:
        cfg = load_config(path)
        mod = load_signal(cfg["signal"])
        split = cfg["split"]
        key = (args.data or cfg["data"], split["test_frac"], split["seed"])
        if key not in cache:
            examples = load_examples(key[0])
            cache[key] = split_examples(examples, key[1], key[2])
        train, test = cache[key]
        m_in = run_backtest(mod, cfg["params"], train, return_scores=True)
        m_out = run_backtest(mod, cfg["params"], test, return_scores=True)
        theta, bal, prec, rec = threshold_transfer(m_in, m_out)
        rows.append({
            "config": path.stem,
            "signal": cfg["signal"],
            "in_auroc": m_in["auroc"], "out_auroc": m_out["auroc"],
            "gap": m_in["auroc"] - m_out["auroc"],
            "theta": theta, "bal": bal, "prec": prec, "rec": rec,
            "params": cfg["params"],
        })

    rows.sort(key=lambda r: -r["out_auroc"])
    hdr = (f"{'config':<28} {'in AUROC':>9} {'out AUROC':>10} {'gap':>7} "
           f"{'θ(in)':>7} {'balAcc@θ':>9} {'P@θ':>6} {'R@θ':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['config']:<28} {r['in_auroc']:>9.3f} {r['out_auroc']:>10.3f} "
              f"{r['gap']:>+7.3f} {r['theta']:>7.3f} {r['bal']:>9.3f} "
              f"{r['prec']:>6.3f} {r['rec']:>6.3f}")
    print("\n(θ tuned on train by balanced accuracy, frozen, applied to the "
          "validation scores; balanced accuracy weights hallucinations and "
          "correct answers equally, so flag-nothing scores 0.500)")
    print()
    best = rows[0]
    print(f"best on validation: {best['config']}  params={best['params']}")


if __name__ == "__main__":
    main()
