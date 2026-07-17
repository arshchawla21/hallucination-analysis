"""Backtest engine for hallucination signals.

The trading engine replays a price history day by day and asks a strategy
for positions; this one replays each response's uncertainty stream token by
token and asks a signal for a running hallucination score. The final score
of each response is the signal's "position", and instead of PnL we settle
against the judge's label:

    label 1 = fail (hallucinated)      label 0 = succ (correct)

Headline objective is AUROC (threshold-free, insensitive to the 625/163
class imbalance); average precision, best-F1 and the accuracy at that
threshold are also reported.

Splits are stratified by label with a fixed seed so every signal is
optimised and judged on identical train/test responses.
"""

import json
from pathlib import Path

import numpy as np

DATA_PATH = "data/results.json"


# ------------------------------------------------------------------- data

def load_examples(path=DATA_PATH):
    """List of {surprisal, entropy, label, question} dicts.

    succ -> 0, fail -> 1. `null` (refusals) are excluded: the model declined
    to answer, so there is nothing to grade as hallucinated-or-not.
    """
    with open(path) as f:
        raw = json.load(f)
    examples = []
    for label, key in ((0, "succ"), (1, "fail")):
        for r in raw[key]:
            if not r["surprisal"]:
                continue
            examples.append({
                "surprisal": np.asarray(r["surprisal"], dtype=float),
                "entropy": np.asarray(r["entropy"], dtype=float),
                "label": label,
                "question": r.get("question", ""),
            })
    return examples


def split_examples(examples, test_frac=0.3, seed=0):
    """Stratified (train, test) split -- the in/out-sample boundary."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for label in (0, 1):
        group = [ex for ex in examples if ex["label"] == label]
        idx = rng.permutation(len(group))
        n_test = int(round(test_frac * len(group)))
        test += [group[i] for i in idx[:n_test]]
        train += [group[i] for i in idx[n_test:]]
    return train, test


# ---------------------------------------------------------------- metrics

def _rankdata(x):
    """Average ranks (1-based), mergesort for stable ties."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def auroc(scores, labels):
    """Mann-Whitney AUROC: P(score of a random fail > score of a random succ)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    ranks = _rankdata(scores)
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    n_pos = labels.sum()
    if n_pos == 0:
        return 0.0
    return float((precision * y).sum() / n_pos)


def best_f1(scores, labels):
    """(f1, threshold, accuracy) at the F1-optimal decision threshold."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    n = len(labels)
    n_pos = labels.sum()
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    s = scores[order]
    tp = np.cumsum(y)
    k = np.arange(1, n + 1)             # predict positive for the top-k scores
    f1 = 2 * tp / (k + n_pos)           # = 2TP / (2TP + FP + FN)
    # only cut between distinct score values
    valid = np.append(s[:-1] > s[1:], True)
    f1 = np.where(valid, f1, -1)
    best = int(np.argmax(f1))
    acc = (tp[best] + (n - n_pos - (k[best] - tp[best]))) / n
    return float(f1[best]), float(s[best]), float(acc)


def roc_curve(scores, labels):
    """(fpr, tpr) arrays for plotting."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    n_pos = max(int(labels.sum()), 1)
    n_neg = max(int(len(labels) - labels.sum()), 1)
    return (np.concatenate([[0], fp / n_neg]),
            np.concatenate([[0], tp / n_pos]))


# ----------------------------------------------------------------- engine

def run_backtest(mod, params, examples, return_scores=False):
    """Stream every response through a fresh signal instance; score & settle.

    mod:    a signals.<name> module (needs make(**params))
    params: signal parameters for this trial
    """
    scores = np.empty(len(examples))
    labels = np.empty(len(examples), dtype=int)
    for i, ex in enumerate(examples):
        sig = mod.make(**params)
        score = 0.0
        for s, e in zip(ex["surprisal"], ex["entropy"]):
            score = sig.update(s, e)
        scores[i] = score
        labels[i] = ex["label"]

    f1, thr, acc = best_f1(scores, labels)
    metrics = {
        "auroc": auroc(scores, labels),
        "ap": average_precision(scores, labels),
        "f1": f1,
        "threshold": thr,
        "accuracy": acc,
        "base_rate": float(labels.mean()),
        "n": len(examples),
    }
    if return_scores:
        metrics["scores"] = scores
        metrics["labels"] = labels
    return metrics


def format_metrics(m, title=""):
    head = f"{title:<14}" if title else ""
    return (f"{head}AUROC {m['auroc']:.3f}   AP {m['ap']:.3f}   "
            f"F1 {m['f1']:.3f}   acc {m['accuracy']:.3f}   "
            f"(n={m['n']}, base rate {m['base_rate']:.2f})")
