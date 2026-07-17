"""Order-free summary statistics -- the "raw ML" strawman.

Score = a single running statistic of the uncertainty stream (mean / max /
sum / last). This is what most perplexity-style detectors reduce to, and it
is invariant to token ORDER by construction (except `last`). Every temporal
signal in this repo has to beat this to justify existing, and MCPT's
token-shuffle null leaves this signal untouched -- which is exactly the
point of the test.
"""

DEFAULTS = {
    "FEATURE": "surprisal",   # surprisal | entropy
    "STAT": "mean",           # mean | max | sum | last
}

_FEATURES = ("surprisal", "entropy")
_STATS = ("mean", "max", "sum", "last")


class Baseline:
    def __init__(self, **params):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown Baseline params: {sorted(unknown)}")
        self.params = {**DEFAULTS, **params}
        self.FEATURE = self.params["FEATURE"]
        self.STAT = self.params["STAT"]
        if self.FEATURE not in _FEATURES:
            raise ValueError(f"FEATURE must be one of {_FEATURES}")
        if self.STAT not in _STATS:
            raise ValueError(f"STAT must be one of {_STATS}")

        self._n = 0
        self._sum = 0.0
        self._max = float("-inf")
        self._last = 0.0

    def update(self, surprisal, entropy):
        x = surprisal if self.FEATURE == "surprisal" else entropy
        self._n += 1
        self._sum += x
        self._max = max(self._max, x)
        self._last = x
        if self.STAT == "mean":
            return self._sum / self._n
        if self.STAT == "max":
            return self._max
        if self.STAT == "sum":
            return self._sum
        return self._last


def make(**params):
    return Baseline(**params)
