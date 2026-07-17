"""Worst uncertainty burst -- max rolling-K mean.

Score = the highest mean uncertainty over any K consecutive tokens seen so
far. K = 1 degenerates to the (order-free) max token; K > 1 rewards
CLUSTERED uncertainty: a confabulated span (invented name, fake title)
should light up several adjacent tokens, while an isolated odd word choice
lights up one. Shuffling token order scatters the cluster, so any edge
beyond K = 1 is genuinely temporal.

Trading analog: the worst K-day losing streak as a risk signal.
"""

from collections import deque

DEFAULTS = {
    "FEATURE": "entropy",   # entropy | surprisal
    "K": 4,                 # burst length in tokens
}


class Streak:
    def __init__(self, **params):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown Streak params: {sorted(unknown)}")
        self.params = {**DEFAULTS, **params}
        self.FEATURE = self.params["FEATURE"]
        self.K = int(self.params["K"])
        if self.K < 1:
            raise ValueError("K must be >= 1")

        self._win = deque(maxlen=self.K)
        self._sum = 0.0
        self._best = float("-inf")

    def update(self, surprisal, entropy):
        x = entropy if self.FEATURE == "entropy" else surprisal
        if len(self._win) == self.K:
            self._sum -= self._win[0]
        self._win.append(x)
        self._sum += x
        # full-K bursts only; a response shorter than K falls back to the
        # mean of what it has, so every response still gets a score
        if len(self._win) == self.K:
            self._best = max(self._best, self._sum / self.K)
        if self._best == float("-inf"):
            return self._sum / len(self._win)
        return self._best


def make(**params):
    return Streak(**params)
