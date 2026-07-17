"""Relative spike -- current token vs its own trailing baseline, as a ratio.

The recency winner is end-weighted, but "the commitment tokens come last"
is a property of short trivia answers, not of hallucination: in longer
conversation the fabrication can happen anywhere, and early is worse. This
signal drops the position dependence. For every token, compare it to the
rolling mean of the PREVIOUS N tokens (never including itself):

    r_t = x_t / max(mean(x_{t-N} .. x_{t-1}), FLOOR)

r_t asks "how many times its own recent baseline is this token?", the
proportional version of a Bollinger z-score. The FLOOR keeps a near-silent
baseline from turning routine tokens into infinite ratios.

The ratio series is then aggregated position-independently:

    max        -- the single worst relative spike anywhere (pure spike)
    ewma_final -- EWMA of the ratios, final value (pure recency, on ratios)
    ewma_max   -- running max of that EWMA (a sustained relative burst
                  ANYWHERE in the response -- the spike/recency blend)
"""

from collections import deque

DEFAULTS = {
    "FEATURE": "entropy",   # entropy | surprisal
    "N": 5,                 # trailing baseline window (previous tokens only)
    "MIN_OBS": 2,           # baseline tokens required before judging
    "FLOOR": 0.5,           # baseline floor (nats)
    "MODE": "ewma_max",     # max | ewma_final | ewma_max
    "ALPHA": 0.35,          # EWMA weight for the ewma_* modes
}

_MODES = ("max", "ewma_final", "ewma_max")


class RelSpike:
    def __init__(self, **params):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown RelSpike params: {sorted(unknown)}")
        self.params = {**DEFAULTS, **params}
        self.FEATURE = self.params["FEATURE"]
        self.N = int(self.params["N"])
        self.MIN_OBS = int(self.params["MIN_OBS"])
        self.FLOOR = self.params["FLOOR"]
        self.MODE = self.params["MODE"]
        self.ALPHA = self.params["ALPHA"]
        if self.MODE not in _MODES:
            raise ValueError(f"MODE must be one of {_MODES}")
        if self.FLOOR <= 0:
            raise ValueError("FLOOR must be > 0 (it is the ratio denominator)")
        if not 0 < self.ALPHA <= 1:
            raise ValueError("ALPHA must be in (0, 1]")
        if self.MIN_OBS < 1:
            raise ValueError("MIN_OBS must be >= 1")

        self._win = deque(maxlen=self.N)
        self._sum = 0.0
        self._max_r = 0.0
        self._ewma = 0.0
        self._max_ewma = 0.0

    def update(self, surprisal, entropy):
        x = entropy if self.FEATURE == "entropy" else surprisal

        if len(self._win) >= self.MIN_OBS:
            base = max(self._sum / len(self._win), self.FLOOR)
            r = x / base
        else:
            r = 0.0   # no baseline yet, nothing to judge against

        self._max_r = max(self._max_r, r)
        self._ewma += self.ALPHA * (r - self._ewma)
        self._max_ewma = max(self._max_ewma, self._ewma)

        if len(self._win) == self.N:
            self._sum -= self._win[0]
        self._win.append(x)
        self._sum += x

        if self.MODE == "max":
            return self._max_r
        if self.MODE == "ewma_final":
            return self._ewma
        return self._max_ewma


def make(**params):
    return RelSpike(**params)
