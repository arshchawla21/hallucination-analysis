"""Uncertainty momentum -- EWMA crossover on the entropy stream.

The moving-average crossover, verbatim from trading: a fast EWMA tracks the
model's current uncertainty, a slow EWMA tracks its baseline over the whole
response. When the fast average pulls above the slow one and stays there,
the model has entered a sustained high-uncertainty REGIME -- distinct from
the single-token spikes the other signals hunt. Confabulated multi-token
spans (invented names, dates, titles) should look like exactly this.

Score = the largest fast-over-slow gap seen so far (MODE=max_gap), or the
accumulated positive gap (MODE=area, a length-sensitive variant).
"""

DEFAULTS = {
    "FEATURE": "entropy",   # entropy | surprisal
    "FAST": 0.6,            # fast EWMA alpha (reacts in ~1/alpha tokens)
    "SLOW": 0.1,            # slow EWMA alpha (the response's baseline)
    "MODE": "max_gap",      # max_gap | area
}

_MODES = ("max_gap", "area")


class EwmaCrossover:
    def __init__(self, **params):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown EwmaCrossover params: {sorted(unknown)}")
        self.params = {**DEFAULTS, **params}
        self.FEATURE = self.params["FEATURE"]
        self.FAST = self.params["FAST"]
        self.SLOW = self.params["SLOW"]
        self.MODE = self.params["MODE"]
        if not self.SLOW < self.FAST:
            raise ValueError(f"SLOW={self.SLOW} must be < FAST={self.FAST}")
        if self.MODE not in _MODES:
            raise ValueError(f"MODE must be one of {_MODES}")

        self._fast = None
        self._slow = None
        self._max_gap = 0.0
        self._area = 0.0

    def update(self, surprisal, entropy):
        x = entropy if self.FEATURE == "entropy" else surprisal
        if self._fast is None:
            self._fast = self._slow = x
        else:
            self._fast += self.FAST * (x - self._fast)
            self._slow += self.SLOW * (x - self._slow)

        gap = self._fast - self._slow
        if gap > 0:
            self._area += gap
        self._max_gap = max(self._max_gap, gap)
        return self._max_gap if self.MODE == "max_gap" else self._area


def make(**params):
    return EwmaCrossover(**params)
