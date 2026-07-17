"""Adaptive spike detection -- Bollinger bands on the uncertainty stream.

A direct port of the trading Bollinger strategy: instead of an absolute
spike threshold, each token is judged against the response's OWN recent
baseline. A rolling window over the last WINDOW tokens gives a local
mean/std; a token whose z-score breaches ENTRY_Z is a band breach.

The hypothesis this encodes (and what makes it temporal rather than
marginal): a spike after a run of confident tokens -- the model cruising
through fluent text and suddenly reaching -- is more diagnostic than the
same surprisal value inside an already-noisy stretch.

Short responses are a reality here (some are 4 tokens), so scoring starts
after MIN_OBS tokens rather than a full window, and the std gets a floor
(MIN_STD) so a near-constant warm-up doesn't make every next token an
infinite-z breach.
"""

import math
from collections import deque

DEFAULTS = {
    "FEATURE": "surprisal",   # surprisal | entropy
    "WINDOW": 10,             # rolling baseline length (tokens)
    "MIN_OBS": 3,             # tokens of history required before scoring
    "ENTRY_Z": 2.0,           # breach when z exceeds this
    "MIN_STD": 0.5,           # std floor (nats) -- tames the calm-warm-up blowup
    "WEIGHTED": True,         # True: accumulate excess z; False: count breaches
    "NORM": "sqrt",           # none | sqrt | len
}

_NORMS = ("none", "sqrt", "len")


class BollingerBreach:
    def __init__(self, **params):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown BollingerBreach params: {sorted(unknown)}")
        self.params = {**DEFAULTS, **params}
        self.FEATURE = self.params["FEATURE"]
        self.WINDOW = int(self.params["WINDOW"])
        self.MIN_OBS = int(self.params["MIN_OBS"])
        self.ENTRY_Z = self.params["ENTRY_Z"]
        self.MIN_STD = self.params["MIN_STD"]
        self.WEIGHTED = self.params["WEIGHTED"]
        self.NORM = self.params["NORM"]
        if self.NORM not in _NORMS:
            raise ValueError(f"NORM must be one of {_NORMS}")
        if self.MIN_OBS < 2:
            raise ValueError("MIN_OBS must be >= 2 (std needs 2 points)")

        self._win = deque(maxlen=self.WINDOW)
        self._t = 0
        self._mass = 0.0

    def update(self, surprisal, entropy):
        x = surprisal if self.FEATURE == "surprisal" else entropy
        self._t += 1

        # judge the current token against the PREVIOUS tokens only (causal)
        if len(self._win) >= self.MIN_OBS:
            n = len(self._win)
            mean = sum(self._win) / n
            var = sum((v - mean) ** 2 for v in self._win) / n
            std = max(math.sqrt(var), self.MIN_STD)
            z = (x - mean) / std
            if z > self.ENTRY_Z:
                self._mass += (z - self.ENTRY_Z) if self.WEIGHTED else 1.0

        self._win.append(x)

        if self.NORM == "len":
            return self._mass / self._t
        if self.NORM == "sqrt":
            return self._mass / math.sqrt(self._t)
        return self._mass


def make(**params):
    return BollingerBreach(**params)
