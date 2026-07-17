"""Fixed-threshold spike counting.

The core HALT-style observation: hallucinations show up as SPIKES in
surprisal -- isolated tokens the model itself found very unlikely (it
"reached" for a fact it didn't have). Count tokens above an absolute
threshold and normalise by response length so long answers aren't penalised
just for having more tokens.

Trading analogy: counting days a stock gaps beyond a fixed move -- the
simplest possible event study.
"""

import math

DEFAULTS = {
    "FEATURE": "surprisal",   # surprisal | entropy
    "THRESHOLD": 3.0,         # nats; a token with p < e^-3 ~ 5% counts as a spike
    "NORM": "sqrt",           # none | sqrt | len -- length normalisation
    "WEIGHTED": False,        # True: add the excess (x - THRESHOLD), not just 1
}

_NORMS = ("none", "sqrt", "len")


class SpikeCount:
    def __init__(self, **params):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown SpikeCount params: {sorted(unknown)}")
        self.params = {**DEFAULTS, **params}
        self.FEATURE = self.params["FEATURE"]
        self.THRESHOLD = self.params["THRESHOLD"]
        self.NORM = self.params["NORM"]
        self.WEIGHTED = self.params["WEIGHTED"]
        if self.NORM not in _NORMS:
            raise ValueError(f"NORM must be one of {_NORMS}")

        self._t = 0
        self._mass = 0.0

    def update(self, surprisal, entropy):
        x = surprisal if self.FEATURE == "surprisal" else entropy
        self._t += 1
        if x > self.THRESHOLD:
            self._mass += (x - self.THRESHOLD) if self.WEIGHTED else 1.0
        if self.NORM == "len":
            return self._mass / self._t
        if self.NORM == "sqrt":
            return self._mass / math.sqrt(self._t)
        return self._mass


def make(**params):
    return SpikeCount(**params)
