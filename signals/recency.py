"""EWMA regime estimate -- what uncertainty regime is the model in NOW?

An exponentially weighted moving average over the uncertainty stream (the
volatility estimator from risk desks). At every token the score estimates
the model's current uncertainty regime over roughly its last 1/ALPHA
tokens, and it FORGETS: transient wobbles decay away within a few tokens.

Strictly causal: no knowledge of where the response ends (decoding is
open-ended, so distance-to-end is not information a running signal can
have). Classification simply reads the estimate in whatever state it is
when the stream stops. Order-sensitive: shuffling tokens moves mass in and
out of the memory window. FLOOR soft-thresholds each token first so calm
filler doesn't dilute the estimate.
"""

DEFAULTS = {
    "FEATURE": "entropy",   # entropy | surprisal
    "ALPHA": 0.1,           # EWMA weight; effective memory ~ 1/ALPHA tokens
    "FLOOR": 0.0,           # per-token soft threshold: max(x - FLOOR, 0)
}


class Recency:
    def __init__(self, **params):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown Recency params: {sorted(unknown)}")
        self.params = {**DEFAULTS, **params}
        self.FEATURE = self.params["FEATURE"]
        self.ALPHA = self.params["ALPHA"]
        self.FLOOR = self.params["FLOOR"]
        if not 0 < self.ALPHA <= 1:
            raise ValueError("ALPHA must be in (0, 1]")

        self._ewma = None

    def update(self, surprisal, entropy):
        x = entropy if self.FEATURE == "entropy" else surprisal
        x = max(x - self.FLOOR, 0.0)
        if self._ewma is None:
            self._ewma = x
        else:
            self._ewma += self.ALPHA * (x - self._ewma)
        return self._ewma


def make(**params):
    return Recency(**params)
