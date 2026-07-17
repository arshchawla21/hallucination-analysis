"""Generalised recency -- the EWMA's peak, its close, or any blend.

The recency signal scores a response by where its uncertainty EWMA ENDS,
which quietly assumes fabrication happens late. This signal keeps the same
EWMA but scores

    score = MIX * max_t(ewma_t) + (1 - MIX) * ewma_T

MIX = 0 is exactly the recency champion (position-dependent, trusts the
close); MIX = 1 is fully position-independent (the worst sustained
uncertainty regime wherever it occurred, early hallucinations included).
The optimiser choosing MIX tells us how much of recency's edge is "late
matters" versus "a sustained burst happened at all".
"""

DEFAULTS = {
    "FEATURE": "entropy",   # entropy | surprisal
    "ALPHA": 0.2,           # EWMA weight; effective memory ~ 1/ALPHA tokens
    "FLOOR": 0.5,           # per-token soft threshold: max(x - FLOOR, 0)
    "MIX": 0.5,             # 0 = final value (recency) .. 1 = running peak
}


class Peak:
    def __init__(self, **params):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown Peak params: {sorted(unknown)}")
        self.params = {**DEFAULTS, **params}
        self.FEATURE = self.params["FEATURE"]
        self.ALPHA = self.params["ALPHA"]
        self.FLOOR = self.params["FLOOR"]
        self.MIX = self.params["MIX"]
        if not 0 < self.ALPHA <= 1:
            raise ValueError("ALPHA must be in (0, 1]")
        if not 0 <= self.MIX <= 1:
            raise ValueError("MIX must be in [0, 1]")

        self._ewma = None
        self._peak = 0.0

    def update(self, surprisal, entropy):
        x = entropy if self.FEATURE == "entropy" else surprisal
        x = max(x - self.FLOOR, 0.0)
        if self._ewma is None:
            self._ewma = x
        else:
            self._ewma += self.ALPHA * (x - self._ewma)
        self._peak = max(self._peak, self._ewma)
        return self.MIX * self._peak + (1 - self.MIX) * self._ewma


def make(**params):
    return Peak(**params)
