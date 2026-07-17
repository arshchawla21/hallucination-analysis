"""Marginal + temporal ensemble.

Mean entropy is the proven order-free workhorse; the Bollinger breach score
is the best temporal candidate. Rather than making them compete, stack them:

    score = mean(entropy) + W * bollinger_breach_score

W is the whole experiment. If validation prefers W > 0, timing information
adds value ON TOP of the marginal level; if W = 0 wins, the temporal story
is redundant. Trading analog: adding a momentum overlay to a value book and
letting the optimiser size the sleeve.
"""

from signals.baseline import Baseline
from signals.bollinger import BollingerBreach

DEFAULTS = {
    "W": 0.2,             # weight of the temporal sleeve
    "FEATURE": "entropy",  # feature both legs run on
    "WINDOW": 10,          # bollinger leg
    "ENTRY_Z": 1.0,
    "MIN_STD": 0.5,
}


class Ensemble:
    def __init__(self, **params):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown Ensemble params: {sorted(unknown)}")
        self.params = {**DEFAULTS, **params}
        self.W = self.params["W"]
        self._base = Baseline(FEATURE=self.params["FEATURE"], STAT="mean")
        self._boll = BollingerBreach(
            FEATURE=self.params["FEATURE"], WINDOW=self.params["WINDOW"],
            ENTRY_Z=self.params["ENTRY_Z"], MIN_STD=self.params["MIN_STD"],
            WEIGHTED=True, NORM="sqrt")

    def update(self, surprisal, entropy):
        base = self._base.update(surprisal, entropy)
        boll = self._boll.update(surprisal, entropy)
        return base + self.W * boll


def make(**params):
    return Ensemble(**params)
