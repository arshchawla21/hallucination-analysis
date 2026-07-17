"""Hallucination signals.

Each module in this package is one "strategy": a class that consumes an LLM's
per-token uncertainty stream (surprisal, entropy) causally -- one token at a
time, never looking ahead -- and maintains a running hallucination score.
Higher score = more likely the response is hallucinated.

Contract (mirrors the trading repo's strategies/):

- a class that owns ALL of its state as instance attributes;
- params in a module-level DEFAULTS dict (UPPERCASE keys), validated in
  __init__;
- update(surprisal, entropy) -> float : feed one token, get the running score;
- module-level make(**params) returning a fresh instance (the optimiser and
  the backtest engine call this once per trial / per response).
"""

import importlib


def load_signal(name):
    """Import signals.<name> and sanity-check its contract."""
    mod = importlib.import_module(f"signals.{name}")
    for attr in ("DEFAULTS", "make"):
        if not hasattr(mod, attr):
            raise AttributeError(f"signals.{name} is missing `{attr}`")
    return mod
