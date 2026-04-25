"""
Diğer stratejilerin 5x leverage versiyonları.

Her birinin orijinal mantığı + leverage=5 ile patch.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from cnlib.base_strategy import BaseStrategy, COINS

from strategies.technical_ensemble import TechnicalEnsemble
from strategies.enhanced_momentum import EnhancedMomentum
from strategies.momentum_hysteresis import MomentumHysteresis


def _apply_leverage(decisions: list[dict], leverage: int) -> list[dict]:
    """signal != 0 olanlara leverage uygula; signal=0 için leverage=1 (neutral)."""
    patched = []
    for d in decisions:
        new = dict(d)
        if new["signal"] != 0:
            new["leverage"] = leverage
        else:
            new["leverage"] = 1
        patched.append(new)
    return patched


class Ensemble5x(TechnicalEnsemble):
    def predict(self, data):
        d = super().predict(data)
        return _apply_leverage(d, 5)


class Enhanced5x(EnhancedMomentum):
    def predict(self, data):
        d = super().predict(data)
        return _apply_leverage(d, 5)


class Hysteresis5x(MomentumHysteresis):
    def predict(self, data):
        d = super().predict(data)
        return _apply_leverage(d, 5)
