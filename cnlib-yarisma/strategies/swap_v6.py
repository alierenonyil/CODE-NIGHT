"""
Swap V6 — magnitude-weighted + multi-lag confirmation.

10^50 hedefi için tekniği yoğunlaştırıyoruz:

  1. Signal: last return direction (V3 ile aynı)
  2. Magnitude-weighted allocation:
     3 coin'in |r1| büyüklüklerine göre pay dağılımı.
     Yüksek magnitude = daha güçlü signal → daha çok alloc.
  3. Multi-lag confidence (sadece alloc için, leverage sabit):
     r1 ve r2 aynı yönde + magnitude yüksek → full 10x alloc
     Farklı yönde → küçük alloc
  4. Damper (V3 fine-grid optimum):
     Broad market -7.5% drop → 4 candle 5x mode
  5. Extreme filter:
     Son 5 candle tek gün >%12 hareket → 1 candle flat

Allocation max: 0.999 total (validator 1.0 cap altı).
Leverage: 10x default, damper 5x, extreme flat.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from cnlib.base_strategy import BaseStrategy, COINS


class SwapV6(BaseStrategy):
    MIN_MAGNITUDE = 0.0055
    DAMPER_THRESHOLD = -0.075
    DAMPER_CANDLES = 4
    EXTREME_THRESHOLD = 0.12
    EXTREME_WINDOW = 5

    CONFIDENCE_BONUS = 1.3       # r1 ve r2 aynı yönde → alloc 30% bonus
    MAX_PER_COIN = 0.5           # tek coin'e %50 alloc yapabilir (diğerleri düştüyse)
    TOTAL_CAP = 0.999

    def __init__(self):
        super().__init__()
        self.damper_remaining = 0

    def predict(self, data: dict[str, Any]) -> list[dict]:
        # Broad market damper
        recent_drops = []
        for coin in COINS:
            df = data[coin]
            if len(df) >= 2:
                r = float(df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1)
                recent_drops.append(r)
        if len(recent_drops) == 3 and sum(recent_drops) / 3 < self.DAMPER_THRESHOLD:
            self.damper_remaining = self.DAMPER_CANDLES
        damper_active = self.damper_remaining > 0
        if damper_active:
            self.damper_remaining -= 1
            current_lev = 5
        else:
            current_lev = 10

        # Per-coin signal + magnitude + confidence weight
        signals: dict[str, int] = {}
        weights: dict[str, float] = {}

        for coin in COINS:
            df = data[coin]
            if len(df) < 3:
                signals[coin] = 0
                weights[coin] = 0.0
                continue

            close = df["Close"]
            r1 = float(close.iloc[-1] / close.iloc[-2] - 1)
            r2 = float(close.iloc[-2] / close.iloc[-3] - 1)

            if abs(r1) < self.MIN_MAGNITUDE:
                signals[coin] = 0
                weights[coin] = 0.0
                continue

            # Extreme single-day filter
            ext = close.pct_change().iloc[-self.EXTREME_WINDOW:].abs().max()
            if pd.notna(ext) and ext > self.EXTREME_THRESHOLD:
                signals[coin] = 0
                weights[coin] = 0.0
                continue

            direction = 1 if r1 > 0 else -1
            mag = abs(r1)
            # Multi-lag confidence bonus
            if (r1 * r2) > 0:
                mag *= self.CONFIDENCE_BONUS
            weights[coin] = mag
            signals[coin] = direction

        # Magnitude-weighted allocation
        total_weight = sum(weights.values())
        decisions: list[dict] = []

        if total_weight > 0:
            # Her coin için alloc hesapla, cap'lere sığdır
            raw_alloc = {
                c: (weights[c] / total_weight) * self.TOTAL_CAP
                for c in COINS if weights[c] > 0
            }
            # Per-coin cap
            capped = {c: min(a, self.MAX_PER_COIN) for c, a in raw_alloc.items()}
            # Yeniden normalize et (cap sonrası toplam çok azsa)
            total_capped = sum(capped.values())
            if total_capped > self.TOTAL_CAP:
                scale = self.TOTAL_CAP / total_capped
                capped = {c: a * scale for c, a in capped.items()}

            for coin in COINS:
                if signals[coin] != 0 and coin in capped:
                    decisions.append({
                        "coin": coin, "signal": signals[coin],
                        "allocation": capped[coin],
                        "leverage": current_lev,
                    })
                else:
                    decisions.append({
                        "coin": coin, "signal": 0,
                        "allocation": 0.0, "leverage": 1,
                    })
        else:
            decisions = [{"coin": c, "signal": 0, "allocation": 0.0, "leverage": 1}
                         for c in COINS]

        return decisions


class SwapV6NoBonus(SwapV6):
    """Confidence bonus kapalı — sadece raw magnitude weighting test."""
    CONFIDENCE_BONUS = 1.0


class SwapV6HighBonus(SwapV6):
    """Multi-lag confidence'a daha çok pay."""
    CONFIDENCE_BONUS = 2.0


class SwapV6MaxConcentration(SwapV6):
    """Tek coin'e %80 alloc verebilir (ultra concentrated)."""
    MAX_PER_COIN = 0.8
    CONFIDENCE_BONUS = 1.5
