"""
Swap Optimized — 10^40+ hedef, rakipler seviyesi.

Scoring = final portfolio value. Rakipler $10^40 civarında.
Bizim Swap 10x: $1.45×10^40 (ama 10 liquidation).

Bu dosyadaki varyantlar:

  SwapV1 — Swap 10x + full allocation (0.333 per coin, 0.999 total)
            Max compound, cash buffer minimal.

  SwapV2 — Swap 10x + volatility-aware leverage
            Vol high (>5%) → 5x geçici; vol extreme (>12%) → flat
            Liquidation azalır, compound daha temiz.

  SwapV3 — Swap 10x + post-loss recovery damper
            Son candle kayıpsa (liquidation sonrası) bir sonraki
            pozisyonda 5x ile başla. Risk-off moment.

  SwapV4 — Kombinasyon: V1 max alloc + V2 vol-aware + V3 damper
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from cnlib.base_strategy import BaseStrategy, COINS


class SwapV1(BaseStrategy):
    """Max allocation Swap 10x. Cash buffer minimize."""
    MIN_MAGNITUDE = 0.005
    LEVERAGE = 10
    MAX_PER_COIN = 0.333
    TOTAL_CAP = 0.999

    def predict(self, data: dict[str, Any]) -> list[dict]:
        signals: dict[str, int] = {}
        for coin in COINS:
            df = data[coin]
            if len(df) < 2:
                signals[coin] = 0
                continue
            close = df["Close"]
            last_r = float(close.iloc[-1] / close.iloc[-2] - 1)
            if abs(last_r) < self.MIN_MAGNITUDE:
                signals[coin] = 0
            elif last_r > 0:
                signals[coin] = 1
            else:
                signals[coin] = -1

        active = [c for c, s in signals.items() if s != 0]
        k = len(active)
        per_alloc = min(self.MAX_PER_COIN, self.TOTAL_CAP / k) if k > 0 else 0.0

        decisions: list[dict] = []
        for coin in COINS:
            s = signals[coin]
            if s != 0:
                decisions.append({
                    "coin": coin, "signal": s,
                    "allocation": per_alloc, "leverage": self.LEVERAGE,
                })
            else:
                decisions.append({"coin": coin, "signal": 0,
                                  "allocation": 0.0, "leverage": 1})
        return decisions


class SwapV2(BaseStrategy):
    """Volatility-aware 10x. Vol high → 5x, extreme → flat."""
    MIN_MAGNITUDE = 0.005
    VOL_WINDOW = 10
    EXTREME_WINDOW = 5
    VOL_THRESHOLD = 0.05
    EXTREME_THRESHOLD = 0.12
    MAX_PER_COIN = 0.333
    TOTAL_CAP = 0.999

    def predict(self, data: dict[str, Any]) -> list[dict]:
        signals: dict[str, int] = {}
        leverages: dict[str, int] = {}

        for coin in COINS:
            df = data[coin]
            if len(df) < self.VOL_WINDOW + 1:
                signals[coin] = 0
                leverages[coin] = 1
                continue

            close = df["Close"]
            last_r = float(close.iloc[-1] / close.iloc[-2] - 1)

            if abs(last_r) < self.MIN_MAGNITUDE:
                signals[coin] = 0
                leverages[coin] = 1
                continue

            r = close.pct_change()
            ext = r.iloc[-self.EXTREME_WINDOW:].abs().max()
            if pd.notna(ext) and ext > self.EXTREME_THRESHOLD:
                signals[coin] = 0
                leverages[coin] = 1
                continue

            vol = r.iloc[-self.VOL_WINDOW:].abs().mean()
            if pd.notna(vol) and vol > self.VOL_THRESHOLD:
                lev = 5
            else:
                lev = 10

            signals[coin] = 1 if last_r > 0 else -1
            leverages[coin] = lev

        active = [c for c, s in signals.items() if s != 0]
        k = len(active)
        per_alloc = min(self.MAX_PER_COIN, self.TOTAL_CAP / k) if k > 0 else 0.0

        decisions: list[dict] = []
        for coin in COINS:
            s = signals[coin]
            if s != 0:
                decisions.append({
                    "coin": coin, "signal": s,
                    "allocation": per_alloc, "leverage": leverages[coin],
                })
            else:
                decisions.append({"coin": coin, "signal": 0,
                                  "allocation": 0.0, "leverage": 1})
        return decisions


class SwapV3(BaseStrategy):
    """10x with post-loss damper. Liquidation sonrası 3 candle 5x."""
    MIN_MAGNITUDE = 0.005
    MAX_PER_COIN = 0.333
    TOTAL_CAP = 0.999
    DAMPER_CANDLES = 3

    def __init__(self):
        super().__init__()
        self.prev_value: float | None = None
        self.damper_remaining: int = 0

    def predict(self, data: dict[str, Any]) -> list[dict]:
        # Damper aktif mi?
        if self.damper_remaining > 0:
            current_lev = 5
            self.damper_remaining -= 1
        else:
            current_lev = 10

        # Not: strateji portfolio'yu göremez; basit proxy — son candle'da tüm
        # 3 coin birlikte -5%+ düştüyse damper devreye sok
        close_avg_drop = []
        for coin in COINS:
            df = data[coin]
            if len(df) >= 2:
                r = float(df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1)
                close_avg_drop.append(r)
        if len(close_avg_drop) == 3 and sum(close_avg_drop) / 3 < -0.05:
            self.damper_remaining = self.DAMPER_CANDLES
            current_lev = 5

        signals: dict[str, int] = {}
        for coin in COINS:
            df = data[coin]
            if len(df) < 2:
                signals[coin] = 0
                continue
            close = df["Close"]
            last_r = float(close.iloc[-1] / close.iloc[-2] - 1)
            if abs(last_r) < self.MIN_MAGNITUDE:
                signals[coin] = 0
            elif last_r > 0:
                signals[coin] = 1
            else:
                signals[coin] = -1

        active = [c for c, s in signals.items() if s != 0]
        k = len(active)
        per_alloc = min(self.MAX_PER_COIN, self.TOTAL_CAP / k) if k > 0 else 0.0

        decisions: list[dict] = []
        for coin in COINS:
            s = signals[coin]
            if s != 0:
                decisions.append({
                    "coin": coin, "signal": s,
                    "allocation": per_alloc, "leverage": current_lev,
                })
            else:
                decisions.append({"coin": coin, "signal": 0,
                                  "allocation": 0.0, "leverage": 1})
        return decisions


class SwapV5(BaseStrategy):
    """
    V3 damper + 2-gün confirmation lag.
    r_{t-1} ve r_{t-2} aynı yönde ise 10x (yüksek güven).
    Farklı yönde ise 5x (karışık signal).
    Düşük magnitude threshold — daha çok gün yakala.
    Post-loss damper.
    """
    MIN_MAGNITUDE = 0.003      # %0.3 — daha çok işlem
    EXTREME_WINDOW = 5
    EXTREME_THRESHOLD = 0.12
    MAX_PER_COIN = 0.333
    TOTAL_CAP = 0.999
    DAMPER_CANDLES = 2

    def __init__(self):
        super().__init__()
        self.damper_remaining = 0

    def predict(self, data: dict[str, Any]) -> list[dict]:
        # Broad market damper (3 coin ortalama -%5+ düşüş)
        recent_drops = []
        for coin in COINS:
            df = data[coin]
            if len(df) >= 2:
                r = float(df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1)
                recent_drops.append(r)
        if len(recent_drops) == 3 and sum(recent_drops) / 3 < -0.05:
            self.damper_remaining = self.DAMPER_CANDLES
        damper_active = self.damper_remaining > 0
        if damper_active:
            self.damper_remaining -= 1

        signals: dict[str, int] = {}
        leverages: dict[str, int] = {}

        for coin in COINS:
            df = data[coin]
            if len(df) < 3:
                signals[coin] = 0
                leverages[coin] = 1
                continue

            close = df["Close"]
            r1 = float(close.iloc[-1] / close.iloc[-2] - 1)
            r2 = float(close.iloc[-2] / close.iloc[-3] - 1)

            if abs(r1) < self.MIN_MAGNITUDE:
                signals[coin] = 0
                leverages[coin] = 1
                continue

            # Extreme filter
            ext = close.pct_change().iloc[-self.EXTREME_WINDOW:].abs().max()
            if pd.notna(ext) and ext > self.EXTREME_THRESHOLD:
                signals[coin] = 0
                leverages[coin] = 1
                continue

            direction = 1 if r1 > 0 else -1
            same_direction = (r1 * r2) > 0

            if damper_active:
                lev = 3
            elif same_direction:
                lev = 10       # yüksek güven — autocorr teyitli
            else:
                lev = 5        # karışık — dikkatli

            signals[coin] = direction
            leverages[coin] = lev

        active = [c for c, s in signals.items() if s != 0]
        k = len(active)
        per_alloc = min(self.MAX_PER_COIN, self.TOTAL_CAP / k) if k > 0 else 0.0

        decisions: list[dict] = []
        for coin in COINS:
            s = signals[coin]
            if s != 0:
                decisions.append({
                    "coin": coin, "signal": s,
                    "allocation": per_alloc, "leverage": leverages[coin],
                })
            else:
                decisions.append({"coin": coin, "signal": 0,
                                  "allocation": 0.0, "leverage": 1})
        return decisions


class SwapV4(BaseStrategy):
    """Kombinasyon: max alloc + vol-aware + extreme filter + regime damper."""
    MIN_MAGNITUDE = 0.005
    VOL_WINDOW = 10
    EXTREME_WINDOW = 5
    VOL_THRESHOLD = 0.05
    EXTREME_THRESHOLD = 0.12
    MAX_PER_COIN = 0.333
    TOTAL_CAP = 0.999
    DAMPER_CANDLES = 2

    def __init__(self):
        super().__init__()
        self.damper_remaining = 0

    def predict(self, data: dict[str, Any]) -> list[dict]:
        # Post-loss damper (broad market drop detection)
        recent_drops = []
        for coin in COINS:
            df = data[coin]
            if len(df) >= 2:
                r = float(df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1)
                recent_drops.append(r)
        if len(recent_drops) == 3 and sum(recent_drops) / 3 < -0.05:
            self.damper_remaining = self.DAMPER_CANDLES
        damper_active = self.damper_remaining > 0
        if damper_active:
            self.damper_remaining -= 1

        signals: dict[str, int] = {}
        leverages: dict[str, int] = {}

        for coin in COINS:
            df = data[coin]
            if len(df) < self.VOL_WINDOW + 1:
                signals[coin] = 0
                leverages[coin] = 1
                continue

            close = df["Close"]
            last_r = float(close.iloc[-1] / close.iloc[-2] - 1)

            if abs(last_r) < self.MIN_MAGNITUDE:
                signals[coin] = 0
                leverages[coin] = 1
                continue

            r = close.pct_change()
            ext = r.iloc[-self.EXTREME_WINDOW:].abs().max()
            if pd.notna(ext) and ext > self.EXTREME_THRESHOLD:
                signals[coin] = 0
                leverages[coin] = 1
                continue

            vol = r.iloc[-self.VOL_WINDOW:].abs().mean()
            if damper_active:
                lev = 3
            elif pd.notna(vol) and vol > self.VOL_THRESHOLD:
                lev = 5
            else:
                lev = 10

            signals[coin] = 1 if last_r > 0 else -1
            leverages[coin] = lev

        active = [c for c, s in signals.items() if s != 0]
        k = len(active)
        per_alloc = min(self.MAX_PER_COIN, self.TOTAL_CAP / k) if k > 0 else 0.0

        decisions: list[dict] = []
        for coin in COINS:
            s = signals[coin]
            if s != 0:
                decisions.append({
                    "coin": coin, "signal": s,
                    "allocation": per_alloc, "leverage": leverages[coin],
                })
            else:
                decisions.append({"coin": coin, "signal": 0,
                                  "allocation": 0.0, "leverage": 1})
        return decisions
