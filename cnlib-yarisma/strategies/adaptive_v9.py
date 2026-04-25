"""
Adaptive V9 — V3 + smart bear protection.

V3'ün robust mode 2x leverage tutuyor (bull dönem optimum).
V9: Eğer aktif coin volatil + bear işaretlerinde → 1x'e otomatik düşer.

Robust mode'da per-coin leverage:
  Eğer son 10 candle vol > %4 AND close SMA50'den fazla uzakta → 1x
  Aksi halde V3 default 2x

Bu yf_2018 ve volatil-bear datalarda sermayeyi korur, bull dönemde V3 ile aynı.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from strategies.adaptive_v3 import AdaptiveV3
from cnlib.base_strategy import COINS


class AdaptiveV9(AdaptiveV3):
    BEAR_VOL_THRESHOLD = 0.04
    BEAR_DRIFT_THRESHOLD = -0.10  # son 30 günde %10+ düştüyse "bear-ish"

    def _predict_robust(self, data):
        # V3 robust ile aynı, ama leverage per-coin dinamik
        per_coin_bull = {}
        bull_count = 0
        for coin in COINS:
            df = data[coin]
            if len(df) < self.ROBUST_SMA_LONG:
                per_coin_bull[coin] = False
                continue
            close = df["Close"]
            sma50 = float(close.iloc[-self.ROBUST_SMA_LONG:].mean())
            is_bull = float(close.iloc[-1]) > sma50
            per_coin_bull[coin] = is_bull
            if is_bull:
                bull_count += 1

        if bull_count == 0:
            return [self._flat(c) for c in COINS]

        signals = {}
        leverages = {}
        for coin in COINS:
            df = data[coin]
            if not per_coin_bull.get(coin, False) or len(df) < 21:
                signals[coin] = 0
                leverages[coin] = 1
                continue

            close = df["Close"]
            last = float(close.iloc[-1])
            sma20 = float(close.iloc[-20:].mean())
            if last <= sma20:
                signals[coin] = 0
                leverages[coin] = 1
                continue

            # Extreme single-day filter
            ext = close.pct_change().iloc[-5:].abs().max()
            if pd.notna(ext) and ext > self.ROBUST_VOL_EXTREME:
                signals[coin] = 0
                leverages[coin] = 1
                continue

            # YENİ: bear-volatile detect
            recent_vol = close.pct_change().iloc[-10:].abs().mean()
            recent_drift = (close.iloc[-1] / close.iloc[-30] - 1) if len(close) >= 30 else 0
            is_bear_volatile = (
                pd.notna(recent_vol) and recent_vol > self.BEAR_VOL_THRESHOLD
                and recent_drift < self.BEAR_DRIFT_THRESHOLD
            )

            signals[coin] = 1
            leverages[coin] = 1 if is_bear_volatile else self.ROBUST_MAX_LEV

        active = [c for c, s in signals.items() if s != 0]
        k = len(active)
        per_alloc = min(self.MAX_PER_COIN, self.TOTAL_CAP / k) if k > 0 else 0.0

        decisions = []
        for coin in COINS:
            s = signals[coin]
            if s != 0:
                decisions.append({
                    "coin": coin, "signal": s,
                    "allocation": per_alloc, "leverage": leverages[coin],
                })
            else:
                decisions.append(self._flat(coin))
        return decisions
