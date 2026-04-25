"""
Adaptive V2 — V6 MaxConc swap modu + robust fallback.

V1 farkı:
  - Swap modu artık V6 MaxConc (magnitude-weighted, %80 tek-coin concentration)
  - Hybrid modu da güçlendirildi
  - Regime thresholds tune edilebilir
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from cnlib.base_strategy import BaseStrategy, COINS


class AdaptiveV2(BaseStrategy):
    WARMUP = 30
    AUTOCORR_WINDOW = 60
    REGIME_REFRESH = 30
    AUTOCORR_SWAP_THRESHOLD = 0.30
    AUTOCORR_HYBRID_THRESHOLD = 0.10

    # Swap modu (V6 MaxConc)
    SWAP_MIN_MAG = 0.0055
    SWAP_DAMPER_THRESH = -0.075
    SWAP_DAMPER_CANDLES = 4
    SWAP_EXTREME_THRESH = 0.12
    SWAP_MAX_PER_COIN = 0.8       # concentration
    SWAP_CONFIDENCE_BONUS = 1.5

    # Hybrid mod
    HYBRID_MIN_MAG = 0.008
    HYBRID_MAX_LEV = 5
    HYBRID_CONCENTRATION = 0.6

    # Robust mod
    ROBUST_SMA_SHORT = 20
    ROBUST_SMA_LONG = 50
    ROBUST_MAX_LEV = 2
    ROBUST_VOL_EXTREME = 0.12

    MAX_PER_COIN = 0.333
    TOTAL_CAP = 0.999

    def __init__(self):
        super().__init__()
        self.current_regime = "robust"
        self.last_regime_check = -1
        self.damper_remaining = 0
        self._last_autocorr = 0.0

    def _measure_autocorr(self, data):
        corrs = []
        for coin in COINS:
            df = data[coin]
            if len(df) < self.AUTOCORR_WINDOW + 2:
                continue
            close = df["Close"].iloc[-self.AUTOCORR_WINDOW:]
            r = close.pct_change().dropna()
            if len(r) < 10:
                continue
            ac = r.autocorr(lag=1)
            if pd.notna(ac):
                corrs.append(ac)
        return sum(corrs) / len(corrs) if corrs else 0.0

    def _refresh_regime(self, data):
        ac = self._measure_autocorr(data)
        self._last_autocorr = ac
        if ac >= self.AUTOCORR_SWAP_THRESHOLD:
            self.current_regime = "swap"
        elif ac >= self.AUTOCORR_HYBRID_THRESHOLD:
            self.current_regime = "hybrid"
        else:
            self.current_regime = "robust"

    def predict(self, data: dict[str, Any]) -> list[dict]:
        i = self.candle_index
        if i < self.WARMUP:
            return [self._flat(c) for c in COINS]

        if self.last_regime_check < 0 or (i - self.last_regime_check) >= self.REGIME_REFRESH:
            self._refresh_regime(data)
            self.last_regime_check = i

        if self.current_regime == "swap":
            return self._predict_swap_v6maxconc(data)
        elif self.current_regime == "hybrid":
            return self._predict_hybrid_v2(data)
        else:
            return self._predict_robust(data)

    def _predict_swap_v6maxconc(self, data):
        # Broad market damper
        recent_drops = []
        for coin in COINS:
            df = data[coin]
            if len(df) >= 2:
                r = float(df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1)
                recent_drops.append(r)
        if len(recent_drops) == 3 and sum(recent_drops) / 3 < self.SWAP_DAMPER_THRESH:
            self.damper_remaining = self.SWAP_DAMPER_CANDLES
        damper_active = self.damper_remaining > 0
        if damper_active:
            self.damper_remaining -= 1
            lev = 5
        else:
            lev = 10

        signals = {}
        weights = {}
        for coin in COINS:
            df = data[coin]
            if len(df) < 6:
                signals[coin] = 0
                weights[coin] = 0.0
                continue
            close = df["Close"]
            r1 = float(close.iloc[-1] / close.iloc[-2] - 1)
            r2 = float(close.iloc[-2] / close.iloc[-3] - 1) if len(close) >= 3 else 0.0

            if abs(r1) < self.SWAP_MIN_MAG:
                signals[coin] = 0
                weights[coin] = 0.0
                continue

            ext = close.pct_change().iloc[-5:].abs().max()
            if pd.notna(ext) and ext > self.SWAP_EXTREME_THRESH:
                signals[coin] = 0
                weights[coin] = 0.0
                continue

            direction = 1 if r1 > 0 else -1
            mag = abs(r1)
            if (r1 * r2) > 0:
                mag *= self.SWAP_CONFIDENCE_BONUS
            weights[coin] = mag
            signals[coin] = direction

        return self._magnitude_alloc(signals, weights, lev, self.SWAP_MAX_PER_COIN)

    def _predict_hybrid_v2(self, data):
        signals = {}
        weights = {}
        for coin in COINS:
            df = data[coin]
            if len(df) < 21:
                signals[coin] = 0
                weights[coin] = 0.0
                continue

            close = df["Close"]
            last = float(close.iloc[-1])
            sma20 = float(close.iloc[-20:].mean())

            if last <= sma20:
                signals[coin] = 0
                weights[coin] = 0.0
                continue

            r1 = float(close.iloc[-1] / close.iloc[-2] - 1)
            if abs(r1) < self.HYBRID_MIN_MAG:
                signals[coin] = 0
                weights[coin] = 0.0
                continue

            ext = close.pct_change().iloc[-5:].abs().max()
            if pd.notna(ext) and ext > self.SWAP_EXTREME_THRESH:
                signals[coin] = 0
                weights[coin] = 0.0
                continue

            signals[coin] = 1
            weights[coin] = max(abs(r1), 0.001)

        # Vol-based leverage
        vol_avg = 0.0
        cnt = 0
        for coin in COINS:
            if weights.get(coin, 0) > 0:
                v = data[coin]["Close"].pct_change().iloc[-10:].abs().mean()
                if pd.notna(v):
                    vol_avg += float(v)
                    cnt += 1
        vol_avg = vol_avg / cnt if cnt > 0 else 0.0
        lev = 3 if vol_avg > 0.05 else 5

        return self._magnitude_alloc(signals, weights, lev, self.HYBRID_CONCENTRATION)

    def _predict_robust(self, data):
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

            ext = close.pct_change().iloc[-5:].abs().max()
            if pd.notna(ext) and ext > self.ROBUST_VOL_EXTREME:
                signals[coin] = 0
                leverages[coin] = 1
                continue

            signals[coin] = 1
            leverages[coin] = self.ROBUST_MAX_LEV

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

    def _magnitude_alloc(self, signals, weights, leverage, max_per_coin):
        total_w = sum(weights.values())
        allocations = {}
        if total_w > 0:
            raw = {c: (weights[c] / total_w) * self.TOTAL_CAP
                   for c in COINS if weights[c] > 0}
            capped = {c: min(a, max_per_coin) for c, a in raw.items()}
            total_c = sum(capped.values())
            if total_c > self.TOTAL_CAP:
                scale = self.TOTAL_CAP / total_c
                capped = {c: a * scale for c, a in capped.items()}
            allocations = capped

        decisions = []
        for coin in COINS:
            s = signals.get(coin, 0)
            a = allocations.get(coin, 0.0)
            if s != 0 and a > 0:
                decisions.append({
                    "coin": coin, "signal": s,
                    "allocation": a, "leverage": leverage,
                })
            else:
                decisions.append(self._flat(coin))
        return decisions

    @staticmethod
    def _flat(coin):
        return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}
