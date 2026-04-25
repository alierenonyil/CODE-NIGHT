"""
Adaptive V7 — per-coin rejim.

V3: 3 coin'in ORTALAMA autocorr/vol → TEK rejim seç
V7: Her coin kendi autocorr/vol'una göre KENDİ rejimi seçer

Avantaj: Bir coin bull diğeri bear ise ayrı davranış.

Mantık:
  Her coin için (her REGIME_REFRESH'te):
    autocorr_{coin}, vol_{coin} ölç → coin için rejim seç
  predict() içinde her coin'i kendi rejimiyle işle.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from cnlib.base_strategy import BaseStrategy, COINS


class AdaptiveV7(BaseStrategy):
    WARMUP = 30
    AUTOCORR_WINDOW = 60
    REGIME_REFRESH = 30
    AUTOCORR_SWAP_THRESHOLD = 0.20
    AUTOCORR_HYBRID_THRESHOLD = 0.10
    ULTRA_VOL_THRESHOLD = 0.045

    SWAP_MIN_MAG = 0.008
    SWAP_DAMPER_THRESH = -0.075
    SWAP_DAMPER_CANDLES = 4
    SWAP_EXTREME_THRESH = 0.12
    SWAP_MAX_PER_COIN = 0.95
    SWAP_CONFIDENCE_BONUS = 3.0

    HYBRID_MIN_MAG = 0.01
    HYBRID_CONCENTRATION = 0.5

    ROBUST_SMA_SHORT = 20
    ROBUST_SMA_LONG = 50
    ROBUST_MAX_LEV = 2
    ROBUST_VOL_EXTREME = 0.12

    SAFE_SMA_LONG = 50
    SAFE_MAX_LEV = 1
    SAFE_EXTREME = 0.10

    MAX_PER_COIN = 0.333
    TOTAL_CAP = 0.96

    def __init__(self):
        super().__init__()
        self.coin_regimes: dict[str, str] = {c: "robust" for c in COINS}
        self.last_regime_check = -1
        self.damper_remaining: dict[str, int] = {c: 0 for c in COINS}

    def _measure_coin(self, df: pd.DataFrame) -> tuple[float, float]:
        if len(df) < self.AUTOCORR_WINDOW + 2:
            return 0.0, 0.0
        close = df["Close"].iloc[-self.AUTOCORR_WINDOW:]
        r = close.pct_change().dropna()
        if len(r) < 10:
            return 0.0, 0.0
        ac = r.autocorr(lag=1)
        return float(ac) if pd.notna(ac) else 0.0, float(r.std())

    def _refresh_regimes(self, data):
        for coin in COINS:
            ac, vol = self._measure_coin(data[coin])
            if vol > self.ULTRA_VOL_THRESHOLD:
                self.coin_regimes[coin] = "safe"
            elif ac >= self.AUTOCORR_SWAP_THRESHOLD:
                self.coin_regimes[coin] = "swap"
            elif ac >= self.AUTOCORR_HYBRID_THRESHOLD:
                self.coin_regimes[coin] = "hybrid"
            else:
                self.coin_regimes[coin] = "robust"

    def predict(self, data: dict[str, Any]) -> list[dict]:
        i = self.candle_index
        if i < self.WARMUP:
            return [self._flat(c) for c in COINS]

        if self.last_regime_check < 0 or (i - self.last_regime_check) >= self.REGIME_REFRESH:
            self._refresh_regimes(data)
            self.last_regime_check = i

        # Her coin için kendi rejiminde signal + leverage
        per_coin_decision = {}
        for coin in COINS:
            regime = self.coin_regimes[coin]
            if regime == "safe":
                sig, lev, w = self._coin_safe(coin, data[coin])
            elif regime == "swap":
                sig, lev, w = self._coin_swap(coin, data[coin], data)
            elif regime == "hybrid":
                sig, lev, w = self._coin_hybrid(coin, data[coin])
            else:
                sig, lev, w = self._coin_robust(coin, data[coin])
            per_coin_decision[coin] = (sig, lev, w)

        # Alloc: aktif coinler, magnitude-weighted
        active = [(c, lev, w) for c, (sig, lev, w) in per_coin_decision.items() if sig != 0]
        if not active:
            return [self._flat(c) for c in COINS]

        total_w = sum(w for _, _, w in active) or 1.0
        decisions = []
        for coin in COINS:
            sig, lev, w = per_coin_decision[coin]
            if sig == 0:
                decisions.append(self._flat(coin))
            else:
                alloc = min(self.MAX_PER_COIN, (w / total_w) * self.TOTAL_CAP)
                decisions.append({
                    "coin": coin, "signal": sig,
                    "allocation": alloc, "leverage": lev,
                })
        return decisions

    # ------------------------------------------------------------------
    # Per-coin regime implementations
    # ------------------------------------------------------------------

    def _coin_safe(self, coin, df):
        if len(df) < self.SAFE_SMA_LONG:
            return 0, 1, 0.0
        close = df["Close"]
        last = float(close.iloc[-1])
        sma50 = float(close.iloc[-self.SAFE_SMA_LONG:].mean())
        sma20 = float(close.iloc[-20:].mean())
        if last <= sma50 or last <= sma20:
            return 0, 1, 0.0
        ext = close.pct_change().iloc[-5:].abs().max()
        if pd.notna(ext) and ext > self.SAFE_EXTREME:
            return 0, 1, 0.0
        return 1, self.SAFE_MAX_LEV, 1.0

    def _coin_swap(self, coin, df, all_data):
        # Broad damper check
        recent_drops = []
        for c in COINS:
            d = all_data[c]
            if len(d) >= 2:
                recent_drops.append(float(d["Close"].iloc[-1] / d["Close"].iloc[-2] - 1))
        if len(recent_drops) == 3 and sum(recent_drops) / 3 < self.SWAP_DAMPER_THRESH:
            self.damper_remaining[coin] = self.SWAP_DAMPER_CANDLES
        if self.damper_remaining[coin] > 0:
            self.damper_remaining[coin] -= 1
            lev = 5
        else:
            lev = 10

        if len(df) < 6:
            return 0, 1, 0.0
        close = df["Close"]
        r1 = float(close.iloc[-1] / close.iloc[-2] - 1)
        r2 = float(close.iloc[-2] / close.iloc[-3] - 1) if len(close) >= 3 else 0.0

        if abs(r1) < self.SWAP_MIN_MAG:
            return 0, 1, 0.0
        ext = close.pct_change().iloc[-5:].abs().max()
        if pd.notna(ext) and ext > self.SWAP_EXTREME_THRESH:
            return 0, 1, 0.0

        sig = 1 if r1 > 0 else -1
        mag = abs(r1)
        if (r1 * r2) > 0:
            mag *= self.SWAP_CONFIDENCE_BONUS
        return sig, lev, mag

    def _coin_hybrid(self, coin, df):
        if len(df) < 21:
            return 0, 1, 0.0
        close = df["Close"]
        last = float(close.iloc[-1])
        sma20 = float(close.iloc[-20:].mean())
        if last <= sma20:
            return 0, 1, 0.0
        r1 = float(close.iloc[-1] / close.iloc[-2] - 1)
        if abs(r1) < self.HYBRID_MIN_MAG:
            return 0, 1, 0.0
        ext = close.pct_change().iloc[-5:].abs().max()
        if pd.notna(ext) and ext > self.SWAP_EXTREME_THRESH:
            return 0, 1, 0.0
        vol = close.pct_change().iloc[-10:].abs().mean()
        lev = 3 if (pd.notna(vol) and vol > 0.05) else 5
        return 1, lev, max(abs(r1), 0.001)

    def _coin_robust(self, coin, df):
        if len(df) < self.ROBUST_SMA_LONG:
            return 0, 1, 0.0
        close = df["Close"]
        last = float(close.iloc[-1])
        sma50 = float(close.iloc[-self.ROBUST_SMA_LONG:].mean())
        sma20 = float(close.iloc[-self.ROBUST_SMA_SHORT:].mean())
        if last <= sma50 or last <= sma20:
            return 0, 1, 0.0
        ext = close.pct_change().iloc[-5:].abs().max()
        if pd.notna(ext) and ext > self.ROBUST_VOL_EXTREME:
            return 0, 1, 0.0
        return 1, self.ROBUST_MAX_LEV, 1.0

    @staticmethod
    def _flat(coin):
        return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}
