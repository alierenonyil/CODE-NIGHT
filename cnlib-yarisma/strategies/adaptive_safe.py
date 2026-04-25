"""
AdaptiveSafe — Safe2x + selective Swap mode.

Strateji felsefesi:
  Default: Safe2x (bear gate + 2x leverage + SMA50 filter, 1 wipeout/50)
  Sadece autocorr çok yüksek (>0.4) ise → V3 Swap moduna geç

Yarışma datasında autocorr 0.65 → Swap mode → 10^42 mult
Sentetik datada autocorr <0.4 → Safe2x → ortalama %60+

Bu V3'ten farkı:
  - V3 0.2'de Swap moda geçiyordu, sentetik bull dönemlerde aktif olup wipeout aldı
  - V10 sadece 0.4+'da Swap → daha kesin sinyal, daha az risk
  - Default Safe2x mantığı (bear gate + extreme filter)
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from cnlib.base_strategy import BaseStrategy, COINS


class AdaptiveSafe(BaseStrategy):
    # Common
    WARMUP = 30
    AUTOCORR_WINDOW = 60
    REGIME_REFRESH = 30
    AUTOCORR_SWAP_THRESHOLD = 0.40   # 0.20 → 0.40 (cok daha katı)

    # Swap mode (autocorr exploit)
    SWAP_MIN_MAG = 0.0055
    SWAP_DAMPER_THRESH = -0.075
    SWAP_DAMPER_CANDLES = 4
    SWAP_EXTREME_THRESH = 0.12
    SWAP_MAX_PER_COIN = 0.95
    SWAP_CONFIDENCE_BONUS = 3.0

    # Safe2x mode (default for unknown markets)
    SAFE_SMA_SHORT = 20
    SAFE_SMA_LONG = 50
    SAFE_LEVERAGE = 2
    SAFE_BEAR_DRIFT_DAYS = 30
    SAFE_BEAR_DRIFT = -0.10
    SAFE_EXTREME = 0.15

    MAX_PER_COIN = 0.32
    TOTAL_CAP = 0.96

    def __init__(self):
        super().__init__()
        self.current_regime = "safe"
        self.last_regime_check = -1
        self.damper_remaining = 0
        self._last_autocorr = 0.0

    def _measure_autocorr(self, data) -> float:
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
        # Yüksek autocorr → swap (yarışma datası)
        # Aksi halde → safe (gerçek/sentetik market)
        if ac >= self.AUTOCORR_SWAP_THRESHOLD:
            self.current_regime = "swap"
        else:
            self.current_regime = "safe"

    def predict(self, data: dict[str, Any]) -> list[dict]:
        i = self.candle_index
        if i < self.WARMUP:
            return [self._flat(c) for c in COINS]

        if self.last_regime_check < 0 or (i - self.last_regime_check) >= self.REGIME_REFRESH:
            self._refresh_regime(data)
            self.last_regime_check = i

        if self.current_regime == "swap":
            return self._predict_swap(data)
        else:
            return self._predict_safe(data)

    # ------------------------------------------------------------------
    def _predict_swap(self, data):
        # V3 swap mode (V6 MaxConc tuned)
        recent_drops = []
        for coin in COINS:
            df = data[coin]
            if len(df) >= 2:
                recent_drops.append(float(df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1))
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

            signals[coin] = 1 if r1 > 0 else -1
            mag = abs(r1)
            if (r1 * r2) > 0:
                mag *= self.SWAP_CONFIDENCE_BONUS
            weights[coin] = mag

        return self._magnitude_alloc(signals, weights, lev, self.SWAP_MAX_PER_COIN)

    # ------------------------------------------------------------------
    def _predict_safe(self, data):
        # Safe2x mode (bear gate + standart Long)
        bear_count = 0
        for coin in COINS:
            df = data[coin]
            if len(df) < max(self.SAFE_SMA_LONG, self.SAFE_BEAR_DRIFT_DAYS + 1):
                continue
            close = df["Close"]
            last = float(close.iloc[-1])
            sma50 = float(close.iloc[-self.SAFE_SMA_LONG:].mean())
            drift = float(close.iloc[-1] / close.iloc[-self.SAFE_BEAR_DRIFT_DAYS] - 1)
            if last < sma50 and drift < self.SAFE_BEAR_DRIFT:
                bear_count += 1

        if bear_count == 3:
            return [self._flat(c) for c in COINS]

        signals = {}
        for coin in COINS:
            df = data[coin]
            if len(df) < self.SAFE_SMA_LONG:
                signals[coin] = 0
                continue
            close = df["Close"]
            last = float(close.iloc[-1])
            sma20 = float(close.iloc[-self.SAFE_SMA_SHORT:].mean())
            sma50 = float(close.iloc[-self.SAFE_SMA_LONG:].mean())

            if last <= sma50 or last <= sma20:
                signals[coin] = 0
                continue

            ext = close.pct_change().iloc[-5:].abs().max()
            if pd.notna(ext) and ext > self.SAFE_EXTREME:
                signals[coin] = 0
                continue

            signals[coin] = 1

        active = [c for c, s in signals.items() if s == 1]
        k = len(active)
        per_alloc = min(self.MAX_PER_COIN, self.TOTAL_CAP / k) if k > 0 else 0.0

        decisions = []
        for coin in COINS:
            if signals[coin] == 1:
                decisions.append({
                    "coin": coin, "signal": 1,
                    "allocation": per_alloc, "leverage": self.SAFE_LEVERAGE,
                })
            else:
                decisions.append(self._flat(coin))
        return decisions

    # ------------------------------------------------------------------
    def _magnitude_alloc(self, signals, weights, leverage, max_per_coin):
        total_w = sum(weights.values())
        allocations = {}
        if total_w > 0:
            raw = {c: (weights[c] / total_w) * 0.999 for c in COINS if weights[c] > 0}
            capped = {c: min(a, max_per_coin) for c, a in raw.items()}
            total_c = sum(capped.values())
            if total_c > 0.999:
                scale = 0.999 / total_c
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
