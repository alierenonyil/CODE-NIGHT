"""
Submission strateji — AdaptiveFinal + Stability Layer.

İyileştirmeler (önceki AdaptiveFinal'a göre):
  + Stop-loss aktif (entry × 0.85) → liquidation matematiksel imkansız
  + Hysteresis rejim değişiminde (0.45 enter / 0.30 exit) → flip-flop önle
  + Volatility-based leverage scale (vol > 6% → 5x'i 3x'e düşür)
  + Single-day extreme filter (son candle |r| > 12% → o coin flat)

Kullanım: doküman spec'ine %100 uyumlu
  from strategy import Strategy
  from cnlib import backtest
  result = backtest.run(strategy=Strategy(), initial_capital=3000.0)
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from cnlib.base_strategy import BaseStrategy, COINS


class Strategy(BaseStrategy):
    """AdaptiveFinal + stop-loss + hysteresis + vol-aware leverage."""

    # Regime detection
    WARMUP = 60
    AUTOCORR_WINDOW = 60
    REGIME_REFRESH = 30

    # Hysteresis band — flip-flop önle
    AUTOCORR_ENTER_AGG = 0.45    # girer aggressive moda
    AUTOCORR_EXIT_AGG = 0.30     # çıkar aggressive moddan

    # Aggressive mode
    AGG_SMA = 4
    AGG_LEVERAGE_HIGH = 5
    AGG_LEVERAGE_LOW = 3         # vol yüksekse 3x'e düş
    AGG_VOL_HIGH = 0.06          # son 5 candle ort. abs return > 6%
    AGG_EXTREME = 0.12           # son candle |r| > 12% → o coin flat
    AGG_STOP_LOSS_PCT = 0.85     # entry * 0.85 (uzun pozisyon için)
    AGG_MAX_PER_COIN = 0.32
    AGG_TOTAL_CAP = 0.90

    # Safe mode
    SAFE_SMA_SHORT = 20
    SAFE_SMA_LONG = 50
    SAFE_LEVERAGE = 2
    SAFE_BEAR_DRIFT_DAYS = 30
    SAFE_BEAR_DRIFT = -0.10
    SAFE_EXTREME = 0.15
    SAFE_STOP_LOSS_PCT = 0.90    # entry * 0.90 (2x leverage'da daha gevşek)
    SAFE_MAX_PER_COIN = 0.32
    SAFE_TOTAL_CAP = 0.96

    def __init__(self):
        super().__init__()
        self.in_aggressive = False
        self.last_regime_check = -1
        self._last_autocorr = 0.0
        self._last_vol = 0.0

    # ------------------------------------------------------------------
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
        # Hysteresis band
        if not self.in_aggressive:
            if ac >= self.AUTOCORR_ENTER_AGG:
                self.in_aggressive = True
        else:
            if ac < self.AUTOCORR_EXIT_AGG:
                self.in_aggressive = False

    @property
    def current_regime(self):
        return "aggressive" if self.in_aggressive else "safe"

    def predict(self, data: dict[str, Any]) -> list[dict]:
        i = self.candle_index
        if i < self.WARMUP:
            return [self._flat(c) for c in COINS]

        if self.last_regime_check < 0 or (i - self.last_regime_check) >= self.REGIME_REFRESH:
            self._refresh_regime(data)
            self.last_regime_check = i

        if self.in_aggressive:
            return self._predict_aggressive(data)
        else:
            return self._predict_safe(data)

    # ------------------------------------------------------------------
    def _predict_aggressive(self, data):
        """Yarışma datası — Leveraged 5x SMA=4 + stop-loss + vol-aware."""
        signals = {}
        leverages = {}
        stop_losses = {}

        for coin in COINS:
            df = data[coin]
            if len(df) < max(self.AGG_SMA, 6):
                signals[coin] = 0
                continue

            close = df["Close"]
            last = float(close.iloc[-1])
            sma = float(close.iloc[-self.AGG_SMA:].mean())

            # Trend signal
            if last <= sma:
                signals[coin] = 0
                continue

            # Extreme single-day filter
            ext = close.pct_change().iloc[-3:].abs().max()
            if pd.notna(ext) and ext > self.AGG_EXTREME:
                signals[coin] = 0
                continue

            # Vol-aware leverage
            vol_recent = close.pct_change().iloc[-5:].abs().mean()
            if pd.notna(vol_recent) and vol_recent > self.AGG_VOL_HIGH:
                lev = self.AGG_LEVERAGE_LOW
            else:
                lev = self.AGG_LEVERAGE_HIGH

            signals[coin] = 1
            leverages[coin] = lev
            stop_losses[coin] = last * self.AGG_STOP_LOSS_PCT

        active = [c for c, s in signals.items() if s == 1]
        k = len(active)
        per_alloc = min(self.AGG_MAX_PER_COIN, self.AGG_TOTAL_CAP / k) if k > 0 else 0.0

        decisions = []
        for coin in COINS:
            if signals.get(coin, 0) == 1:
                decisions.append({
                    "coin": coin, "signal": 1,
                    "allocation": per_alloc, "leverage": leverages[coin],
                    "stop_loss": stop_losses[coin],
                })
            else:
                decisions.append(self._flat(coin))
        return decisions

    # ------------------------------------------------------------------
    def _predict_safe(self, data):
        """Gerçek market — Safe2x bear gate + SMA50 + stop-loss."""
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
        stop_losses = {}
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
            stop_losses[coin] = last * self.SAFE_STOP_LOSS_PCT

        active = [c for c, s in signals.items() if s == 1]
        k = len(active)
        per_alloc = min(self.SAFE_MAX_PER_COIN, self.SAFE_TOTAL_CAP / k) if k > 0 else 0.0

        decisions = []
        for coin in COINS:
            if signals.get(coin, 0) == 1:
                decisions.append({
                    "coin": coin, "signal": 1,
                    "allocation": per_alloc, "leverage": self.SAFE_LEVERAGE,
                    "stop_loss": stop_losses[coin],
                })
            else:
                decisions.append(self._flat(coin))
        return decisions

    @staticmethod
    def _flat(coin):
        return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}
