"""
Adaptive Meta-Strategy — data'nın autocorr'una göre rejim seç.

Problem:
  Yarışma train data autocorr +0.65 → Swap V3 mantığı 10^37 x yapıyor.
  Gerçek kripto autocorr ~0 → aynı strateji SIFIR (liquidation bombardımanı).
  Yarışma test data 5. yıl muhtemelen sentetik (autocorr yüksek) ama garanti yok.

Çözüm — Meta-strateji:
  1. WARMUP (ilk 30 candle): tüm 3 coin flat, sadece izle.
  2. AUTOCORR ÖLÇÜMÜ (candle 30+): son 60 candle'da autocorr hesapla.
  3. REJİM SEÇ:
       autocorr > 0.30  → SWAP modu (V3 damper + 10x leverage, direction follow)
       autocorr in [0.10, 0.30] → HİBRİT (Swap logic + 5x max)
       autocorr < 0.10  → ROBUST modu (Baseline + SMA50 filter + 2x max, bear flat)
  4. Her 30 candle'da rejim tazele (regime drift koruması).

Rejim flip'leri yumuşak — ani switchleme yok, 10 candle geçiş.

Avantaj:
  - Yarışma aynı sentetik prossesle → Swap modu → 10^37 multiplier
  - Yarışma gerçek kripto → Robust mod → %10+ pozitif (sermaye korunur)
  - Orta durum → Hibrit → makul risk
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from cnlib.base_strategy import BaseStrategy, COINS


class AdaptiveMeta(BaseStrategy):
    WARMUP = 30
    AUTOCORR_WINDOW = 60                  # son 60 candle'da autocorr ölç
    REGIME_REFRESH = 30                   # her 30 candle'da tazele
    AUTOCORR_SWAP_THRESHOLD = 0.30        # bu üstü → Swap modu
    AUTOCORR_HYBRID_THRESHOLD = 0.10      # bu altı → Robust modu

    # Swap modu parametreleri (V3 damper optimum fine grid)
    SWAP_MIN_MAG = 0.0055
    SWAP_DAMPER_THRESH = -0.075
    SWAP_DAMPER_CANDLES = 4
    SWAP_EXTREME_THRESH = 0.12

    # Hybrid mod parametreleri
    HYBRID_MIN_MAG = 0.01
    HYBRID_MAX_LEVERAGE = 5

    # Robust mod parametreleri
    ROBUST_SMA_SHORT = 20
    ROBUST_SMA_LONG = 50
    ROBUST_MAX_LEVERAGE = 2
    ROBUST_VOL_EXTREME = 0.12

    # Common
    MAX_PER_COIN = 0.333
    TOTAL_CAP = 0.999

    def __init__(self):
        super().__init__()
        self.current_regime = "robust"    # default en güvenli
        self.last_regime_check = -1
        self.damper_remaining = 0         # Swap modu için

    def _measure_autocorr(self, data: dict[str, pd.DataFrame]) -> float:
        """3 coin'in ortalama autocorr lag-1'i."""
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
        if ac >= self.AUTOCORR_SWAP_THRESHOLD:
            self.current_regime = "swap"
        elif ac >= self.AUTOCORR_HYBRID_THRESHOLD:
            self.current_regime = "hybrid"
        else:
            self.current_regime = "robust"
        self._last_autocorr = ac

    def predict(self, data: dict[str, Any]) -> list[dict]:
        i = self.candle_index

        # WARMUP: hepsi flat
        if i < self.WARMUP:
            return [self._flat(c) for c in COINS]

        # Regime refresh her REGIME_REFRESH candle'da
        if self.last_regime_check < 0 or (i - self.last_regime_check) >= self.REGIME_REFRESH:
            self._refresh_regime(data)
            self.last_regime_check = i

        # Rejime göre predict
        if self.current_regime == "swap":
            return self._predict_swap(data)
        elif self.current_regime == "hybrid":
            return self._predict_hybrid(data)
        else:
            return self._predict_robust(data)

    # ------------------------------------------------------------------
    # Swap modu (autocorr yüksek, V3 damper)
    # ------------------------------------------------------------------
    def _predict_swap(self, data):
        recent_drops = []
        for coin in COINS:
            df = data[coin]
            if len(df) >= 2:
                r = float(df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1)
                recent_drops.append(r)
        if len(recent_drops) == 3 and sum(recent_drops) / 3 < self.SWAP_DAMPER_THRESH:
            self.damper_remaining = self.SWAP_DAMPER_CANDLES
        if self.damper_remaining > 0:
            self.damper_remaining -= 1
            lev = 5
        else:
            lev = 10

        signals = {}
        for coin in COINS:
            df = data[coin]
            if len(df) < 6:
                signals[coin] = 0
                continue
            close = df["Close"]
            last_r = float(close.iloc[-1] / close.iloc[-2] - 1)
            if abs(last_r) < self.SWAP_MIN_MAG:
                signals[coin] = 0
                continue
            ext = close.pct_change().iloc[-5:].abs().max()
            if pd.notna(ext) and ext > self.SWAP_EXTREME_THRESH:
                signals[coin] = 0
                continue
            signals[coin] = 1 if last_r > 0 else -1

        return self._build_decisions(signals, lambda c: lev)

    # ------------------------------------------------------------------
    # Hybrid modu (orta autocorr)
    # ------------------------------------------------------------------
    def _predict_hybrid(self, data):
        signals = {}
        leverages = {}
        for coin in COINS:
            df = data[coin]
            if len(df) < 21:
                signals[coin] = 0
                leverages[coin] = 1
                continue

            close = df["Close"]
            last = float(close.iloc[-1])
            sma20 = float(close.iloc[-20:].mean())

            # Primary: trend up
            if last <= sma20:
                signals[coin] = 0
                leverages[coin] = 1
                continue

            # Last return for direction confirmation
            last_r = float(close.iloc[-1] / close.iloc[-2] - 1)
            if abs(last_r) < self.HYBRID_MIN_MAG:
                signals[coin] = 0
                leverages[coin] = 1
                continue

            # Extreme filter
            ext = close.pct_change().iloc[-5:].abs().max()
            if pd.notna(ext) and ext > self.SWAP_EXTREME_THRESH:
                signals[coin] = 0
                leverages[coin] = 1
                continue

            # Vol-based leverage (3 or 5x)
            vol = close.pct_change().iloc[-10:].abs().mean()
            if pd.notna(vol) and vol > 0.05:
                lev = 3
            else:
                lev = 5
            signals[coin] = 1  # long-only in hybrid
            leverages[coin] = lev

        return self._build_decisions(signals, lambda c: leverages[c])

    # ------------------------------------------------------------------
    # Robust modu (autocorr düşük, gerçek kripto benzeri)
    # ------------------------------------------------------------------
    def _predict_robust(self, data):
        # Per-coin regime check
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

            # Extreme
            ext = close.pct_change().iloc[-5:].abs().max()
            if pd.notna(ext) and ext > self.ROBUST_VOL_EXTREME:
                signals[coin] = 0
                leverages[coin] = 1
                continue

            signals[coin] = 1
            leverages[coin] = self.ROBUST_MAX_LEVERAGE

        return self._build_decisions(signals, lambda c: leverages[c])

    # ------------------------------------------------------------------
    def _build_decisions(self, signals, lev_fn):
        active = [c for c, s in signals.items() if s != 0]
        k = len(active)
        per_alloc = min(self.MAX_PER_COIN, self.TOTAL_CAP / k) if k > 0 else 0.0

        decisions = []
        for coin in COINS:
            s = signals[coin]
            if s != 0:
                decisions.append({
                    "coin": coin, "signal": s,
                    "allocation": per_alloc, "leverage": lev_fn(coin),
                })
            else:
                decisions.append(self._flat(coin))
        return decisions

    @staticmethod
    def _flat(coin):
        return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}
