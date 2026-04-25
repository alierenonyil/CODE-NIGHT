"""
Robust Hybrid — Out-of-sample 5. yıl için dayanıklı strateji.

Felsefe:
  Yarışma puanlaması out-of-sample 5. yılda → train overfit = diskalifiye.
  Her sentetik senaryoda (normal/crash/pump/mixed) pozitif kalmalı.
  Rakipler train'de katrilyon yapmış olabilir ama çoğu out-of-sample'da
  patlayacak. Tutarlı pozitif + yüksek ortalama = gerçek kazanan.

Katmanlar:

  1) MARKET REGIME (bear koruması)
     Kaç coin SMA50 üstünde? (0, 1, 2, 3)
       0 → bear → tümüyle flat (crash hayatta kalma)
       1 → zayıf → o coin 2x
       2 → orta → aktif coinler 3x
       3 → agresif → aktif coinler 5x
     10x ASLA (out-of-sample liquidation riski bilinmez)

  2) PER-COIN SİNYAL
     Coin SMA50 üstü AND Close > SMA20 → long aday
     Aksi halde flat

  3) VOLATILITY SANITY
     Son 5 candle |return| max > 12% → o coin flat (extreme event)
     Son 10 candle |return| ortalama > 5% → leverage bir kademe düş

  4) ALLOCATION
     Aktif coin sayısı k → per_alloc = min(0.33, 0.96/k)  (%4 cash buffer)
     Bear'de alloc zaten 0 (flat).

Long-only + flat (short döküman izinli ama v1'de risk minimize).
Stop-loss YOK (trend kestirir).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from cnlib.base_strategy import BaseStrategy, COINS


VALID_LEVERAGES = [1, 2, 3, 5, 10]


def _clamp_leverage(lev: int) -> int:
    """En yakın küçük-eşit valid leverage."""
    allowed = [v for v in VALID_LEVERAGES if v <= lev]
    return max(allowed) if allowed else 1


class RobustHybrid(BaseStrategy):
    SMA_SHORT = 20
    SMA_LONG = 50
    VOL_WINDOW = 10
    EXTREME_WINDOW = 5

    VOL_HIGH = 0.05              # son 10 candle ort. abs >5% → leverage düş
    VOL_EXTREME = 0.12           # son 5 candle maks abs >12% → flat

    MAX_PER_COIN = 0.33
    TOTAL_CAP = 0.96

    def predict(self, data: dict[str, Any]) -> list[dict]:
        # Katman 1: Market regime
        per_coin_bull: dict[str, bool] = {}
        bull_count = 0
        for coin in COINS:
            df: pd.DataFrame = data[coin]
            if len(df) < self.SMA_LONG:
                per_coin_bull[coin] = False
                continue
            close = df["Close"]
            sma50 = float(close.iloc[-self.SMA_LONG:].mean())
            is_bull = float(close.iloc[-1]) > sma50
            per_coin_bull[coin] = is_bull
            if is_bull:
                bull_count += 1

        # 0 coin bull → total bear → flat everywhere
        if bull_count == 0:
            return [self._flat(c) for c in COINS]

        # Base leverage by regime strength
        if bull_count == 3:
            base_leverage = 5
        elif bull_count == 2:
            base_leverage = 3
        else:  # == 1
            base_leverage = 2

        # Katman 2 + 3: Per-coin signal + vol sanity
        signals: dict[str, int] = {}
        leverages: dict[str, int] = {}
        for coin in COINS:
            df = data[coin]

            # Coin kendisi bull değilse flat
            if not per_coin_bull[coin]:
                signals[coin] = 0
                leverages[coin] = 1
                continue

            if len(df) < max(self.SMA_SHORT, self.VOL_WINDOW + 1):
                signals[coin] = 0
                leverages[coin] = 1
                continue

            close = df["Close"]
            last = float(close.iloc[-1])
            sma20 = float(close.iloc[-self.SMA_SHORT:].mean())

            # Short-term momentum filter (baseline mantığı)
            if last <= sma20:
                signals[coin] = 0
                leverages[coin] = 1
                continue

            # Volatility sanity
            r = close.pct_change()
            recent_extreme = r.iloc[-self.EXTREME_WINDOW:].abs().max()
            recent_avg = r.iloc[-self.VOL_WINDOW:].abs().mean()

            # Ani patlama → flat
            if pd.notna(recent_extreme) and recent_extreme > self.VOL_EXTREME:
                signals[coin] = 0
                leverages[coin] = 1
                continue

            # Yüksek vol → leverage bir kademe düşür (valid leverage'e snap)
            effective_lev = base_leverage
            if pd.notna(recent_avg) and recent_avg > self.VOL_HIGH:
                # 5→3, 3→2, 2→1
                fallback = {5: 3, 3: 2, 2: 1, 1: 1}
                effective_lev = fallback.get(effective_lev, 1)
            effective_lev = _clamp_leverage(effective_lev)

            signals[coin] = 1
            leverages[coin] = effective_lev

        # Katman 4: Allocation
        active = [c for c, s in signals.items() if s == 1]
        k = len(active)
        per_alloc = min(self.MAX_PER_COIN, self.TOTAL_CAP / k) if k > 0 else 0.0

        decisions: list[dict] = []
        for coin in COINS:
            if signals[coin] == 1:
                decisions.append({
                    "coin":       coin,
                    "signal":     1,
                    "allocation": per_alloc,
                    "leverage":   leverages[coin],
                })
            else:
                decisions.append(self._flat(coin))
        return decisions

    @staticmethod
    def _flat(coin: str) -> dict:
        return {
            "coin":       coin,
            "signal":     0,
            "allocation": 0.0,
            "leverage":   1,
        }
