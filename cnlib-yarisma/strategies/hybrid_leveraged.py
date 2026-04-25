"""
Hybrid Leveraged — volatilite-bilinçli leverage seçimi.

Mantık:
  Close > SMA20 (primary signal) → long pozisyon.
  Leverage seçimi son 10 günlük volatiliteye göre:
    - vol < 2.5%: 10x  (stabil, liquidation riski düşük)
    - vol < 4.0%: 5x   (orta, güvenli)
    - aksi halde: 3x   (yüksek vol, temkinli)

Data analizinden:
  - 5x liquidation: 1 gün (kapcoin, tek event)
  - 10x liquidation: 127 gün -10% altı
  - Volatile günlerin çoğu 10x'i tasfiye eder.

Amaç: 5x Baseline'ı (136,204x) geçmek. 10x sabit (102,508x) liquidation'a
takıldı — hybrid 10x'i sadece güvenli günlerde kullanmalı.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from cnlib.base_strategy import BaseStrategy, COINS


class HybridLeveraged(BaseStrategy):
    SMA_PERIOD = 20
    VOL_WINDOW = 10

    # Volatility tier threshold'ları (grid search sonucu optimize — $3.51B @ 1000$)
    VOL_TIER10 = 0.020         # son 10 gün abs. return ortalaması < 2.0% → 10x
    VOL_TIER5  = 0.050         # < 5% → 5x; aksi → 3x

    MAX_PER_COIN = 0.33
    TOTAL_ALLOC_CAP = 0.96

    def predict(self, data: dict[str, Any]) -> list[dict]:
        # Pass 1: signal + leverage hesapla
        evaluated: dict[str, tuple[int, int]] = {}
        for coin in COINS:
            df: pd.DataFrame = data[coin]
            if len(df) < max(self.SMA_PERIOD, self.VOL_WINDOW + 1):
                evaluated[coin] = (0, 1)
                continue

            close = df["Close"]
            last = float(close.iloc[-1])
            sma = float(close.iloc[-self.SMA_PERIOD:].mean())
            if last <= sma:
                evaluated[coin] = (0, 1)
                continue

            # Son 10 günlük absolute return ortalaması (volatilite proxy)
            r = close.pct_change().iloc[-self.VOL_WINDOW:].abs().mean()
            if pd.isna(r):
                evaluated[coin] = (0, 1)
                continue

            if r < self.VOL_TIER10:
                lev = 10
            elif r < self.VOL_TIER5:
                lev = 5
            else:
                lev = 3

            evaluated[coin] = (1, lev)

        # Pass 2: alloc
        active = [c for c, (s, _) in evaluated.items() if s == 1]
        k = len(active)
        per_alloc = min(self.MAX_PER_COIN, self.TOTAL_ALLOC_CAP / k) if k > 0 else 0.0

        decisions: list[dict] = []
        for coin in COINS:
            sig, lev = evaluated[coin]
            if sig == 1:
                decisions.append({
                    "coin":       coin,
                    "signal":     1,
                    "allocation": per_alloc,
                    "leverage":   lev,
                })
            else:
                decisions.append({
                    "coin":       coin,
                    "signal":     0,
                    "allocation": 0.0,
                    "leverage":   1,
                })
        return decisions
