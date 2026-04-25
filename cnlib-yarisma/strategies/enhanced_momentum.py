"""
Enhanced Momentum — Baseline'ı geçmek için.

Felsefe:
  Baseline (Close > SMA20) güçlü çünkü bull-heavy datada
  her trend fırsatını yakalıyor. Ensemble'ın ekstra filtreleri
  fırsat kaçırdı ve return düşürdü. Burada:

    1. Baseline'ın sinyalini olduğu gibi koru (Close > SMA20 → long).
    2. Üstüne confidence skoru ekle (3 teyit feature):
       - SMA_fast(10) > SMA_slow(50): uzun vadeli trend uyumu
       - RSI(14) > 50: momentum teyidi
       - ADX(14) > 25: güçlü trend
    3. Confidence'a göre alloc:
       - Conf 3: 0.32 (agresif)
       - Conf 2: 0.30
       - Conf 1: 0.27
       - Conf 0: 0.22  (primary sinyal var ama confirmation yok → küçük ama pozisyonda)
    4. 3 coin toplam alloc max 0.96 (%4 cash buffer) — failed_open riski düşük.
    5. ATR-inverse weighting: düşük volatiliteli coine orantısal daha çok pay
       (risk-parity benzeri; aynı confidence tier içinde yeniden dağıtım).

Leverage = 1 (Ali Eren'in izni olmadan artırma yok — brief kuralı).
Long-only + flat.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from cnlib.base_strategy import BaseStrategy, COINS
from utils.indicators import rsi, adx, atr


class EnhancedMomentum(BaseStrategy):
    """Baseline + confidence + ATR-weighted sizing."""

    WARMUP = 50                    # SMA50 dolması için
    SMA_FAST = 10
    SMA_MAIN = 20                  # Baseline ile aynı
    SMA_SLOW = 50
    RSI_PERIOD = 14
    ADX_PERIOD = 14
    ATR_PERIOD = 14

    RSI_BULL = 50
    ADX_STRONG = 25

    # Confidence tier → hedef alloc per coin
    ALLOC_BY_CONF = {3: 0.32, 2: 0.30, 1: 0.27, 0: 0.22}
    TOTAL_ALLOC_CAP = 0.96         # 3 coin × 0.32 = 0.96 → validator 1.0 altı

    def predict(self, data: dict[str, Any]) -> list[dict]:
        # Pass 1: her coin için sinyal + confidence + atr_pct
        evaluated: dict[str, tuple[int, int, float]] = {}
        for coin in COINS:
            df: pd.DataFrame = data[coin]
            if len(df) < self.WARMUP:
                evaluated[coin] = (0, 0, 0.0)
                continue
            evaluated[coin] = self._evaluate(df)

        # Pass 2: aktif coinler için alloc hesapla (confidence + ATR weighting)
        active = {c: (conf, atr_pct) for c, (sig, conf, atr_pct) in evaluated.items()
                  if sig == 1}

        allocations = self._compute_allocations(active)

        # Pass 3: decisions üret
        decisions: list[dict] = []
        for coin in COINS:
            alloc = allocations.get(coin, 0.0)
            if alloc > 0.0:
                decisions.append({
                    "coin":       coin,
                    "signal":     1,
                    "allocation": alloc,
                    "leverage":   1,
                })
            else:
                decisions.append({
                    "coin":       coin,
                    "signal":     0,
                    "allocation": 0.0,
                    "leverage":   1,
                })
        return decisions

    # ------------------------------------------------------------------
    # İç mantık
    # ------------------------------------------------------------------

    def _evaluate(self, df: pd.DataFrame) -> tuple[int, int, float]:
        """
        Returns: (signal, confidence 0-3, atr_pct).
        signal=1 → Close > SMA20; aksi 0.
        confidence → 3 ek feature'dan kaç tanesi teyit veriyor.
        atr_pct = ATR / Close (volatilite metric'i).
        """
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        last = float(close.iloc[-1])
        sma_main = float(close.iloc[-self.SMA_MAIN:].mean())

        # Primary (baseline): Close > SMA20?
        if last <= sma_main:
            return (0, 0, 0.0)

        conf = 0

        # Feature 1: SMA_fast > SMA_slow → uzun vadeli trend uyumu
        sma_fast = float(close.iloc[-self.SMA_FAST:].mean())
        sma_slow = float(close.iloc[-self.SMA_SLOW:].mean())
        if sma_fast > sma_slow:
            conf += 1

        # Feature 2: RSI > 50 → momentum pozitif
        rsi_s = rsi(close, self.RSI_PERIOD)
        rsi_last = rsi_s.iloc[-1]
        if not pd.isna(rsi_last) and rsi_last > self.RSI_BULL:
            conf += 1

        # Feature 3: ADX > 25 → güçlü trend
        adx_s, _, _ = adx(high, low, close, self.ADX_PERIOD)
        adx_last = adx_s.iloc[-1]
        if not pd.isna(adx_last) and adx_last > self.ADX_STRONG:
            conf += 1

        # ATR_pct — volatilite ölçütü
        atr_s = atr(high, low, close, self.ATR_PERIOD)
        atr_last = atr_s.iloc[-1]
        if pd.isna(atr_last) or last <= 0:
            atr_pct = 0.05  # fallback
        else:
            atr_pct = float(atr_last / last)

        return (1, conf, atr_pct)

    def _compute_allocations(
        self,
        active: dict[str, tuple[int, float]],
    ) -> dict[str, float]:
        """
        Confidence-based base alloc + ATR-inverse yeniden dağıtım.

        Algoritma:
          1. Her aktif coin için base_alloc = ALLOC_BY_CONF[conf]
          2. Toplam alloc > TOTAL_ALLOC_CAP ise proportional küçült.
          3. ATR-inverse: aynı confidence tier içinde düşük volatiliteli
             coine (daha stabil) +%10'a kadar relatif pay kaydır.
        """
        if not active:
            return {}

        # 1) Base alloc
        base = {c: self.ALLOC_BY_CONF[conf] for c, (conf, _) in active.items()}

        # 2) Total cap kontrolü
        total = sum(base.values())
        if total > self.TOTAL_ALLOC_CAP:
            scale = self.TOTAL_ALLOC_CAP / total
            base = {c: v * scale for c, v in base.items()}

        # 3) ATR-inverse ince ayar (toplam sabit tut, dağılımı değiştir)
        #    Düşük ATR → yüksek inv → biraz fazla pay
        atr_pcts = {c: max(a, 1e-6) for c, (_, a) in active.items()}
        inv = {c: 1.0 / a for c, a in atr_pcts.items()}
        avg_inv = sum(inv.values()) / len(inv)
        # Relatif çarpan [0.9, 1.1] arasında tut (fazla kaydırma yok)
        adjust = {c: max(0.9, min(1.1, inv[c] / avg_inv)) for c in inv}

        # Uygula
        raw = {c: base[c] * adjust[c] for c in base}
        # Cap check sonrası
        total_raw = sum(raw.values())
        if total_raw > self.TOTAL_ALLOC_CAP:
            scale = self.TOTAL_ALLOC_CAP / total_raw
            raw = {c: v * scale for c, v in raw.items()}

        return raw
