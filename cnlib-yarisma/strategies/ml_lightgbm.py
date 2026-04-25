"""
Adım 3 — LightGBM Walk-Forward ML Stratejisi.

Mantık:
  - Her coin için ayrı LightGBM binary classifier
  - Target: 5 gün sonra fiyat bugünden yüksek mi? (forward-shift)
  - Warmup: ilk 500 candle flat (lookahead yok, data olgunlaşması)
  - Walk-forward: her 90 candle'da re-train (sadece o ana kadarki history ile)
  - Inference: her candle predict_proba → probability → allocation tier

Probability → allocation tier (long-only):
  p >= 0.70: alloc 0.30 (yüksek güven)
  p >= 0.60: alloc 0.22
  p >= 0.55: alloc 0.15
  p <  0.55: flat

Total alloc max 0.90 (3 coin × 0.30), validator 1.0 cap altı.
Leverage 1 (brief kuralı).

Docker/offline uyumu:
  - İnternet çağrısı yok
  - Tüm eğitim içeride, LightGBM deterministic (seed sabit)
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMClassifier
except ImportError as exc:
    raise ImportError(
        "LightGBM kurulu değil. `pip install lightgbm` çalıştır."
    ) from exc

from cnlib.base_strategy import BaseStrategy, COINS
from utils.features import build_features, build_training_set


warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


class MLLightGBM(BaseStrategy):
    """Walk-forward LightGBM, long-only, leverage 1."""

    WARMUP = 500                   # ilk eğitim için minimum candle
    RETRAIN_FREQUENCY = 90         # her 90 candle'da yeniden eğit
    HORIZON = 5                    # target: 5 gün sonraki hareket

    # Probability tier → allocation
    PROB_TIERS = [
        (0.70, 0.30),  # >= 0.70 → 0.30
        (0.60, 0.22),  # >= 0.60 → 0.22
        (0.55, 0.15),  # >= 0.55 → 0.15
    ]
    MIN_PROB = 0.55                # < 0.55 → flat
    TOTAL_ALLOC_CAP = 0.90

    RANDOM_SEED = 42

    def __init__(self) -> None:
        super().__init__()
        self.models: dict[str, LGBMClassifier] = {}
        self.last_train_at: dict[str, int] = {c: -1 for c in COINS}
        # Cache feature DataFrame tüm data için — her predict'te sadece son satırı al
        self._feature_cache: dict[str, pd.DataFrame] = {}
        self._last_feature_update: dict[str, int] = {c: -1 for c in COINS}

    # ------------------------------------------------------------------
    # Eğitim
    # ------------------------------------------------------------------

    def _train_coin(self, coin: str, up_to_index: int) -> None:
        """
        0..up_to_index (dahil) geçmişi kullanarak model fit et.
        Target forward-shift olduğu için son HORIZON satır otomatik atılır.
        """
        df = self._full_data[coin].iloc[: up_to_index + 1]
        X, y = build_training_set(df, horizon=self.HORIZON)
        if len(X) < 100:
            return  # çok az sample

        model = LGBMClassifier(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.05,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.5,
            class_weight="balanced",
            random_state=self.RANDOM_SEED,
            verbose=-1,
        )
        model.fit(X, y)
        self.models[coin] = model
        self.last_train_at[coin] = up_to_index

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _compute_features_row(self, coin: str, df: pd.DataFrame) -> np.ndarray | None:
        """
        Son candle için feature vektörü hesapla.
        NaN varsa None döner.
        """
        # Cache stratejisi: feature tablosunu bir kez hesapla, tekrar kullan
        # Ama coin_data her predict'te slicing yapıyor → df boyutu artıyor
        # Basit: sadece son 100 candle üzerinde features hesapla (SMA50 warmup bile yeterli)
        last_n = min(len(df), 120)
        tail = df.iloc[-last_n:]
        feats = build_features(tail)
        last_row = feats.iloc[-1]
        if last_row.isna().any():
            return None
        return last_row.values.reshape(1, -1)

    # ------------------------------------------------------------------
    # Ana predict
    # ------------------------------------------------------------------

    def predict(self, data: dict[str, Any]) -> list[dict]:
        i = self.candle_index

        # Warmup: yeterli history yok → hepsi flat
        if i < self.WARMUP:
            return [self._flat(c) for c in COINS]

        # Re-train gerekli mi?
        for coin in COINS:
            if (
                self.last_train_at[coin] < 0
                or (i - self.last_train_at[coin]) >= self.RETRAIN_FREQUENCY
            ):
                self._train_coin(coin, i)

        # Inference — her coin için probability hesapla
        probs: dict[str, float] = {}
        for coin in COINS:
            if coin not in self.models:
                probs[coin] = 0.0
                continue
            feat_row = self._compute_features_row(coin, data[coin])
            if feat_row is None:
                probs[coin] = 0.0
                continue
            proba = self.models[coin].predict_proba(feat_row)[0]
            # Sınıf 1 = pozitif (fiyat artacak)
            # predict_proba kolon sırası sınıf etiketlerine göre
            class_idx = list(self.models[coin].classes_).index(1) if 1 in self.models[coin].classes_ else -1
            probs[coin] = float(proba[class_idx]) if class_idx >= 0 else 0.0

        # Probability → allocation tier
        raw_alloc = {}
        for coin, p in probs.items():
            raw_alloc[coin] = self._prob_to_alloc(p)

        # Toplam cap — aşıyorsa proportional küçült
        total = sum(raw_alloc.values())
        if total > self.TOTAL_ALLOC_CAP and total > 0:
            scale = self.TOTAL_ALLOC_CAP / total
            raw_alloc = {c: v * scale for c, v in raw_alloc.items()}

        # Decisions
        decisions: list[dict] = []
        for coin in COINS:
            alloc = raw_alloc[coin]
            if alloc > 0.0:
                decisions.append({
                    "coin":       coin,
                    "signal":     1,
                    "allocation": alloc,
                    "leverage":   1,
                })
            else:
                decisions.append(self._flat(coin))
        return decisions

    # ------------------------------------------------------------------
    # Yardımcılar
    # ------------------------------------------------------------------

    def _prob_to_alloc(self, p: float) -> float:
        """Probability'den allocation tier'ı döner."""
        if p < self.MIN_PROB:
            return 0.0
        for threshold, alloc in self.PROB_TIERS:
            if p >= threshold:
                return alloc
        return 0.0

    @staticmethod
    def _flat(coin: str) -> dict:
        return {
            "coin":       coin,
            "signal":     0,
            "allocation": 0.0,
            "leverage":   1,
        }
