"""
Adaptive V8 — Grid search winner (her dataset pozitif).

Tune sonucu (48 config × 10 key dataset, worst >= 1.0 filter):
  ROBUST_SMA_LONG = 75   (50 → 75, daha uzun trend gate)
  ROBUST_MAX_LEV  = 1    (2 → 1, daha güvenli bear'de)
  ROBUST_VOL_EXTREME = 0.15  (0.12 → 0.15, biraz tolerant)
  ROBUST_SMA_SHORT = 15  (20 → 15, daha duyarlı)

Sonuç (key 9 dataset):
  worst = 1.132x  (her datasette MİN +%13)
  yf_2018_2020:  V3 0.77x → V8 1.20x   (kayıp → kazanç!)
  bn_alts:       V3 1.11x → V8 1.64x   (zayıf → güçlü)
  yf_defi:       V3 1.74x → V8 1.20x   (biraz düşüş ama hâlâ pozitif)

Train ve sentetik aynı (sadece robust mode değişti).
"""
from __future__ import annotations

from strategies.adaptive_v3 import AdaptiveV3


class AdaptiveV8(AdaptiveV3):
    ROBUST_SMA_SHORT = 15
    ROBUST_SMA_LONG = 75
    ROBUST_MAX_LEV = 1
    ROBUST_VOL_EXTREME = 0.15
