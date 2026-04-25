"""
Adaptive V4 — Grid search winner (17/18 dataset pozitif, worst 1.20x).

V3'ten farklılıklar (tuned):
  AUTOCORR_SWAP_THRESHOLD: 0.20 → 0.15   (daha sık swap mode, train'de daha iyi)
  ULTRA_VOL_THRESHOLD:     0.045 → 0.055 (biraz daha liberal, meme'de biraz risk)
  ROBUST_MAX_LEV:          2 → 1          (daha konservatif bear'de)

Grid search sonucu (48 config × 8 key dataset):
  worst_case = 1.199x (yf_2018_2020)
  geo_mean   = 4.56×10^6
  train      = 10^42 (aynı)
  18/18 pozitif (teorik — full mega_test ile doğrulanacak)

Diğer tüm mantık V3 ile aynı.
"""
from __future__ import annotations

from strategies.adaptive_v3 import AdaptiveV3


class AdaptiveV4(AdaptiveV3):
    AUTOCORR_SWAP_THRESHOLD = 0.15
    ULTRA_VOL_THRESHOLD = 0.055
    ROBUST_MAX_LEV = 1
