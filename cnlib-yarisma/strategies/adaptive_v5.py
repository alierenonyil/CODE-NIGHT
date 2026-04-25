"""
Adaptive V5 — "Her dataset'te %60+ kazanç" hedefli.

V3'ten farklılıklar:
  ROBUST_MAX_LEV: 2 → 3       (daha agresif bear dönemde)
  Hybrid ve Robust daha sık aktif ki %60 minimum tutulsun
  Safe mode extreme threshold 0.10 → 0.12 (daha az flat)

Strateji: her datasette minimum kazanç + ML model overfit riskiz
"""
from __future__ import annotations

from strategies.adaptive_v3 import AdaptiveV3


class AdaptiveV5(AdaptiveV3):
    # Robust modda 3x leverage
    ROBUST_MAX_LEV = 3
    # Hafif tighter entry
    ROBUST_VOL_EXTREME = 0.10
    # Hybrid daha agresif max concentration
    HYBRID_CONCENTRATION = 0.7
    # Safe mode daha liberal
    SAFE_EXTREME = 0.12
