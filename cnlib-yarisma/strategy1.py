"""
Strategy v3.2 PRODUCTION — Tight Crash Gating.

CRASH DETECTION (v3.2 fix — false positive azaltıldı):
  v3.1'de OR vardı: vol spike VEYA drift bear → crash. Vol-only false positive
  v3.2: vol AND drift gerek + 2 candle persistence (flash spike'lar trigger etmez)

  CRASH = (avg_vol_10d > 0.06) AND (avg_30d_drift < -0.10)
  Persistence: 2 ardışık candle koşullar sağlanmalı — flash gürültü filtresi
  Hysteresis: exit için her iki koşul ayrı ayrı yokluğu (sticky korundu)

4 REGIMES:
  AGGRESSIVE (autocorr > 0.45): long-only, 5x SMA(4)
  BALANCED   (autocorr 0.10–0.30): long-only, 3x SMA(10)
  SAFE       (autocorr < 0.10): long-only, 2x SMA(20/50) + bear gate
  CRASH      (vol spike OR deep drift): SHORT-only, 1-2x, momentum confirm

CNLIB-compliant: BaseStrategy, predict every candle, 3 coin tüm decisions,
signal {1,-1,0}, allocation toplam ≤ 1.0, leverage {1,2,3,5,10}.

Risk korundu: stop-loss, DD circuit breaker, min-hold, correlation damper,
vol-inverse allocation, emergency vol switch.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from cnlib.base_strategy import BaseStrategy, COINS


class Strategy(BaseStrategy):
    """v3.1 SAFE — simple crash detection, robust regime hybrid."""

    # ---------------- Regime detection ----------------
    WARMUP = 60
    AC_WINDOW = 60
    REGIME_REFRESH = 20

    AC_AGGR_HIGH = 0.45
    AC_AGGR_LOW  = 0.30
    AC_BAL_HIGH  = 0.10

    # Crash detection (v3.2: tight AND + persistence)
    CRASH_VOL_SPIKE = 0.06         # avg vol 10d > 6% (sıkılaştı: 0.05 → 0.06)
    CRASH_DRIFT_DEEP = -0.10       # avg 30d drift < -10% (gevşedi: AND koşulu)
    CRASH_PERSISTENCE = 2          # 2 ardışık candle koşullar sağlanmalı

    VOL_HIGH = 0.05
    EMERGENCY_VOL = 0.08
    EMERGENCY_VOL_RECOVER = 0.06
    EXTREME_DAY = 0.12

    # ---------------- Mode parameters ----------------
    AGG_SMA = 4
    AGG_LEV_NORMAL = 5
    AGG_LEV_VOL = 3
    AGG_PER_COIN_MAX = 0.30
    AGG_TOTAL_CAP = 0.85

    BAL_SMA = 10
    BAL_LEV_NORMAL = 3
    BAL_LEV_VOL = 2
    BAL_PER_COIN_MAX = 0.30
    BAL_TOTAL_CAP = 0.85

    SAFE_SMA_FAST = 20
    SAFE_SMA_SLOW = 50
    SAFE_BEAR_DAYS = 30
    SAFE_BEAR_DRIFT = -0.10
    SAFE_LEV_NORMAL = 2
    SAFE_LEV_VOL = 1
    SAFE_PER_COIN_MAX = 0.32
    SAFE_TOTAL_CAP = 0.92

    # CRASH mode (basit, momentum confirm)
    CRASH_LEV_NORMAL = 2
    CRASH_LEV_VOL = 1
    CRASH_PER_COIN_MAX = 0.32
    CRASH_TOTAL_CAP = 0.85
    CRASH_MOMENTUM_3D = -0.03      # son 3 günde cum return < -3% → SHORT OK

    # ---------------- Risk management ----------------
    SL_SAFETY = 0.70
    DD_CB_THRESHOLD = -0.35
    DD_CB_RECOVER = -0.15
    MIN_HOLD_CANDLES = 2
    CORR_WINDOW = 15
    CORR_HIGH = 0.80
    WEIGHT_EPS = 0.01

    def __init__(self):
        super().__init__()
        # State (v3.2: + crash_streak counter for persistence)
        self.global_regime = "safe"
        self.coin_regimes: dict[str, str] = {c: "safe" for c in COINS}
        self.last_regime_check = -1
        self._in_crash = False
        self._crash_streak = 0          # v3.2: AND koşulu kaç ardışık candle sağlandı
        self._dd_cb_active = False
        # Equity tracking
        self._equity_proxy = 1.0
        self._equity_peak = 1.0
        self._emergency_safe = False
        # Min-hold
        self._open_candles: dict[str, int] = {c: 0 for c in COINS}
        self._last_signal: dict[str, int] = {c: 0 for c in COINS}
        self._last_lev: dict[str, int] = {c: 1 for c in COINS}
        self._last_alloc: dict[str, float] = {c: 0.0 for c in COINS}
        # Telemetry
        self.current_regime = "safe"
        self._last_autocorr = 0.0
        self._last_vol = 0.0

    # ============================================================
    # Core helpers
    # ============================================================

    def _coin_autocorr(self, df) -> float:
        if len(df) < self.AC_WINDOW + 2:
            return 0.0
        r = df["Close"].iloc[-self.AC_WINDOW:].pct_change().dropna()
        if len(r) < 10:
            return 0.0
        ac = r.autocorr(lag=1)
        return float(ac) if pd.notna(ac) else 0.0

    def _coin_vol(self, df, window=20) -> float:
        if len(df) < window + 1:
            return 0.05
        r = df["Close"].iloc[-window:].pct_change().dropna()
        s = r.std()
        return float(s) if pd.notna(s) and s > 0 else 0.05

    def _coin_drift(self, df, days=30) -> float:
        if len(df) < days + 1:
            return 0.0
        return float(df["Close"].iloc[-1] / df["Close"].iloc[-days] - 1)

    def _max_corr(self, data) -> float:
        if min(len(data[c]) for c in COINS) < self.CORR_WINDOW + 1:
            return 0.0
        df = pd.DataFrame({
            c: data[c]["Close"].iloc[-self.CORR_WINDOW:].pct_change().values
            for c in COINS
        }).dropna()
        if len(df) < 8:
            return 0.0
        m = df.corr().values
        n = len(COINS)
        out = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if pd.notna(m[i][j]):
                    out = max(out, abs(m[i][j]))
        return out

    # ============================================================
    # CRASH detection — SIMPLIFIED (1 OR condition)
    # ============================================================

    def _is_crash_now(self, data) -> tuple[bool, bool]:
        """Returns (vol_spike, deep_drift) — basit kontrol."""
        vols = [self._coin_vol(data[c], 10) for c in COINS]
        avg_vol = sum(vols) / len(vols) if vols else 0.0

        drifts = []
        for c in COINS:
            if len(data[c]) >= 31:
                drifts.append(self._coin_drift(data[c], 30))
        avg_drift = sum(drifts) / len(drifts) if drifts else 0.0

        vol_spike = avg_vol > self.CRASH_VOL_SPIKE
        deep_drift = avg_drift < self.CRASH_DRIFT_DEEP
        return vol_spike, deep_drift

    def _update_crash_state(self, data):
        """
        v3.2: AND-based detection + 2-candle persistence.
        Enter when: vol AND drift met for CRASH_PERSISTENCE consecutive candles.
        Exit when: BOTH conditions clear (asymmetric / sticky).
        """
        vol_spike, deep_drift = self._is_crash_now(data)
        both_now = vol_spike and deep_drift

        if not self._in_crash:
            # Increment streak only when BOTH met
            if both_now:
                self._crash_streak += 1
            else:
                self._crash_streak = 0
            # Enter only after persistence threshold reached
            if self._crash_streak >= self.CRASH_PERSISTENCE:
                self._in_crash = True
        else:
            # Exit when BOTH conditions clear (sticky)
            if not vol_spike and not deep_drift:
                self._in_crash = False
                self._crash_streak = 0

    # ============================================================
    # Regime classification (autocorr-based, hysteresis)
    # ============================================================

    def _classify_coin_regime(self, ac: float, current: str) -> str:
        if current == "aggressive":
            if ac < self.AC_AGGR_LOW:
                return "balanced" if ac >= self.AC_BAL_HIGH else "safe"
            return "aggressive"
        elif current == "balanced":
            if ac >= self.AC_AGGR_HIGH:
                return "aggressive"
            if ac < self.AC_BAL_HIGH:
                return "safe"
            return "balanced"
        else:
            if ac >= self.AC_AGGR_HIGH:
                return "aggressive"
            if ac >= self.AC_BAL_HIGH:
                return "balanced"
            return "safe"

    def _refresh_regimes(self, data):
        acs, vols = [], []
        for c in COINS:
            ac = self._coin_autocorr(data[c])
            v = self._coin_vol(data[c])
            acs.append(ac); vols.append(v)
            self.coin_regimes[c] = self._classify_coin_regime(ac, self.coin_regimes[c])

        self._last_autocorr = sum(acs) / 3
        self._last_vol = sum(vols) / 3

        # Crash overrides everything
        if self._in_crash:
            self.global_regime = "crash"
            for c in COINS:
                self.coin_regimes[c] = "crash"
            self.current_regime = "crash"
            return

        # DD CB → safe forced
        if self._dd_cb_active:
            self.global_regime = "safe"
            for c in COINS:
                self.coin_regimes[c] = "safe"
            self.current_regime = "safe"
            return

        # Majority vote
        regs = list(self.coin_regimes.values())
        if regs.count("aggressive") >= 2:
            self.global_regime = "aggressive"
        elif regs.count("safe") >= 2:
            self.global_regime = "safe"
        else:
            self.global_regime = "balanced"
        self.current_regime = self.global_regime

    # ============================================================
    # Emergency vol + DD CB
    # ============================================================

    def _check_emergency_vol(self, data) -> bool:
        vols = []
        for c in COINS:
            if len(data[c]) < 6:
                continue
            r = data[c]["Close"].iloc[-5:].pct_change().dropna().abs()
            if len(r):
                vols.append(float(r.mean()))
        if not vols:
            return self._emergency_safe
        avg = sum(vols) / len(vols)
        if not self._emergency_safe and avg > self.EMERGENCY_VOL:
            return True
        if self._emergency_safe and avg < self.EMERGENCY_VOL_RECOVER:
            return False
        return self._emergency_safe

    def _update_equity_proxy(self, data):
        delta = 0.0
        for c in COINS:
            sig = self._last_signal[c]
            if sig == 0:
                continue
            df = data[c]
            if len(df) < 2:
                continue
            r = float(df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1)
            delta += sig * r * self._last_lev[c] * self._last_alloc[c]
        self._equity_proxy *= max(1.0 + delta, 0.01)
        if self._equity_proxy > self._equity_peak:
            self._equity_peak = self._equity_proxy
            self._dd_cb_active = False
            return
        dd = (self._equity_proxy - self._equity_peak) / self._equity_peak
        if dd <= self.DD_CB_THRESHOLD:
            self._dd_cb_active = True
        elif dd >= self.DD_CB_RECOVER:
            self._dd_cb_active = False

    # ============================================================
    # Core predict
    # ============================================================

    def predict(self, data: dict[str, Any]) -> list[dict]:
        i = self.candle_index
        if i < self.WARMUP:
            self._reset_last()
            return [self._flat(c) for c in COINS]

        self._update_equity_proxy(data)
        self._update_crash_state(data)
        self._emergency_safe = self._check_emergency_vol(data)

        if self.last_regime_check < 0 or (i - self.last_regime_check) >= self.REGIME_REFRESH:
            self._refresh_regimes(data)
            self.last_regime_check = i

        # Crash always overrides (after refresh too)
        if self._in_crash:
            self.global_regime = "crash"
            for c in COINS:
                self.coin_regimes[c] = "crash"
            self.current_regime = "crash"
        elif self._emergency_safe:
            self.global_regime = "safe"
            for c in COINS:
                self.coin_regimes[c] = "safe"
            self.current_regime = "safe"

        # Per-coin signal
        sigs, levs, sls, weights = {}, {}, {}, {}
        for c in COINS:
            sig, lev, sl, w = self._coin_signal(data[c], self.coin_regimes[c])
            sigs[c] = sig; levs[c] = lev; sls[c] = sl; weights[c] = w

        # Min-hold
        for c in COINS:
            prev = self._last_signal[c]
            cur = sigs[c]
            if prev != 0 and cur != prev and cur != 0 and self._open_candles[c] < self.MIN_HOLD_CANDLES:
                sigs[c] = prev
                levs[c] = self._last_lev[c]
                if weights[c] == 0:
                    weights[c] = 1.0
            if sigs[c] != 0:
                self._open_candles[c] += 1
            else:
                self._open_candles[c] = 0

        # Allocation
        active = [c for c in COINS if sigs[c] != 0]
        allocations = {c: 0.0 for c in COINS}
        if active:
            total_cap = self._mode_total_cap()
            per_max = self._mode_per_coin_max()
            if self._in_crash:
                total_cap *= 0.80
            elif self._max_corr(data) > self.CORR_HIGH:
                total_cap *= 0.75
            tw = sum(weights[c] for c in active) or 1.0
            for c in active:
                raw = (weights[c] / tw) * total_cap
                allocations[c] = min(raw, per_max)
            tot = sum(allocations.values())
            if tot > total_cap and tot > 0:
                k = total_cap / tot
                for c in allocations:
                    allocations[c] *= k

        decisions = []
        for c in COINS:
            if sigs[c] != 0 and allocations[c] > 0:
                d = {
                    "coin": c, "signal": sigs[c],
                    "allocation": allocations[c],
                    "leverage": levs[c],
                }
                if sls[c] is not None:
                    d["stop_loss"] = sls[c]
                decisions.append(d)
            else:
                decisions.append(self._flat(c))

        self._last_signal = {c: sigs[c] for c in COINS}
        self._last_lev = {c: levs[c] if sigs[c] != 0 else 1 for c in COINS}
        self._last_alloc = {c: allocations[c] for c in COINS}
        return decisions

    # ============================================================
    # Per-coin signals per mode
    # ============================================================

    def _coin_signal(self, df, regime):
        if regime == "aggressive":
            return self._sig_long(df, self.AGG_SMA, self.AGG_LEV_NORMAL, self.AGG_LEV_VOL)
        elif regime == "balanced":
            return self._sig_long(df, self.BAL_SMA, self.BAL_LEV_NORMAL, self.BAL_LEV_VOL)
        elif regime == "crash":
            return self._sig_crash(df)
        else:
            return self._sig_safe(df)

    def _sig_long(self, df, sma_period, lev_normal, lev_vol):
        if len(df) < max(sma_period, 4):
            return 0, 1, None, 0.0
        close = df["Close"]
        last = float(close.iloc[-1])
        sma = float(close.iloc[-sma_period:].mean())
        if last <= sma:
            return 0, 1, None, 0.0
        ext = close.pct_change().iloc[-3:].abs().max()
        if pd.notna(ext) and ext > self.EXTREME_DAY:
            return 0, 1, None, 0.0
        vol = self._coin_vol(df, 10)
        lev = lev_vol if vol > self.VOL_HIGH else lev_normal
        sl_dist = (1.0 / lev) * self.SL_SAFETY
        sl = last * (1.0 - sl_dist)
        w = 1.0 / (vol + self.WEIGHT_EPS)
        return 1, lev, sl, w

    def _sig_safe(self, df):
        if len(df) < self.SAFE_SMA_SLOW:
            return 0, 1, None, 0.0
        close = df["Close"]
        last = float(close.iloc[-1])
        sma_fast = float(close.iloc[-self.SAFE_SMA_FAST:].mean())
        sma_slow = float(close.iloc[-self.SAFE_SMA_SLOW:].mean())
        if len(close) >= self.SAFE_BEAR_DAYS + 1:
            drift = float(close.iloc[-1] / close.iloc[-self.SAFE_BEAR_DAYS] - 1)
            if last < sma_slow and drift < self.SAFE_BEAR_DRIFT:
                return 0, 1, None, 0.0
        if last <= sma_fast or last <= sma_slow:
            return 0, 1, None, 0.0
        ext = close.pct_change().iloc[-5:].abs().max()
        if pd.notna(ext) and ext > 0.15:
            return 0, 1, None, 0.0
        vol = self._coin_vol(df, 20)
        lev = self.SAFE_LEV_VOL if vol > self.VOL_HIGH else self.SAFE_LEV_NORMAL
        sl_dist = (1.0 / lev) * self.SL_SAFETY
        sl = last * (1.0 - sl_dist)
        w = 1.0 / (vol + self.WEIGHT_EPS)
        return 1, lev, sl, w

    def _sig_crash(self, df):
        """SHORT in crash regime — single momentum confirm + extreme filter."""
        if len(df) < 5:
            return 0, 1, None, 0.0
        close = df["Close"]
        last = float(close.iloc[-1])
        # Single confirmation: 3-day cumulative return < -3%
        momentum_3d = float(close.iloc[-1] / close.iloc[-4] - 1)
        if momentum_3d > self.CRASH_MOMENTUM_3D:
            return 0, 1, None, 0.0
        # Extreme single-day filter (squeeze koruması)
        ext = close.pct_change().iloc[-3:].abs().max()
        if pd.notna(ext) and ext > self.EXTREME_DAY:
            return 0, 1, None, 0.0
        vol = self._coin_vol(df, 10)
        lev = self.CRASH_LEV_VOL if vol > self.VOL_HIGH else self.CRASH_LEV_NORMAL
        sl_dist = (1.0 / lev) * self.SL_SAFETY if lev > 1 else 0.20
        sl = last * (1.0 + sl_dist)
        w = 1.0 / (vol + self.WEIGHT_EPS)
        return -1, lev, sl, w

    # ============================================================
    # Mode caps
    # ============================================================

    def _mode_total_cap(self):
        return {
            "aggressive": self.AGG_TOTAL_CAP,
            "balanced":   self.BAL_TOTAL_CAP,
            "safe":       self.SAFE_TOTAL_CAP,
            "crash":      self.CRASH_TOTAL_CAP,
        }[self.global_regime]

    def _mode_per_coin_max(self):
        return {
            "aggressive": self.AGG_PER_COIN_MAX,
            "balanced":   self.BAL_PER_COIN_MAX,
            "safe":       self.SAFE_PER_COIN_MAX,
            "crash":      self.CRASH_PER_COIN_MAX,
        }[self.global_regime]

    def _reset_last(self):
        self._last_signal = {c: 0 for c in COINS}
        self._last_lev = {c: 1 for c in COINS}
        self._last_alloc = {c: 0.0 for c in COINS}

    @staticmethod
    def _flat(coin):
        return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}


if __name__ == "__main__":
    from cnlib import backtest
    result = backtest.run(strategy=Strategy(), initial_capital=3000.0)
    result.print_summary()
