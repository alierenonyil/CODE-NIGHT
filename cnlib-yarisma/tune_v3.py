"""
V3 damper parametre grid search.

Params:
  MIN_MAGNITUDE:         0.002, 0.003, 0.005, 0.008, 0.010
  DAMPER_THRESHOLD:      -0.03, -0.05, -0.07, -0.10  (broad market drop trigger)
  DAMPER_CANDLES:        1, 2, 3, 4                   (damper süresi)
  EXTREME_THRESHOLD:     0.10, 0.12, 0.15, 1.00       (single-candle spike filter)

Strategy: phase 1 = real data'da hızlı sweep; phase 2 = top 10 full stres test.
Target: real log10 maksimum + tüm sentetik senaryoda pozitif kalma.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import pandas as pd

from cnlib import backtest
from cnlib.base_strategy import BaseStrategy, COINS


SYNTH = Path(__file__).parent / "synthetic_data"
SCENARIOS = [
    ("real",   None,              0),
    ("normal", SYNTH / "normal",  1570),
    ("crash",  SYNTH / "crash",   1570),
    ("pump",   SYNTH / "pump",    1570),
    ("mixed",  SYNTH / "mixed",   1570),
]


class SwapV3Param(BaseStrategy):
    """V3 damper with parametrized knobs."""

    MIN_MAGNITUDE = 0.005
    DAMPER_THRESHOLD = -0.05
    DAMPER_CANDLES = 2
    EXTREME_THRESHOLD = 0.12
    EXTREME_WINDOW = 5
    MAX_PER_COIN = 0.333
    TOTAL_CAP = 0.999

    def __init__(self):
        super().__init__()
        self.damper_remaining = 0

    def predict(self, data: dict[str, Any]) -> list[dict]:
        recent_drops = []
        for coin in COINS:
            df = data[coin]
            if len(df) >= 2:
                r = float(df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1)
                recent_drops.append(r)
        if len(recent_drops) == 3 and sum(recent_drops) / 3 < self.DAMPER_THRESHOLD:
            self.damper_remaining = self.DAMPER_CANDLES
        damper_active = self.damper_remaining > 0
        if damper_active:
            self.damper_remaining -= 1
            current_lev = 5
        else:
            current_lev = 10

        signals: dict[str, int] = {}
        for coin in COINS:
            df = data[coin]
            if len(df) < self.EXTREME_WINDOW + 1:
                signals[coin] = 0
                continue

            close = df["Close"]
            last_r = float(close.iloc[-1] / close.iloc[-2] - 1)

            if abs(last_r) < self.MIN_MAGNITUDE:
                signals[coin] = 0
                continue

            ext = close.pct_change().iloc[-self.EXTREME_WINDOW:].abs().max()
            if pd.notna(ext) and ext > self.EXTREME_THRESHOLD:
                signals[coin] = 0
                continue

            signals[coin] = 1 if last_r > 0 else -1

        active = [c for c, s in signals.items() if s != 0]
        k = len(active)
        per_alloc = min(self.MAX_PER_COIN, self.TOTAL_CAP / k) if k > 0 else 0.0

        decisions: list[dict] = []
        for coin in COINS:
            s = signals[coin]
            if s != 0:
                decisions.append({
                    "coin": coin, "signal": s,
                    "allocation": per_alloc, "leverage": current_lev,
                })
            else:
                decisions.append({"coin": coin, "signal": 0,
                                  "allocation": 0.0, "leverage": 1})
        return decisions


def make_class(mm: float, dt: float, dc: int, et: float) -> type:
    return type(
        "V3Tuned",
        (SwapV3Param,),
        {
            "MIN_MAGNITUDE":     mm,
            "DAMPER_THRESHOLD":  dt,
            "DAMPER_CANDLES":    dc,
            "EXTREME_THRESHOLD": et,
        },
    )


def run_backtest(cls, scn: str, ddir: Path | None, start: int, initial: float = 3000.0) -> tuple[float, int]:
    strat = cls()
    r = backtest.run(strategy=strat, initial_capital=initial, data_dir=ddir, start_candle=start, silent=True)
    mult = max(r.final_portfolio_value / initial, 1e-6)
    return mult, r.total_liquidations


def main() -> None:
    # Phase 1 — real only, grid sweep
    MMs  = [0.002, 0.003, 0.005, 0.008, 0.010]
    DTs  = [-0.03, -0.05, -0.07, -0.10]
    DCs  = [1, 2, 3, 4]
    ETs  = [0.10, 0.12, 0.15, 1.00]

    combos = [(mm, dt, dc, et) for mm in MMs for dt in DTs for dc in DCs for et in ETs]
    print(f"Phase 1: {len(combos)} configs on REAL only @ \$3000")

    t0 = time.time()
    real_results = []
    for i, (mm, dt, dc, et) in enumerate(combos):
        cls = make_class(mm, dt, dc, et)
        mult, liq = run_backtest(cls, "real", None, 0)
        real_results.append({
            "mm": mm, "dt": dt, "dc": dc, "et": et,
            "real_mult": mult, "liq": liq,
            "log_real": math.log10(mult),
        })
        if i % 50 == 0:
            print(f"  {i}/{len(combos)}  ({time.time()-t0:.1f}s)  "
                  f"log={math.log10(mult):+.2f}  "
                  f"params=mm={mm} dt={dt} dc={dc} et={et}")

    # Top 10 real
    real_results.sort(key=lambda x: -x["log_real"])
    top = real_results[:10]
    print(f"\nPhase 1 done in {time.time()-t0:.1f}s. Top 10 on REAL:")
    print(f"{'rank':<5}{'log':>8} {'mult':>14}{'liq':>4}  mm       dt       dc   et")
    for rank, r in enumerate(top, 1):
        print(f"  {rank:<3}{r['log_real']:+8.2f}  {r['real_mult']:>10.2e} {r['liq']:>4}  "
              f"mm={r['mm']:<5} dt={r['dt']:<5} dc={r['dc']:<2} et={r['et']}")

    # Phase 2 — top 10 full scenario
    print(f"\nPhase 2: top 10 across all scenarios")
    phase2 = []
    for r in top:
        cls = make_class(r["mm"], r["dt"], r["dc"], r["et"])
        logs = {}
        liqs = {}
        for scn, ddir, start in SCENARIOS:
            mult, liq = run_backtest(cls, scn, ddir, start)
            logs[scn] = math.log10(mult)
            liqs[scn] = liq
        log_sum = sum(logs.values())
        phase2.append({**r, "logs": logs, "liqs": liqs, "log_sum": log_sum})

    phase2.sort(key=lambda x: -x["log_sum"])
    print(f"\n{'rank':<5}{'logSUM':>8}  real     normal   crash    pump     mixed     params")
    for rank, r in enumerate(phase2, 1):
        scns = " ".join(f"{r['logs'][s[0]]:+6.2f}" for s in SCENARIOS)
        params = f"mm={r['mm']} dt={r['dt']} dc={r['dc']} et={r['et']}"
        print(f"  {rank:<3}{r['log_sum']:>8.2f}  {scns}   {params}")

    # En iyi parametrelerle son value hesapla
    winner = phase2[0]
    cls = make_class(winner["mm"], winner["dt"], winner["dc"], winner["et"])
    mult, liq = run_backtest(cls, "real", None, 0)
    final_value = 3000.0 * mult
    print(f"\n>>> WINNER: mm={winner['mm']} dt={winner['dt']} "
          f"dc={winner['dc']} et={winner['et']}")
    print(f"    Real @ \$3000 \u2192 \${final_value:.4e}  ({mult:.4e}x)  liq={liq}")


if __name__ == "__main__":
    main()
