"""
Mega Stress Test — Adaptive V3 (+ baseline) tüm data setlerinde.

16+ dataset:
  - Train (yarışma)
  - 4 sentetik senaryo
  - 2 gerçek (yf_real, bn_majors, bn_altcoins)
  - 10 yahoo çoklu dönem (2018-2026)

Her dataset için: multiplier, sharpe, max DD, liquidation, regime_sabitliği
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from cnlib import backtest
from strategies.adaptive_v3 import AdaptiveV3
from strategies.adaptive_v4 import AdaptiveV4
from strategies.robust_hybrid import RobustHybrid
from strategies.leveraged_baseline import LeveragedBaseline3x, LeveragedBaseline5x
from strategies.swap_optimized import SwapV3
from strategies.baseline_momentum import BaselineMomentum


ROOT = Path(__file__).parent
SYNTH = ROOT / "synthetic_data"
MULTI = ROOT / "multi_data"

SCENARIOS = [
    ("train",            None,                       0),
    ("s_normal",         SYNTH / "normal",           1570),
    ("s_crash",          SYNTH / "crash",            1570),
    ("s_pump",           SYNTH / "pump",             1570),
    ("s_mixed",          SYNTH / "mixed",            1570),
    ("real_yf_2y",       ROOT / "real_crypto_data",  0),
    ("bn_majors_2y",     ROOT / "binance_data/majors",    0),
    ("bn_altcoins_2y",   ROOT / "binance_data/altcoins",  0),
]

# Multi-data ek senaryolar
for d in sorted(MULTI.glob("*/")):
    SCENARIOS.append((d.name, d, 0))


STRATEGIES = [
    ("Baseline",      BaselineMomentum),
    ("Base5x",        LeveragedBaseline5x),
    ("SwapV3",        SwapV3),
    ("Robust",        RobustHybrid),
    ("AdaptiveV3",    AdaptiveV3),
    ("AdaptiveV4",    AdaptiveV4),
]


def _metrics(r):
    df = pd.DataFrame(r.portfolio_series)
    if len(df) < 2:
        return 0.0, 0.0
    pv = df["portfolio_value"]
    daily = pv.pct_change().dropna()
    sharpe = (daily.mean() / daily.std()) * (365 ** 0.5) if daily.std() > 0 else 0.0
    cummax = pv.cummax()
    dd = ((pv - cummax) / cummax).min()
    return float(sharpe), float(dd)


def main() -> None:
    print(f"Stratejiler: {len(STRATEGIES)}")
    print(f"Datasetler:  {len(SCENARIOS)}")
    print()

    # strateji x dataset matris
    matrix = {}
    for sname, cls in STRATEGIES:
        matrix[sname] = {}
        for dsname, ddir, start in SCENARIOS:
            try:
                strat = cls()
                r = backtest.run(strategy=strat, initial_capital=3000.0,
                                 data_dir=ddir, start_candle=start, silent=True)
                mult = max(r.final_portfolio_value / 3000.0, 1e-9)
                sh, dd = _metrics(r)
                matrix[sname][dsname] = {
                    "mult": mult,
                    "sharpe": sh,
                    "dd": dd,
                    "liq": r.total_liquidations,
                    "trades": r.total_trades,
                }
            except Exception as e:
                matrix[sname][dsname] = {"error": str(e)}

    # Print: multiplier table
    print("=" * 250)
    print("  MEGA STRESS TEST — MULTIPLIER @ $3000")
    print("=" * 250)
    header = f"{'Dataset':<26}"
    for sname, _ in STRATEGIES:
        header += f"{sname:>16}"
    print(header)
    print("-" * 250)

    # Each row = dataset
    for dsname, _, _ in SCENARIOS:
        line = f"{dsname:<26}"
        for sname, _ in STRATEGIES:
            res = matrix[sname][dsname]
            if "error" in res:
                line += f"{'ERR':>16}"
                continue
            m = res["mult"]
            if m > 1e6:
                cell = f"{m:.1e}"
            elif m > 100:
                cell = f"{m:,.0f}x"
            else:
                cell = f"{m:.3f}x"
            if res["liq"] > 0:
                cell = cell[:-1] + f"L{res['liq']}"
            line += f"{cell:>16}"
        print(line)

    # Summary — geo mean + worst per strategy
    print()
    print("=" * 120)
    print("  STRATEGY SUMMARY — geo mean, worst-case, pozitif dataset sayisi")
    print("=" * 120)
    summary_rows = []
    for sname, _ in STRATEGIES:
        mults = []
        for dsname, _, _ in SCENARIOS:
            res = matrix[sname][dsname]
            if "error" in res:
                continue
            mults.append(res["mult"])
        if not mults:
            continue
        log_mults = [math.log10(max(m, 1e-9)) for m in mults]
        geo_mean = 10 ** (sum(log_mults) / len(log_mults))
        worst = min(mults)
        positive_count = sum(1 for m in mults if m >= 1.0)
        summary_rows.append((sname, geo_mean, worst, positive_count, len(mults)))

    summary_rows.sort(key=lambda x: -x[1])  # geo_mean desc
    print(f"{'rank':<5}{'Strateji':<13}{'Geo Mean':>18}{'Worst':>14}{'Pozitif':>12}")
    print("-" * 120)
    for rank, (s, geo, w, pos, total) in enumerate(summary_rows, 1):
        print(f"  {rank:<3}{s:<13}{geo:>18.4e}{w:>14.4f}{pos:>5}/{total}")


if __name__ == "__main__":
    main()
