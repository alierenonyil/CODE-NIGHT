"""
Tüm stratejileri real + 4 sentetik senaryoda test et.

Her senaryo, sadece sentetik kısım (son 365 candle) üzerinde ölçülür
(start_candle=1570). "real" hariç — real için tüm 1570 candle.

Çıktı:
  results/stress_matrix.json   — [strategy][scenario] → {return, sharpe, dd}
  Konsolda karşılaştırmalı tablo.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cnlib import backtest
from strategies.baseline_momentum import BaselineMomentum
from strategies.technical_ensemble import TechnicalEnsemble
from strategies.enhanced_momentum import EnhancedMomentum
from strategies.momentum_hysteresis import MomentumHysteresis
from strategies.ml_lightgbm import MLLightGBM


PROJ_DIR = Path(__file__).parent
SYNTH_DIR = PROJ_DIR / "synthetic_data"
RESULTS_DIR = PROJ_DIR / "results"

STRATEGIES = [
    ("baseline",   BaselineMomentum),
    ("ensemble",   TechnicalEnsemble),
    ("enhanced",   EnhancedMomentum),
    ("hysteresis", MomentumHysteresis),
    ("ml",         MLLightGBM),
]

# (name, data_dir, start_candle)
SCENARIOS = [
    ("real",    None,                    0),
    ("normal",  SYNTH_DIR / "normal",    1570),
    ("crash",   SYNTH_DIR / "crash",     1570),
    ("pump",    SYNTH_DIR / "pump",      1570),
    ("mixed",   SYNTH_DIR / "mixed",     1570),
]


def compute_metrics(result) -> dict:
    df = pd.DataFrame(result.portfolio_series)
    if len(df) < 2:
        return {
            "return_pct": result.return_pct,
            "sharpe":     0.0,
            "sortino":    0.0,
            "max_dd":     0.0,
        }
    pv = df["portfolio_value"]
    daily = pv.pct_change().dropna()
    sharpe = (daily.mean() / daily.std()) * (365 ** 0.5) if daily.std() > 0 else 0.0
    down = daily[daily < 0]
    sortino = (daily.mean() / down.std()) * (365 ** 0.5) if len(down) and down.std() > 0 else 0.0
    cummax = pv.cummax()
    dd = (pv - cummax) / cummax
    return {
        "return_pct":          float(result.return_pct),
        "final_value":         float(result.final_portfolio_value),
        "sharpe":              float(sharpe),
        "sortino":             float(sortino),
        "max_dd":              float(dd.min()),
        "trades":              int(result.total_trades),
        "liquidations":        int(result.total_liquidations),
        "validation_errors":   int(result.validation_errors),
        "strategy_errors":     int(result.strategy_errors),
        "failed_opens":        int(result.failed_opens),
    }


def main() -> None:
    matrix: dict[str, dict[str, dict]] = {}

    print("=" * 100)
    print("  STRES TEST\u0130 \u2014 5 strateji \u00d7 5 senaryo (real + 4 sentetik)")
    print("=" * 100)

    total_t0 = time.time()
    for strat_name, StratCls in STRATEGIES:
        matrix[strat_name] = {}
        for scn_name, data_dir, start_candle in SCENARIOS:
            print(f"\n>> [{strat_name:<10}] @ [{scn_name:<7}] ...", end="", flush=True)
            t0 = time.time()
            strategy = StratCls()
            result = backtest.run(
                strategy=strategy,
                initial_capital=3000.0,
                data_dir=data_dir,
                start_candle=start_candle,
                silent=True,
            )
            m = compute_metrics(result)
            m["seconds"] = round(time.time() - t0, 1)
            matrix[strat_name][scn_name] = m
            print(f" ret={m['return_pct']:+8.2f}%  sharpe={m['sharpe']:+.2f}  "
                  f"dd={m['max_dd']*100:+6.2f}%  ({m['seconds']}s)")

    print(f"\nToplam s\u00fcre: {time.time() - total_t0:.1f}s")

    # JSON'a kaydet
    (RESULTS_DIR / "stress_matrix.json").write_text(
        json.dumps(matrix, indent=2), encoding="utf-8"
    )

    # Karşılaştırma tablosu
    _print_matrix(matrix, "return_pct", "RETURN %", "{:>+10.2f}%")
    _print_matrix(matrix, "sharpe",     "SHARPE",   "{:>+10.3f}")
    _print_matrix(matrix, "max_dd",     "MAX DD",   "{:>+10.2%}")

    # Ranking — her senaryo için 1-5 sırala
    print("\n" + "=" * 100)
    print("  SIRALAMA (return baz\u0131nda)")
    print("=" * 100)
    for scn_name, _, _ in SCENARIOS:
        ranked = sorted(
            matrix.items(),
            key=lambda kv: -kv[1][scn_name]["return_pct"]
        )
        print(f"\n[{scn_name}]")
        for rank, (strat, data) in enumerate(ranked, 1):
            m = data[scn_name]
            print(f"  {rank}. {strat:<12} {m['return_pct']:+8.2f}%  "
                  f"sharpe={m['sharpe']:+.3f}  dd={m['max_dd']*100:+6.2f}%")


def _print_matrix(matrix, key, label, fmt):
    print("\n" + "=" * 100)
    print(f"  {label}")
    print("=" * 100)
    scn_names = [s[0] for s in SCENARIOS]
    name_w = 14
    col_w = 14
    print(f"{'Strateji':<{name_w}}" + "".join(f"{s:>{col_w}}" for s in scn_names))
    print("-" * (name_w + col_w * len(scn_names)))
    for strat_name, _ in STRATEGIES:
        row = f"{strat_name:<{name_w}}"
        for scn in scn_names:
            v = matrix[strat_name][scn][key]
            row += f"{fmt.format(v):>{col_w}}"
        print(row)


if __name__ == "__main__":
    main()
