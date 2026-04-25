"""Leveraged Baseline — 2x, 3x, 5x, 10x karşılaştırması ($1000 başlangıç)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cnlib import backtest
from strategies.leveraged_baseline import (
    LeveragedBaseline2x,
    LeveragedBaseline3x,
    LeveragedBaseline5x,
    LeveragedBaseline10x,
)


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _metrics(df: pd.DataFrame) -> dict:
    pv = df["portfolio_value"]
    daily = pv.pct_change().dropna()
    sharpe = (daily.mean() / daily.std()) * (365 ** 0.5) if daily.std() > 0 else 0.0
    cummax = pv.cummax()
    dd = (pv - cummax) / cummax
    return {
        "sharpe": float(sharpe),
        "max_drawdown": float(dd.min()),
    }


def run_one(cls, label: str) -> dict:
    strategy = cls()
    result = backtest.run(strategy=strategy, initial_capital=1000.0, silent=True)
    df = pd.DataFrame(result.portfolio_series)
    mx = _metrics(df)
    return {
        "label":                label,
        "final_value":          result.final_portfolio_value,
        "multiplier":           result.final_portfolio_value / 1000.0,
        "return_pct":           result.return_pct,
        "sharpe":               mx["sharpe"],
        "max_dd":               mx["max_drawdown"],
        "total_trades":         result.total_trades,
        "liquidations":         result.total_liquidations,
        "liquidation_loss":     result.total_liquidation_loss,
        "failed_opens":         result.failed_opens,
    }


def main() -> None:
    print("=" * 80)
    print("  LEVERAGED BASELINE \u2014 $1000 ba\u015flang\u0131\u00e7, 1570 candle")
    print("=" * 80)

    variants = [
        (LeveragedBaseline2x,  "Baseline 2x"),
        (LeveragedBaseline3x,  "Baseline 3x"),
        (LeveragedBaseline5x,  "Baseline 5x"),
        (LeveragedBaseline10x, "Baseline 10x"),
    ]

    results = []
    for cls, label in variants:
        r = run_one(cls, label)
        results.append(r)
        print(f"\n  [{label:<13}] "
              f"${r['final_value']:>15,.2f}  ({r['multiplier']:>12,.2f}x)  "
              f"return={r['return_pct']:+10.2f}%  "
              f"sharpe={r['sharpe']:+.2f}  "
              f"dd={r['max_dd']*100:+6.2f}%  "
              f"liq={r['liquidations']} (${r['liquidation_loss']:.0f})")

    # Kaydet
    (RESULTS_DIR / "leveraged_sweep.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"\nJSON: results/leveraged_sweep.json")

    # Hedefe ne kadar yak\u0131n?
    print("\n" + "=" * 80)
    print("  HEDEF KAR\u015eILA\u015eTIRMA")
    print("=" * 80)
    target = 80_000 * 10**15  # 80K katrilyon
    target_mult = target / 1000
    print(f"  Rakip hedef:  $1000 \u2192 ${target:>30,.0f}  ({target_mult:,.2e}x)")
    for r in results:
        gap = target_mult / r['multiplier']
        print(f"  {r['label']:<15} ${r['final_value']:>30,.2f}  "
              f"({r['multiplier']:>12,.2f}x)  gap: {gap:,.2e}")


if __name__ == "__main__":
    main()
