"""
Baseline (20-gün momentum) backtest runner.

Kullanım:
    python run_baseline.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from cnlib import backtest
from strategies.baseline_momentum import BaselineMomentum


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main() -> None:
    print("=" * 55)
    print("  BASELINE MOMENTUM — 20-gün SMA")
    print("=" * 55)

    strategy = BaselineMomentum()
    result = backtest.run(strategy=strategy, initial_capital=3000.0, silent=False)

    print()
    result.print_summary()

    # Detaylı metrikler
    df = result.portfolio_dataframe()
    if len(df) > 1:
        pv = df["portfolio_value"]
        daily_returns = pv.pct_change().dropna()

        # Sharpe (risk-free = 0, günlük → yıllık)
        if daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * (365 ** 0.5)
        else:
            sharpe = 0.0

        # Max drawdown
        cummax = pv.cummax()
        drawdown = (pv - cummax) / cummax
        max_dd = float(drawdown.min())

        # Sortino (downside-only std)
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = (daily_returns.mean() / downside.std()) * (365 ** 0.5)
        else:
            sortino = 0.0

        print("-" * 55)
        print(f"  Sharpe (y\u0131ll\u0131k)      : {sharpe:>13.4f}")
        print(f"  Sortino (y\u0131ll\u0131k)     : {sortino:>13.4f}")
        print(f"  Max Drawdown        : {max_dd*100:>12.2f}%")
        print(f"  \u0130\u015flem yap\u0131lan candle : {len(result.trade_history):>13,}")
        print("=" * 55)

    # JSON'a kaydet
    summary = {
        "strategy":              "baseline_momentum",
        "initial_capital":       result.initial_capital,
        "final_portfolio_value": result.final_portfolio_value,
        "net_pnl":               result.net_pnl,
        "return_pct":            result.return_pct,
        "total_candles":         result.total_candles,
        "total_trades":          result.total_trades,
        "total_liquidations":    result.total_liquidations,
        "validation_errors":     result.validation_errors,
        "strategy_errors":       result.strategy_errors,
        "failed_opens":          result.failed_opens,
    }
    if len(df) > 1:
        summary.update({
            "sharpe":       float(sharpe),
            "sortino":      float(sortino),
            "max_drawdown": float(max_dd),
        })

    (RESULTS_DIR / "baseline_momentum.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nJSON \u00f6zet: results/baseline_momentum.json")

    # Grafik
    if len(df) > 1:
        fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        ax[0].plot(df["candle_index"], df["portfolio_value"], color="#1f77b4", lw=1.2)
        ax[0].axhline(result.initial_capital, color="gray", ls="--", lw=0.8, label="Initial")
        ax[0].set_ylabel("Portf\u00f6y De\u011feri ($)")
        ax[0].set_title(f"Baseline Momentum — Return: {result.return_pct:+.2f}%")
        ax[0].legend(loc="upper left")
        ax[0].grid(alpha=0.3)

        ax[1].fill_between(df["candle_index"], drawdown * 100, 0, color="red", alpha=0.4)
        ax[1].set_ylabel("Drawdown (%)")
        ax[1].set_xlabel("Candle Index")
        ax[1].grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(RESULTS_DIR / "baseline_momentum.png", dpi=120)
        print(f"Grafik:   results/baseline_momentum.png")


if __name__ == "__main__":
    main()
