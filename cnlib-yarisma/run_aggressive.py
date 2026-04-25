"""Aggressive Leveraged ($1000 initial) backtest runner."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from cnlib import backtest
from strategies.aggressive_leveraged import AggressiveLeveraged


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _metrics(df: pd.DataFrame) -> dict:
    pv = df["portfolio_value"]
    daily = pv.pct_change().dropna()
    sharpe = (daily.mean() / daily.std()) * (365 ** 0.5) if daily.std() > 0 else 0.0
    down = daily[daily < 0]
    sortino = (daily.mean() / down.std()) * (365 ** 0.5) if len(down) and down.std() > 0 else 0.0
    cummax = pv.cummax()
    dd = (pv - cummax) / cummax
    return {
        "sharpe":       float(sharpe),
        "sortino":      float(sortino),
        "max_drawdown": float(dd.min()),
        "drawdown":     dd,
    }


def main() -> None:
    print("=" * 55)
    print("  AGGRESSIVE LEVERAGED \u2014 10x/5x/3x + stop-loss")
    print("  Initial Capital: $1000 (yar\u0131\u015fma ayar\u0131)")
    print("=" * 55)

    strategy = AggressiveLeveraged()
    result = backtest.run(strategy=strategy, initial_capital=1000.0, silent=False)

    print()
    result.print_summary()

    df = result.portfolio_dataframe()
    mx = _metrics(df)

    print("-" * 55)
    print(f"  Sharpe (y\u0131ll\u0131k)      : {mx['sharpe']:>13.4f}")
    print(f"  Sortino (y\u0131ll\u0131k)     : {mx['sortino']:>13.4f}")
    print(f"  Max Drawdown        : {mx['max_drawdown']*100:>12.2f}%")
    print(f"  Final ($)           : ${result.final_portfolio_value:>13,.2f}")
    print(f"  Multiplier          : {result.final_portfolio_value / 1000:>13,.2f}x")
    print("=" * 55)

    summary = {
        "strategy":              "aggressive_leveraged",
        "initial_capital":       result.initial_capital,
        "final_portfolio_value": result.final_portfolio_value,
        "net_pnl":               result.net_pnl,
        "return_pct":            result.return_pct,
        "multiplier":            result.final_portfolio_value / result.initial_capital,
        "total_candles":         result.total_candles,
        "total_trades":          result.total_trades,
        "total_liquidations":    result.total_liquidations,
        "total_liquidation_loss": result.total_liquidation_loss,
        "validation_errors":     result.validation_errors,
        "strategy_errors":       result.strategy_errors,
        "failed_opens":          result.failed_opens,
        "sharpe":                mx["sharpe"],
        "sortino":               mx["sortino"],
        "max_drawdown":          mx["max_drawdown"],
    }
    (RESULTS_DIR / "aggressive_leveraged.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nJSON \u00f6zet: results/aggressive_leveraged.json")

    # Grafik (log y-scale — katrilyon hedefi için)
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(df["candle_index"], df["portfolio_value"],
               color="#e74c3c", lw=1.5,
               label=f"Aggressive 10x ({result.return_pct:+.1f}%)")
    ax[0].axhline(result.initial_capital, color="gray", ls="--", lw=0.8)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Portf\u00f6y De\u011feri ($, log scale)")
    ax[0].set_title(f"Aggressive Leveraged — $1000 \u2192 ${result.final_portfolio_value:,.0f} "
                    f"({result.final_portfolio_value/1000:.1f}x)")
    ax[0].legend(loc="upper left")
    ax[0].grid(alpha=0.3, which="both")

    ax[1].fill_between(df["candle_index"], mx["drawdown"] * 100, 0, color="red", alpha=0.4)
    ax[1].set_ylabel("Drawdown (%)")
    ax[1].set_xlabel("Candle Index")
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "aggressive_leveraged.png", dpi=120)
    print(f"Grafik:   results/aggressive_leveraged.png")


if __name__ == "__main__":
    main()
