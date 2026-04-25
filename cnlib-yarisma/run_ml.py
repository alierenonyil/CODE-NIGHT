"""LightGBM walk-forward ML backtest runner."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from cnlib import backtest
from strategies.ml_lightgbm import MLLightGBM


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _metrics(df: pd.DataFrame) -> dict:
    pv = df["portfolio_value"]
    daily = pv.pct_change().dropna()
    sharpe = (daily.mean() / daily.std()) * (365 ** 0.5) if daily.std() > 0 else 0.0
    downside = daily[daily < 0]
    sortino = ((daily.mean() / downside.std()) * (365 ** 0.5)
               if len(downside) > 0 and downside.std() > 0 else 0.0)
    cummax = pv.cummax()
    drawdown = (pv - cummax) / cummax
    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(drawdown.min()),
        "drawdown": drawdown,
    }


def main() -> None:
    print("=" * 55)
    print("  ML LIGHTGBM \u2014 walk-forward, long-only, leverage 1")
    print("=" * 55)

    t0 = time.time()
    strategy = MLLightGBM()
    result = backtest.run(strategy=strategy, initial_capital=3000.0, silent=False)
    elapsed = time.time() - t0

    print()
    result.print_summary()

    df = result.portfolio_dataframe()
    mx = _metrics(df)

    print("-" * 55)
    print(f"  Sharpe (y\u0131ll\u0131k)      : {mx['sharpe']:>13.4f}")
    print(f"  Sortino (y\u0131ll\u0131k)     : {mx['sortino']:>13.4f}")
    print(f"  Max Drawdown        : {mx['max_drawdown']*100:>12.2f}%")
    print(f"  \u0130\u015flem yap\u0131lan candle : {len(result.trade_history):>13,}")
    print(f"  Backtest s\u00fcresi     : {elapsed:>12.1f}s")
    print("=" * 55)

    summary = {
        "strategy":              "ml_lightgbm",
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
        "sharpe":                mx["sharpe"],
        "sortino":               mx["sortino"],
        "max_drawdown":          mx["max_drawdown"],
        "backtest_seconds":      elapsed,
    }
    (RESULTS_DIR / "ml_lightgbm.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nJSON \u00f6zet: results/ml_lightgbm.json")

    # Grafik
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(df["candle_index"], df["portfolio_value"],
               color="#9467bd", lw=1.2, label=f"ML ({result.return_pct:+.1f}%)")
    ax[0].axhline(result.initial_capital, color="gray", ls="--", lw=0.8)

    baseline_file = RESULTS_DIR / "baseline_momentum.json"
    if baseline_file.exists():
        bj = json.loads(baseline_file.read_text(encoding="utf-8"))
        ax[0].axhline(bj["final_portfolio_value"], color="#1f77b4", ls=":",
                      lw=0.8, label=f"Baseline ({bj['return_pct']:+.1f}%)")

    ax[0].set_ylabel("Portf\u00f6y De\u011feri ($)")
    ax[0].set_title("ML LightGBM vs Baseline")
    ax[0].legend(loc="upper left")
    ax[0].grid(alpha=0.3)

    ax[1].fill_between(df["candle_index"], mx["drawdown"] * 100, 0, color="red", alpha=0.4)
    ax[1].set_ylabel("Drawdown (%)")
    ax[1].set_xlabel("Candle Index")
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "ml_lightgbm.png", dpi=120)
    print(f"Grafik:   results/ml_lightgbm.png")


if __name__ == "__main__":
    main()
