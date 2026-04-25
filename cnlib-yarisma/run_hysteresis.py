"""Momentum Hysteresis backtest runner."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from cnlib import backtest
from strategies.momentum_hysteresis import MomentumHysteresis


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
    print("  MOMENTUM HYSTERESIS \u2014 SMA20 double-band")
    print("=" * 55)

    strategy = MomentumHysteresis()
    result = backtest.run(strategy=strategy, initial_capital=3000.0, silent=False)

    print()
    result.print_summary()

    df = result.portfolio_dataframe()
    mx = _metrics(df)

    print("-" * 55)
    print(f"  Sharpe (y\u0131ll\u0131k)      : {mx['sharpe']:>13.4f}")
    print(f"  Sortino (y\u0131ll\u0131k)     : {mx['sortino']:>13.4f}")
    print(f"  Max Drawdown        : {mx['max_drawdown']*100:>12.2f}%")
    print(f"  \u0130\u015flem yap\u0131lan candle : {len(result.trade_history):>13,}")
    print("=" * 55)

    summary = {
        "strategy":              "momentum_hysteresis",
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
    }
    (RESULTS_DIR / "momentum_hysteresis.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nJSON \u00f6zet: results/momentum_hysteresis.json")


if __name__ == "__main__":
    main()
