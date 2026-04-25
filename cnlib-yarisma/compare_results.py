"""
Strateji sonuçlarını yan yana koyup özet tablo basar.
"""
from __future__ import annotations

import json
from pathlib import Path


RESULTS_DIR = Path(__file__).parent / "results"

FIELDS = [
    ("Final Portfolio",    "final_portfolio_value",   "${:>12,.2f}"),
    ("Return %",           "return_pct",              "{:>+12.2f}%"),
    ("Sharpe (y)",         "sharpe",                  "{:>13.4f}"),
    ("Sortino (y)",        "sortino",                 "{:>13.4f}"),
    ("Max Drawdown",       "max_drawdown",            "{:>+12.2%}"),
    ("Total Trades",       "total_trades",            "{:>13,}"),
    ("Liquidations",       "total_liquidations",      "{:>13,}"),
    ("Validation Errors",  "validation_errors",       "{:>13,}"),
    ("Strategy Errors",    "strategy_errors",         "{:>13,}"),
    ("Failed Opens",       "failed_opens",            "{:>13,}"),
]

STRATEGIES = [
    ("Baseline Momentum",   "baseline_momentum.json"),
    ("Technical Ensemble",  "technical_ensemble.json"),
    ("Enhanced Momentum",   "enhanced_momentum.json"),
    ("Hysteresis",          "momentum_hysteresis.json"),
    ("ML LightGBM",         "ml_lightgbm.json"),
]


def main() -> None:
    results = []
    for name, file in STRATEGIES:
        path = RESULTS_DIR / file
        if not path.exists():
            print(f"Eksik: {file}")
            continue
        results.append((name, json.loads(path.read_text(encoding="utf-8"))))

    if not results:
        print("Sonu\u00e7 bulunamad\u0131.")
        return

    # Header
    name_w = 20
    col_w = 18
    print("=" * (name_w + col_w * len(results) + 2))
    print(f"{'Metrik':<{name_w}}" + "".join(f"{n:>{col_w}}" for n, _ in results))
    print("=" * (name_w + col_w * len(results) + 2))

    for label, key, fmt in FIELDS:
        row = f"{label:<{name_w}}"
        for _, r in results:
            if key in r:
                row += f"{fmt.format(r[key]):>{col_w}}"
            else:
                row += f"{'N/A':>{col_w}}"
        print(row)
    print("=" * (name_w + col_w * len(results) + 2))


if __name__ == "__main__":
    main()
