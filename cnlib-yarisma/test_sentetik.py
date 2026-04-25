"""
Sentetik klasörünündeki 20 dataset'te tüm stratejileri test et.
"""
from __future__ import annotations

import math
from pathlib import Path

from cnlib import backtest
from strategies.adaptive_v3 import AdaptiveV3
from strategies.adaptive_v8 import AdaptiveV8
from strategies.robust_hybrid import RobustHybrid
from strategies.leveraged_baseline import LeveragedBaseline3x


SENT = Path(__file__).parent / "sentetik"


def main() -> None:
    folders = sorted([d for d in SENT.iterdir() if d.is_dir()])

    strategies = [
        ("V3",       AdaptiveV3),
        ("V8",       AdaptiveV8),
        ("Robust",   RobustHybrid),
        ("Base3x",   LeveragedBaseline3x),
    ]

    print(f"\n{'Dataset':<28}", end="")
    for name, _ in strategies:
        print(f"{name:>14}", end="")
    print()
    print("-" * 95)

    counters = {n: {"pos": 0, "60+": 0, "neg": 0} for n, _ in strategies}
    all_results = {}

    for d in folders:
        row = f"{d.name:<28}"
        all_results[d.name] = {}
        for sname, cls in strategies:
            try:
                strat = cls()
                r = backtest.run(strategy=strat, initial_capital=3000.0,
                                 data_dir=d, start_candle=0, silent=True)
                m = max(r.final_portfolio_value / 3000.0, 1e-9)
                ret_pct = (m - 1) * 100
                all_results[d.name][sname] = ret_pct
                if m >= 1:
                    counters[sname]["pos"] += 1
                if m >= 1.6:
                    counters[sname]["60+"] += 1
                if m < 1:
                    counters[sname]["neg"] += 1
                if abs(ret_pct) > 1e6:
                    cell = f"{ret_pct:.2e}"
                else:
                    cell = f"{ret_pct:+.1f}%"
                row += f"{cell:>14}"
            except Exception as e:
                row += f"{'ERR':>14}"
        print(row)

    print()
    print(f"{'Strateji':<14}{'pozitif':>10}{'%60+':>10}{'negatif':>10}")
    print("-" * 45)
    for sname, c in counters.items():
        print(f"{sname:<14}{c['pos']:>3}/{len(folders):<6}{c['60+']:>3}/{len(folders):<6}{c['neg']:>3}/{len(folders)}")


if __name__ == "__main__":
    main()
