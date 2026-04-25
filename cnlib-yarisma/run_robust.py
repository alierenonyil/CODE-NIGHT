"""
Robust Hybrid — full stres test ($1000, real + 4 sentetik).

Ek olarak:
  - Composite skor: her senaryoda 1-5 ranking → ortalama sıra
  - Worst-case guard: en kötü senaryodaki multiplier'a dikkat
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cnlib import backtest
from strategies.robust_hybrid import RobustHybrid
from strategies.leveraged_baseline import LeveragedBaseline5x, LeveragedBaseline3x
from strategies.hybrid_leveraged import HybridLeveraged


SYNTH_DIR = Path(__file__).parent / "synthetic_data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SCENARIOS = [
    ("real",    None,                   0),
    ("normal",  SYNTH_DIR / "normal",   1570),
    ("crash",   SYNTH_DIR / "crash",    1570),
    ("pump",    SYNTH_DIR / "pump",     1570),
    ("mixed",   SYNTH_DIR / "mixed",    1570),
]

STRATEGIES = [
    ("Baseline 3x",  LeveragedBaseline3x),
    ("Baseline 5x",  LeveragedBaseline5x),
    ("Hybrid",       HybridLeveraged),
    ("Robust",       RobustHybrid),
]


def _metrics(pv: pd.Series) -> tuple[float, float]:
    daily = pv.pct_change().dropna()
    sharpe = (daily.mean() / daily.std()) * (365 ** 0.5) if daily.std() > 0 else 0.0
    cummax = pv.cummax()
    dd = ((pv - cummax) / cummax).min()
    return float(sharpe), float(dd)


def main() -> None:
    matrix: dict[str, dict[str, dict]] = {}

    print("=" * 120)
    print("  OUT-OF-SAMPLE ROBUSTNESS TEST \u2014 $1000 ba\u015flang\u0131\u00e7")
    print("=" * 120)

    for strat_name, cls in STRATEGIES:
        matrix[strat_name] = {}
        for scn_name, ddir, start in SCENARIOS:
            strat = cls()
            r = backtest.run(
                strategy=strat,
                initial_capital=1000.0,
                data_dir=ddir,
                start_candle=start,
                silent=True,
            )
            df = pd.DataFrame(r.portfolio_series)
            sharpe, dd = _metrics(df["portfolio_value"]) if len(df) > 1 else (0.0, 0.0)
            matrix[strat_name][scn_name] = {
                "final":       float(r.final_portfolio_value),
                "multiplier":  float(r.final_portfolio_value / 1000.0),
                "return_pct":  float(r.return_pct),
                "sharpe":      sharpe,
                "max_dd":      dd,
                "liquidations": int(r.total_liquidations),
            }

    # ------------------- MULTIPLIER TABLE -------------------
    print(f"\n{'Strateji':<15}" + "".join(f"{s[0]:>18}" for s in SCENARIOS))
    print("-" * 105)
    for strat, _ in STRATEGIES:
        line = f"{strat:<15}"
        for scn, _, _ in SCENARIOS:
            m = matrix[strat][scn]
            line += f"{m['multiplier']:>15,.1f}x"
            if m["liquidations"] > 0:
                line += f"(L{m['liquidations']})"
            else:
                line += "    "
        print(line)

    # ------------------- WORST CASE -------------------
    print("\n" + "=" * 120)
    print("  WORST-CASE ANAL\u0130Z (en k\u00f6t\u00fc senaryodaki multiplier)")
    print("=" * 120)
    for strat, _ in STRATEGIES:
        by_mult = [matrix[strat][s][0] if isinstance(matrix[strat][s], tuple) else
                   (matrix[strat][s]["multiplier"], s) for s, _, _ in SCENARIOS]
        by_mult = [(matrix[strat][s]["multiplier"], s) for s, _, _ in SCENARIOS]
        by_mult.sort()
        worst_mult, worst_scn = by_mult[0]
        best_mult, best_scn = by_mult[-1]
        avg = sum(m for m, _ in by_mult) / len(by_mult)
        geo_mean = 1.0
        for m, _ in by_mult:
            geo_mean *= max(m, 0.01)  # log hatasından kaçın
        geo_mean = geo_mean ** (1.0 / len(by_mult))
        print(f"  {strat:<13} worst: {worst_mult:>10,.2f}x ({worst_scn})   "
              f"best: {best_mult:>12,.1f}x ({best_scn})   "
              f"arith.avg: {avg:>12,.1f}x   "
              f"geo.mean: {geo_mean:>10,.2f}x")

    # ------------------- COMPOSITE SCORE -------------------
    # Geometric mean en iyi robust metric (çarpımsal senaryolar için)
    print("\n" + "=" * 120)
    print("  COMPOSITE SKOR (geometric mean of multipliers \u2014 robust comparison)")
    print("=" * 120)
    scored = []
    for strat, _ in STRATEGIES:
        mults = [matrix[strat][s]["multiplier"] for s, _, _ in SCENARIOS]
        geo = 1.0
        for m in mults:
            geo *= max(m, 0.01)
        geo = geo ** (1.0 / len(mults))
        scored.append((strat, geo))
    scored.sort(key=lambda x: -x[1])
    for rank, (strat, geo) in enumerate(scored, 1):
        print(f"  {rank}. {strat:<13} geo_mean = {geo:>12,.2f}x")

    # Kaydet
    (RESULTS_DIR / "robustness_matrix.json").write_text(
        json.dumps(matrix, indent=2), encoding="utf-8"
    )
    print(f"\nJSON: results/robustness_matrix.json")


if __name__ == "__main__":
    main()
