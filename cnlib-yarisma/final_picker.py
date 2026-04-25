"""
Final strategy picker — tüm aday stratejiler full protocol'da yarışsın.

Score formülü (Ali Eren):
  3*holdout + 2*walk_forward + train - 2*MDD - liq_pen - overfit_pen - low_trade_pen

Filter:
  - holdout_mult > 1.0 zorunlu
  - 0 liquidation (train+val+holdout)
  - 0 validation errors
  - overfit_gap < 8 (train-val log10 diff)

Profit max: ACCEPTED'lerden en yüksek holdout return.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Type

import pandas as pd

from cnlib import backtest


ROOT = Path(__file__).parent
SPLITS_DIR = ROOT / "splits"


def fold_test(strat_cls: Type, data_dir: Path, start: int) -> dict:
    strat = strat_cls()
    r = backtest.run(strategy=strat, initial_capital=3000.0,
                     data_dir=data_dir, start_candle=start, silent=True)
    df = pd.DataFrame(r.portfolio_series)
    pv = df["portfolio_value"]
    if len(pv) > 1:
        cummax = pv.cummax()
        max_dd = float(((pv - cummax) / cummax).min())
    else:
        max_dd = 0.0
    return {
        "final": float(r.final_portfolio_value),
        "mult": float(r.final_portfolio_value / 3000.0),
        "trades": int(r.total_trades),
        "liq": int(r.total_liquidations),
        "val_err": int(r.validation_errors),
        "max_dd": max_dd,
    }


def evaluate(name: str, cls: Type, splits: dict[str, Path]) -> dict:
    train = fold_test(cls, splits["train_60"], 0)
    val = fold_test(cls, splits["trainval_80"], 942)
    holdout = fold_test(cls, splits["full_100"], 1256)

    # Walk-forward 5 fold expanding window içeride
    full_dir = splits["trainval_80"]
    wf_logs = []
    for ts in [400, 600, 800, 1000, 1100]:
        f = fold_test(cls, full_dir, ts)
        wf_logs.append(math.log10(max(f["mult"], 1e-9)))
    wf_avg = sum(wf_logs) / len(wf_logs)

    train_log = math.log10(max(train["mult"], 1e-9))
    val_log = math.log10(max(val["mult"], 1e-9))
    holdout_log = math.log10(max(holdout["mult"], 1e-9))

    overfit_gap = train_log - val_log
    overfit_pen = max(0, overfit_gap - 5) * 2

    liq_pen = (train["liq"] + val["liq"] + holdout["liq"]) * 5
    low_trade_pen = sum(3 for r in [train, val, holdout] if r["trades"] < 5)

    worst_dd = min(train["max_dd"], val["max_dd"], holdout["max_dd"])
    mdd_pen = abs(worst_dd) * 2

    score = (
        3 * holdout_log + 2 * wf_avg + train_log
        - mdd_pen - liq_pen - overfit_pen - low_trade_pen
    )

    accepted = (
        holdout["mult"] > 1.0 and
        train["liq"] == 0 and val["liq"] == 0 and holdout["liq"] == 0 and
        train["val_err"] == 0 and val["val_err"] == 0 and holdout["val_err"] == 0 and
        overfit_gap < 8
    )

    return {
        "name": name,
        "train": train,
        "val": val,
        "holdout": holdout,
        "wf_avg_log": wf_avg,
        "train_log": train_log,
        "val_log": val_log,
        "holdout_log": holdout_log,
        "overfit_gap": overfit_gap,
        "score": score,
        "accepted": accepted,
    }


def main() -> None:
    splits = {
        "train_60":    SPLITS_DIR / "train_60",
        "trainval_80": SPLITS_DIR / "trainval_80",
        "full_100":    SPLITS_DIR / "full_100",
    }
    if not splits["train_60"].exists():
        print("Splits eksik — holdout_protocol.py once calistir.")
        return

    from strategies.baseline_momentum import BaselineMomentum
    from strategies.leveraged_baseline import (
        LeveragedBaseline2x, LeveragedBaseline3x, LeveragedBaseline5x,
    )
    from strategies.safe_baseline import SafeBaseline, SafeBaseline3x, SafeBaseline1x

    # Cesitli SafeBaseline + ek varyantlar
    class SafeBaseline5x(SafeBaseline):
        LEVERAGE = 5

    class LeveragedBaseline5xSafer(LeveragedBaseline5x):
        TOTAL_ALLOC_CAP = 0.85   # daha düşük cap, liquidation riski azalt

    class SafeBaseline2xWideMA(SafeBaseline):
        LEVERAGE = 2
        SMA_LONG = 100

    candidates = [
        ("Baseline 1x",          BaselineMomentum),
        ("Leveraged 2x",         LeveragedBaseline2x),
        ("Leveraged 3x",         LeveragedBaseline3x),
        ("Leveraged 5x",         LeveragedBaseline5x),
        ("Leveraged 5x SafeAlloc", LeveragedBaseline5xSafer),
        ("Safe 1x",              SafeBaseline1x),
        ("Safe 2x",              SafeBaseline),
        ("Safe 3x",              SafeBaseline3x),
        ("Safe 5x",              SafeBaseline5x),
        ("Safe 2x WideMA(100)",  SafeBaseline2xWideMA),
    ]

    print("=" * 100)
    print("  FINAL PICKER — score formula + accept filter")
    print("=" * 100)

    results = []
    for name, cls in candidates:
        try:
            r = evaluate(name, cls, splits)
            results.append(r)
            t = r["train"]
            v = r["val"]
            h = r["holdout"]
            verdict = "OK" if r["accepted"] else "REJ"
            print(f"  [{verdict}] {name:<24} score={r['score']:+6.2f}  "
                  f"holdout={h['mult']:>8.4f}x  train={t['mult']:.2e}x  "
                  f"liq={t['liq']}/{v['liq']}/{h['liq']}  gap={r['overfit_gap']:.2f}")
        except Exception as e:
            print(f"  ERROR {name}: {e}")

    accepted = [r for r in results if r["accepted"]]
    accepted.sort(key=lambda x: -x["holdout"]["mult"])  # PROFIT MAX

    print("\n" + "=" * 100)
    print("  ACCEPTED — sorted by HOLDOUT mult (profit max)")
    print("=" * 100)
    for rank, r in enumerate(accepted, 1):
        h = r["holdout"]
        v = r["val"]
        t = r["train"]
        print(f"  {rank}. {r['name']:<24}  "
              f"holdout={h['mult']:>9.4f}x ($3000→${h['final']:>12,.0f})  "
              f"val={v['mult']:.2f}x  train={t['mult']:.2e}x  "
              f"score={r['score']:+.2f}  gap={r['overfit_gap']:.2f}")

    if accepted:
        winner = accepted[0]
        print("\n" + "=" * 100)
        print(f"  ★ WINNER (max holdout): {winner['name']}")
        print("=" * 100)
        h = winner["holdout"]
        v = winner["val"]
        t = winner["train"]
        wf = winner["wf_avg_log"]
        print(f"  TRAIN return:    {(t['mult']-1)*100:+,.2f}%  (${3000 * t['mult']:,.0f})")
        print(f"  VAL return:      {(v['mult']-1)*100:+,.2f}%  (${3000 * v['mult']:,.0f})")
        print(f"  HOLDOUT return:  {(h['mult']-1)*100:+,.2f}%  (${3000 * h['mult']:,.0f})")
        print(f"  Walk-forward avg log: {wf:+.2f}")
        print(f"  Total trades (train/val/holdout): {t['trades']}/{v['trades']}/{h['trades']}")
        print(f"  Max DD (train/val/holdout): {t['max_dd']*100:.2f}% / {v['max_dd']*100:.2f}% / {h['max_dd']*100:.2f}%")
        print(f"  Liquidations: 0/0/0")
        print(f"  Overfit risk: {'LOW' if winner['overfit_gap'] < 3 else 'MEDIUM'} (gap={winner['overfit_gap']:.2f})")


if __name__ == "__main__":
    main()
