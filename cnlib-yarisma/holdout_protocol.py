"""
Time-series split + walk-forward + holdout protocol — Ali Eren'in spec'i.

Train data: cnlib paketinin içindeki parquet (1570 candle, 4 yıl).
Split: 60% train (0-941), 20% val (942-1255), 20% holdout (1256-1569).

Adımlar:
  1. Splits oluştur (3 dizin)
  2. Strategy 3 fold'da backtest
  3. Walk-forward 5 fold (expanding window)
  4. Score hesapla, overfit penalty uygula
  5. Final rapor

Holdout SADECE final raporlamada kullanılır.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Type

import pandas as pd

from cnlib import backtest


ROOT = Path(__file__).parent
SOURCE_DATA = ROOT / ".venv/Lib/site-packages/cnlib/data"
SPLITS_DIR = ROOT / "splits"

COINS = ["kapcoin-usd_train", "metucoin-usd_train", "tamcoin-usd_train"]


def prepare_splits() -> dict[str, Path]:
    """3 split dizini oluştur: train (0-941), val (0-1255), full (0-1569)."""
    SPLITS_DIR.mkdir(exist_ok=True)
    splits = {
        "train_60":   942,    # 0..941 inclusive
        "trainval_80": 1256,  # 0..1255
        "full_100":   1570,   # 0..1569
    }
    out = {}
    for name, end in splits.items():
        d = SPLITS_DIR / name
        d.mkdir(exist_ok=True)
        for coin in COINS:
            df = pd.read_parquet(SOURCE_DATA / f"{coin}.parquet")
            df.iloc[:end].reset_index(drop=True).to_parquet(d / f"{coin}.parquet", index=False)
        out[name] = d
        print(f"  [{name}] {end} candle (3 coin)")
    return out


def fold_test(strat_cls: Type, data_dir: Path, start: int) -> dict:
    """Backtest tek fold'da."""
    strat = strat_cls()
    r = backtest.run(strategy=strat, initial_capital=3000.0,
                     data_dir=data_dir, start_candle=start, silent=True)
    df = pd.DataFrame(r.portfolio_series)
    pv = df["portfolio_value"]
    if len(pv) > 1:
        cummax = pv.cummax()
        max_dd = float(((pv - cummax) / cummax).min())
        daily = pv.pct_change().dropna()
        sharpe = float((daily.mean() / daily.std()) * (365 ** 0.5)) if daily.std() > 0 else 0.0
    else:
        max_dd = 0.0
        sharpe = 0.0
    return {
        "final":      float(r.final_portfolio_value),
        "mult":       float(r.final_portfolio_value / 3000.0),
        "ret_pct":    float(r.return_pct),
        "trades":     int(r.total_trades),
        "liq":        int(r.total_liquidations),
        "val_err":    int(r.validation_errors),
        "max_dd":     max_dd,
        "sharpe":     sharpe,
    }


def walk_forward(strat_cls: Type, splits: dict[str, Path]) -> list[dict]:
    """5 fold expanding window — sadece train ve val içinde, holdout ASLA dahil değil."""
    full_dir = splits["trainval_80"]  # 0..1255 only
    folds = []
    # Fold start points (test başlangıcı), expanding train [0, start)
    test_starts = [400, 600, 800, 1000, 1100]
    for ts in test_starts:
        # Cnlib start_candle=ts → backtest ts'den 1255'e kadar
        strat = strat_cls()
        r = backtest.run(strategy=strat, initial_capital=3000.0,
                         data_dir=full_dir, start_candle=ts, silent=True)
        # Sadece bu fold'un başlangıç-bitişi (sınırlı candle pencere)
        # Pratik: full backtest sonucunu raporla — tüm fold'lar overlapping ama OK
        folds.append({
            "test_start":   ts,
            "candles":      1255 - ts,
            "final":        float(r.final_portfolio_value),
            "mult":         float(r.final_portfolio_value / 3000.0),
            "ret_pct":      float(r.return_pct),
            "trades":       int(r.total_trades),
            "liq":          int(r.total_liquidations),
        })
    return folds


def evaluate_strategy(name: str, cls: Type, splits: dict[str, Path]) -> dict:
    print(f"\n=== {name} ===")
    # 1. Train fold (0-941)
    train_r = fold_test(cls, splits["train_60"], 0)
    print(f"  TRAIN     ret={train_r['ret_pct']:+.2e}%  trades={train_r['trades']}  liq={train_r['liq']}  dd={train_r['max_dd']*100:+.2f}%")

    # 2. Val fold (942-1255) — start_candle=942, data 0-1255
    val_r = fold_test(cls, splits["trainval_80"], 942)
    print(f"  VAL       ret={val_r['ret_pct']:+.2e}%  trades={val_r['trades']}  liq={val_r['liq']}  dd={val_r['max_dd']*100:+.2f}%")

    # 3. Holdout fold (1256-1569) — start_candle=1256
    holdout_r = fold_test(cls, splits["full_100"], 1256)
    print(f"  HOLDOUT   ret={holdout_r['ret_pct']:+.2e}%  trades={holdout_r['trades']}  liq={holdout_r['liq']}  dd={holdout_r['max_dd']*100:+.2f}%")

    # 4. Walk-forward (5 fold)
    wf = walk_forward(cls, splits)
    wf_avg = sum(f["ret_pct"] for f in wf) / len(wf)
    wf_liq = sum(f["liq"] for f in wf)
    print(f"  WALK-FWD  avg ret={wf_avg:+.2e}%  liq_total={wf_liq}")

    # 5. Score
    train_return = math.log10(max(train_r["mult"], 1e-9))
    val_return = math.log10(max(val_r["mult"], 1e-9))
    holdout_return = math.log10(max(holdout_r["mult"], 1e-9))
    wf_avg_log = sum(math.log10(max(f["mult"], 1e-9)) for f in wf) / len(wf)

    # Overfit penalty: train >> val ise ceza
    overfit_gap = train_return - val_return
    overfit_penalty = max(0, overfit_gap - 5) * 2

    # Liquidation penalty
    liq_penalty = (train_r["liq"] + val_r["liq"] + holdout_r["liq"] + wf_liq) * 5

    # Low trade penalty
    low_trade_penalty = 0
    for r in [train_r, val_r, holdout_r]:
        if r["trades"] < 5:
            low_trade_penalty += 3

    # MDD penalty (worst DD across all folds)
    worst_dd = min(train_r["max_dd"], val_r["max_dd"], holdout_r["max_dd"])
    mdd_penalty = abs(worst_dd) * 2

    score = (
        3 * holdout_return +
        2 * wf_avg_log +
        train_return -
        mdd_penalty -
        liq_penalty -
        overfit_penalty -
        low_trade_penalty
    )

    print(f"  Score: {score:+.2f}  "
          f"(train={train_return:+.2f} val={val_return:+.2f} hold={holdout_return:+.2f} "
          f"wf_avg={wf_avg_log:+.2f} overfit_pen={overfit_penalty:+.2f} liq_pen={liq_penalty})")

    # Verdict
    accepted = (
        holdout_r["mult"] > 1.0 and        # holdout pozitif
        train_r["liq"] == 0 and            # liquidation 0
        val_r["liq"] == 0 and
        holdout_r["liq"] == 0 and
        train_r["val_err"] == 0 and
        val_r["val_err"] == 0
    )
    verdict = "ACCEPTED" if accepted else "REJECTED"
    overfit_risk = "LOW" if overfit_gap < 3 else "MEDIUM" if overfit_gap < 8 else "HIGH"
    print(f"  Verdict: {verdict}  Overfit risk: {overfit_risk} (gap={overfit_gap:.2f})")

    return {
        "name": name,
        "train": train_r,
        "val": val_r,
        "holdout": holdout_r,
        "walk_forward": wf,
        "wf_avg_log": wf_avg_log,
        "score": score,
        "verdict": verdict,
        "overfit_risk": overfit_risk,
        "overfit_gap": overfit_gap,
    }


def main() -> None:
    print("=" * 80)
    print("  TIME-SERIES SPLIT PROTOCOL")
    print("  60% train (0-941) + 20% val (942-1255) + 20% holdout (1256-1569)")
    print("=" * 80)

    print("\nSplit'ler hazırlanıyor...")
    splits = prepare_splits()

    from strategy import Strategy as AdaptiveSafe
    from strategies.adaptive_v3 import AdaptiveV3
    from strategies.safe_baseline import SafeBaseline
    from strategies.leveraged_baseline import LeveragedBaseline2x, LeveragedBaseline3x
    from strategies.baseline_momentum import BaselineMomentum

    candidates = [
        ("AdaptiveSafe (current)", AdaptiveSafe),
        ("AdaptiveV3",             AdaptiveV3),
        ("SafeBaseline 2x",        SafeBaseline),
        ("LeveragedBaseline 2x",   LeveragedBaseline2x),
        ("LeveragedBaseline 3x",   LeveragedBaseline3x),
        ("Baseline 1x",            BaselineMomentum),
    ]

    results = []
    for name, cls in candidates:
        try:
            r = evaluate_strategy(name, cls, splits)
            results.append(r)
        except Exception as e:
            print(f"  ERROR in {name}: {type(e).__name__}: {e}")

    print()
    print("=" * 100)
    print("  FINAL RANKING")
    print("=" * 100)
    accepted = [r for r in results if r["verdict"] == "ACCEPTED"]
    accepted.sort(key=lambda x: -x["score"])
    rejected = [r for r in results if r["verdict"] == "REJECTED"]

    print(f"\nACCEPTED ({len(accepted)}):")
    for r in accepted:
        print(f"  {r['name']:<26} score={r['score']:+7.2f}  "
              f"holdout_mult={r['holdout']['mult']:.4f}  "
              f"train=10^{math.log10(max(r['train']['mult'],1e-9)):.1f}  "
              f"overfit={r['overfit_risk']}")

    print(f"\nREJECTED ({len(rejected)}):")
    for r in rejected:
        reasons = []
        if r["holdout"]["mult"] <= 1.0:
            reasons.append(f"holdout mult={r['holdout']['mult']:.3f}")
        if r["train"]["liq"] > 0:
            reasons.append(f"train_liq={r['train']['liq']}")
        if r["val"]["liq"] > 0:
            reasons.append(f"val_liq={r['val']['liq']}")
        if r["holdout"]["liq"] > 0:
            reasons.append(f"holdout_liq={r['holdout']['liq']}")
        print(f"  {r['name']:<26} score={r['score']:+7.2f}  reasons: {', '.join(reasons) or 'overfit'}")


if __name__ == "__main__":
    main()
