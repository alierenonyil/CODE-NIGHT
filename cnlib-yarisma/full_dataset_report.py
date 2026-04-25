"""
Tüm dataset klasörlerini tara, strategy.py'yi her birinde test et.

Klasörler:
  - sentetik/ (Ali Eren'in 50 sentetik dataset)
  - synthetic_data/ (Claude'un 4 senaryo sentetik)
  - multi_data/ (yfinance 10 farklı dönem)
  - real_crypto_data/ (yfinance BTC/ETH/SOL son 2 yıl)
  - binance_data/majors, altcoins, futures_1h (Binance gerçek)
  - splits/ (train/val/holdout splits)
  - cnlib paket data (yarışma train data)

$3000 başlangıç ile her dataset'te final portfolio + return %.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from cnlib import backtest
from strategy1 import Strategy   # v2.1 with 3 fixes


ROOT = Path(__file__).parent


def has_three_coins(d: Path) -> bool:
    coins = ["kapcoin-usd_train", "metucoin-usd_train", "tamcoin-usd_train"]
    return all((d / f"{c}.parquet").exists() for c in coins)


def find_all_datasets() -> list[tuple[str, Path, int]]:
    """(label, path, start_candle) listesi."""
    datasets = []

    # 1. cnlib package data (yarışma train data)
    cnlib_data = ROOT / ".venv/Lib/site-packages/cnlib/data"
    if has_three_coins(cnlib_data):
        datasets.append(("00_yarisma_train", cnlib_data, 0))

    # 2. Splits (train/val/holdout)
    splits = ROOT / "splits"
    if splits.exists():
        for sub in sorted(splits.iterdir()):
            if sub.is_dir() and has_three_coins(sub):
                datasets.append((f"split_{sub.name}", sub, 0))
        # Holdout-only
        full = splits / "full_100"
        if has_three_coins(full):
            datasets.append(("split_holdout_only", full, 1256))
            datasets.append(("split_val_only", splits / "trainval_80", 942))

    # 3. sentetik/ (50 senaryo)
    sent = ROOT / "sentetik"
    if sent.exists():
        for d in sorted(sent.iterdir()):
            if d.is_dir() and has_three_coins(d):
                datasets.append((f"sent_{d.name}", d, 0))

    # 4. synthetic_data/ (4 senaryo)
    syn = ROOT / "synthetic_data"
    if syn.exists():
        for d in sorted(syn.iterdir()):
            if d.is_dir() and has_three_coins(d):
                datasets.append((f"syn_{d.name}", d, 1570))  # son 365 candle

    # 5. multi_data/ (yfinance multi-period)
    multi = ROOT / "multi_data"
    if multi.exists():
        for d in sorted(multi.iterdir()):
            if d.is_dir() and has_three_coins(d):
                datasets.append((f"multi_{d.name}", d, 0))

    # 6. real_crypto_data/
    real = ROOT / "real_crypto_data"
    if has_three_coins(real):
        datasets.append(("real_btc_eth_sol_2y", real, 0))

    # 7. binance_data/
    bn = ROOT / "binance_data"
    if bn.exists():
        for d in sorted(bn.iterdir()):
            if d.is_dir() and has_three_coins(d):
                datasets.append((f"binance_{d.name}", d, 0))

    return datasets


def test_one(data_dir: Path, start: int) -> dict:
    strat = Strategy()
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
        "ret_pct": float(r.return_pct),
        "trades": int(r.total_trades),
        "liq": int(r.total_liquidations),
        "val_err": int(r.validation_errors),
        "max_dd": max_dd,
    }


def main() -> None:
    datasets = find_all_datasets()
    print(f"Toplam dataset: {len(datasets)}\n")

    print(f"{'#':<4}{'Dataset':<42}{'Final $':>22}{'Mult':>14}{'Return %':>16}{'Trades':>8}{'Liq':>5}{'DD':>9}")
    print("-" * 130)

    results = []
    pos_count = 0
    p60_count = 0
    neg_count = 0
    wipe_count = 0
    liq_total = 0

    for i, (label, ddir, start) in enumerate(datasets, 1):
        try:
            r = test_one(ddir, start)
            results.append((label, r))
            pos = r["mult"] >= 1.0
            if pos: pos_count += 1
            if r["mult"] >= 1.6: p60_count += 1
            if not pos: neg_count += 1
            if r["mult"] < 0.1: wipe_count += 1
            liq_total += r["liq"]

            mult_s = f"{r['mult']:.2e}" if r['mult'] > 1e6 or r['mult'] < 0.001 else f"{r['mult']:.4f}"
            ret_s = f"{r['ret_pct']:.2e}%" if abs(r['ret_pct']) > 1e6 else f"{r['ret_pct']:+.2f}%"
            final_s = f"${r['final']:.2e}" if r['final'] > 1e9 else f"${r['final']:>14,.2f}"
            print(f"{i:<4}{label[:40]:<42}{final_s:>22}{mult_s:>14}{ret_s:>16}{r['trades']:>8}{r['liq']:>5}{r['max_dd']*100:>+8.2f}%")
        except Exception as e:
            print(f"{i:<4}{label[:40]:<42}  ERROR: {e}")

    n = len(results)
    print()
    print("=" * 130)
    print("  ÖZET")
    print("=" * 130)
    print(f"  Toplam dataset:           {n}")
    print(f"  Pozitif kazanç (≥%0):     {pos_count}/{n}  ({100*pos_count/n:.1f}%)")
    print(f"  %60+ kazanç (≥1.6x):      {p60_count}/{n}  ({100*p60_count/n:.1f}%)")
    print(f"  Negatif (kayıp):          {neg_count}/{n}  ({100*neg_count/n:.1f}%)")
    print(f"  Wipeout (-%90+):          {wipe_count}/{n}  ({100*wipe_count/n:.1f}%)")
    print(f"  Toplam liquidation:       {liq_total}")
    print()

    # Top kazanç
    sorted_by_mult = sorted(results, key=lambda x: -x[1]["mult"])
    print("  EN ÇOK KAZANANLAR:")
    for label, r in sorted_by_mult[:10]:
        print(f"    {label[:40]:<42}  $3000 → ${r['final']:.2e}  ({r['mult']:.2e}x)")

    print()
    print("  EN ÇOK KAYBEDENLER:")
    for label, r in sorted_by_mult[-10:]:
        print(f"    {label[:40]:<42}  $3000 → ${r['final']:>13,.2f}  ({r['mult']:.4f}x)")


if __name__ == "__main__":
    main()
