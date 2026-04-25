"""
Robust modu parametre grid search.

Amaç: V3'ün tek kayıp senaryosu yf_btc_eth_sol_2018_2020 (0.771x)
      + defi'deki düşük performansı iyileştirmek.

Robust parametreleri:
  ROBUST_SMA_LONG: 30, 50, 100, 150
  ROBUST_MAX_LEV:  1, 2, 3
  ROBUST_VOL_EXTREME: 0.08, 0.10, 0.12, 0.15
  YENİ: ROBUST_SMA_LONGER (ek uzun gate): 100, 200 veya None

Swap/Hybrid/Safe modları aynı kalacak — Swap train'de zaten 10^42.

Key dataset'ler robust davranışa bağımlı:
  - yf_btc_eth_sol_2018_2020 (crypto winter)
  - yf_bear_2022_2024
  - yf_defi (altcoin bear)
  - bn_altcoins_2y
  - yf_meme (volatile + negative)
  - real_yf_2y (yakın zaman bear-ish)

Train autocorr yüksek → Swap modunda zaten; robust değişikliği etkilemez.
"""
from __future__ import annotations

import math
from pathlib import Path

from cnlib import backtest
from strategies.adaptive_v3 import AdaptiveV3


ROOT = Path(__file__).parent
MULTI = ROOT / "multi_data"
SYNTH = ROOT / "synthetic_data"

ROBUST_KEY_DATASETS = [
    ("yf_2018_2020",    MULTI / "yf_btc_eth_sol_2018_2020",   0),
    ("yf_bear_2022",    MULTI / "yf_bear_2022_2024",          0),
    ("yf_defi",         MULTI / "yf_defi",                     0),
    ("bn_altcoins",     ROOT / "binance_data/altcoins",        0),
    ("yf_meme",         MULTI / "yf_meme",                     0),
    ("yf_recent",       MULTI / "yf_recent_2024_2026",         0),
    ("real_yf_2y",      ROOT / "real_crypto_data",             0),
    ("yf_classic",      MULTI / "yf_classic_assets",           0),
    ("yf_majors_alt",   MULTI / "yf_majors_alt",               0),
    # Train sanity: bozmayalım diye kontrol
    ("train",           None,                                  0),
]


def make_class(sma_long, max_lev, vol_ext, sma_short=20):
    return type(
        "AV3R",
        (AdaptiveV3,),
        {
            "ROBUST_SMA_LONG": sma_long,
            "ROBUST_MAX_LEV": max_lev,
            "ROBUST_VOL_EXTREME": vol_ext,
            "ROBUST_SMA_SHORT": sma_short,
        },
    )


def run_one(cls) -> tuple[float, float, list[float]]:
    mults = []
    for name, ddir, start in ROBUST_KEY_DATASETS:
        try:
            strat = cls()
            r = backtest.run(strategy=strat, initial_capital=3000.0,
                             data_dir=ddir, start_candle=start, silent=True)
            mults.append(max(r.final_portfolio_value / 3000.0, 1e-9))
        except Exception:
            mults.append(1e-9)
    log_sum = sum(math.log10(m) for m in mults)
    worst = min(mults)
    return log_sum, worst, mults


def main() -> None:
    # Grid
    SMAs = [30, 50, 75, 100, 150]
    LEVs = [1, 2, 3]
    VOLs = [0.08, 0.10, 0.12, 0.15]
    SHORTs = [15, 20, 30]

    configs = [(sl, ml, ve, ss) for sl in SMAs for ml in LEVs for ve in VOLs for ss in SHORTs]
    print(f"Configs: {len(configs)}")

    results = []
    for i, (sl, ml, ve, ss) in enumerate(configs):
        if sl <= ss:  # uzun SMA kısadan büyük olmalı
            continue
        cls = make_class(sl, ml, ve, ss)
        log_sum, worst, mults = run_one(cls)
        results.append((log_sum, worst, sl, ml, ve, ss, mults))
        if i % 40 == 0:
            print(f"  {i}/{len(configs)}  log_sum={log_sum:+.2f} worst={worst:.3f}")

    # En iyi: log_sum max + worst >= 1.0 (all positive)
    positive_results = [r for r in results if r[1] >= 1.0]
    positive_results.sort(key=lambda x: -x[0])

    all_results = list(results)
    all_results.sort(key=lambda x: -x[0])

    print("\n" + "=" * 120)
    print("  TOP 10 - All positive (worst >= 1.0)")
    print("=" * 120)
    print(f"{'rank':<5}{'score':>9}{'worst':>9}  sl   ml   ve     ss   details")
    for rank, (ls, w, sl, ml, ve, ss, ms) in enumerate(positive_results[:10], 1):
        dets = " ".join(f"{m:.2f}" for m in ms[:-1])  # train hariç
        print(f"  {rank:<3}{ls:+8.2f}{w:>9.3f}   sl={sl:<3} ml={ml} ve={ve:<5} ss={ss:<3}  {dets}")

    print("\n" + "=" * 120)
    print("  TOP 10 - By score (may have worst < 1.0)")
    print("=" * 120)
    for rank, (ls, w, sl, ml, ve, ss, ms) in enumerate(all_results[:10], 1):
        dets = " ".join(f"{m:.2f}" for m in ms[:-1])
        print(f"  {rank:<3}{ls:+8.2f}{w:>9.3f}   sl={sl:<3} ml={ml} ve={ve:<5} ss={ss:<3}  {dets}")


if __name__ == "__main__":
    main()
