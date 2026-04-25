"""
4 yıllık real datanın derin analizi.

Çıkarılan metrikler (coin başına):
  - Temel istatistikler: mean, std, min, max, skewness, kurtosis
  - Günlük return dağılımı: pozitif/negatif gün sayısı, medyan, extreme'ler
  - Volatilite rejimleri: rolling std 30-günlük üzerinden
  - Pikler ve çöküşler: top-5 günlük yükseliş/düşüş
  - Trend dönemleri: uzun pozitif/negatif momentum serileri
  - Autocorrelation: return lag-1, lag-5
  - Drawdown analizi: max peak-to-trough, toparlanma süreleri

Çıktı: JSON + konsolda tablo
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).parent.parent / ".venv" / "Lib" / "site-packages" / "cnlib" / "data"
COINS = ["kapcoin-usd_train", "metucoin-usd_train", "tamcoin-usd_train"]


def load_coin(coin: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / f"{coin}.parquet")
    df = df.sort_values("Date").reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def analyze_coin(df: pd.DataFrame) -> dict[str, Any]:
    close = df["Close"]
    r = close.pct_change().dropna()
    log_r = np.log(close / close.shift(1)).dropna()

    # Temel istatistikler
    stats = {
        "candle_count":  int(len(df)),
        "date_start":    str(df["Date"].iloc[0].date()),
        "date_end":      str(df["Date"].iloc[-1].date()),
        "price_min":     float(close.min()),
        "price_max":     float(close.max()),
        "price_start":   float(close.iloc[0]),
        "price_end":     float(close.iloc[-1]),
        "total_return":  float(close.iloc[-1] / close.iloc[0] - 1),
    }

    # Günlük return dağılımı
    stats["daily_return_mean"] = float(r.mean())
    stats["daily_return_std"]  = float(r.std())
    stats["daily_return_median"] = float(r.median())
    stats["daily_return_skew"] = float(r.skew())
    stats["daily_return_kurt"] = float(r.kurtosis())
    stats["positive_days_pct"] = float((r > 0).mean())
    stats["negative_days_pct"] = float((r < 0).mean())

    # Extreme günler
    stats["best_day_pct"]  = float(r.max())
    stats["worst_day_pct"] = float(r.min())
    stats["best_day_date"] = str(df["Date"].iloc[r.idxmax() + 1].date())
    stats["worst_day_date"] = str(df["Date"].iloc[r.idxmin() + 1].date())

    # Top-5 yükseliş ve düşüş
    top_up = r.nlargest(5)
    top_down = r.nsmallest(5)
    stats["top5_up_pct"] = [round(float(x) * 100, 2) for x in top_up.values]
    stats["top5_down_pct"] = [round(float(x) * 100, 2) for x in top_down.values]

    # Volatilite rejimleri (30-günlük rolling std)
    vol30 = r.rolling(30).std() * np.sqrt(365)  # yıllıklaştırılmış
    stats["vol_annualized_mean"]   = float(vol30.mean())
    stats["vol_annualized_p10"]    = float(vol30.quantile(0.1))
    stats["vol_annualized_p90"]    = float(vol30.quantile(0.9))
    stats["vol_annualized_max"]    = float(vol30.max())

    # Autocorrelation
    stats["autocorr_lag1"] = float(r.autocorr(lag=1))
    stats["autocorr_lag5"] = float(r.autocorr(lag=5))

    # Drawdown analizi
    pv = close / close.iloc[0] * 100  # 100 başlangıçlı
    cummax = pv.cummax()
    drawdown = (pv - cummax) / cummax
    stats["max_drawdown_pct"]   = float(drawdown.min())
    stats["drawdown_end_date"]  = str(df["Date"].iloc[drawdown.idxmin()].date())

    # Trend serileri — ardışık pozitif/negatif günlerin en uzunu
    pos_run = _longest_run((r > 0).astype(int).values)
    neg_run = _longest_run((r < 0).astype(int).values)
    stats["longest_up_streak"]   = int(pos_run)
    stats["longest_down_streak"] = int(neg_run)

    # Bull/bear rejim — SMA50 üstü/altı gün yüzdesi
    sma50 = close.rolling(50).mean()
    above_sma = (close > sma50).dropna()
    stats["days_above_sma50_pct"] = float(above_sma.mean())

    return stats


def _longest_run(binary_seq: np.ndarray) -> int:
    """Ardışık 1'lerin en uzun serisi."""
    if len(binary_seq) == 0:
        return 0
    best = current = 0
    for v in binary_seq:
        if v == 1:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def main() -> None:
    print("=" * 80)
    print("  4 YILLIK DATA ANAL\u0130Z\u0130")
    print("=" * 80)

    analyses = {}
    for coin in COINS:
        df = load_coin(coin)
        analyses[coin] = analyze_coin(df)

    # Konsol tablosu
    fields = [
        ("Candle count",          "candle_count",            "{:>12,}"),
        ("Tarih aral\u0131\u011f\u0131",          "_date_range",             "{:>24}"),
        ("Start / End",           "_start_end",              "{:>22}"),
        ("Total return",          "total_return",            "{:>+11.2%}"),
        ("Mean daily return",     "daily_return_mean",       "{:>+12.4%}"),
        ("Std daily return",      "daily_return_std",        "{:>12.4%}"),
        ("Skew",                  "daily_return_skew",       "{:>12.4f}"),
        ("Kurt (excess)",         "daily_return_kurt",       "{:>12.4f}"),
        ("Pozitif g\u00fcn %",          "positive_days_pct",       "{:>12.2%}"),
        ("Negatif g\u00fcn %",          "negative_days_pct",       "{:>12.2%}"),
        ("En iyi g\u00fcn",             "best_day_pct",            "{:>+12.2%}"),
        ("En k\u00f6t\u00fc g\u00fcn",            "worst_day_pct",           "{:>+12.2%}"),
        ("Vol (y\u0131l\u0131k) ort.",      "vol_annualized_mean",     "{:>+12.2%}"),
        ("Vol (y\u0131l\u0131k) maks.",     "vol_annualized_max",      "{:>+12.2%}"),
        ("Max DD",                "max_drawdown_pct",        "{:>+12.2%}"),
        ("Autocorr lag-1",        "autocorr_lag1",           "{:>+12.4f}"),
        ("En uzun up streak",     "longest_up_streak",       "{:>12}"),
        ("En uzun down streak",   "longest_down_streak",     "{:>12}"),
        ("SMA50 \u00fcst\u00fc g\u00fcn %",     "days_above_sma50_pct",    "{:>12.2%}"),
    ]

    # Header
    name_w = 22
    col_w = 18
    print(f"{'Metrik':<{name_w}}" + "".join(f"{c.split('-')[0]:>{col_w}}" for c in COINS))
    print("-" * (name_w + col_w * len(COINS)))

    for label, key, fmt in fields:
        row = f"{label:<{name_w}}"
        for coin in COINS:
            a = analyses[coin]
            if key == "_date_range":
                val = f"{a['date_start']}->{a['date_end']}"
            elif key == "_start_end":
                val = f"{a['price_start']:.1f}->{a['price_end']:.1f}"
            else:
                val = fmt.format(a[key])
            row += f"{val:>{col_w}}"
        print(row)

    # Top-5 listesi
    print("\n" + "=" * 80)
    print("  TOP 5 EXTREME G\u00dcN")
    print("=" * 80)
    for coin in COINS:
        a = analyses[coin]
        print(f"\n{coin}:")
        print(f"  En b\u00fcy\u00fck 5 art\u0131\u015f:  {a['top5_up_pct']}%")
        print(f"  En b\u00fcy\u00fck 5 d\u00fc\u015f\u00fc\u015f: {a['top5_down_pct']}%")

    # JSON'a kaydet
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "data_analysis.json").write_text(
        json.dumps(analyses, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nJSON: results/data_analysis.json")


if __name__ == "__main__":
    main()
