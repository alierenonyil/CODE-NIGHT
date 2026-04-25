"""
Binance public API ile kline data indir.

Key gerektirmez — public endpoint.

URL: https://api.binance.com/api/v3/klines
Limit max 1000. Daha fazla için time window ile loop.

Saved to: binance_data/{target_name}.parquet  (cnlib format)
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


BASE = "https://api.binance.com/api/v3/klines"
OUT_DIR = Path(__file__).parent.parent / "binance_data"
OUT_DIR.mkdir(exist_ok=True)

# Binance symbol → cnlib coin slot
MAPPING = [
    ("BTCUSDT", "kapcoin-usd_train"),
    ("ETHUSDT", "metucoin-usd_train"),
    ("SOLUSDT", "tamcoin-usd_train"),
]

# Alt coin seti (test #10 için daha volatil)
ALT_MAPPING = [
    ("ADAUSDT",  "kapcoin-usd_train"),   # Cardano
    ("AVAXUSDT", "metucoin-usd_train"),  # Avalanche
    ("DOTUSDT",  "tamcoin-usd_train"),   # Polkadot
]


def fetch_klines(symbol: str, start_ms: int, end_ms: int, interval: str = "1d") -> list:
    """Tek sayfa kline çek (max 1000 row)."""
    params = {
        "symbol":    symbol,
        "interval":  interval,
        "startTime": start_ms,
        "endTime":   end_ms,
        "limit":     1000,
    }
    resp = requests.get(BASE, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_range(symbol: str, start: datetime, end: datetime, interval: str = "1d") -> pd.DataFrame:
    """Verilen tarih aralığı için tüm günlük candle'ları çek."""
    all_rows = []
    cur = start
    while cur < end:
        start_ms = int(cur.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        chunk = fetch_klines(symbol, start_ms, end_ms, interval)
        if not chunk:
            break
        all_rows.extend(chunk)
        # Son candle'ın close zamanından sonraki ms'den devam et
        last_close = chunk[-1][6]
        next_time = datetime.fromtimestamp(last_close / 1000) + timedelta(milliseconds=1)
        if next_time <= cur:
            break
        cur = next_time
        if len(chunk) < 1000:
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=[
        "OpenTime", "Open", "High", "Low", "Close", "Volume",
        "CloseTime", "QuoteAssetVolume", "NumberOfTrades",
        "TakerBuyBaseVolume", "TakerBuyQuoteVolume", "Ignore",
    ])
    df["Date"] = pd.to_datetime(df["OpenTime"], unit="ms")
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].astype({
        "Open":   float,
        "High":   float,
        "Low":    float,
        "Close":  float,
        "Volume": float,
    })
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def download_set(mapping: list[tuple[str, str]], out_subdir: str, years: int = 2) -> Path:
    out = OUT_DIR / out_subdir
    out.mkdir(parents=True, exist_ok=True)

    end = datetime.utcnow()
    start = end - timedelta(days=years * 365 + 30)

    dfs = {}
    print(f"Tarih araligi: {start.date()} -> {end.date()}")
    for symbol, target in mapping:
        print(f"  {symbol} ...", end=" ", flush=True)
        df = fetch_range(symbol, start, end)
        if df.empty:
            print("HATA")
            continue
        print(f"{len(df)} candle  {df['Close'].iloc[0]:.2f}->{df['Close'].iloc[-1]:.2f}  "
              f"({(df['Close'].iloc[-1]/df['Close'].iloc[0]-1)*100:+.2f}%)")
        dfs[target] = df

    if len(dfs) != len(mapping):
        print("!! eksik data")
        return out

    # Eşit uzunluk (cnlib shart)
    min_len = min(len(d) for d in dfs.values())
    print(f"En kisa: {min_len} candle, hepsini trim ediyorum")
    for target, df in dfs.items():
        trimmed = df.iloc[-min_len:].reset_index(drop=True)
        path = out / f"{target}.parquet"
        trimmed.to_parquet(path, index=False)

    return out


def main() -> None:
    print("=" * 80)
    print("BINANCE BTC/ETH/SOL (majors)")
    print("=" * 80)
    d1 = download_set(MAPPING, "majors", years=2)
    print(f"\nKaydedildi: {d1}")

    print("\n" + "=" * 80)
    print("BINANCE ADA/AVAX/DOT (altcoins - daha volatil)")
    print("=" * 80)
    d2 = download_set(ALT_MAPPING, "altcoins", years=2)
    print(f"\nKaydedildi: {d2}")

    # Autocorr check
    print("\n" + "=" * 80)
    print("AUTOCORR LAG-1 KONTROL")
    print("=" * 80)
    for dir_path in [d1, d2]:
        print(f"\n{dir_path.name}:")
        for target in ["kapcoin-usd_train", "metucoin-usd_train", "tamcoin-usd_train"]:
            df = pd.read_parquet(dir_path / f"{target}.parquet")
            r = df["Close"].pct_change().dropna()
            print(f"  {target:<22} autocorr={r.autocorr(1):+.4f}  "
                  f"std_daily={r.std()*100:.2f}%  "
                  f"worst={r.min()*100:+.2f}%")


if __name__ == "__main__":
    main()
