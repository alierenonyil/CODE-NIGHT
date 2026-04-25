"""
Gerçek kripto 2 yıllık OHLCV data indir.

BTC, ETH, SOL → yfinance → parquet (cnlib format)

cnlib hardcoded coin isimleri: kapcoin-usd_train, metucoin-usd_train, tamcoin-usd_train
BTC → kapcoin slot (en yüksek cap)
ETH → metucoin slot
SOL → tamcoin slot
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


OUT_DIR = Path(__file__).parent.parent / "real_crypto_data"
OUT_DIR.mkdir(exist_ok=True)

COIN_MAPPING = [
    ("BTC-USD", "kapcoin-usd_train"),
    ("ETH-USD", "metucoin-usd_train"),
    ("SOL-USD", "tamcoin-usd_train"),
]


def main() -> None:
    import yfinance as yf

    end = date.today()
    start = end - timedelta(days=2 * 365 + 30)  # 2 yıl + buffer

    print(f"Indirilen aralik: {start} -> {end}")
    dfs = {}

    for ticker, target_name in COIN_MAPPING:
        print(f"\n  {ticker} indiriliyor...")
        df = yf.download(
            ticker,
            start=str(start),
            end=str(end),
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        if df is None or len(df) == 0:
            print(f"    HATA: {ticker} indirilemedi")
            sys.exit(1)

        # yfinance multi-index kolonlar döndürüyor (ticker'a göre)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        # Standart kolonlar
        df = df.rename(columns={"Date": "Date"})
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        df["Date"] = pd.to_datetime(df["Date"])

        # NaN satırları at
        df = df.dropna(subset=["Close"]).reset_index(drop=True)
        # Tarih sırala
        df = df.sort_values("Date").reset_index(drop=True)

        print(f"    {len(df)} candle, {df['Date'].iloc[0].date()} -> {df['Date'].iloc[-1].date()}")
        print(f"    Close: {float(df['Close'].iloc[0]):.2f} -> {float(df['Close'].iloc[-1]):.2f}")
        print(f"    Total return: {float(df['Close'].iloc[-1]/df['Close'].iloc[0]-1)*100:+.2f}%")

        dfs[target_name] = df

    # Tüm coinlerin uzunlukları eşit olmalı (cnlib shart koşuyor)
    min_len = min(len(d) for d in dfs.values())
    print(f"\nEn kisa: {min_len} candle. Hepsini buna trim ediyorum (sondan).")

    for target_name, df in dfs.items():
        trimmed = df.iloc[-min_len:].reset_index(drop=True)
        out_path = OUT_DIR / f"{target_name}.parquet"
        trimmed.to_parquet(out_path, index=False)
        print(f"  Kaydedildi: {out_path}")

    # Kontrol analizi
    print("\n" + "=" * 80)
    print("  GERCEK KRIPTO 2 YIL - OZET ISTATISTIKLER")
    print("=" * 80)
    for target_name, df in dfs.items():
        trimmed = df.iloc[-min_len:].reset_index(drop=True)
        r = trimmed["Close"].pct_change().dropna()
        print(f"\n{target_name}:")
        print(f"  Total return: {float(trimmed['Close'].iloc[-1]/trimmed['Close'].iloc[0]-1)*100:+.2f}%")
        print(f"  Mean daily:   {float(r.mean())*100:+.4f}%")
        print(f"  Std daily:    {float(r.std())*100:.4f}%")
        print(f"  Best day:     {float(r.max())*100:+.2f}%")
        print(f"  Worst day:    {float(r.min())*100:+.2f}%")
        print(f"  Autocorr lag1: {float(r.autocorr(1)):+.4f}")


if __name__ == "__main__":
    main()
