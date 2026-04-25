"""
Çok çeşitli gerçek data toplayıcı.

- Yahoo: BTC/ETH/SOL farklı dönemler (2018-2020, 2020-2022, 2022-2024)
- Yahoo: diğer majors (DOGE, LINK, BNB, UNI, MATIC)
- Klasik varlıklar (kripto dışı): GLD (altın), SPY (S&P500)
- Binance (public, key'siz): daha fazla coin + farklı zaman dilimleri

Tüm çıktılar cnlib format'ına uyarlanır:
  {target}/kapcoin-usd_train.parquet
  {target}/metucoin-usd_train.parquet
  {target}/tamcoin-usd_train.parquet
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent.parent
MULTI_DIR = ROOT / "multi_data"
MULTI_DIR.mkdir(exist_ok=True)


def save_set(dfs: dict[str, pd.DataFrame], subdir: str) -> bool:
    """3 coin'i cnlib format'a uyarla ve kaydet."""
    if len(dfs) != 3:
        print(f"  [{subdir}] HATA: 3 coin eksik ({len(dfs)} var)")
        return False
    min_len = min(len(d) for d in dfs.values())
    if min_len < 100:
        print(f"  [{subdir}] HATA: cok az candle ({min_len})")
        return False
    out = MULTI_DIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    targets = ["kapcoin-usd_train", "metucoin-usd_train", "tamcoin-usd_train"]
    for i, (name, df) in enumerate(dfs.items()):
        trimmed = df.iloc[-min_len:].reset_index(drop=True)
        trimmed = trimmed[["Date", "Open", "High", "Low", "Close", "Volume"]]
        trimmed.to_parquet(out / f"{targets[i]}.parquet", index=False)
    print(f"  [{subdir}] OK: {min_len} candle per coin")
    return True


def yf_download(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """yfinance wrapper, kolon temizliği."""
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, interval="1d",
                     auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df = df.rename(columns={"Date": "Date"})
    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]


def fetch_yahoo_period(target_subdir: str, tickers: list[str],
                       start: str, end: str) -> None:
    print(f"\n[YF] {target_subdir}: {tickers} {start}->{end}")
    dfs = {}
    for t in tickers:
        d = yf_download(t, start, end)
        if d is None or len(d) < 100:
            print(f"  {t} eksik")
            continue
        dfs[t] = d
        print(f"  {t}: {len(d)} candle  {d['Close'].iloc[0]:.2f}->{d['Close'].iloc[-1]:.2f}  "
              f"({(d['Close'].iloc[-1]/d['Close'].iloc[0]-1)*100:+.1f}%)")
    if len(dfs) == 3:
        save_set(dfs, target_subdir)


def main() -> None:
    sets = [
        # (subdir, tickers, start, end)
        ("yf_btc_eth_sol_2018_2020", ["BTC-USD", "ETH-USD", "LTC-USD"], "2018-01-01", "2020-01-01"),
        ("yf_btc_eth_ltc_2019_2021", ["BTC-USD", "ETH-USD", "LTC-USD"], "2019-01-01", "2021-01-01"),
        ("yf_bull_2020_2022",        ["BTC-USD", "ETH-USD", "SOL-USD"], "2020-01-01", "2022-01-01"),
        ("yf_bear_2022_2024",        ["BTC-USD", "ETH-USD", "SOL-USD"], "2022-01-01", "2024-01-01"),
        ("yf_recent_2024_2026",      ["BTC-USD", "ETH-USD", "SOL-USD"], "2024-04-25", "2026-04-25"),
        ("yf_meme",                  ["DOGE-USD", "SHIB-USD", "PEPE24478-USD"], "2024-04-25", "2026-04-25"),
        ("yf_majors_alt",            ["BNB-USD", "XRP-USD", "ADA-USD"], "2022-01-01", "2024-01-01"),
        ("yf_defi",                  ["UNI-USD", "LINK-USD", "AAVE-USD"], "2022-01-01", "2024-01-01"),
        ("yf_classic_assets",        ["GLD", "SPY", "QQQ"], "2022-01-01", "2024-01-01"),
        ("yf_big4_2021",             ["BTC-USD", "ETH-USD", "ADA-USD"], "2020-06-01", "2022-06-01"),
    ]
    for subdir, tickers, start, end in sets:
        fetch_yahoo_period(subdir, tickers, start, end)


if __name__ == "__main__":
    main()
