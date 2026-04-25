"""
Sentetik 1 yıllık data generator.

Mantık:
  1. Real datanın istatistiklerini ölç (mean return, std, autocorr, skew, kurt,
     max extreme)
  2. Stationary Block Bootstrap: rastgele blok boyutlu bootstrap autocorrelation
     ve volatility clustering'i korur
  3. Stress injection: 3-5 adet extreme event (flash crash, pump) zorunlu
  4. High/Low/Open: Close etrafında gerçek datadan öğrenilen range'lerle
  5. Volume: rolling mean * lognormal noise

Çıktı:
  synthetic_data/{coin}_synth_{scenario}.parquet  (cnlib format uyumlu)

Scenario'lar:
  - normal:  standart bootstrap, hafif stress
  - crash:   bear rejim ağırlıklı (ortalama return negatif)
  - pump:    aşırı bull (ortalama return pozitif yüksek)
  - mixed:   ani rejim değişimleri (bull->bear->bull)
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).parent.parent / ".venv" / "Lib" / "site-packages" / "cnlib" / "data"
SYNTH_DIR = Path(__file__).parent.parent / "synthetic_data"
COINS = ["kapcoin-usd_train", "metucoin-usd_train", "tamcoin-usd_train"]

Scenario = Literal["normal", "crash", "pump", "mixed"]


def load_real(coin: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / f"{coin}.parquet")
    df = df.sort_values("Date").reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def stationary_bootstrap(
    returns: np.ndarray,
    n_steps: int,
    avg_block: int = 20,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Politis-Romano (1994) stationary bootstrap.
    Rastgele başlangıç + geometric blok boyutu → autocorr korur.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(returns)
    p = 1.0 / avg_block  # her adımda yeni blok başlatma olasılığı
    out = np.empty(n_steps, dtype=float)
    idx = rng.integers(0, n)
    for t in range(n_steps):
        if rng.random() < p or idx >= n:
            idx = rng.integers(0, n)
        out[t] = returns[idx]
        idx += 1
    return out


def inject_stress(
    returns: np.ndarray,
    scenario: Scenario,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Scenario'ya göre stress event'ler enjekte et.

    normal: 2-3 flash crash (-8% ile -12% arası), 2-3 pump (+7% ile +11%)
    crash:  5-7 flash crash, ortalama return'ü -0.3% shift et
    pump:   5-7 pump, ortalama +0.4% shift
    mixed:  ardışık rejim değişimleri (her 60-90 günde bir flip)
    """
    out = returns.copy()
    n = len(out)

    def sample_crash():
        return rng.uniform(-0.12, -0.08)

    def sample_pump():
        return rng.uniform(0.07, 0.11)

    if scenario == "normal":
        n_crashes = rng.integers(2, 4)
        n_pumps = rng.integers(2, 4)
        for _ in range(n_crashes):
            out[rng.integers(0, n)] = sample_crash()
        for _ in range(n_pumps):
            out[rng.integers(0, n)] = sample_pump()

    elif scenario == "crash":
        out = out - 0.003  # -0.3% günlük drift
        n_crashes = rng.integers(5, 8)
        for _ in range(n_crashes):
            out[rng.integers(0, n)] = sample_crash()

    elif scenario == "pump":
        out = out + 0.004  # +0.4% günlük drift
        n_pumps = rng.integers(5, 8)
        for _ in range(n_pumps):
            out[rng.integers(0, n)] = sample_pump()

    elif scenario == "mixed":
        # 60-90 günlük bloklar halinde pozitif/negatif drift
        pos = 0
        regime = 1  # +1 = bull, -1 = bear
        while pos < n:
            block_len = rng.integers(60, 91)
            end = min(pos + block_len, n)
            drift = 0.003 * regime
            out[pos:end] = out[pos:end] + drift
            # Blok başında 1-2 extreme event
            for _ in range(rng.integers(1, 3)):
                out[rng.integers(pos, end)] = (
                    sample_pump() if regime > 0 else sample_crash()
                )
            pos = end
            regime *= -1

    return out


def generate_ohlcv(
    real: pd.DataFrame,
    scenario: Scenario,
    n_days: int = 365,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Real data'yı baz alarak sentetik OHLCV üret.
    """
    rng = np.random.default_rng(seed)

    close = real["Close"].values
    log_ret = np.diff(np.log(close))

    # Bootstrap log returns
    synth_log_r = stationary_bootstrap(log_ret, n_days, avg_block=15, rng=rng)
    # Stress enjekte et (log return uzayında: log(1+r) yaklaşık r küçük r için)
    synth_r = np.exp(synth_log_r) - 1.0
    synth_r = inject_stress(synth_r, scenario, rng)
    # Tekrar clip (fazla agresif stress patlaması olmasın): ±%20 sınır
    synth_r = np.clip(synth_r, -0.20, 0.25)

    # Close serisi
    start_price = float(real["Close"].iloc[-1])
    synth_close = start_price * np.cumprod(1.0 + synth_r)

    # High/Low/Open — real datanın günlük range istatistiklerini öğren
    hl_range = (real["High"] - real["Low"]) / real["Close"]
    hl_mean = float(hl_range.mean())
    hl_std = float(hl_range.std())

    # Open genelde close_{t-1}'e yakın (gap var)
    # Gap = open_t - close_{t-1}, küçük: mean/std
    gap = (real["Open"] - real["Close"].shift(1)) / real["Close"].shift(1)
    gap_mean = float(gap.dropna().mean())
    gap_std = float(gap.dropna().std())

    # Close position in range (0=low, 1=high)
    close_pos = ((real["Close"] - real["Low"]) / (real["High"] - real["Low"])).clip(0, 1)
    cp_mean = float(close_pos.dropna().mean())
    cp_std = float(close_pos.dropna().std())

    highs = np.empty(n_days)
    lows  = np.empty(n_days)
    opens = np.empty(n_days)

    prev_close = start_price
    for t in range(n_days):
        c = synth_close[t]
        # Günlük range (close'a göre yüzde)
        rng_pct = max(abs(rng.normal(hl_mean, hl_std)), 0.005)
        rng_val = c * rng_pct

        # Close'un range içindeki pozisyonu
        cp = np.clip(rng.normal(cp_mean, cp_std), 0.05, 0.95)
        l = c - rng_val * cp
        h = c + rng_val * (1 - cp)

        # Open: previous close + gap
        g = rng.normal(gap_mean, gap_std)
        o = prev_close * (1 + g)
        # Open high-low içinde tutulur
        o = np.clip(o, l * 1.001, h * 0.999)

        highs[t] = h
        lows[t]  = l
        opens[t] = o
        prev_close = c

    # Volume — rolling mean'den lognormal noise
    real_vol = real["Volume"].values
    real_vol_mean = real_vol[-180:].mean()  # son 6 ay ortalaması baz
    real_vol_std_log = np.std(np.log(real_vol[real_vol > 0]))
    vol = np.exp(np.log(real_vol_mean) + rng.normal(0, real_vol_std_log, n_days))

    # Tarih: son real tarihten devam
    start_date = real["Date"].iloc[-1] + pd.Timedelta(days=1)
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")

    synth = pd.DataFrame({
        "Date":   dates,
        "Open":   opens,
        "High":   highs,
        "Low":    lows,
        "Close":  synth_close,
        "Volume": vol,
    })
    return synth


def build_combined_dataset(
    scenario: Scenario,
    seed: int = 42,
    output_dir: Path | None = None,
) -> Path:
    """
    Real 4-yıllık + sentetik 1-yıllık birleşik dataset üret.
    Her coin için ayrı parquet yazar. Dizin yolu döner.
    """
    out_dir = output_dir or (SYNTH_DIR / scenario)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, coin in enumerate(COINS):
        real = load_real(coin)
        synth = generate_ohlcv(real, scenario, n_days=365, seed=seed + i)
        combined = pd.concat([real, synth], ignore_index=True)
        combined.to_parquet(out_dir / f"{coin}.parquet", index=False)

    return out_dir


def summarize_synthetic(synth: pd.DataFrame) -> dict:
    r = synth["Close"].pct_change().dropna()
    return {
        "n_days":                   len(synth),
        "return_pct":               float(synth["Close"].iloc[-1] / synth["Close"].iloc[0] - 1),
        "mean_daily":               float(r.mean()),
        "std_daily":                float(r.std()),
        "autocorr_lag1":            float(r.autocorr(lag=1)),
        "best_day":                 float(r.max()),
        "worst_day":                float(r.min()),
        "skew":                     float(r.skew()),
    }


def main() -> None:
    scenarios: list[Scenario] = ["normal", "crash", "pump", "mixed"]
    print("=" * 80)
    print("  SENTET\u0130K DATA \u00dcRET\u0130M\u0130 \u2014 365 g\u00fcn, 4 senaryo")
    print("=" * 80)

    for scenario in scenarios:
        out_dir = build_combined_dataset(scenario, seed=42)
        # \u00d6zet
        print(f"\n[{scenario}] Dizin: {out_dir}")
        for coin in COINS:
            full = pd.read_parquet(out_dir / f"{coin}.parquet")
            # Sadece son 365 g\u00fcn (sentetik k\u0131s\u0131m)
            synth_only = full.iloc[-365:].reset_index(drop=True)
            stats = summarize_synthetic(synth_only)
            print(f"  {coin[:18]:<18} ret={stats['return_pct']*100:+7.2f}%  "
                  f"vol={stats['std_daily']*100:.2f}%  "
                  f"auto={stats['autocorr_lag1']:+.3f}  "
                  f"best={stats['best_day']*100:+.2f}%  "
                  f"worst={stats['worst_day']*100:+.2f}%")


if __name__ == "__main__":
    main()
