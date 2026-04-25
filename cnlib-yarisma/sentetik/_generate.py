"""
20 sentetik dataset üretici — model eğitimi için optimum çeşitlilik.

Her dataset 3 coin x 1935 gün (2022-11-27 → 2028-03-14).
Her dataset farklı bir market rejimini öğretir:
  - trend yapıları (bull/bear/sideways)
  - volatilite rejimleri (low/high/clustering)
  - olay tabanlı (flash crash, black swan, pump-dump)
  - korelasyon yapıları (decoupled/anti/lead-lag/perfect)
  - örüntü tabanlı (accumulation/distribution/mean-revert/momentum)

Üretilen dosyalar: sentetik/{scenario}/{coin}-usd_train.parquet
Şema mevcut synthetic_data ile birebir aynı:
  Date(datetime64), Close, High, Low, Open, Volume (float64)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------------------------------------------------------
# Sabit parametreler
# -----------------------------------------------------------------------------
START_DATE = "2022-11-27"
N_DAYS = 1935
OUT_DIR = Path(__file__).parent

COIN_START = {
    "kapcoin": 57.82034766,
    "metucoin": 849.79805416,
    "tamcoin": 291.89164915,
}
COIN_VOL_BASE = {
    "kapcoin": 1.05e10,
    "metucoin": 1.85e10,
    "tamcoin": 3.50e9,
}
COINS = ["kapcoin", "metucoin", "tamcoin"]


# -----------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# -----------------------------------------------------------------------------
def correlated_normals(rng, n, corr):
    """3 coin için korelasyonlu standart normal innovasyonlar üret."""
    corr = np.asarray(corr, dtype=float)
    L = np.linalg.cholesky(corr)
    Z = rng.standard_normal((n, 3))
    return Z @ L.T  # (n, 3)


def garch_vol(rng, n, sigma_long, persistence=0.92, vol_of_vol=0.15):
    """Volatilite kümelemesi için basit AR(1) varyans süreci."""
    sigma = np.empty(n)
    sigma[0] = sigma_long
    eps = rng.standard_normal(n) * vol_of_vol
    for t in range(1, n):
        sigma[t] = max(0.003, sigma_long + persistence * (sigma[t - 1] - sigma_long) + eps[t] * sigma_long * 0.4)
    return sigma


def add_jumps(returns, rng, frequency_per_year=2, mean_size=0.0, std_size=0.06, n_days=N_DAYS, asymmetry=0.0):
    """Poisson-jump ile fat-tail eklemeleri. asymmetry < 0 → daha çok aşağı sıçrama."""
    p = frequency_per_year / 365.0
    mask = rng.random(n_days) < p
    sizes = rng.normal(mean_size, std_size, size=n_days)
    if asymmetry != 0.0:
        sizes = np.where(rng.random(n_days) < 0.5 + asymmetry, -np.abs(sizes), np.abs(sizes))
    returns[mask] += sizes[mask]
    return returns


def piecewise(n, segments):
    """Birden fazla aralığı [(start_frac, end_frac, value), ...] birleştirip uzunluk-n dizi döndür."""
    out = np.zeros(n)
    for s, e, v in segments:
        i0 = int(round(s * n))
        i1 = int(round(e * n))
        if callable(v):
            out[i0:i1] = v(np.linspace(0, 1, i1 - i0))
        else:
            out[i0:i1] = v
    return out


def smooth(x, window=10):
    """Hareketli ortalamayla parametre eğrisini yumuşat."""
    if window <= 1:
        return x
    pad = window // 2
    extended = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    kernel = np.ones(window) / window
    return np.convolve(extended, kernel, mode="valid")[: len(x)]


def build_ohlcv(coin, log_rets, rng, intraday_amp=1.5, vol_base_mult=1.0):
    """Log getirilerden OHLCV inşa et. OHLC tutarlılığını garanti altına al."""
    n = len(log_rets)
    start = COIN_START[coin]
    vol_base = COIN_VOL_BASE[coin] * vol_base_mult

    close = start * np.exp(np.cumsum(log_rets))

    # Open = bir önceki Close + küçük overnight gap
    overnight_gap = rng.normal(0.0, 0.0035, size=n)
    open_ = np.empty(n)
    open_[0] = start
    open_[1:] = close[:-1] * np.exp(overnight_gap[1:])

    # Intraday menzil → mutlak getiri ile orantılı + taban gürültü
    abs_ret = np.abs(log_rets)
    base_range = abs_ret * intraday_amp + rng.uniform(0.012, 0.045, size=n)

    upper_extra = rng.uniform(0.0, 1.0, size=n) * base_range
    lower_extra = rng.uniform(0.0, 1.0, size=n) * base_range

    high = np.maximum(open_, close) * (1.0 + upper_extra)
    low = np.minimum(open_, close) * (1.0 - lower_extra)
    low = np.maximum(low, start * 1e-4)  # zemin

    # Volume: vol-of-day ile orantılı, lognormal noise
    range_pct = (high - low) / np.maximum(close, 1e-6)
    vol_factor = 1.0 + 5.0 * abs_ret + 3.0 * range_pct
    vol_noise = rng.lognormal(0.0, 0.45, size=n)
    volume = vol_base * vol_factor * vol_noise

    return open_, high, low, close, volume


def save_scenario(name, coin_data):
    out = OUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(start=START_DATE, periods=N_DAYS, freq="D")
    for coin, (o, h, l, c, v) in coin_data.items():
        df = pd.DataFrame(
            {
                "Date": dates,
                "Close": c,
                "High": h,
                "Low": l,
                "Open": o,
                "Volume": v,
            }
        )
        df.to_parquet(out / f"{coin}-usd_train.parquet", index=False)


# -----------------------------------------------------------------------------
# Korelasyon matrisleri
# -----------------------------------------------------------------------------
CORR_NORMAL = [[1.0, 0.55, 0.45], [0.55, 1.0, 0.50], [0.45, 0.50, 1.0]]
CORR_HIGH = [[1.0, 0.92, 0.90], [0.92, 1.0, 0.88], [0.90, 0.88, 1.0]]
CORR_LOW = [[1.0, 0.10, 0.05], [0.10, 1.0, 0.08], [0.05, 0.08, 1.0]]
CORR_NEG = [[1.0, -0.65, 0.20], [-0.65, 1.0, -0.30], [0.20, -0.30, 1.0]]


def make_returns(rng, mu_seq_per_coin, sigma_seq_per_coin, corr=CORR_NORMAL):
    """3 coin için korelasyonlu, zamanla değişen mu/sigma'ya sahip log-getiri matrisi."""
    n = N_DAYS
    Z = correlated_normals(rng, n, corr)  # (n, 3)
    rets = np.empty_like(Z)
    for i in range(3):
        rets[:, i] = mu_seq_per_coin[i] + sigma_seq_per_coin[i] * Z[:, i]
    return rets  # (n, 3)


# =============================================================================
# 20 SENARYO TANIMLARI
# =============================================================================

def scenario_01_bull_steady():
    """Düzenli boğa: yumuşak yükseliş, düşük volatilite, +6x net."""
    rng = np.random.default_rng(101)
    mu = np.full(N_DAYS, 0.0024)
    sigma = garch_vol(rng, N_DAYS, sigma_long=0.018, persistence=0.85, vol_of_vol=0.10)
    rets_per = []
    for i, coin in enumerate(COINS):
        m = mu * (0.9 + 0.15 * i)
        rets_per.append((m, sigma * (0.95 + 0.05 * i)))
    mu_seq = [r[0] for r in rets_per]
    sig_seq = [r[1] for r in rets_per]
    R = make_returns(rng, mu_seq, sig_seq, corr=CORR_NORMAL)
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i]
        r = add_jumps(r, rng, frequency_per_year=1, mean_size=0.0, std_size=0.04)
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.4)
    return out


def scenario_02_bull_parabolic():
    """Parabolik boğa: erken sakin, orta rally, geç parabolik, son blow-off + crash."""
    rng = np.random.default_rng(202)
    mu = piecewise(
        N_DAYS,
        [
            (0.00, 0.30, 0.0010),
            (0.30, 0.55, 0.0030),
            (0.55, 0.85, 0.0060),
            (0.85, 0.92, 0.0090),
            (0.92, 1.00, -0.0070),
        ],
    )
    sigma = piecewise(
        N_DAYS,
        [
            (0.00, 0.30, 0.018),
            (0.30, 0.55, 0.025),
            (0.55, 0.85, 0.040),
            (0.85, 0.92, 0.060),
            (0.92, 1.00, 0.075),
        ],
    )
    mu = smooth(mu, 15)
    sigma = smooth(sigma, 15)
    R = make_returns(rng, [mu, mu * 1.05, mu * 0.95], [sigma, sigma * 1.05, sigma * 1.0], corr=CORR_HIGH)
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i]
        r = add_jumps(r, rng, frequency_per_year=4, std_size=0.05, asymmetry=-0.1)
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.7)
    return out


def scenario_03_bear_capitulation():
    """Bear + kapitülasyon: dağıtım, yavaş düşüş, panik flush, yavaş toparlanma."""
    rng = np.random.default_rng(303)
    mu = piecewise(
        N_DAYS,
        [
            (0.00, 0.15, 0.0010),
            (0.15, 0.50, -0.0035),
            (0.50, 0.65, -0.0120),
            (0.65, 1.00, 0.0015),
        ],
    )
    sigma = piecewise(
        N_DAYS,
        [
            (0.00, 0.15, 0.020),
            (0.15, 0.50, 0.025),
            (0.50, 0.65, 0.055),
            (0.65, 1.00, 0.024),
        ],
    )
    mu = smooth(mu, 12)
    sigma = smooth(sigma, 12)
    R = make_returns(rng, [mu, mu * 1.10, mu * 0.95], [sigma, sigma * 1.05, sigma], corr=CORR_HIGH)
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i]
        r = add_jumps(r, rng, frequency_per_year=6, std_size=0.07, asymmetry=-0.35)
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.8)
    return out


def scenario_04_bear_grinding():
    """Yavaş bear: -%70 düşüş, düşük vol, ısrarlı."""
    rng = np.random.default_rng(404)
    mu = np.full(N_DAYS, -0.0022)
    sigma = garch_vol(rng, N_DAYS, sigma_long=0.018, persistence=0.88, vol_of_vol=0.08)
    R = make_returns(rng, [mu, mu * 0.95, mu * 1.05], [sigma, sigma, sigma * 1.05], corr=CORR_NORMAL)
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i]
        r = add_jumps(r, rng, frequency_per_year=2, std_size=0.04, asymmetry=-0.15)
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.4)
    return out


def scenario_05_crab_market():
    """Yatay piyasa: log-fiyat OU ile sıfır etrafında salınır, fake breakouts."""
    rng = np.random.default_rng(505)
    n = N_DAYS
    sigma_arr = garch_vol(rng, n, sigma_long=0.022, persistence=0.85, vol_of_vol=0.10)
    L = np.linalg.cholesky(np.array(CORR_NORMAL))
    Z = rng.standard_normal((n, 3)) @ L.T  # korelasyonlu innovasyonlar
    R = np.zeros((n, 3))
    kappa = 0.025  # log-fiyat MR hızı
    for i in range(3):
        log_p = 0.0
        for t in range(n):
            innov = sigma_arr[t] * Z[t, i]
            new_log = log_p * (1 - kappa) + innov
            R[t, i] = new_log - log_p
            log_p = new_log
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i].copy()
        # geçici fakeout breakout drift'leri
        for _ in range(6):
            day = int(rng.integers(50, n - 50))
            r[day:day + 8] += rng.normal(0.005, 0.003)
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.5)
    return out


def scenario_06_high_vol_chaos():
    """Sürekli yüksek volatilite. Net yatay ama vahşi salınım."""
    rng = np.random.default_rng(606)
    sigma = garch_vol(rng, N_DAYS, sigma_long=0.045, persistence=0.90, vol_of_vol=0.20)
    mu = np.full(N_DAYS, 0.0003)
    R = make_returns(rng, [mu, mu, mu], [sigma, sigma * 1.05, sigma * 0.95], corr=CORR_NORMAL)
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i]
        r = add_jumps(r, rng, frequency_per_year=12, std_size=0.07)
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=2.0)
    return out


def scenario_07_low_vol_compression():
    """Vol sıkışması → patlama → sıkışma. Bollinger squeeze öğretici."""
    rng = np.random.default_rng(707)
    sigma = piecewise(
        N_DAYS,
        [
            (0.00, 0.20, 0.020),
            (0.20, 0.40, 0.008),  # sıkışma
            (0.40, 0.55, 0.045),  # patlama
            (0.55, 0.75, 0.012),  # tekrar sıkışma
            (0.75, 0.90, 0.040),
            (0.90, 1.00, 0.015),
        ],
    )
    sigma = smooth(sigma, 18)
    mu = np.full(N_DAYS, 0.0008)
    R = make_returns(rng, [mu, mu * 1.05, mu * 0.95], [sigma, sigma * 1.05, sigma * 0.95], corr=CORR_HIGH)
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i]
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.6)
    return out


def scenario_08_regime_switch_vol():
    """3 rejim döngüsü: bull-low / bear-high / flat-mid. Rejim tespiti öğretici."""
    rng = np.random.default_rng(808)
    mu = piecewise(
        N_DAYS,
        [
            (0.00, 0.25, 0.0030),
            (0.25, 0.45, -0.0050),
            (0.45, 0.65, 0.0000),
            (0.65, 0.85, 0.0040),
            (0.85, 1.00, -0.0030),
        ],
    )
    sigma = piecewise(
        N_DAYS,
        [
            (0.00, 0.25, 0.015),
            (0.25, 0.45, 0.045),
            (0.45, 0.65, 0.022),
            (0.65, 0.85, 0.018),
            (0.85, 1.00, 0.038),
        ],
    )
    mu = smooth(mu, 12)
    sigma = smooth(sigma, 12)
    R = make_returns(rng, [mu, mu * 1.05, mu * 0.95], [sigma, sigma * 1.05, sigma], corr=CORR_HIGH)
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i]
        r = add_jumps(r, rng, frequency_per_year=5, std_size=0.06, asymmetry=-0.1)
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.7)
    return out


def scenario_09_flash_crash():
    """Normal trend + 5 ani flash crash (-25% bir günde, ertesi gün kısmi recovery)."""
    rng = np.random.default_rng(909)
    mu = np.full(N_DAYS, 0.0014)
    sigma = garch_vol(rng, N_DAYS, sigma_long=0.020, persistence=0.85, vol_of_vol=0.10)
    R = make_returns(rng, [mu, mu * 1.05, mu * 0.95], [sigma, sigma, sigma * 1.05], corr=CORR_HIGH)
    crash_days = np.sort(rng.choice(np.arange(60, N_DAYS - 60), size=5, replace=False))
    for d in crash_days:
        R[d, :] += rng.uniform(-0.32, -0.22, size=3)
        R[d + 1, :] += rng.uniform(0.10, 0.18, size=3)
        R[d + 2, :] += rng.uniform(0.03, 0.07, size=3)
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i]
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.7)
    return out


def scenario_10_black_swan():
    """Sık black swan olayları, ağır kuyruklu getiri dağılımı."""
    rng = np.random.default_rng(1010)
    mu = np.full(N_DAYS, 0.0005)
    sigma = garch_vol(rng, N_DAYS, sigma_long=0.022, persistence=0.92, vol_of_vol=0.15)
    # Student-t innovations (df=5) ile fat-tail
    L = np.linalg.cholesky(np.array(CORR_HIGH))
    T = rng.standard_t(df=5, size=(N_DAYS, 3)) / np.sqrt(5 / 3)  # std=1
    T = T @ L.T
    R = np.zeros((N_DAYS, 3))
    for i in range(3):
        R[:, i] = mu * (1.0 + 0.05 * (i - 1)) + sigma * T[:, i]
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i].copy()
        r = add_jumps(r, rng, frequency_per_year=8, std_size=0.09, asymmetry=-0.30)
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=2.0)
    return out


def scenario_11_pump_dump_cycles():
    """5 döngü: 30 gün +60% pump, 25 gün -45% dump, 20 gün dinlenme."""
    rng = np.random.default_rng(1111)
    n = N_DAYS
    cycle_len = n // 5
    mu = np.zeros(n)
    sigma = np.full(n, 0.022)
    for k in range(5):
        s = k * cycle_len
        mu[s:s + 30] = 0.018          # pump
        sigma[s:s + 30] = 0.030
        mu[s + 30:s + 55] = -0.022    # dump
        sigma[s + 30:s + 55] = 0.045
        mu[s + 55:s + 75] = 0.0       # dinlenme
        sigma[s + 55:s + 75] = 0.018
    mu = smooth(mu, 5)
    sigma = smooth(sigma, 5)
    R = make_returns(rng, [mu, mu * 1.10, mu * 0.95], [sigma, sigma * 1.05, sigma], corr=CORR_HIGH)
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i]
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.8)
    return out


def scenario_12_news_driven():
    """Sakin arka plan + 30-40 günde bir ±%10 haber spike'ı."""
    rng = np.random.default_rng(1212)
    mu = np.full(N_DAYS, 0.0008)
    sigma = garch_vol(rng, N_DAYS, sigma_long=0.016, persistence=0.80, vol_of_vol=0.08)
    R = make_returns(rng, [mu, mu * 1.0, mu * 1.0], [sigma, sigma, sigma], corr=CORR_HIGH)
    spike_days = np.arange(20, N_DAYS, 35) + rng.integers(-5, 5, size=len(np.arange(20, N_DAYS, 35)))
    spike_days = spike_days[(spike_days >= 0) & (spike_days < N_DAYS)]
    for d in spike_days:
        sign = rng.choice([-1, 1], p=[0.45, 0.55])
        magnitude = rng.uniform(0.07, 0.13) * sign
        R[d, :] += magnitude * np.array([1.0, 1.05, 0.95])
    out = {}
    for i, coin in enumerate(COINS):
        out[coin] = build_ohlcv(coin, R[:, i], rng, intraday_amp=1.6)
    return out


def scenario_13_decoupled():
    """Her coin bağımsız; kapcoin bull, metucoin bear, tamcoin sideways."""
    rng = np.random.default_rng(1313)
    n = N_DAYS
    mu_kap = np.full(n, 0.0014)
    mu_met = np.full(n, -0.0020)
    mu_tam = np.full(n, 0.0001)
    sig_kap = garch_vol(rng, n, sigma_long=0.020)
    sig_met = garch_vol(np.random.default_rng(1314), n, sigma_long=0.024)
    sig_tam = garch_vol(np.random.default_rng(1315), n, sigma_long=0.022)
    R = make_returns(
        rng,
        [mu_kap, mu_met, mu_tam],
        [sig_kap, sig_met, sig_tam],
        corr=CORR_LOW,
    )
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i].copy()
        r = add_jumps(r, rng, frequency_per_year=3, std_size=0.05)
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.5)
    return out


def scenario_14_anti_correlated():
    """kapcoin ↔ metucoin ters korelasyon, tamcoin orta."""
    rng = np.random.default_rng(1414)
    n = N_DAYS
    sigma = garch_vol(rng, n, sigma_long=0.024)
    mu = piecewise(n, [(0.0, 0.5, 0.0015), (0.5, 1.0, -0.0010)])
    mu = smooth(mu, 20)
    R = make_returns(
        rng,
        [mu, -mu, mu * 0.5],
        [sigma, sigma * 1.05, sigma * 0.9],
        corr=CORR_NEG,
    )
    out = {}
    for i, coin in enumerate(COINS):
        out[coin] = build_ohlcv(coin, R[:, i], rng, intraday_amp=1.6)
    return out


def scenario_15_lead_lag():
    """tamcoin önder; kapcoin 3 gün gecikmeli, metucoin 5 gün gecikmeli takip."""
    rng = np.random.default_rng(1515)
    n = N_DAYS
    sigma = garch_vol(rng, n, sigma_long=0.022)
    mu = piecewise(n, [(0.0, 0.3, 0.001), (0.3, 0.6, 0.004), (0.6, 0.85, -0.002), (0.85, 1.0, 0.003)])
    mu = smooth(mu, 15)

    leader = mu + sigma * rng.standard_normal(n)
    lag3 = np.concatenate([np.zeros(3), leader[:-3]])
    lag5 = np.concatenate([np.zeros(5), leader[:-5]])

    # idiosyncratic noise
    noise_kap = 0.7 * sigma * rng.standard_normal(n)
    noise_met = 0.7 * sigma * rng.standard_normal(n)

    R = np.zeros((n, 3))
    R[:, 0] = 0.6 * lag3 + noise_kap          # kapcoin lagged 3
    R[:, 1] = 0.6 * lag5 + noise_met          # metucoin lagged 5
    R[:, 2] = leader                           # tamcoin = leader

    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i]
        r = add_jumps(r, rng, frequency_per_year=2, std_size=0.04)
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.6)
    return out


def scenario_16_perfect_corr():
    """Tüm coinler birlikte hareket eder (~0.95 korelasyon). Sistemik risk."""
    rng = np.random.default_rng(1616)
    n = N_DAYS
    mu = piecewise(n, [(0.0, 0.4, 0.0025), (0.4, 0.7, -0.0035), (0.7, 1.0, 0.0020)])
    mu = smooth(mu, 18)
    sigma = garch_vol(rng, n, sigma_long=0.025, persistence=0.92)
    R = make_returns(
        rng,
        [mu, mu * 1.08, mu * 0.95],
        [sigma, sigma * 1.05, sigma * 0.98],
        corr=CORR_HIGH,
    )
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i]
        r = add_jumps(r, rng, frequency_per_year=4, std_size=0.05, asymmetry=-0.1)
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.6)
    return out


def scenario_17_accumulation_breakout():
    """Uzun dip-yapımı (taban) → patlayıcı kırılım → blow-off. Wyckoff şeması."""
    rng = np.random.default_rng(1717)
    n = N_DAYS
    mu = piecewise(
        n,
        [
            (0.00, 0.55, 0.0001),
            (0.55, 0.85, 0.0040),
            (0.85, 0.95, 0.0025),
            (0.95, 1.00, -0.0060),
        ],
    )
    sigma = piecewise(
        n,
        [
            (0.00, 0.55, 0.013),
            (0.55, 0.85, 0.028),
            (0.85, 0.95, 0.045),
            (0.95, 1.00, 0.060),
        ],
    )
    mu = smooth(mu, 18)
    sigma = smooth(sigma, 18)
    R = make_returns(rng, [mu, mu * 1.05, mu * 0.95], [sigma, sigma * 1.05, sigma], corr=CORR_HIGH)
    # Accumulation aşamasında log-fiyat MR (range-bound)
    end_acc = int(0.55 * n)
    kappa = 0.04
    for i in range(3):
        log_p = 0.0
        for t in range(end_acc):
            new_log = log_p * (1 - kappa) + R[t, i]
            R[t, i] = new_log - log_p
            log_p = new_log
    out = {}
    for i, coin in enumerate(COINS):
        out[coin] = build_ohlcv(coin, R[:, i], rng, intraday_amp=1.5)
    return out


def scenario_18_distribution_breakdown():
    """Yükseliş → dağıtım (yatay zirve) → kırılma. Tepe formasyonu öğretici."""
    rng = np.random.default_rng(1818)
    n = N_DAYS
    mu = piecewise(
        n,
        [
            (0.00, 0.30, 0.0050),
            (0.30, 0.65, 0.0001),
            (0.65, 0.95, -0.0050),
            (0.95, 1.00, -0.0020),
        ],
    )
    sigma = piecewise(
        n,
        [
            (0.00, 0.30, 0.022),
            (0.30, 0.65, 0.018),
            (0.65, 0.95, 0.040),
            (0.95, 1.00, 0.025),
        ],
    )
    mu = smooth(mu, 18)
    sigma = smooth(sigma, 18)
    R = make_returns(rng, [mu, mu * 1.05, mu * 0.95], [sigma, sigma * 1.05, sigma], corr=CORR_HIGH)
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i]
        r = add_jumps(r, rng, frequency_per_year=4, std_size=0.06, asymmetry=-0.2)
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.6)
    return out


def scenario_19_mean_reverting():
    """Güçlü mean reversion: log-fiyat OU(kappa=0.05) ile başlangıç etrafında salınır."""
    rng = np.random.default_rng(1919)
    n = N_DAYS
    sigma = garch_vol(rng, n, sigma_long=0.022, persistence=0.80, vol_of_vol=0.10)
    L = np.linalg.cholesky(np.array(CORR_NORMAL))
    Z = rng.standard_normal((n, 3)) @ L.T
    R = np.zeros((n, 3))
    kappa = 0.05  # güçlü mean reversion
    for i in range(3):
        log_p = 0.0
        for t in range(n):
            innov = sigma[t] * Z[t, i]
            new_log = log_p * (1 - kappa) + innov
            R[t, i] = new_log - log_p
            log_p = new_log
    out = {}
    for i, coin in enumerate(COINS):
        out[coin] = build_ohlcv(coin, R[:, i], rng, intraday_amp=1.6)
    return out


def scenario_20_trending_persistent():
    """Güçlü momentum (AR(1) phi=0.20). Trend takip stratejileri için."""
    rng = np.random.default_rng(2020)
    n = N_DAYS
    sigma = garch_vol(rng, n, sigma_long=0.022, persistence=0.90, vol_of_vol=0.12)
    mu_drift = 0.0006  # base drift; AR(1) bunu büyütür
    R = np.zeros((n, 3))
    L = np.linalg.cholesky(np.array(CORR_HIGH))
    Z = rng.standard_normal((n, 3)) @ L.T
    phi = 0.20  # autocorrelation (gerçekçi momentum)
    for i in range(3):
        path = np.empty(n)
        prev = 0.0
        s_mult = 1.0 + 0.05 * (i - 1)
        for t in range(n):
            innov = sigma[t] * s_mult * Z[t, i]
            path[t] = mu_drift + phi * prev + innov
            prev = path[t]
        R[:, i] = path
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i].copy()
        r = add_jumps(r, rng, frequency_per_year=3, std_size=0.05)
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.7)
    return out


# =============================================================================
SCENARIOS = {
    "01_bull_steady":            scenario_01_bull_steady,
    "02_bull_parabolic":         scenario_02_bull_parabolic,
    "03_bear_capitulation":      scenario_03_bear_capitulation,
    "04_bear_grinding":          scenario_04_bear_grinding,
    "05_crab_market":            scenario_05_crab_market,
    "06_high_vol_chaos":         scenario_06_high_vol_chaos,
    "07_low_vol_compression":    scenario_07_low_vol_compression,
    "08_regime_switch":          scenario_08_regime_switch_vol,
    "09_flash_crash":            scenario_09_flash_crash,
    "10_black_swan":             scenario_10_black_swan,
    "11_pump_dump_cycles":       scenario_11_pump_dump_cycles,
    "12_news_driven":            scenario_12_news_driven,
    "13_decoupled":              scenario_13_decoupled,
    "14_anti_correlated":        scenario_14_anti_correlated,
    "15_lead_lag":               scenario_15_lead_lag,
    "16_perfect_corr":           scenario_16_perfect_corr,
    "17_accumulation_breakout":  scenario_17_accumulation_breakout,
    "18_distribution_breakdown": scenario_18_distribution_breakdown,
    "19_mean_reverting":         scenario_19_mean_reverting,
    "20_trending_persistent":    scenario_20_trending_persistent,
}


def main():
    print(f"Çıktı dizini: {OUT_DIR.resolve()}")
    print(f"Her senaryo: 3 coin x {N_DAYS} gun ({START_DATE} -> +{N_DAYS}d)")
    print()
    for idx, (name, func) in enumerate(SCENARIOS.items(), 1):
        coin_data = func()
        save_scenario(name, coin_data)
        # özet stat
        kc = coin_data["kapcoin"]
        close = kc[3]
        rets = np.diff(np.log(close))
        total_ret = (close[-1] / close[0] - 1) * 100
        max_dd = ((close - np.maximum.accumulate(close)) / np.maximum.accumulate(close)).min() * 100
        print(
            f"[{idx:>2}/20] {name:<28} "
            f"kapcoin: ret={total_ret:+8.1f}%  "
            f"σ={rets.std() * 100:5.2f}%/d  "
            f"MaxDD={max_dd:+6.1f}%"
        )
    print()
    print("Tamamlandı.")


if __name__ == "__main__":
    main()
