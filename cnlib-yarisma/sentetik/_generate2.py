"""
21-50 numaralı sentetik dataset üretici (toplam 50'ye tamamlar).

Üç ana grup:
  21-30  GERÇEK DATA TABANLI (real_crypto_data/multi_data bootstrap + transforms)
  31-40  KLASİK CHART PATERNLERİ (H&S, double top, wedge, triangle vs.)
  41-50  MİKRO-YAPI / ÇOKLU-VARLIK / MEVSİMSELLİK

Tüm üretim seeded; mevcut Bitcoin (real_crypto_data/kapcoin) datasını DEĞİŞTİRMEDEN
referans olarak kullanır. Çıktı şeması mevcut synthetic_data ile aynı.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from _generate import (
    N_DAYS, START_DATE, OUT_DIR, COIN_START, COIN_VOL_BASE, COINS,
    correlated_normals, garch_vol, add_jumps, piecewise, smooth,
    build_ohlcv, save_scenario, make_returns,
    CORR_NORMAL, CORR_HIGH, CORR_LOW, CORR_NEG,
)

REAL_DIR = Path(__file__).parent.parent / "real_crypto_data"
MULTI_DIR = Path(__file__).parent.parent / "multi_data"
BINANCE_DIR = Path(__file__).parent.parent / "binance_data"


# -----------------------------------------------------------------------------
# Gerçek-data yardımcıları
# -----------------------------------------------------------------------------
def load_real_returns(path):
    """Gerçek parquet'ten log-getiri serisi yükle."""
    df = pd.read_parquet(path)
    return np.diff(np.log(df["Close"].values))


def block_bootstrap(returns, n_target, block_size, rng):
    """Stationary-block bootstrap: rastgele başlangıçlı bloklar birleştir."""
    n_blocks = (n_target // block_size) + 2
    starts = rng.integers(0, len(returns) - block_size + 1, size=n_blocks)
    chunks = [returns[s : s + block_size] for s in starts]
    out = np.concatenate(chunks)
    return out[:n_target]


def correlate_three(rets_three_uncorr, corr, rng):
    """3 bağımsız getiri serisini Cholesky ile hedef korelasyona dönüştür."""
    Z = np.column_stack(rets_three_uncorr)  # (n, 3)
    # Z'leri standardize et
    Zs = (Z - Z.mean(0)) / Z.std(0)
    L = np.linalg.cholesky(np.array(corr))
    Zc = Zs @ L.T
    # Geri ölçek (orijinal volatiliteyi koru)
    return Zc * Z.std(0) + Z.mean(0)


# =============================================================================
# 21-30 GERÇEK DATA TABANLI
# =============================================================================

def scenario_21_real_btc_bootstrap():
    """Gerçek BTC günlük getirilerinden block-bootstrap. Yapı korunur."""
    rng = np.random.default_rng(2121)
    real = load_real_returns(REAL_DIR / "kapcoin-usd_train.parquet")
    out = {}
    for i, coin in enumerate(COINS):
        # Her coin için farklı seed ile bootstrap
        sub_rng = np.random.default_rng(2121 + i * 7)
        r = block_bootstrap(real, N_DAYS, block_size=20, rng=sub_rng)
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.5)
    return out


def scenario_22_real_btc_amplified_bull():
    """Gerçek BTC örüntüsü + günde +0.0015 ek drift. Aşırı boğa."""
    rng = np.random.default_rng(2222)
    real = load_real_returns(REAL_DIR / "kapcoin-usd_train.parquet")
    out = {}
    for i, coin in enumerate(COINS):
        sub_rng = np.random.default_rng(2222 + i * 11)
        r = block_bootstrap(real, N_DAYS, block_size=15, rng=sub_rng)
        r = r + 0.0015 * (1.0 + 0.05 * (i - 1))
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.6)
    return out


def scenario_23_real_btc_inverted():
    """Gerçek BTC getirileri × -1: bull → bear ayna görüntüsü."""
    rng = np.random.default_rng(2323)
    real = load_real_returns(REAL_DIR / "kapcoin-usd_train.parquet")
    out = {}
    for i, coin in enumerate(COINS):
        sub_rng = np.random.default_rng(2323 + i * 13)
        r = block_bootstrap(real, N_DAYS, block_size=25, rng=sub_rng)
        r = -r * (1.0 + 0.03 * (i - 1))  # ters çevir, hafif farklılaştır
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.6)
    return out


def scenario_24_real_btc_volatility_amplified():
    """Gerçek getiriler × 1.6 vol scaling. Aynı örüntü, daha vahşi."""
    rng = np.random.default_rng(2424)
    real = load_real_returns(REAL_DIR / "kapcoin-usd_train.parquet")
    real_centered = real - real.mean()
    out = {}
    for i, coin in enumerate(COINS):
        sub_rng = np.random.default_rng(2424 + i * 17)
        r = block_bootstrap(real_centered, N_DAYS, block_size=20, rng=sub_rng)
        r = r * 1.6 + 0.0008 * (1.0 + 0.05 * (i - 1))
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=2.0)
    return out


def scenario_25_real_btc_compressed():
    """Gerçek getiriler × 0.5: düşük-vol BTC örüntüsü, ayrıştırma testi."""
    rng = np.random.default_rng(2525)
    real = load_real_returns(REAL_DIR / "kapcoin-usd_train.parquet")
    real_centered = real - real.mean()
    out = {}
    for i, coin in enumerate(COINS):
        sub_rng = np.random.default_rng(2525 + i * 19)
        r = block_bootstrap(real_centered, N_DAYS, block_size=30, rng=sub_rng)
        r = r * 0.5 + 0.0006 * (1.0 + 0.05 * (i - 1))
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.3)
    return out


def scenario_26_real_three_coins_mixed():
    """3 coin için 3 farklı gerçek dataset bootstrap'ı (BTC/ETH/SOL benzeri)."""
    rng = np.random.default_rng(2626)
    src_kap = load_real_returns(REAL_DIR / "kapcoin-usd_train.parquet")
    src_met = load_real_returns(REAL_DIR / "metucoin-usd_train.parquet")
    src_tam = load_real_returns(REAL_DIR / "tamcoin-usd_train.parquet")
    out = {}
    for src, coin, seed in [(src_kap, "kapcoin", 2626), (src_met, "metucoin", 2627), (src_tam, "tamcoin", 2628)]:
        sub_rng = np.random.default_rng(seed)
        r = block_bootstrap(src, N_DAYS, block_size=20, rng=sub_rng)
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.6)
    return out


def scenario_27_2017_parabolic_top():
    """2017 BTC tarzı parabolik tepe: 12 ay +1500%, sonra 12 ay -85%."""
    rng = np.random.default_rng(2727)
    n = N_DAYS
    # 365 gün sakin, 365 gün rally, 270 gün parabolik, 360 gün crash, kalan toparlanma
    mu = piecewise(
        n,
        [
            (0.00, 0.19, 0.0010),
            (0.19, 0.38, 0.0040),
            (0.38, 0.52, 0.0090),
            (0.52, 0.71, -0.0080),
            (0.71, 0.85, -0.0020),
            (0.85, 1.00, 0.0015),
        ],
    )
    sigma = piecewise(
        n,
        [
            (0.00, 0.19, 0.025),
            (0.19, 0.38, 0.040),
            (0.38, 0.52, 0.060),
            (0.52, 0.71, 0.055),
            (0.71, 0.85, 0.030),
            (0.85, 1.00, 0.025),
        ],
    )
    mu = smooth(mu, 18)
    sigma = smooth(sigma, 18)
    R = make_returns(rng, [mu, mu * 1.10, mu * 1.05], [sigma, sigma * 1.05, sigma], corr=CORR_HIGH)
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i].copy()
        r = add_jumps(r, rng, frequency_per_year=5, std_size=0.06, asymmetry=-0.1)
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.7)
    return out


def scenario_28_covid_crash_recovery():
    """Mart 2020 tarzı: 14 günde -50%, sonra 18 ay +1000% V-recovery."""
    rng = np.random.default_rng(2828)
    n = N_DAYS
    crash_start = int(0.15 * n)
    crash_end = crash_start + 14
    recovery_end = int(0.55 * n)

    mu = np.full(n, 0.0010)
    sigma = np.full(n, 0.022)

    mu[:crash_start] = 0.0008
    sigma[:crash_start] = 0.020

    mu[crash_start:crash_end] = -0.045  # -50%/14d
    sigma[crash_start:crash_end] = 0.090

    mu[crash_end:recovery_end] = 0.0055  # V-recovery
    sigma[crash_end:recovery_end] = 0.045

    mu[recovery_end:] = 0.0008  # post-recovery cooldown
    sigma[recovery_end:] = 0.025

    mu = smooth(mu, 5)
    sigma = smooth(sigma, 5)

    R = make_returns(rng, [mu, mu * 1.10, mu * 1.05], [sigma, sigma * 1.05, sigma], corr=CORR_HIGH)
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i].copy()
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.8)
    return out


def scenario_29_crypto_winter_grinding():
    """2022 tarzı kripto kışı: Ocak +20% short squeeze + LUNA tarzı 2 collapse + grind."""
    rng = np.random.default_rng(2929)
    n = N_DAYS
    mu = piecewise(
        n,
        [
            (0.00, 0.05, 0.0050),       # short squeeze rally
            (0.05, 0.25, -0.0030),      # initial decline
            (0.25, 0.30, -0.0150),      # LUNA-style collapse 1
            (0.30, 0.50, -0.0010),      # grind down
            (0.50, 0.55, -0.0140),      # FTX-style collapse 2
            (0.55, 0.85, -0.0005),      # bottom grinding
            (0.85, 1.00, 0.0020),       # recovery
        ],
    )
    sigma = piecewise(
        n,
        [
            (0.00, 0.05, 0.030),
            (0.05, 0.25, 0.025),
            (0.25, 0.30, 0.080),
            (0.30, 0.50, 0.022),
            (0.50, 0.55, 0.075),
            (0.55, 0.85, 0.018),
            (0.85, 1.00, 0.022),
        ],
    )
    mu = smooth(mu, 8)
    sigma = smooth(sigma, 8)
    R = make_returns(rng, [mu, mu * 1.05, mu * 1.10], [sigma, sigma * 1.05, sigma * 1.10], corr=CORR_HIGH)
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i].copy()
        r = add_jumps(r, rng, frequency_per_year=8, std_size=0.07, asymmetry=-0.4)
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.7)
    return out


def scenario_30_alt_season_rotation():
    """Alt season: önce kapcoin (BTC) liderlik, sonra para alt'lara akar (met/tam)."""
    rng = np.random.default_rng(3030)
    n = N_DAYS
    # Phase 1 (0-40%): kapcoin lider, alts geride
    # Phase 2 (40-65%): rotation — alts pump, kap sideways
    # Phase 3 (65-100%): hep beraber dağıtım

    mu_kap = piecewise(n, [(0.00, 0.40, 0.0040), (0.40, 0.65, 0.0001), (0.65, 1.00, -0.0020)])
    mu_met = piecewise(n, [(0.00, 0.40, 0.0010), (0.40, 0.65, 0.0080), (0.65, 1.00, -0.0035)])
    mu_tam = piecewise(n, [(0.00, 0.40, 0.0008), (0.40, 0.65, 0.0095), (0.65, 1.00, -0.0040)])

    sig_kap = piecewise(n, [(0.00, 0.40, 0.022), (0.40, 0.65, 0.020), (0.65, 1.00, 0.030)])
    sig_met = piecewise(n, [(0.00, 0.40, 0.028), (0.40, 0.65, 0.045), (0.65, 1.00, 0.040)])
    sig_tam = piecewise(n, [(0.00, 0.40, 0.030), (0.40, 0.65, 0.052), (0.65, 1.00, 0.045)])

    mu_kap, mu_met, mu_tam = smooth(mu_kap, 15), smooth(mu_met, 15), smooth(mu_tam, 15)
    sig_kap, sig_met, sig_tam = smooth(sig_kap, 15), smooth(sig_met, 15), smooth(sig_tam, 15)

    R = make_returns(
        rng,
        [mu_kap, mu_met, mu_tam],
        [sig_kap, sig_met, sig_tam],
        corr=CORR_NORMAL,
    )
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i].copy()
        r = add_jumps(r, rng, frequency_per_year=4, std_size=0.05)
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.6)
    return out


# =============================================================================
# 31-40 KLASİK CHART PATERNLERİ
# =============================================================================

def _pattern_drift(shape, n):
    """Belirli şekildeki yumuşak fiyat trajektörisini günlük drift'e çevirir.
    shape: target log-price path (length n+1)."""
    log_p = np.array(shape)
    return np.diff(log_p)


def scenario_31_head_and_shoulders():
    """Sol omuz, baş, sağ omuz, sonra boyun çizgisini kırma. Klasik dönüş."""
    rng = np.random.default_rng(3131)
    n = N_DAYS
    t = np.linspace(0, 1, n + 1)
    # 5 pik: sol omuz (1.5), baş (2.2), sağ omuz (1.5), sonra düşüş -50%
    shape = np.zeros(n + 1)
    # Build with peaks
    for center, height, width in [
        (0.18, 0.5, 0.05),    # left shoulder
        (0.35, 0.85, 0.07),   # head
        (0.52, 0.5, 0.05),    # right shoulder
    ]:
        shape += height * np.exp(-((t - center) ** 2) / (2 * width ** 2))
    # post-pattern: breakdown -60%
    breakdown_mask = t > 0.60
    shape[breakdown_mask] -= np.linspace(0, 0.85, breakdown_mask.sum())
    # smooth
    shape = smooth(shape, 8)

    drift = _pattern_drift(shape, n)
    sigma = garch_vol(rng, n, sigma_long=0.025, persistence=0.85)
    L = np.linalg.cholesky(np.array(CORR_HIGH))
    Z = rng.standard_normal((n, 3)) @ L.T
    R = np.zeros((n, 3))
    for i in range(3):
        R[:, i] = drift + sigma * Z[:, i]
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i].copy()
        r = add_jumps(r, rng, frequency_per_year=3, std_size=0.05, asymmetry=-0.2)
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.6)
    return out


def scenario_32_inverse_head_and_shoulders():
    """Tabanda ters H&S — boğa dönüşü. Net pozitif sonuç."""
    rng = np.random.default_rng(3232)
    n = N_DAYS
    t = np.linspace(0, 1, n + 1)
    shape = np.zeros(n + 1)
    # Hafif downtrend giriş (0 → -0.30 by t=0.10)
    pre_mask = t < 0.10
    shape[pre_mask] = -0.30 * (t[pre_mask] / 0.10)
    shape[t >= 0.10] = -0.30
    # Inverse H&S 3 trough (boyun çizgisi seviyesi ~ -0.30)
    for center, depth, width in [
        (0.20, 0.30, 0.05),
        (0.37, 0.55, 0.07),
        (0.54, 0.30, 0.05),
    ]:
        shape -= depth * np.exp(-((t - center) ** 2) / (2 * width ** 2))
    # Breakout up: -0.30 → +1.10 (toplam ~+300% logaritmik)
    breakout_mask = t > 0.62
    shape[breakout_mask] = -0.30 + np.linspace(0, 1.40, breakout_mask.sum())
    shape = smooth(shape, 8)

    drift = _pattern_drift(shape, n)
    sigma = garch_vol(rng, n, sigma_long=0.022, persistence=0.85)
    L = np.linalg.cholesky(np.array(CORR_HIGH))
    Z = rng.standard_normal((n, 3)) @ L.T
    R = np.zeros((n, 3))
    for i in range(3):
        R[:, i] = drift + sigma * Z[:, i]
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i].copy()
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.5)
    return out


def scenario_33_double_top():
    """İki eşit zirve, boyun çizgisi kırılır → ayı."""
    rng = np.random.default_rng(3333)
    n = N_DAYS
    t = np.linspace(0, 1, n + 1)
    shape = np.zeros(n + 1)
    peak_h = 0.65
    # Build piecewise linear shape: 0 → 0.65 → 0.45 → 0.65 → 0.45 → -0.50
    # Anchor noktaları
    anchors = [
        (0.00, 0.00),
        (0.30, peak_h),
        (0.42, 0.45),
        (0.55, peak_h),
        (0.65, 0.45),
        (1.00, -0.55),
    ]
    xs, ys = zip(*anchors)
    shape = np.interp(t, xs, ys)
    shape = smooth(shape, 15)

    drift = _pattern_drift(shape, n)
    sigma = garch_vol(rng, n, sigma_long=0.024, persistence=0.85)
    L = np.linalg.cholesky(np.array(CORR_HIGH))
    Z = rng.standard_normal((n, 3)) @ L.T
    R = np.zeros((n, 3))
    for i in range(3):
        R[:, i] = drift + sigma * Z[:, i]
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i].copy()
        r = add_jumps(r, rng, frequency_per_year=3, std_size=0.05, asymmetry=-0.2)
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.6)
    return out


def scenario_34_double_bottom():
    """W tabanı: iki eşit dip, neckline kırılır → boğa."""
    rng = np.random.default_rng(3437)  # seed change for cleaner realization
    n = N_DAYS
    t = np.linspace(0, 1, n + 1)
    bottom_d = -0.55
    anchors = [
        (0.00, 0.00),
        (0.30, bottom_d),
        (0.42, -0.30),
        (0.55, bottom_d),
        (0.65, -0.30),
        (1.00, 1.40),   # daha güçlü breakout
    ]
    xs, ys = zip(*anchors)
    shape = np.interp(t, xs, ys)
    shape = smooth(shape, 15)

    drift = _pattern_drift(shape, n)
    sigma = garch_vol(rng, n, sigma_long=0.025, persistence=0.85)
    L = np.linalg.cholesky(np.array(CORR_HIGH))
    Z = rng.standard_normal((n, 3)) @ L.T
    R = np.zeros((n, 3))
    for i in range(3):
        R[:, i] = drift + sigma * Z[:, i]
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i].copy()
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.6)
    return out


def scenario_35_cup_and_handle():
    """Yumuşak U formu (cup) + küçük kulp + kırılım."""
    rng = np.random.default_rng(3535)
    n = N_DAYS
    t = np.linspace(0, 1, n + 1)
    shape = np.zeros(n + 1)
    # Cup: U-shape from 0 to 0.55
    cup_mask = t < 0.55
    cup_t = t[cup_mask] / 0.55
    shape[cup_mask] = -0.5 * (1 - (2 * cup_t - 1) ** 2)  # parabolic U, depth -0.5
    # Handle: small dip 0.55-0.70
    handle_mask = (t >= 0.55) & (t < 0.70)
    handle_t = (t[handle_mask] - 0.55) / 0.15
    shape[handle_mask] = -0.15 * np.sin(np.pi * handle_t)
    # Breakout 0.70 onwards
    bo_mask = t >= 0.70
    shape[bo_mask] = np.linspace(0, 1.2, bo_mask.sum())
    shape = smooth(shape, 15)

    drift = _pattern_drift(shape, n)
    sigma = garch_vol(rng, n, sigma_long=0.022, persistence=0.85)
    L = np.linalg.cholesky(np.array(CORR_HIGH))
    Z = rng.standard_normal((n, 3)) @ L.T
    R = np.zeros((n, 3))
    for i in range(3):
        R[:, i] = drift + sigma * Z[:, i]
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i].copy()
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.5)
    return out


def scenario_36_ascending_triangle():
    """Düz tepe + yükselen dipler → yukarı kırılım."""
    rng = np.random.default_rng(3636)
    n = N_DAYS
    t = np.linspace(0, 1, n + 1)
    shape = np.zeros(n + 1)
    # Triangle phase 0-0.65: oscillation between rising lows and flat top (~0.5)
    flat_top = 0.5
    in_tri = t < 0.65
    tri_t = t[in_tri] / 0.65
    rising_low = -0.2 + tri_t * 0.6  # rising from -0.2 to 0.4
    osc = np.sin(2 * np.pi * 5 * tri_t) * 0.5 * (1 - tri_t)  # oscillation
    shape[in_tri] = (flat_top + rising_low) / 2 + osc
    # Breakout up 0.65 onwards
    bo_mask = t >= 0.65
    shape[bo_mask] = flat_top + np.linspace(0, 1.0, bo_mask.sum())
    shape = smooth(shape, 10)

    drift = _pattern_drift(shape, n)
    sigma = garch_vol(rng, n, sigma_long=0.022, persistence=0.83)
    L = np.linalg.cholesky(np.array(CORR_HIGH))
    Z = rng.standard_normal((n, 3)) @ L.T
    R = np.zeros((n, 3))
    for i in range(3):
        R[:, i] = drift + sigma * Z[:, i]
    out = {}
    for i, coin in enumerate(COINS):
        out[coin] = build_ohlcv(coin, R[:, i], rng, intraday_amp=1.6)
    return out


def scenario_37_descending_triangle():
    """Düz dip + alçalan tepeler → aşağı kırılım."""
    rng = np.random.default_rng(3737)
    n = N_DAYS
    t = np.linspace(0, 1, n + 1)
    shape = np.zeros(n + 1)
    flat_bottom = -0.4
    in_tri = t < 0.65
    tri_t = t[in_tri] / 0.65
    falling_high = 0.4 - tri_t * 0.6  # falling from 0.4 to -0.2
    osc = np.sin(2 * np.pi * 5 * tri_t) * 0.5 * (1 - tri_t)
    shape[in_tri] = (flat_bottom + falling_high) / 2 + osc
    bd_mask = t >= 0.65
    shape[bd_mask] = flat_bottom - np.linspace(0, 0.9, bd_mask.sum())
    shape = smooth(shape, 10)

    drift = _pattern_drift(shape, n)
    sigma = garch_vol(rng, n, sigma_long=0.024, persistence=0.85)
    L = np.linalg.cholesky(np.array(CORR_HIGH))
    Z = rng.standard_normal((n, 3)) @ L.T
    R = np.zeros((n, 3))
    for i in range(3):
        R[:, i] = drift + sigma * Z[:, i]
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i].copy()
        r = add_jumps(r, rng, frequency_per_year=2, std_size=0.05, asymmetry=-0.3)
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.6)
    return out


def scenario_38_symmetric_triangle():
    """Yakınsayan trendlines, kırılım yönü rastgele (burada yukarı)."""
    rng = np.random.default_rng(3838)
    n = N_DAYS
    t = np.linspace(0, 1, n + 1)
    shape = np.zeros(n + 1)
    in_tri = t < 0.65
    tri_t = t[in_tri] / 0.65
    amplitude = 0.5 * (1 - tri_t)  # daralan menzil
    osc = np.sin(2 * np.pi * 6 * tri_t) * amplitude
    shape[in_tri] = osc
    bo_mask = t >= 0.65
    shape[bo_mask] = np.linspace(0, 1.1, bo_mask.sum())
    shape = smooth(shape, 8)

    drift = _pattern_drift(shape, n)
    sigma = garch_vol(rng, n, sigma_long=0.022, persistence=0.83)
    L = np.linalg.cholesky(np.array(CORR_HIGH))
    Z = rng.standard_normal((n, 3)) @ L.T
    R = np.zeros((n, 3))
    for i in range(3):
        R[:, i] = drift + sigma * Z[:, i]
    out = {}
    for i, coin in enumerate(COINS):
        out[coin] = build_ohlcv(coin, R[:, i], rng, intraday_amp=1.5)
    return out


def scenario_39_falling_wedge():
    """Daralan aşağı eğimli kanal → boğa kırılımı."""
    rng = np.random.default_rng(3939)
    n = N_DAYS
    t = np.linspace(0, 1, n + 1)
    shape = np.zeros(n + 1)
    in_w = t < 0.65
    wt = t[in_w] / 0.65
    midline = -0.5 * wt  # downward sloping
    amplitude = 0.4 * (1 - wt)  # narrowing
    osc = np.sin(2 * np.pi * 5 * wt) * amplitude
    shape[in_w] = midline + osc
    bo_mask = t >= 0.65
    shape[bo_mask] = -0.5 + np.linspace(0, 1.4, bo_mask.sum())
    shape = smooth(shape, 10)

    drift = _pattern_drift(shape, n)
    sigma = garch_vol(rng, n, sigma_long=0.025, persistence=0.85)
    L = np.linalg.cholesky(np.array(CORR_HIGH))
    Z = rng.standard_normal((n, 3)) @ L.T
    R = np.zeros((n, 3))
    for i in range(3):
        R[:, i] = drift + sigma * Z[:, i]
    out = {}
    for i, coin in enumerate(COINS):
        out[coin] = build_ohlcv(coin, R[:, i], rng, intraday_amp=1.6)
    return out


def scenario_40_rising_wedge():
    """Daralan yukarı eğimli kanal → ayı kırılımı."""
    rng = np.random.default_rng(4040)
    n = N_DAYS
    t = np.linspace(0, 1, n + 1)
    shape = np.zeros(n + 1)
    in_w = t < 0.65
    wt = t[in_w] / 0.65
    midline = 0.6 * wt  # upward sloping
    amplitude = 0.35 * (1 - wt)
    osc = np.sin(2 * np.pi * 5 * wt) * amplitude
    shape[in_w] = midline + osc
    bd_mask = t >= 0.65
    shape[bd_mask] = 0.6 - np.linspace(0, 1.2, bd_mask.sum())
    shape = smooth(shape, 10)

    drift = _pattern_drift(shape, n)
    sigma = garch_vol(rng, n, sigma_long=0.025, persistence=0.85)
    L = np.linalg.cholesky(np.array(CORR_HIGH))
    Z = rng.standard_normal((n, 3)) @ L.T
    R = np.zeros((n, 3))
    for i in range(3):
        R[:, i] = drift + sigma * Z[:, i]
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i].copy()
        r = add_jumps(r, rng, frequency_per_year=3, std_size=0.05, asymmetry=-0.3)
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.6)
    return out


# =============================================================================
# 41-50 MİKRO-YAPI / ÇOKLU-VARLIK / MEVSİMSELLİK
# =============================================================================

def scenario_41_volume_leads_price():
    """Volume önden gelir, fiyat 2-5 gün gecikmeli takip eder. Predictive volume."""
    rng = np.random.default_rng(4141)
    n = N_DAYS
    sigma = garch_vol(rng, n, sigma_long=0.022, persistence=0.85)
    mu = np.full(n, 0.0010)
    R = make_returns(rng, [mu, mu * 1.05, mu * 0.95], [sigma, sigma * 1.05, sigma], corr=CORR_HIGH)
    out = {}
    dates = pd.date_range(start=START_DATE, periods=n, freq="D")
    for i, coin in enumerate(COINS):
        r = R[:, i].copy()
        # Build OHLC normally
        o, h, l, c, v = build_ohlcv(coin, r, rng, intraday_amp=1.6)
        # Volume'u 3 gün geri kaydır (volume liderlik etsin)
        # Forward-shift: today's volume = volume that "predicts" today (so come from future)
        v_shifted = np.concatenate([v[3:], v[-3:] * rng.uniform(0.8, 1.2, size=3)])
        # Boost volume on days before big moves
        abs_ret = np.abs(r)
        for t_idx in range(3, n):
            if abs_ret[t_idx] > 0.04:
                v_shifted[t_idx - 3] *= 1.8  # spike volume 3 days before big move
        out[coin] = (o, h, l, c, v_shifted)
    return out


def scenario_42_volume_climax():
    """Volume tepelerde/diplerde patlar (capitulation/euphoria volume)."""
    rng = np.random.default_rng(4242)
    n = N_DAYS
    mu = piecewise(n, [(0.0, 0.4, 0.0030), (0.4, 0.55, 0.0080), (0.55, 0.70, -0.0080), (0.70, 1.00, 0.0010)])
    sigma = garch_vol(rng, n, sigma_long=0.024, persistence=0.85)
    mu = smooth(mu, 15)
    R = make_returns(rng, [mu, mu * 1.05, mu * 0.95], [sigma, sigma * 1.05, sigma], corr=CORR_HIGH)
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i].copy()
        o, h, l, c, v = build_ohlcv(coin, r, rng, intraday_amp=1.7)
        # Top region (~0.55) ve dip (~0.70) civarına volume spike
        t_arr = np.arange(n)
        top_boost = 1 + 2.5 * np.exp(-((t_arr - 0.55 * n) ** 2) / (2 * (n * 0.02) ** 2))
        bot_boost = 1 + 3.0 * np.exp(-((t_arr - 0.70 * n) ** 2) / (2 * (n * 0.02) ** 2))
        v = v * top_boost * bot_boost
        out[coin] = (o, h, l, c, v)
    return out


def scenario_43_gap_continuation():
    """Overnight gap'ler aynı yönde devam eder (momentum gap'leri)."""
    rng = np.random.default_rng(4343)
    n = N_DAYS
    sigma = garch_vol(rng, n, sigma_long=0.022, persistence=0.85)
    mu = np.full(n, 0.0008)
    R = make_returns(rng, [mu, mu * 1.05, mu * 0.95], [sigma, sigma * 1.05, sigma], corr=CORR_HIGH)
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i].copy()
        # ZERO-mean gap'ler — yön rastgele, büyüklük 1-4%
        gap_days = rng.random(n) < 0.15
        gap_signs = rng.choice([-1, 1], size=n)
        gap_mag = rng.uniform(0.015, 0.045, size=n)
        gap_values = gap_signs * gap_mag
        r[gap_days] += gap_values[gap_days]
        # Continuation: gap'ten sonraki 1-2 gün aynı YÖNDE küçük takip
        cont_idx = np.where(gap_days)[0]
        for t_idx in cont_idx:
            sign = gap_signs[t_idx]
            for offset in [1, 2]:
                if t_idx + offset < n:
                    r[t_idx + offset] += sign * rng.uniform(0.005, 0.015)
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.7)
    return out


def scenario_44_gap_fill():
    """Gap'ler ortalama 1-3 gün içinde dolar (mean-reversion gap'leri)."""
    rng = np.random.default_rng(4444)
    n = N_DAYS
    sigma = garch_vol(rng, n, sigma_long=0.020, persistence=0.85)
    mu = np.full(n, 0.0006)
    R = make_returns(rng, [mu, mu * 1.05, mu * 0.95], [sigma, sigma * 1.05, sigma], corr=CORR_HIGH)
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i].copy()
        gap_days = rng.random(n) < 0.12
        gap_size = rng.normal(0.0, 0.030, size=n)
        gap_size[gap_size == 0] = 0.001
        r[gap_days] += gap_size[gap_days]
        # Fill in next 1-3 days
        for t_idx in np.where(gap_days)[0]:
            fill_horizon = min(rng.integers(1, 4), n - t_idx - 1)
            if fill_horizon > 0:
                fill_per_day = -gap_size[t_idx] * 0.8 / fill_horizon
                r[t_idx + 1 : t_idx + 1 + fill_horizon] += fill_per_day
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.5)
    return out


def scenario_45_cross_asset_lead_lag_volume():
    """tamcoin volume'u kapcoin & metucoin için 5 gün önden sinyal verir."""
    rng = np.random.default_rng(4545)
    n = N_DAYS
    sigma = garch_vol(rng, n, sigma_long=0.024, persistence=0.85)
    # tamcoin'in büyük hamleleri 5 gün sonra kap/met'e yansır
    mu_tam = np.zeros(n)
    # rastgele büyük hamleler tamcoin'de
    move_days = rng.random(n) < 0.06
    move_signs = rng.choice([-1, 1], size=n)
    move_sizes = rng.uniform(0.04, 0.10, size=n)
    mu_tam[move_days] = move_signs[move_days] * move_sizes[move_days]
    # propagate to kap/met with lag 5
    mu_kap = np.zeros(n)
    mu_met = np.zeros(n)
    mu_kap[5:] = mu_tam[:-5] * 0.6
    mu_met[5:] = mu_tam[:-5] * 0.7
    base_mu = 0.0005

    L = np.linalg.cholesky(np.array(CORR_LOW))
    Z = rng.standard_normal((n, 3)) @ L.T
    R = np.zeros((n, 3))
    R[:, 0] = base_mu + mu_kap + sigma * Z[:, 0]
    R[:, 1] = base_mu + mu_met + sigma * Z[:, 1]
    R[:, 2] = base_mu + mu_tam + sigma * Z[:, 2]

    out = {}
    for i, coin in enumerate(COINS):
        o, h, l, c, v = build_ohlcv(coin, R[:, i], rng, intraday_amp=1.7)
        if coin == "tamcoin":
            # Volume tamcoin'de önden patlar
            v = v.copy()
            for t_idx in np.where(move_days)[0]:
                if t_idx >= 0 and t_idx < n:
                    v[t_idx] *= 2.5
        out[coin] = (o, h, l, c, v)
    return out


def scenario_46_flight_to_quality():
    """Crash'te kapcoin (BTC tarzı) görece holds, met/tam çok daha kötü."""
    rng = np.random.default_rng(4646)
    n = N_DAYS
    # 4 ana crash dönemi
    mu_kap = np.full(n, 0.0008)
    mu_met = np.full(n, 0.0010)
    mu_tam = np.full(n, 0.0012)
    sig_kap = np.full(n, 0.020)
    sig_met = np.full(n, 0.026)
    sig_tam = np.full(n, 0.030)

    for crash_start_frac, crash_len in [(0.20, 25), (0.45, 20), (0.70, 30), (0.88, 15)]:
        s = int(crash_start_frac * n)
        e = min(s + crash_len, n)
        mu_kap[s:e] = -0.012   # mild
        mu_met[s:e] = -0.025   # heavy
        mu_tam[s:e] = -0.035   # very heavy
        sig_kap[s:e] = 0.040
        sig_met[s:e] = 0.060
        sig_tam[s:e] = 0.075

    R = make_returns(
        rng,
        [mu_kap, mu_met, mu_tam],
        [sig_kap, sig_met, sig_tam],
        corr=CORR_HIGH,
    )
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i].copy()
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.7)
    return out


def scenario_47_btc_dominance_cycles():
    """5 dominans döngüsü: kap → alt → kap → alt → kap. Cycle başına net ~0."""
    rng = np.random.default_rng(4747)
    n = N_DAYS
    n_cycles = 5
    cycle_len = n // n_cycles
    mu_kap = np.zeros(n)
    mu_met = np.zeros(n)
    mu_tam = np.zeros(n)
    for k in range(n_cycles):
        s = k * cycle_len
        mid = s + cycle_len // 2
        e = s + cycle_len
        if k % 2 == 0:  # kap dominant: kap yukarı, alt'lar aşağı (cycle nötr)
            mu_kap[s:mid] = 0.0040
            mu_met[s:mid] = -0.0010
            mu_tam[s:mid] = -0.0015
            mu_kap[mid:e] = -0.0035   # düzeltme
            mu_met[mid:e] = 0.0010
            mu_tam[mid:e] = 0.0010
        else:  # alt dominant: kap düz, alt'lar yukarı (sonra düzeltme)
            mu_kap[s:mid] = 0.0005
            mu_met[s:mid] = 0.0050
            mu_tam[s:mid] = 0.0060
            mu_kap[mid:e] = -0.0005
            mu_met[mid:e] = -0.0045
            mu_tam[mid:e] = -0.0055
    mu_kap = smooth(mu_kap, 10)
    mu_met = smooth(mu_met, 10)
    mu_tam = smooth(mu_tam, 10)
    sigma = garch_vol(rng, n, sigma_long=0.024, persistence=0.85)
    R = make_returns(rng, [mu_kap, mu_met, mu_tam], [sigma, sigma * 1.10, sigma * 1.15], corr=CORR_NORMAL)
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i].copy()
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.6)
    return out


def scenario_48_q4_seasonal_rally():
    """Yıllık Q4 rally + Q1 ayı + Q2/Q3 düşük vol. 5+ yıllık döngü."""
    rng = np.random.default_rng(4848)
    n = N_DAYS
    dates = pd.date_range(start=START_DATE, periods=n, freq="D")
    months = dates.month.values

    mu = np.zeros(n)
    sigma = np.zeros(n)
    for t_idx, m in enumerate(months):
        if m in (10, 11, 12):  # Q4 rally
            mu[t_idx] = 0.0050
            sigma[t_idx] = 0.030
        elif m in (1, 2, 3):  # Q1 hangover
            mu[t_idx] = -0.0025
            sigma[t_idx] = 0.030
        elif m in (4, 5, 6):  # Q2 recovery
            mu[t_idx] = 0.0010
            sigma[t_idx] = 0.020
        else:  # Q3 chop
            mu[t_idx] = 0.0005
            sigma[t_idx] = 0.018

    mu = smooth(mu, 5)
    sigma = smooth(sigma, 5)
    R = make_returns(rng, [mu, mu * 1.05, mu * 0.95], [sigma, sigma * 1.05, sigma], corr=CORR_HIGH)
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i].copy()
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.6)
    return out


def scenario_49_halving_cycle():
    """4-yıllık halving döngüsü: accumulation → markup → distribution → markdown.
    Cycle başına net hafif pozitif (kripto sekuler trendi)."""
    rng = np.random.default_rng(4949)
    n = N_DAYS
    cycle = 1460  # 4 yıl
    mu = np.zeros(n)
    sigma = np.zeros(n)
    for t_idx in range(n):
        phase = (t_idx % cycle) / cycle  # 0..1
        if phase < 0.25:        # accumulation
            mu[t_idx] = 0.0003
            sigma[t_idx] = 0.018
        elif phase < 0.50:      # markup (post-halving rally)
            mu[t_idx] = 0.0030
            sigma[t_idx] = 0.030
        elif phase < 0.65:      # parabolic blow-off
            mu[t_idx] = 0.0055
            sigma[t_idx] = 0.045
        elif phase < 0.85:      # markdown / bear
            mu[t_idx] = -0.0040
            sigma[t_idx] = 0.040
        else:                   # bottom
            mu[t_idx] = -0.0008
            sigma[t_idx] = 0.022

    mu = smooth(mu, 25)
    sigma = smooth(sigma, 25)
    R = make_returns(rng, [mu, mu * 1.05, mu * 1.10], [sigma, sigma * 1.05, sigma * 1.10], corr=CORR_HIGH)
    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i].copy()
        r = add_jumps(r, rng, frequency_per_year=4, std_size=0.06, asymmetry=-0.1)
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.7)
    return out


def scenario_50_grand_mixed_stress_test():
    """Final stress test: tüm rejimler iç içe — bull, crash, sideways, parabolic, bear, recovery."""
    rng = np.random.default_rng(5050)
    n = N_DAYS
    # 6 segment, her biri farklı rejim
    seg = n // 6

    mu = np.zeros(n)
    sigma = np.zeros(n)

    # 1: Steady bull
    mu[0:seg] = 0.0025
    sigma[0:seg] = 0.020
    # 2: Flash crashes within bull
    mu[seg:2*seg] = 0.0015
    sigma[seg:2*seg] = 0.030
    # 3: Sideways chop
    mu[2*seg:3*seg] = 0.0001
    sigma[2*seg:3*seg] = 0.025
    # 4: Parabolic
    mu[3*seg:4*seg] = 0.0080
    sigma[3*seg:4*seg] = 0.045
    # 5: Crash
    mu[4*seg:5*seg] = -0.0070
    sigma[4*seg:5*seg] = 0.055
    # 6: Recovery
    mu[5*seg:] = 0.0030
    sigma[5*seg:] = 0.030

    mu = smooth(mu, 12)
    sigma = smooth(sigma, 12)
    R = make_returns(rng, [mu, mu * 1.05, mu * 1.10], [sigma, sigma * 1.05, sigma * 1.10], corr=CORR_HIGH)

    # 5 random flash crash günü segment 2'de
    crash_days = np.sort(rng.choice(np.arange(seg + 10, 2 * seg - 10), size=5, replace=False))
    for d in crash_days:
        R[d, :] += rng.uniform(-0.20, -0.12, size=3)
    # 3 black swan günü segment 4'te
    swan_days = np.sort(rng.choice(np.arange(3 * seg + 10, 4 * seg - 10), size=3, replace=False))
    for d in swan_days:
        R[d, :] += rng.uniform(-0.15, 0.15, size=3)

    out = {}
    for i, coin in enumerate(COINS):
        r = R[:, i].copy()
        r = add_jumps(r, rng, frequency_per_year=6, std_size=0.06, asymmetry=-0.15)
        out[coin] = build_ohlcv(coin, r, rng, intraday_amp=1.8)
    return out


# =============================================================================
SCENARIOS_2 = {
    "21_real_btc_bootstrap":         scenario_21_real_btc_bootstrap,
    "22_real_btc_amplified_bull":    scenario_22_real_btc_amplified_bull,
    "23_real_btc_inverted":          scenario_23_real_btc_inverted,
    "24_real_btc_vol_amplified":     scenario_24_real_btc_volatility_amplified,
    "25_real_btc_compressed":        scenario_25_real_btc_compressed,
    "26_real_three_coins_mixed":     scenario_26_real_three_coins_mixed,
    "27_2017_parabolic_top":         scenario_27_2017_parabolic_top,
    "28_covid_crash_recovery":       scenario_28_covid_crash_recovery,
    "29_crypto_winter_grinding":     scenario_29_crypto_winter_grinding,
    "30_alt_season_rotation":        scenario_30_alt_season_rotation,
    "31_head_and_shoulders":         scenario_31_head_and_shoulders,
    "32_inverse_head_and_shoulders": scenario_32_inverse_head_and_shoulders,
    "33_double_top":                 scenario_33_double_top,
    "34_double_bottom":              scenario_34_double_bottom,
    "35_cup_and_handle":             scenario_35_cup_and_handle,
    "36_ascending_triangle":         scenario_36_ascending_triangle,
    "37_descending_triangle":        scenario_37_descending_triangle,
    "38_symmetric_triangle":         scenario_38_symmetric_triangle,
    "39_falling_wedge":              scenario_39_falling_wedge,
    "40_rising_wedge":               scenario_40_rising_wedge,
    "41_volume_leads_price":         scenario_41_volume_leads_price,
    "42_volume_climax":              scenario_42_volume_climax,
    "43_gap_continuation":           scenario_43_gap_continuation,
    "44_gap_fill":                   scenario_44_gap_fill,
    "45_lead_lag_volume":            scenario_45_cross_asset_lead_lag_volume,
    "46_flight_to_quality":          scenario_46_flight_to_quality,
    "47_btc_dominance_cycles":       scenario_47_btc_dominance_cycles,
    "48_q4_seasonal_rally":          scenario_48_q4_seasonal_rally,
    "49_halving_cycle":              scenario_49_halving_cycle,
    "50_grand_stress_test":          scenario_50_grand_mixed_stress_test,
}


def main():
    print(f"Cikti dizini: {OUT_DIR.resolve()}")
    print(f"30 yeni senaryo (21-50). Mevcut Bitcoin (real_crypto_data) referans olarak kullaniliyor.")
    print()
    for idx, (name, func) in enumerate(SCENARIOS_2.items(), 21):
        coin_data = func()
        save_scenario(name, coin_data)
        kc = coin_data["kapcoin"]
        close = kc[3]
        rets = np.diff(np.log(close))
        total_ret = (close[-1] / close[0] - 1) * 100
        max_dd = ((close - np.maximum.accumulate(close)) / np.maximum.accumulate(close)).min() * 100
        print(
            f"[{idx:>2}/50] {name:<32} "
            f"kap: ret={total_ret:+9.1f}%  "
            f"sigma={rets.std() * 100:5.2f}%/d  "
            f"MaxDD={max_dd:+6.1f}%"
        )
    print()
    print("Tamamlandi.")


if __name__ == "__main__":
    main()
