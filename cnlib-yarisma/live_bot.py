"""
Live fake-trade bot — Binance Futures Testnet.

Her 5 dakikada bir loop:
  1. Son 100 saatlik candle (BTC/ETH/SOL) futures public endpoint
  2. Adaptive V3 predict() çağır (strategy.Strategy)
  3. Mevcut pozisyonları testnet'ten çek
  4. Decision-diff uygula: close-then-open pattern

Güvenlik:
  - Sadece isolated margin
  - Her iteration'da max 1 işlem per coin
  - Order hata yakalar, botu durdurmaz
  - 6 saat veya Ctrl+C'ye kadar

Hedef: Adaptive V3'ün live Binance Futures testnet'te strateji
hayatta kalırlığını doğrulamak.

Kullanım:
    python live_bot.py [duration_hours]
"""
from __future__ import annotations

import hashlib
import hmac
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from strategy import Strategy
from cnlib.base_strategy import COINS


load_dotenv()
API_KEY = os.getenv("BINANCE_FUTURES_API_KEY")
API_SECRET = os.getenv("BINANCE_FUTURES_API_SECRET")

# Testnet endpoints
BASE = "https://testnet.binancefuture.com"
FAPI = f"{BASE}/fapi"
PUBLIC_FAPI = "https://fapi.binance.com/fapi"  # kline public MainNet aynı schema

# Coin mapping: cnlib slot → Binance symbol
SYMBOLS = {
    "kapcoin-usd_train":  "BTCUSDT",
    "metucoin-usd_train": "ETHUSDT",
    "tamcoin-usd_train":  "SOLUSDT",
}

LOG_FILE = Path(__file__).parent / "live_bot_log.txt"
STATE_FILE = Path(__file__).parent / "live_bot_state.json"


def log(msg: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------
# Binance helpers
# ------------------------------------------------------------------

def _sign(params: dict) -> dict:
    params = dict(params)
    params["timestamp"] = int(time.time() * 1000)
    params["recvWindow"] = 10000
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    params["signature"] = sig
    return params


def _headers() -> dict:
    return {"X-MBX-APIKEY": API_KEY}


def get_account() -> dict:
    r = requests.get(f"{FAPI}/v2/account", params=_sign({}), headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json()


def get_positions() -> dict[str, dict]:
    """symbol → position dict."""
    r = requests.get(f"{FAPI}/v2/positionRisk", params=_sign({}), headers=_headers(), timeout=10)
    r.raise_for_status()
    out = {}
    for p in r.json():
        amt = float(p.get("positionAmt", 0))
        if amt != 0:
            out[p["symbol"]] = {
                "amount":    amt,
                "entry":     float(p.get("entryPrice", 0)),
                "pnl":       float(p.get("unRealizedProfit", 0)),
                "margin":    float(p.get("isolatedMargin", 0) or p.get("initialMargin", 0)),
                "leverage":  int(p.get("leverage", 1)),
            }
    return out


def fetch_klines(symbol: str, interval: str = "1h", limit: int = 120) -> pd.DataFrame:
    r = requests.get(
        f"{PUBLIC_FAPI}/v1/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "OpenTime", "Open", "High", "Low", "Close", "Volume",
        "CloseTime", "qav", "trades", "tbbv", "tbqv", "ign",
    ])
    df["Date"] = pd.to_datetime(df["OpenTime"], unit="ms")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = df[c].astype(float)
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]


# ------------------------------------------------------------------
# Symbol info (step sizes)
# ------------------------------------------------------------------

_SYMBOL_FILTERS: dict[str, dict] = {}


def load_symbol_filters() -> None:
    r = requests.get(f"{FAPI}/v1/exchangeInfo", timeout=15)
    r.raise_for_status()
    info = r.json()
    for s in info.get("symbols", []):
        sym = s["symbol"]
        if sym not in SYMBOLS.values():
            continue
        step_size = None
        min_qty = None
        min_notional = None
        for f in s.get("filters", []):
            if f["filterType"] == "LOT_SIZE":
                step_size = float(f["stepSize"])
                min_qty = float(f["minQty"])
            elif f["filterType"] == "MIN_NOTIONAL":
                min_notional = float(f.get("notional", 0))
        _SYMBOL_FILTERS[sym] = {
            "step_size":    step_size,
            "min_qty":      min_qty,
            "min_notional": min_notional,
            "price_precision": s.get("pricePrecision", 2),
            "qty_precision":   s.get("quantityPrecision", 3),
        }


def round_qty(symbol: str, qty: float) -> float:
    f = _SYMBOL_FILTERS.get(symbol, {})
    step = f.get("step_size", 0.001)
    rounded = round(qty / step) * step
    return round(rounded, f.get("qty_precision", 3))


# ------------------------------------------------------------------
# Trading actions
# ------------------------------------------------------------------

def set_leverage(symbol: str, leverage: int) -> None:
    try:
        requests.post(
            f"{FAPI}/v1/leverage",
            params=_sign({"symbol": symbol, "leverage": leverage}),
            headers=_headers(),
            timeout=10,
        )
    except Exception as e:
        log(f"  set_leverage({symbol}, {leverage}) HATA: {e}")


def set_isolated(symbol: str) -> None:
    try:
        requests.post(
            f"{FAPI}/v1/marginType",
            params=_sign({"symbol": symbol, "marginType": "ISOLATED"}),
            headers=_headers(),
            timeout=10,
        )
    except Exception:
        pass  # zaten isolated olabilir


def market_order(symbol: str, side: str, quantity: float, reduce_only: bool = False) -> dict:
    params = {
        "symbol":   symbol,
        "side":     side,        # BUY or SELL
        "type":     "MARKET",
        "quantity": quantity,
    }
    if reduce_only:
        params["reduceOnly"] = "true"
    r = requests.post(f"{FAPI}/v1/order", params=_sign(params), headers=_headers(), timeout=10)
    return r.json() if r.status_code == 200 else {"error": r.text, "status": r.status_code}


def close_position(symbol: str, pos: dict) -> None:
    amt = pos["amount"]
    side = "SELL" if amt > 0 else "BUY"
    qty = abs(amt)
    result = market_order(symbol, side, qty, reduce_only=True)
    if "error" in result:
        log(f"  CLOSE {symbol} HATA: {result}")
    else:
        log(f"  CLOSE {symbol} {side} qty={qty} -> OK")


# ------------------------------------------------------------------
# Strategy integration
# ------------------------------------------------------------------

def build_data_dict(hist_limit: int = 100) -> dict[str, pd.DataFrame]:
    """Son N saatlik candle'ı 3 coin için çek."""
    data = {}
    for slot, symbol in SYMBOLS.items():
        df = fetch_klines(symbol, "1h", limit=hist_limit)
        data[slot] = df
    # Eşit uzunluk
    min_len = min(len(d) for d in data.values())
    data = {k: v.iloc[-min_len:].reset_index(drop=True) for k, v in data.items()}
    return data


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

def main(duration_hours: float = 6.0, loop_seconds: int = 300) -> None:
    if not API_KEY or not API_SECRET:
        log("HATA: .env icinde BINANCE_FUTURES_API_KEY/SECRET yok")
        return

    log("=" * 70)
    log("LIVE BOT BASLATILIYOR")
    log(f"  Sure: {duration_hours} saat, interval: {loop_seconds}s")
    log("=" * 70)

    load_symbol_filters()
    log(f"Symbol filters: {list(_SYMBOL_FILTERS.keys())}")

    # Isolated margin + leverage 10 default
    for symbol in SYMBOLS.values():
        set_isolated(symbol)
        set_leverage(symbol, 10)
    log("Leverage 10x + isolated margin ayarlandi.")

    # Initial balance
    acct = get_account()
    start_balance = float(acct["totalWalletBalance"])
    log(f"Baslangic balance: ${start_balance:,.2f}")

    # Stratejinin candle_index'ini saatlik counter ile yürüteceğiz
    strategy = Strategy()
    # Historical data yükleme — get_data() cnlib training data'sını yüklüyor;
    # biz live için manuel data dict geçireceğiz, ama cnlib'in _full_data
    # alanını doldurmak lazım ki strategy.candle_index ile self.coin_data erişilsin.
    # Basit: build_data_dict her seferinde fresh data veriyor, strategy.predict()
    # bunu parametre olarak alıyor. candle_index'i manuel sayıyoruz.

    start_time = time.time()
    end_time = start_time + duration_hours * 3600
    iteration = 0
    candle_idx = 100  # bot warmup'tan sonra başlıyor

    while time.time() < end_time:
        iteration += 1
        elapsed = (time.time() - start_time) / 3600
        log(f"--- Iter #{iteration} (t={elapsed:.2f}h) ---")

        try:
            # Data
            data = build_data_dict(hist_limit=100)

            # Strategy: bu internal candle_index _full_data kullandığı için
            # manuel sayıyoruz; predict() parametre alıyor, OK.
            strategy.candle_index = candle_idx
            # predict() data'yı direkt kullanır, _full_data'ya gerek yok
            decisions = strategy.predict(data)

            # Regime log
            regime = strategy.current_regime
            ac = strategy._last_autocorr
            vol = strategy._last_vol
            log(f"  Regime={regime}  autocorr={ac:+.3f}  vol={vol*100:.2f}%")

            # Current positions
            positions = get_positions()

            # Her coin için karar
            for dec in decisions:
                slot = dec["coin"]
                symbol = SYMBOLS[slot]
                sig = dec["signal"]
                alloc = dec["allocation"]
                lev = dec["leverage"]

                current_pos = positions.get(symbol)
                current_side = None
                if current_pos:
                    current_side = 1 if current_pos["amount"] > 0 else -1

                # Decision logic
                if sig == 0:
                    if current_pos:
                        close_position(symbol, current_pos)
                    continue

                if current_pos is None:
                    # New position
                    set_leverage(symbol, lev)
                    # Current balance
                    acct = get_account()
                    balance = float(acct["availableBalance"])
                    notional = balance * alloc * lev  # leverage × alloc
                    price = float(data[slot]["Close"].iloc[-1])
                    qty = round_qty(symbol, notional / price)
                    if qty <= 0:
                        log(f"  {symbol} qty=0 skip")
                        continue
                    side = "BUY" if sig == 1 else "SELL"
                    result = market_order(symbol, side, qty)
                    if "error" in result:
                        log(f"  OPEN {symbol} {side} qty={qty} HATA: {result}")
                    else:
                        log(f"  OPEN {symbol} {side} qty={qty} lev={lev} alloc={alloc:.2f} -> orderId={result.get('orderId')}")
                elif current_side != sig:
                    # Flip: close + open
                    close_position(symbol, current_pos)
                    time.sleep(0.5)
                    set_leverage(symbol, lev)
                    acct = get_account()
                    balance = float(acct["availableBalance"])
                    notional = balance * alloc * lev
                    price = float(data[slot]["Close"].iloc[-1])
                    qty = round_qty(symbol, notional / price)
                    if qty <= 0:
                        continue
                    side = "BUY" if sig == 1 else "SELL"
                    result = market_order(symbol, side, qty)
                    if "error" in result:
                        log(f"  FLIP {symbol} {side} HATA: {result}")
                    else:
                        log(f"  FLIP {symbol} {side} qty={qty} lev={lev} -> OK")
                else:
                    # Same direction, hold
                    pass

            # Balance snapshot
            acct = get_account()
            bal = float(acct["totalWalletBalance"])
            pnl = bal - start_balance
            pos_count = len(get_positions())
            log(f"  Balance: ${bal:,.2f}  PnL: ${pnl:+,.2f}  "
                f"({(bal/start_balance-1)*100:+.2f}%)  Positions: {pos_count}")

        except Exception as e:
            log(f"  HATA: {type(e).__name__}: {e}")

        candle_idx += 1
        time.sleep(loop_seconds)

    # Final
    log("=" * 70)
    log("BOT DURDURULDU — FINAL RAPOR")
    try:
        acct = get_account()
        bal = float(acct["totalWalletBalance"])
        log(f"  Baslangic: ${start_balance:,.2f}")
        log(f"  Bitis:     ${bal:,.2f}")
        log(f"  Net PnL:   ${bal-start_balance:+,.2f} ({(bal/start_balance-1)*100:+.2f}%)")
        log(f"  Iteration: {iteration}")
    except Exception as e:
        log(f"Final rapor HATA: {e}")


if __name__ == "__main__":
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    loop = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    main(hours, loop)
