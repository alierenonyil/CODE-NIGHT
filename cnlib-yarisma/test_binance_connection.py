"""
Binance testnet bağlantı testi — hangi testnet (spot/futures) çalışıyor anla.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("BINANCE_API_KEY")
SECRET = os.getenv("BINANCE_API_SECRET")

if not KEY or not SECRET:
    print("HATA: .env icinde BINANCE_API_KEY/SECRET yok")
    raise SystemExit(1)

print(f"Key prefix: {KEY[:8]}..., len={len(KEY)}")
print(f"Secret prefix: {SECRET[:8]}..., len={len(SECRET)}")
print()

# --- SPOT TESTNET ---
print("=" * 60)
print(" SPOT TESTNET (testnet.binance.vision)")
print("=" * 60)
try:
    from binance.client import Client
    spot = Client(api_key=KEY, api_secret=SECRET, testnet=True)
    account = spot.get_account()
    print(f"  OK - balances:")
    for b in account.get("balances", []):
        free = float(b["free"])
        if free > 0:
            print(f"    {b['asset']:<10} free={free:,.4f}")
    print(f"  can trade: {account.get('canTrade')}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")

# --- FUTURES TESTNET ---
print()
print("=" * 60)
print(" FUTURES TESTNET (testnet.binancefuture.com)")
print("=" * 60)
try:
    import requests, hmac, hashlib, time
    FAPI = "https://testnet.binancefuture.com/fapi"
    ts = int(time.time() * 1000)
    qs = f"timestamp={ts}"
    sig = hmac.new(SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    r = requests.get(
        f"{FAPI}/v2/account?{qs}&signature={sig}",
        headers={"X-MBX-APIKEY": KEY},
        timeout=10,
    )
    if r.status_code == 200:
        data = r.json()
        print(f"  OK - Total wallet balance: {data.get('totalWalletBalance')}")
        print(f"  Available balance:         {data.get('availableBalance')}")
        print(f"  Assets:")
        for a in data.get("assets", []):
            if float(a.get("walletBalance", 0)) > 0:
                print(f"    {a['asset']:<10} wallet={float(a['walletBalance']):,.4f}  "
                      f"available={float(a['availableBalance']):,.4f}")
    else:
        print(f"  FAIL {r.status_code}: {r.text[:400]}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")
