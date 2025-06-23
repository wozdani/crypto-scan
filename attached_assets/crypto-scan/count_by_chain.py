from utils.data_fetchers import get_symbols_cached
from utils.coingecko import get_contract
import os

symbols = get_symbols_cached()

chains = {}
for s in symbols:
    token_info = get_contract(s)
    if token_info and "chain" in token_info:
        ch = token_info["chain"]
        chains[ch] = chains.get(ch, 0) + 1

print("🔗 Liczba tokenów na Bybit (USDT PERP) z kontraktem CoinGecko według chainów:")
for ch, count in chains.items():
    print(f"  • {ch.upper():<10} → {count} tokenów")
