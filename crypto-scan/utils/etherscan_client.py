"""
Etherscan API V2 Client with Legacy Fallback
Implements unified blockchain scanning with V2-first approach and automatic fallback to chain-specific APIs.
"""

import os
import time
import requests
from typing import Dict, Any, Optional

V2_BASE = "https://api.etherscan.io/v2/api"

LEGACY_BASES = {
    "ethereum": "https://api.etherscan.io/api",
    "bsc":      "https://api.bscscan.com/api",
    "polygon":  "https://api.polygonscan.com/api",
    "arbitrum": "https://api.arbiscan.io/api",
    "optimism": "https://api-optimistic.etherscan.io/api",
    "base":     "https://api.basescan.org/api",
}

LEGACY_KEYS = {
    "ethereum": os.getenv("ETHERSCAN_API_KEY"),
    "bsc":      os.getenv("BSCSCAN_API_KEY"),
    "polygon":  os.getenv("POLYGONSCAN_API_KEY"),
    "arbitrum": os.getenv("ARBISCAN_API_KEY") or os.getenv("ARBITRUMSCAN_API_KEY"),
    "optimism": os.getenv("OPTIMISMSCAN_API_KEY") or os.getenv("OPTIMISM_API_KEY"),
    "base":     os.getenv("BASESCAN_API_KEY") or os.getenv("BASE_API_KEY"),
}

CHAIN_IDS = {
    "ethereum": 1,
    "arbitrum": 42161,
    "base": 8453,
    "optimism": 10,
    "polygon": 137,
    "bsc": 56,
}

class EtherscanClient:
    """
    V2-first Etherscan client with automatic legacy fallback.
    - Tries V2 (one key + chainid)
    - On error or missing key -> falls back to chain-specific legacy host and key
    """
    def __init__(self, rate_per_sec: float = 4.0, timeout: int = 20):
        self.v2_key = os.getenv("ETHERSCAN_V2_API_KEY")
        self.timeout = timeout
        self.delay = max(0.0, 1.0 / rate_per_sec)
        proxy = os.getenv("PROXY_URL")
        self.proxies = {"http": proxy, "https": proxy} if proxy else None

    # ----------------------------- public API -----------------------------

    def balance(self, chain: str, address: str, tag: str = "latest") -> Any:
        """Get account balance for address"""
        return self._get(chain, {"module": "account", "action": "balance", "address": address, "tag": tag})

    def txlist(self, chain: str, address: str, start: int = 0, end: int = 99999999,
               sort: str = "asc", page: int = 1, offset: int = 100) -> Any:
        """Get transaction list for address"""
        return self._get(chain, {
            "module": "account", "action": "txlist",
            "address": address, "startblock": start, "endblock": end,
            "sort": sort, "page": page, "offset": offset
        })

    def tokentx(self, chain: str, contract_address: Optional[str] = None, address: Optional[str] = None,
                start: int = 0, end: int = 99999999, sort: str = "desc", 
                page: int = 1, offset: int = 100) -> Any:
        """Get ERC20 token transfers"""
        params = {
            "module": "account", "action": "tokentx",
            "startblock": start, "endblock": end,
            "sort": sort, "page": page, "offset": offset
        }
        if contract_address:
            params["contractaddress"] = contract_address
        if address:
            params["address"] = address
        
        return self._get(chain, params)

    def get_logs(self, chain: str, contract: Optional[str] = None, topic0: Optional[str] = None,
                 from_block: int = 0, to_block: int = 99999999) -> Any:
        """Get event logs with optional filtering"""
        params = {"module": "logs", "action": "getLogs", "fromBlock": from_block, "toBlock": to_block}
        if contract: 
            params["address"] = contract
        if topic0: 
            params["topic0"] = topic0
        return self._get(chain, params)

    def eth_call(self, chain: str, to: str, data: str, tag: str = "latest") -> Any:
        """Make eth_call to smart contract"""
        return self._get(chain, {"module": "proxy", "action": "eth_call", "to": to, "data": data, "tag": tag})

    # --------------------------- internal helpers -------------------------

    def _sleep(self):
        """Rate limiting delay"""
        if self.delay > 0:
            time.sleep(self.delay)

    def _v2_get(self, chain: str, params: Dict[str, Any]) -> Any:
        """Try V2 API endpoint"""
        if not self.v2_key:
            raise RuntimeError("no_v2_key")
        chainid = CHAIN_IDS.get(chain)
        if not chainid:
            raise RuntimeError(f"unknown_chain:{chain}")
        
        q = {"chainid": chainid, "apikey": self.v2_key, **params}
        
        try:
            r = requests.get(V2_BASE, params=q, timeout=self.timeout, proxies=self.proxies)
            j = r.json()
            
            # Etherscan V2 keeps "status": "1" for success
            if str(j.get("status")) != "1":
                raise RuntimeError(f"v2_error:{j}")
            
            self._sleep()
            return j["result"]
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"v2_request_error:{e}")

    def _legacy_get(self, chain: str, params: Dict[str, Any]) -> Any:
        """Fallback to legacy chain-specific API"""
        key = LEGACY_KEYS.get(chain)
        base = LEGACY_BASES.get(chain)
        if not key or not base:
            raise RuntimeError(f"legacy_missing:{chain}")
        
        q = {"apikey": key, **params}
        
        try:
            r = requests.get(base, params=q, timeout=self.timeout, proxies=self.proxies)
            j = r.json()
            
            if str(j.get("status")) != "1":
                raise RuntimeError(f"legacy_error:{j}")
            
            self._sleep()
            return j["result"]
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"legacy_request_error:{e}")

    def _get(self, chain: str, params: Dict[str, Any]) -> Any:
        """
        Main get method with V2-first strategy and legacy fallback.
        Tries V2 with simple retry/backoff on 429/5xx, else fallback to legacy.
        """
        tries = 2
        
        # Try V2 first
        for i in range(tries):
            try:
                print(f"[ETHERSCAN V2] {chain}: Attempting V2 API call")
                result = self._v2_get(chain, params)
                print(f"[ETHERSCAN V2] {chain}: V2 API success")
                return result
                
            except Exception as e:
                msg = str(e)
                print(f"[ETHERSCAN V2] {chain}: V2 attempt {i+1} failed: {msg}")
                
                if "no_v2_key" in msg:
                    print(f"[ETHERSCAN V2] {chain}: No V2 key available, switching to legacy")
                    break  # go to legacy immediately
                
                # Retry for transient errors
                if any(x in msg for x in ["429", "timeout", "5xx", "request_error"]):
                    if i < tries - 1:  # Not the last attempt
                        backoff_time = 1.5 * (i + 1)
                        print(f"[ETHERSCAN V2] {chain}: Retrying in {backoff_time}s...")
                        time.sleep(backoff_time)
                        continue
                
                # For other errors, proceed to legacy
                break
        
        # Fallback to legacy
        try:
            print(f"[ETHERSCAN LEGACY] {chain}: Attempting legacy API fallback")
            result = self._legacy_get(chain, params)
            print(f"[ETHERSCAN LEGACY] {chain}: Legacy API success")
            return result
            
        except Exception as e:
            print(f"[ETHERSCAN LEGACY] {chain}: Legacy API failed: {e}")
            raise RuntimeError(f"both_apis_failed_for_{chain}:{e}")


# Global singleton instance
_etherscan_client = None

def get_etherscan_client() -> EtherscanClient:
    """Get singleton instance of EtherscanClient"""
    global _etherscan_client
    if _etherscan_client is None:
        _etherscan_client = EtherscanClient(rate_per_sec=4, timeout=20)
    return _etherscan_client