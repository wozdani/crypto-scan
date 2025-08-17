import os, json, time, requests
from typing import Dict, Any, Union

BASE = "https://api.etherscan.io/v2/api"

def _load_aliases() -> Dict[str, int]:
    raw = os.getenv("ETHERSCAN_V2_CHAIN_ALIASES")
    if not raw:
        raise RuntimeError("[CONFIG] Missing ETHERSCAN_V2_CHAIN_ALIASES in .env")
    try:
        # Fix malformed JSON - add braces if missing
        raw = raw.strip()
        if not raw.startswith('{'):
            raw = '{' + raw
        if not raw.endswith('}'):
            raw = raw + '}'
        return {str(k).strip(): int(v) for k, v in json.loads(raw).items()}
    except Exception as e:
        raise RuntimeError(f"[CONFIG] Invalid ETHERSCAN_V2_CHAIN_ALIASES JSON: {e}")

ALIASES = _load_aliases()

class EtherscanV2:
    def __init__(self, rate_per_sec: float = 4.0, timeout: int = 20):
        self.key = os.getenv("ETHERSCAN_V2_API_KEY")
        if not self.key:
            raise RuntimeError("[CONFIG] ETHERSCAN_V2_API_KEY is required.")
        self.delay = max(0.0, 1.0 / rate_per_sec)
        self.timeout = timeout
        proxy = os.getenv("PROXY_URL")
        self.proxies = {"http": proxy, "https": proxy} if proxy else None

    def _sleep(self):
        if self.delay:
            time.sleep(self.delay)

    def _resolve_chainid(self, chain: Union[str, int]) -> int:
        if isinstance(chain, int):
            return chain
        try:
            return int(chain)
        except Exception:
            pass
        if chain not in ALIASES:
            raise ValueError(f"[V2] Unknown chain alias: {chain}. Add it to ETHERSCAN_V2_CHAIN_ALIASES.")
        return ALIASES[chain]

    def _get(self, chain: Union[str, int], params: Dict[str, Any]):
        chainid = self._resolve_chainid(chain)
        q = {"chainid": chainid, "apikey": self.key, **params}
        r = requests.get(BASE, params=q, timeout=self.timeout, proxies=self.proxies)
        j = r.json()
        if str(j.get("status")) != "1":
            raise RuntimeError(f"[V2] Error for chain {chain} (id={chainid}): {j}")
        self._sleep()
        return j["result"]

    # Public API endpoints
    def balance(self, chain, address, tag="latest"):
        return self._get(chain, {"module":"account","action":"balance","address":address,"tag":tag})

    def txlist(self, chain, address, start=0, end=99999999, sort="asc", page=1, offset=100):
        return self._get(chain, {
            "module":"account","action":"txlist","address":address,
            "startblock":start,"endblock":end,"sort":sort,"page":page,"offset":offset
        })

    def get_logs(self, chain, contract=None, topic0=None, from_block=0, to_block=99999999):
        p = {"module":"logs","action":"getLogs","fromBlock":from_block,"toBlock":to_block}
        if contract: p["address"] = contract
        if topic0:   p["topic0"]  = topic0
        return self._get(chain, p)

    def eth_call(self, chain, to, data, tag="latest"):
        return self._get(chain, {"module":"proxy","action":"eth_call","to":to,"data":data,"tag":tag})