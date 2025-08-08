"""
DEX Inflow Detection with BSC Fallback and Swap Classification
Enhanced with eth_getLogs fallback and multi-provider classification
"""

import hashlib
from typing import Dict, List, Set, Optional, Tuple
import sys
import os
import requests
import time

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.dex_routes_bsc import DEX_ROUTERS_BSC
from utils.blockchain_scanners import get_token_transfers_last_24h

# Swap event signature: Swap(address,uint256,uint256,uint256,uint256,address)
# Precomputed keccak256 hash
SWAP_SIG = "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822"

def classify_dex_inflow(logs: List[Dict], router_set: Set[str]) -> Dict:
    """
    Classify DEX inflow from logs using swap signatures and router detection
    
    Args:
        logs: List of transaction logs
        router_set: Set of known DEX router addresses
        
    Returns:
        Dict with USD inflow amount and unique address count
    """
    inflow = 0.0
    addrs = set()
    
    for L in logs:
        try:
            # Check address match
            a = L.get("address", "").lower()
            
            # Check topics for swap signature
            topics = L.get("topics", [])
            has_swap_sig = False
            if topics and len(topics) > 0:
                has_swap_sig = topics[0].lower() == SWAP_SIG.lower()
            
            # Match either router address or swap signature
            if a in router_set or has_swap_sig:
                usd = L.get("amount_usd", 0.0)  # Calculated upstream
                inflow += usd
                
                # Track unique addresses
                if "from" in L:
                    addrs.add(L["from"].lower())
                if "to" in L:
                    addrs.add(L["to"].lower())
                    
        except Exception as e:
            print(f"[DEX CLASSIFY] Error processing log: {e}")
            continue
    
    return {"usd": inflow, "unique": len(addrs)}

def fetch_transfers(contract: str, start_time: int, end_time: int) -> List[Dict]:
    """
    Fetch transfers using multi-provider approach
    
    Args:
        contract: Contract address
        start_time: Start timestamp
        end_time: End timestamp
        
    Returns:
        List of transfer logs with amount_usd calculated
    """
    try:
        # Try primary blockchain scanner
        transfers = get_token_transfers_last_24h(
            symbol="UNKNOWN",  # We have contract, symbol not needed
            chain="bsc",  # Assuming BSC for now
            contract_address=contract
        )
        
        # Filter by time window if available
        filtered_transfers = []
        for transfer in transfers:
            transfer_time = transfer.get("timestamp", 0)
            if start_time <= transfer_time <= end_time:
                # Ensure amount_usd is present
                if "amount_usd" not in transfer:
                    transfer["amount_usd"] = transfer.get("value_usd", 0.0)
                filtered_transfers.append(transfer)
        
        print(f"[DEX FETCH] Retrieved {len(filtered_transfers)} transfers for {contract}")
        return filtered_transfers
        
    except Exception as e:
        print(f"[DEX FETCH] Error fetching transfers: {e}")
        
        # Fallback to RPC method
        return _fetch_via_rpc(contract, start_time, end_time)

def _fetch_via_rpc(contract: str, start_time: int, end_time: int) -> List[Dict]:
    """
    Fallback RPC method using eth_getLogs
    
    Args:
        contract: Contract address
        start_time: Start timestamp
        end_time: End timestamp
        
    Returns:
        List of logs with estimated amount_usd
    """
    try:
        # BSC RPC endpoints
        rpc_urls = [
            "https://bsc-dataseed1.binance.org/",
            "https://bsc-dataseed2.binance.org/",
            "https://bsc-dataseed3.binance.org/"
        ]
        
        # Transfer event topic
        TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
        
        for rpc_url in rpc_urls:
            try:
                # Get latest block
                latest_response = requests.post(rpc_url, json={
                    "jsonrpc": "2.0",
                    "method": "eth_blockNumber",
                    "params": [],
                    "id": 1
                }, timeout=10)
                
                latest_block = int(latest_response.json()["result"], 16)
                from_block = latest_block - 1200  # ~1 hour on BSC (3s blocks)
                
                # Get logs
                logs_response = requests.post(rpc_url, json={
                    "jsonrpc": "2.0", 
                    "method": "eth_getLogs",
                    "params": [{
                        "address": contract,
                        "fromBlock": hex(from_block),
                        "toBlock": "latest",
                        "topics": [TRANSFER_TOPIC]
                    }],
                    "id": 2
                }, timeout=15)
                
                logs = logs_response.json().get("result", [])
                
                # Convert to transfer format
                transfers = []
                for log in logs:
                    try:
                        # Extract data from log
                        data = log.get("data", "0x")
                        topics = log.get("topics", [])
                        
                        if len(topics) >= 3 and len(data) > 2:
                            from_addr = topics[1][-40:]  # Last 40 chars (20 bytes)
                            to_addr = topics[2][-40:]
                            amount_hex = data[-64:]  # Last 32 bytes
                            amount = int(amount_hex, 16)
                            
                            # Rough USD estimation (would need price oracle)
                            amount_usd = amount * 0.000001  # Placeholder conversion
                            
                            transfers.append({
                                "from": f"0x{from_addr}",
                                "to": f"0x{to_addr}",
                                "amount_usd": amount_usd,
                                "address": log.get("address", "").lower(),
                                "topics": topics,
                                "timestamp": int(time.time())  # Approximate
                            })
                    except Exception as parse_error:
                        print(f"[DEX RPC] Error parsing log: {parse_error}")
                        continue
                
                print(f"[DEX RPC] Retrieved {len(transfers)} transfers via RPC")
                return transfers
                
            except Exception as rpc_error:
                print(f"[DEX RPC] Error with {rpc_url}: {rpc_error}")
                continue
        
        # All RPC methods failed
        return []
        
    except Exception as e:
        print(f"[DEX RPC] Fatal error: {e}")
        return []

class TimeWindow:
    """Simple time window class for compatibility"""
    def __init__(self, hours: int = 1):
        current_time = int(time.time())
        self.start = current_time - (hours * 3600)
        self.end = current_time

def run_dex_inflow(contract: str, window: Optional[TimeWindow] = None, providers: Optional[List[str]] = None, router_set: Optional[Set[str]] = None) -> Dict:
    """
    Enhanced DEX inflow detection with multi-provider fallback
    
    Args:
        contract: Token contract address
        window: Time window for analysis (default 1 hour)
        providers: List of provider names (for future extension)
        router_set: Set of known DEX router addresses
        
    Returns:
        Dict with active status, strength, and metadata
    """
    try:
        if window is None:
            window = TimeWindow(1)  # 1 hour default
            
        if router_set is None:
            router_set = set(addr.lower() for addr in DEX_ROUTERS_BSC.values())
        
        # Fetch transfers using multi-provider approach
        tx = fetch_transfers(contract, window.start, window.end)
        
        if not tx:
            return {
                "active": False, 
                "strength": 0.0, 
                "meta": {"status": "NO_DATA", "transfers": 0}
            }
        
        # Classify DEX inflow
        res = classify_dex_inflow(tx, router_set)
        
        if res["usd"] <= 0.0:
            return {
                "active": False, 
                "strength": 0.0, 
                "meta": {"status": "NO_INFLOW", "transfers": len(tx)}
            }
        
        # Normalize strength: $150k USD = strength 1.0 (clamp)
        strength = min(1.0, res["usd"] / 150_000.0)
        
        print(f"[DEX INFLOW] Contract {contract}: ${res['usd']:.0f} inflow, {res['unique']} unique addresses, strength={strength:.3f}")
        
        return {
            "active": True, 
            "strength": strength, 
            "meta": {
                "status": "SUCCESS",
                "inflow_usd": res["usd"], 
                "unique": res["unique"],
                "transfers": len(tx)
            }
        }
        
    except Exception as e:
        print(f"[DEX INFLOW] Error for contract {contract}: {e}")
        return {
            "active": False, 
            "strength": 0.0, 
            "meta": {"status": "UNKNOWN", "error": str(e)}
        }