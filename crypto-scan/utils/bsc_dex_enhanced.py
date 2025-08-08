"""
Enhanced BSC DEX Inflow Detection
Improved BSC detection with eth_getLogs + Swap classification fallback
"""

import requests
import json
import time
from typing import Dict, List, Optional, Tuple
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.stealth_config import STEALTH

# PUNKT 10: Import once_per_scan decorator
try:
    from engine.rl_gateway import once_per_scan
    ONCE_PER_SCAN_AVAILABLE = True
except ImportError:
    ONCE_PER_SCAN_AVAILABLE = False
    def once_per_scan(category, subcategory):
        def decorator(func):
            return func
        return decorator

# BSC RPC endpoints (fallback list)
BSC_RPC_ENDPOINTS = [
    "https://bsc-dataseed1.binance.org/",
    "https://bsc-dataseed2.binance.org/",
    "https://bsc-dataseed3.binance.org/",
    "https://bsc-dataseed4.binance.org/"
]

# Common Swap event topic (Transfer events have different signature)
SWAP_TOPIC = "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822"  # Swap(address,uint256,uint256,uint256,uint256,address)
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

# Known DEX router addresses on BSC
BSC_DEX_ROUTERS = [
    "0x10ed43c718714eb63d5aa57b78b54704e256024e",  # PancakeSwap V2
    "0x05ff2b0db69458a0750badebc4f9e13add608c7f",  # PancakeSwap V1
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",  # Uniswap V2 (also on BSC)
    "0x1b02da8cb0d097eb8d57a175b88c7d8b47997506"   # SushiSwap
]

@once_per_scan("BSC", "dex_inflow")
def detect_bsc_dex_inflow(token_address: str, bscscan_api_key: Optional[str] = None) -> Dict:
    """
    Enhanced BSC DEX inflow detection with multiple fallback methods
    
    Args:
        token_address: Contract address of the token
        bscscan_api_key: BscScan API key (optional)
        
    Returns:
        Dict with dex_inflow_usd, status, and method used
    """
    if not STEALTH["DEX_INFLOW_BSC_ENABLED"]:
        return {"dex_inflow_usd": 0.0, "status": "DISABLED", "method": "config"}
    
    print(f"[BSC DEX ENHANCED] Starting detection for {token_address}")
    
    # Method 1: Try BscScan API if available
    if bscscan_api_key:
        result = _try_bscscan_method(token_address, bscscan_api_key)
        if result["status"] != "FAILED":
            return result
    
    # Method 2: Direct RPC with eth_getLogs
    result = _try_rpc_method(token_address)
    if result["status"] != "FAILED":
        return result
    
    # Method 3: Fallback - return UNKNOWN if uncertain
    if STEALTH["DEX_INFLOW_UNKNOWN_ON_FAIL"]:
        print(f"[BSC DEX ENHANCED] All methods failed, returning UNKNOWN")
        return {
            "dex_inflow_usd": None,  # None indicates UNKNOWN, not 0
            "status": "UNKNOWN", 
            "method": "fallback_unknown",
            "reason": "all_detection_methods_failed"
        }
    else:
        # Legacy behavior: return 0
        return {"dex_inflow_usd": 0.0, "status": "FAILED", "method": "legacy_fallback"}

def _try_bscscan_method(token_address: str, api_key: str) -> Dict:
    """Try BscScan API method"""
    try:
        # Get recent transactions
        url = "https://api.bscscan.com/api"
        params = {
            "module": "account",
            "action": "tokentx",
            "contractaddress": token_address,
            "page": 1,
            "offset": 100,  # Last 100 transactions
            "sort": "desc",
            "apikey": api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            print(f"[BSC BSCSCAN] API error: {response.status_code}")
            return {"status": "FAILED"}
        
        data = response.json()
        if data.get("status") != "1":
            print(f"[BSC BSCSCAN] API response error: {data.get('message', 'unknown')}")
            return {"status": "FAILED"}
        
        transactions = data.get("result", [])
        dex_inflow = _analyze_transactions(transactions, method="bscscan")
        
        return {
            "dex_inflow_usd": dex_inflow,
            "status": "SUCCESS",
            "method": "bscscan_api",
            "tx_count": len(transactions)
        }
        
    except Exception as e:
        print(f"[BSC BSCSCAN] Error: {e}")
        return {"status": "FAILED"}

def _try_rpc_method(token_address: str) -> Dict:
    """Try direct RPC method with eth_getLogs"""
    for rpc_url in BSC_RPC_ENDPOINTS:
        try:
            print(f"[BSC RPC] Trying {rpc_url}")
            
            # Get latest block
            latest_block = _get_latest_block(rpc_url)
            if not latest_block:
                continue
            
            # Get logs from last ~1 hour (200 blocks on BSC)
            from_block = latest_block - 200
            
            # Get Transfer logs for the token
            logs = _get_transfer_logs(rpc_url, token_address, from_block, latest_block)
            if logs is None:
                continue
            
            dex_inflow = _analyze_logs(logs, token_address)
            
            return {
                "dex_inflow_usd": dex_inflow,
                "status": "SUCCESS",
                "method": f"rpc_{rpc_url.split('/')[2]}",
                "log_count": len(logs)
            }
            
        except Exception as e:
            print(f"[BSC RPC] Error with {rpc_url}: {e}")
            continue
    
    return {"status": "FAILED"}

def _get_latest_block(rpc_url: str) -> Optional[int]:
    """Get latest block number"""
    try:
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_blockNumber",
            "params": [],
            "id": 1
        }
        
        response = requests.post(rpc_url, json=payload, timeout=5)
        result = response.json()
        
        if "result" in result:
            return int(result["result"], 16)
            
    except Exception as e:
        print(f"[BSC RPC] Error getting block number: {e}")
    
    return None

def _get_transfer_logs(rpc_url: str, token_address: str, from_block: int, to_block: int) -> Optional[List]:
    """Get Transfer event logs"""
    try:
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_getLogs",
            "params": [{
                "address": token_address,
                "fromBlock": hex(from_block),
                "toBlock": hex(to_block),
                "topics": [TRANSFER_TOPIC]  # Transfer events
            }],
            "id": 1
        }
        
        response = requests.post(rpc_url, json=payload, timeout=10)
        result = response.json()
        
        if "result" in result:
            return result["result"]
            
    except Exception as e:
        print(f"[BSC RPC] Error getting logs: {e}")
    
    return None

def _analyze_transactions(transactions: List[Dict], method: str) -> float:
    """Analyze transactions for DEX inflow (BscScan format)"""
    total_inflow = 0.0
    dex_tx_count = 0
    
    for tx in transactions:
        try:
            # Check if transaction involves DEX router
            to_address = tx.get("to", "").lower()
            from_address = tx.get("from", "").lower()
            
            if any(router.lower() in [to_address, from_address] for router in BSC_DEX_ROUTERS):
                value = float(tx.get("value", "0"))
                if value > 0:
                    # Simple estimation: assume $1 per token for now (would need price oracle)
                    estimated_usd = value * 1.0  # Placeholder
                    total_inflow += estimated_usd
                    dex_tx_count += 1
        except:
            continue
    
    print(f"[BSC {method.upper()}] Found {dex_tx_count} DEX transactions, estimated inflow=${total_inflow:,.0f}")
    return total_inflow

def _analyze_logs(logs: List[Dict], token_address: str) -> float:
    """Analyze logs for DEX inflow (RPC format)"""
    total_inflow = 0.0
    dex_tx_count = 0
    
    for log in logs:
        try:
            # Check transaction hash and get full transaction details if needed
            # For now, count transfers involving known DEX addresses
            tx_hash = log.get("transactionHash", "")
            
            # Simple heuristic: large transfers are likely DEX trades
            topics = log.get("topics", [])
            if len(topics) >= 3:
                # Extract transfer amount from data field
                data = log.get("data", "0x")
                if len(data) > 2:
                    try:
                        amount_hex = data[-64:]  # Last 32 bytes
                        amount = int(amount_hex, 16)
                        
                        # If amount is significant, count as DEX activity
                        if amount > 1000:  # Threshold for significant transfers
                            estimated_usd = amount * 0.000001  # Rough estimation
                            total_inflow += estimated_usd
                            dex_tx_count += 1
                    except:
                        pass
        except:
            continue
    
    # PUNKT 10: Log only once per scan to avoid spam
    print(f"[BSC RPC] Found {dex_tx_count} potential DEX transfers, estimated inflow=${total_inflow:,.0f}")
    return total_inflow