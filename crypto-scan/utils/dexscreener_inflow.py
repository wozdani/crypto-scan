"""
DEX INFLOW 2.0 - DexScreener Integration
Nowy system wykrywania DEX inflow z wykorzystaniem DexScreener API i multi-wallet logic
"""

import requests
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

def get_dexscreener_inflow_score(token_address: str, chain: str = "ethereum") -> Dict:
    """
    Pobiera dane z DexScreener API i oblicza DEX inflow score
    
    Args:
        token_address: Adres kontraktu tokena
        chain: Blockchain (ethereum, bsc, polygon, arbitrum, optimism)
        
    Returns:
        Dict z dex_inflow_score i szczegółowymi danymi
    """
    try:
        # Mapowanie chainów na DexScreener format
        chain_mapping = {
            "ethereum": "ethereum",
            "bsc": "bsc", 
            "polygon": "polygon",
            "arbitrum": "arbitrum",
            "optimism": "optimism"
        }
        
        mapped_chain = chain_mapping.get(chain.lower(), "ethereum")
        
        # API call do DexScreener
        url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"⚠️ DexScreener API error {response.status_code} for {token_address}")
            return _get_empty_dex_result()
            
        data = response.json()
        
        if not data.get('pairs'):
            print(f"📭 No pairs found for {token_address}")
            return _get_empty_dex_result()
        
        # Znajdź najlepszy pair dla danego chain
        best_pair = _find_best_pair(data['pairs'], mapped_chain)
        
        if not best_pair:
            print(f"🔍 No suitable pair found for {token_address} on {chain}")
            return _get_empty_dex_result()
        
        # Oblicz DEX inflow score
        dex_score = _calculate_dex_inflow_score(best_pair)
        
        # Dodaj szczegółowe informacje
        result = {
            "dex_inflow_detected": dex_score["score"] >= 5,  # Minimalny próg
            "dex_inflow_score": dex_score["score"],
            "dex_tags": dex_score["tags"],
            "volume_1h_usd": best_pair.get('volume', {}).get('h1', 0),
            "volume_change_h1": best_pair.get('volumeChange', {}).get('h1', 0),
            "dex_name": best_pair.get('dexId', 'unknown'),
            "pair_address": best_pair.get('pairAddress', ''),
            "chain_id": best_pair.get('chainId', ''),
            "last_trade_ago_min": _calculate_trade_recency(best_pair),
            "verified_pair": _is_verified_pair(best_pair)
        }
        
        if result["dex_inflow_detected"]:
            print(f"🔥 DEX inflow detected for {token_address}: score={dex_score['score']}, volume_1h=${result['volume_1h_usd']:,.0f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in DexScreener API call for {token_address}: {e}")
        return _get_empty_dex_result()

def _find_best_pair(pairs: List[Dict], target_chain: str) -> Optional[Dict]:
    """
    Znajduje najlepszy pair dla danego chain na podstawie volume i aktywności
    """
    suitable_pairs = []
    
    for pair in pairs:
        # Filtruj po chain jeśli określony
        if target_chain != "ethereum":  # ethereum jako domyślny
            if pair.get('chainId') != target_chain:
                continue
        
        # Sprawdź czy pair ma podstawowe dane
        if not pair.get('volume', {}).get('h1'):
            continue
            
        volume_1h = float(pair.get('volume', {}).get('h1', 0))
        
        # Dodaj do kandydatów jeśli volume > 1000 USD
        if volume_1h > 1000:
            suitable_pairs.append({
                'pair': pair,
                'volume_1h': volume_1h,
                'dex_priority': _get_dex_priority(pair.get('dexId', ''))
            })
    
    if not suitable_pairs:
        return None
    
    # Sortuj po volume i priorytecie DEX
    suitable_pairs.sort(key=lambda x: (x['dex_priority'], x['volume_1h']), reverse=True)
    
    return suitable_pairs[0]['pair']

def _calculate_dex_inflow_score(pair: Dict) -> Dict:
    """
    Oblicza DEX inflow score na podstawie danych pair
    """
    score = 0
    tags = []
    
    volume_1h = float(pair.get('volume', {}).get('h1', 0))
    volume_change_h1 = float(pair.get('volumeChange', {}).get('h1', 0))
    
    # Volume 1h > 15K → +3 points
    if volume_1h > 15000:
        score += 3
        tags.append("high_volume_1h")
        
        # Bonus za bardzo wysoki volume
        if volume_1h > 50000:
            score += 2
            tags.append("very_high_volume")
    
    # Volume change > 100% → +5 points  
    if volume_change_h1 > 100:
        score += 5
        tags.append("volume_spike")
        
        # Bonus za ekstremalne wzrosty
        if volume_change_h1 > 500:
            score += 3
            tags.append("extreme_volume_spike")
    
    # TX recency < 3min → +2 points
    trade_recency_min = _calculate_trade_recency(pair)
    if trade_recency_min < 3:
        score += 2
        tags.append("recent_activity")
    
    # Verified pair + known DEX → +2 points
    if _is_verified_pair(pair) and _is_known_dex(pair.get('dexId', '')):
        score += 2
        tags.append("verified_dex")
    
    # DEX-specific bonuses
    dex_id = pair.get('dexId', '').lower()
    if dex_id in ['pancakeswap', 'uniswap', 'sushiswap']:
        tags.append(dex_id)
        if dex_id == 'pancakeswap':
            score += 1  # Bonus dla PancakeSwap
    
    return {
        "score": score,
        "tags": tags
    }

def _calculate_trade_recency(pair: Dict) -> float:
    """
    Oblicza ile minut temu była ostatnia transakcja
    """
    try:
        last_trade = pair.get('lastTrade', {})
        if not last_trade:
            return 999  # Brak danych = stare
            
        timestamp = last_trade.get('timestamp')
        if not timestamp:
            return 999
            
        # Konwertuj timestamp do datetime
        if isinstance(timestamp, (int, float)):
            last_trade_time = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
        else:
            last_trade_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        now = datetime.now(timezone.utc)
        delta = now - last_trade_time
        
        return delta.total_seconds() / 60  # Minuty
        
    except Exception as e:
        print(f"⚠️ Error calculating trade recency: {e}")
        return 999

def _is_verified_pair(pair: Dict) -> bool:
    """
    Sprawdza czy pair jest zweryfikowany
    """
    # Sprawdź różne wskaźniki weryfikacji
    labels = pair.get('labels', [])
    if 'verified' in labels:
        return True
        
    # Sprawdź czy ma logo i podstawowe informacje
    base_token = pair.get('baseToken', {})
    if base_token.get('symbol') and len(base_token.get('symbol', '')) > 1:
        return True
        
    return False

def _is_known_dex(dex_id: str) -> bool:
    """
    Sprawdza czy DEX jest znany i zaufany
    """
    known_dexes = [
        'uniswap', 'pancakeswap', 'sushiswap', 'quickswap', 
        'trader-joe', 'spiritswap', 'spookyswap', 'honeyswap',
        'apeswap', 'bakeryswap', 'camelot', 'kyberswap'
    ]
    
    return dex_id.lower() in known_dexes

def _get_dex_priority(dex_id: str) -> int:
    """
    Zwraca priorytet DEX (wyższy = lepszy)
    """
    dex_priorities = {
        'uniswap': 10,
        'pancakeswap': 9,
        'sushiswap': 8,
        'quickswap': 7,
        'trader-joe': 6,
        'camelot': 5,
        'kyberswap': 4
    }
    
    return dex_priorities.get(dex_id.lower(), 1)

def _get_empty_dex_result() -> Dict:
    """
    Zwraca pusty wynik DEX inflow
    """
    return {
        "dex_inflow_detected": False,
        "dex_inflow_score": 0,
        "dex_tags": [],
        "volume_1h_usd": 0,
        "volume_change_h1": 0,
        "dex_name": "unknown",
        "pair_address": "",
        "chain_id": "",
        "last_trade_ago_min": 999,
        "verified_pair": False
    }

def detect_repeat_value_multi_wallets(transfers: List[Dict], token_address: str) -> Dict:
    """
    Wykrywa powtarzające się transakcje z różnych walletów o podobnej wartości
    
    Args:
        transfers: Lista transferów tokenów
        token_address: Adres kontraktu tokena
        
    Returns:
        Dict z informacjami o multi-wallet repeat pattern
    """
    try:
        if not transfers or len(transfers) < 2:
            return _get_empty_multi_wallet_result()
        
        # Grupuj transfery według wartości USD (±10% tolerance)
        value_groups = []
        
        for transfer in transfers:
            value_usd = float(transfer.get('value_usd', 0))
            from_address = transfer.get('from_address', '').lower()
            to_address = transfer.get('to_address', '').lower()
            
            # Pomiń transfery < $1000
            if value_usd < 1000:
                continue
                
            # Sprawdź czy pasuje do istniejącej grupy (±10%)
            matched_group = None
            for group in value_groups:
                group_avg = sum(t['value_usd'] for t in group['transfers']) / len(group['transfers'])
                tolerance = group_avg * 0.1  # ±10%
                
                if abs(value_usd - group_avg) <= tolerance:
                    matched_group = group
                    break
            
            if matched_group:
                # Sprawdź czy from_address jest różny
                existing_addresses = {t['from_address'].lower() for t in matched_group['transfers']}
                if from_address not in existing_addresses:
                    matched_group['transfers'].append({
                        'value_usd': value_usd,
                        'from_address': from_address,
                        'to_address': to_address,
                        'timestamp': transfer.get('timestamp'),
                        'tx_hash': transfer.get('tx_hash', '')
                    })
            else:
                # Utwórz nową grupę
                value_groups.append({
                    'transfers': [{
                        'value_usd': value_usd,
                        'from_address': from_address,
                        'to_address': to_address,
                        'timestamp': transfer.get('timestamp'),
                        'tx_hash': transfer.get('tx_hash', '')
                    }]
                })
        
        # Znajdź największą grupę z ≥2 różnymi walletami
        best_group = None
        max_wallets = 0
        
        for group in value_groups:
            unique_wallets = len(set(t['from_address'] for t in group['transfers']))
            if unique_wallets >= 2 and unique_wallets > max_wallets:
                max_wallets = unique_wallets
                best_group = group
        
        if not best_group or max_wallets < 2:
            return _get_empty_multi_wallet_result()
        
        # Oblicz statystyki dla najlepszej grupy
        values = [t['value_usd'] for t in best_group['transfers']]
        avg_value = sum(values) / len(values)
        
        result = {
            "multi_wallet_repeat": True,
            "multi_wallet_avg_usd": avg_value,
            "multi_wallet_tx_count": len(best_group['transfers']),
            "multi_wallet_unique_wallets": max_wallets,
            "multi_wallet_value_range": f"${min(values):,.0f} - ${max(values):,.0f}",
            "multi_wallet_addresses": list(set(t['from_address'] for t in best_group['transfers']))[:5]  # Max 5 dla privacy
        }
        
        print(f"🔄 Multi-wallet repeat detected for {token_address}: {max_wallets} wallets, avg ${avg_value:,.0f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in multi-wallet detection for {token_address}: {e}")
        return _get_empty_multi_wallet_result()

def _get_empty_multi_wallet_result() -> Dict:
    """
    Zwraca pusty wynik multi-wallet detection
    """
    return {
        "multi_wallet_repeat": False,
        "multi_wallet_avg_usd": 0,
        "multi_wallet_tx_count": 0,
        "multi_wallet_unique_wallets": 0,
        "multi_wallet_value_range": "",
        "multi_wallet_addresses": []
    }

def test_dexscreener_inflow():
    """
    Test funkcji DexScreener inflow
    """
    print("🧪 Testing DexScreener Inflow 2.0...")
    
    # Test z popularnym tokenem
    test_contracts = [
        ("0xA0b86a33E6441986B84e794473C569c41C4e7332", "ethereum"),  # Sample token
        ("0xe9e7cea3dedca5984780bafc599bd69add087d56", "bsc"),       # BUSD
    ]
    
    for contract, chain in test_contracts:
        print(f"\n🔍 Testing {contract} on {chain}...")
        result = get_dexscreener_inflow_score(contract, chain)
        
        print(f"Result: {json.dumps(result, indent=2)}")
        
        # Test delay
        time.sleep(1)

if __name__ == "__main__":
    test_dexscreener_inflow()