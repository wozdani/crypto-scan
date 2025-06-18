"""
CoinMarketCap Category Cache Builder
Builds local cache of token categories and tags from CoinMarketCap API
"""

import os
import json
import time
import requests
from typing import Dict, List, Any, Optional

CMC_API_BASE = "https://pro-api.coinmarketcap.com"
RATE_LIMIT_DELAY = 2.0  # 2 seconds between batch requests
BATCH_SIZE = 100  # Process 100 symbols per batch
CACHE_FILE = "data/cache/cmc_category_cache.json"


def get_cmc_api_key():
    """Get CoinMarketCap API key from environment"""
    api_key = os.environ.get("CMC_API_KEY")
    if not api_key:
        raise ValueError("CMC_API_KEY environment variable not found")
    return api_key


def ensure_cache_directory():
    """Ensure cache directory exists"""
    cache_dir = os.path.dirname(CACHE_FILE)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[CMC CACHE] Created cache directory: {cache_dir}")


def load_existing_cache() -> Dict[str, Any]:
    """Load existing cache if available"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                print(f"[CMC CACHE] Loaded existing cache with {len(cache)} tokens")
                return cache
        except Exception as e:
            print(f"[CMC CACHE] Error loading existing cache: {e}")
    
    return {}


def save_cache(cache: Dict[str, Any]):
    """Save cache to file"""
    try:
        ensure_cache_directory()
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        print(f"[CMC CACHE] Saved cache with {len(cache)} tokens to {CACHE_FILE}")
    except Exception as e:
        print(f"[CMC CACHE] Error saving cache: {e}")


def fetch_cryptocurrency_map(api_key: str) -> List[Dict[str, Any]]:
    """
    Fetch cryptocurrency map from CoinMarketCap
    Returns list of cryptocurrencies with basic info
    """
    print("[CMC CACHE] Fetching cryptocurrency map...")
    
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': api_key,
    }
    
    params = {
        'listing_status': 'active',
        'start': 1,
        'limit': 5000,  # Get top 5000 cryptocurrencies
        'sort': 'market_cap'
    }
    
    try:
        response = requests.get(
            f"{CMC_API_BASE}/v1/cryptocurrency/map",
            headers=headers,
            params=params,
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        if data.get('status', {}).get('error_code') == 0:
            cryptocurrencies = data.get('data', [])
            print(f"[CMC CACHE] Fetched {len(cryptocurrencies)} cryptocurrencies from map")
            return cryptocurrencies
        else:
            error_msg = data.get('status', {}).get('error_message', 'Unknown error')
            raise Exception(f"CMC API error: {error_msg}")
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error fetching cryptocurrency map: {e}")


def fetch_cryptocurrency_info_batch(api_key: str, symbols: List[str]) -> Dict[str, Any]:
    """
    Fetch cryptocurrency info for a batch of symbols
    Returns dictionary with symbol as key and info as value
    """
    print(f"[CMC CACHE] Fetching info for batch: {symbols[:5]}... ({len(symbols)} total)")
    
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': api_key,
    }
    
    params = {
        'symbol': ','.join(symbols),
        'aux': 'urls,logo,description,tags,platform,date_added,notice,status'
    }
    
    try:
        response = requests.get(
            f"{CMC_API_BASE}/v1/cryptocurrency/info",
            headers=headers,
            params=params,
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        if data.get('status', {}).get('error_code') == 0:
            return data.get('data', {})
        else:
            error_msg = data.get('status', {}).get('error_message', 'Unknown error')
            print(f"[CMC CACHE] Warning: API error for batch {symbols[:3]}...: {error_msg}")
            return {}
            
    except requests.exceptions.RequestException as e:
        print(f"[CMC CACHE] Warning: Network error for batch {symbols[:3]}...: {e}")
        return {}


def extract_category_and_tags(token_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract category and tags from token info
    Returns formatted data for cache
    """
    # Extract category (use platform type or default to 'token')
    category = "token"  # Default category
    
    # Try to determine category from platform info
    platform = token_info.get('platform')
    if platform:
        platform_name = platform.get('name', '').lower()
        if 'ethereum' in platform_name:
            category = "token"
        elif 'binance' in platform_name:
            category = "token"
        elif platform_name in ['bitcoin', 'litecoin', 'dogecoin']:
            category = "coin"
    else:
        # No platform means it's likely a native coin
        category = "coin"
    
    # Extract tags
    tags = token_info.get('tags', [])
    if not isinstance(tags, list):
        tags = []
    
    # Clean up tags - keep only strings
    cleaned_tags = []
    for tag in tags:
        if isinstance(tag, str) and tag.strip():
            cleaned_tags.append(tag.strip())
    
    return {
        "category": category,
        "tags": cleaned_tags
    }


def build_category_cache(update_existing: bool = False) -> Dict[str, Any]:
    """
    Build category cache from CoinMarketCap API
    
    Args:
        update_existing: If True, update existing cache. If False, only process missing tokens.
    
    Returns:
        Dictionary with token categories and tags
    """
    print("[CMC CACHE] Starting category cache build...")
    
    try:
        # Get API key
        api_key = get_cmc_api_key()
        print("[CMC CACHE] API key loaded successfully")
        
        # Load existing cache
        cache = load_existing_cache() if not update_existing else {}
        initial_cache_size = len(cache)
        
        # Fetch cryptocurrency map
        cryptocurrencies = fetch_cryptocurrency_map(api_key)
        
        # Extract symbols that need processing
        symbols_to_process = []
        for crypto in cryptocurrencies:
            symbol = crypto.get('symbol', '').upper()
            if symbol and (update_existing or symbol not in cache):
                symbols_to_process.append(symbol)
        
        print(f"[CMC CACHE] Found {len(symbols_to_process)} symbols to process")
        
        if not symbols_to_process:
            print("[CMC CACHE] No new symbols to process")
            return cache
        
        # Process in batches
        processed_count = 0
        batch_count = 0
        
        for i in range(0, len(symbols_to_process), BATCH_SIZE):
            batch = symbols_to_process[i:i + BATCH_SIZE]
            batch_count += 1
            
            print(f"[CMC CACHE] Processing batch {batch_count} ({len(batch)} symbols)...")
            
            # Fetch info for this batch
            batch_info = fetch_cryptocurrency_info_batch(api_key, batch)
            
            # Process each symbol in the batch
            for symbol in batch:
                if symbol in batch_info:
                    try:
                        token_data = extract_category_and_tags(batch_info[symbol])
                        cache[symbol] = token_data
                        processed_count += 1
                        
                        # Debug output for first few tokens
                        if processed_count <= 5:
                            print(f"[CMC CACHE]   {symbol}: {token_data}")
                            
                    except Exception as e:
                        print(f"[CMC CACHE] Error processing {symbol}: {e}")
                        cache[symbol] = {"category": None, "tags": []}
                else:
                    # Symbol not found in response
                    cache[symbol] = {"category": None, "tags": []}
                    print(f"[CMC CACHE] Symbol {symbol} not found in API response")
            
            # Rate limiting - wait between batches
            if i + BATCH_SIZE < len(symbols_to_process):
                print(f"[CMC CACHE] Rate limiting: waiting {RATE_LIMIT_DELAY}s...")
                time.sleep(RATE_LIMIT_DELAY)
        
        # Save updated cache
        save_cache(cache)
        
        new_tokens = len(cache) - initial_cache_size
        print(f"[CMC CACHE] Cache build completed!")
        print(f"[CMC CACHE] Total tokens in cache: {len(cache)}")
        print(f"[CMC CACHE] New tokens added: {new_tokens}")
        print(f"[CMC CACHE] Processed: {processed_count} tokens")
        
        return cache
        
    except Exception as e:
        print(f"[CMC CACHE] Error building category cache: {e}")
        return load_existing_cache()


def get_token_category_info(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Get category and tags for a specific token
    
    Args:
        symbol: Token symbol (e.g., 'BTC', 'ETH')
    
    Returns:
        Dictionary with category and tags, or None if not found
    """
    try:
        cache = load_existing_cache()
        return cache.get(symbol.upper())
    except Exception as e:
        print(f"[CMC CACHE] Error loading token info for {symbol}: {e}")
        return None


def get_cache_statistics() -> Dict[str, Any]:
    """Get statistics about the category cache"""
    try:
        cache = load_existing_cache()
        
        # Count by category
        category_counts = {}
        tag_counts = {}
        
        for symbol, data in cache.items():
            category = data.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
            
            tags = data.get('tags', [])
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Get top tags
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_tokens": len(cache),
            "categories": category_counts,
            "total_unique_tags": len(tag_counts),
            "top_tags": top_tags,
            "cache_file": CACHE_FILE
        }
        
    except Exception as e:
        print(f"[CMC CACHE] Error getting cache statistics: {e}")
        return {}


def integrate_with_existing_sector_mapping():
    """
    Integrate CMC category data with existing SECTOR_MAPPING
    Updates breakout_cluster_scoring.py with new sector mappings
    """
    try:
        # Load current sector mapping
        sector_file = "utils/breakout_cluster_scoring.py"
        
        if not os.path.exists(sector_file):
            print("[CMC CACHE] Warning: breakout_cluster_scoring.py not found")
            return {}
        
        # Generate suggestions
        suggestions = update_sector_mapping_from_cache()
        
        if not suggestions:
            print("[CMC CACHE] No sector suggestions generated")
            return {}
        
        # Read current file
        with open(sector_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find SECTOR_MAPPING section
        start_marker = "SECTOR_MAPPING = {"
        end_marker = "}"
        
        start_idx = content.find(start_marker)
        if start_idx == -1:
            print("[CMC CACHE] SECTOR_MAPPING not found in file")
            return {}
        
        # Extract existing mapping
        end_idx = content.find(end_marker, start_idx) + 1
        mapping_section = content[start_idx:end_idx]
        
        # Parse existing mapping (simple regex approach)
        import re
        existing_tokens = re.findall(r'"([A-Z]+USDT)": "([^"]+)"', mapping_section)
        existing_mapping = dict(existing_tokens)
        
        print(f"[CMC CACHE] Found {len(existing_mapping)} existing mappings")
        
        # Merge with suggestions (don't overwrite existing)
        new_mappings = {}
        for symbol, sector in suggestions.items():
            if symbol not in existing_mapping:
                new_mappings[symbol] = sector
        
        print(f"[CMC CACHE] Adding {len(new_mappings)} new mappings")
        
        # Generate enhanced mapping report
        enhancement_report = {
            "existing_mappings": len(existing_mapping),
            "new_mappings": len(new_mappings),
            "total_after_merge": len(existing_mapping) + len(new_mappings),
            "new_tokens": new_mappings,
            "sectors_enhanced": {}
        }
        
        # Count by sector
        for sector in new_mappings.values():
            enhancement_report["sectors_enhanced"][sector] = enhancement_report["sectors_enhanced"].get(sector, 0) + 1
        
        # Save enhancement report
        report_file = "data/cache/sector_enhancement_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(enhancement_report, f, indent=2, ensure_ascii=False)
        
        print(f"[CMC CACHE] Enhancement report saved to {report_file}")
        
        return enhancement_report
        
    except Exception as e:
        print(f"[CMC CACHE] Error integrating with sector mapping: {e}")
        return {}


def update_sector_mapping_from_cache():
    """
    Update SECTOR_MAPPING suggestions based on CMC cache
    This function analyzes tags and suggests sector mappings
    """
    try:
        cache = load_existing_cache()
        
        # Define tag to sector mapping
        tag_sector_map = {
            # DeFi tags
            'defi': 'defi',
            'decentralized-exchange': 'defi',
            'yield-farming': 'defi',
            'automated-market-maker': 'defi',
            'lending-borrowing': 'defi',
            'amm': 'defi',
            
            # Gaming tags
            'gaming': 'gaming',
            'metaverse': 'gaming',
            'nft': 'gaming',
            'play-to-earn': 'gaming',
            'virtual-reality': 'gaming',
            
            # Layer 1 tags
            'smart-contracts': 'layer1',
            'platform': 'layer1',
            'blockchain': 'layer1',
            'consensus': 'layer1',
            
            # Layer 2 tags
            'scaling': 'layer2',
            'layer-2': 'layer2',
            'rollup': 'layer2',
            'sidechain': 'layer2',
            
            # AI tags
            'artificial-intelligence': 'ai',
            'machine-learning': 'ai',
            'ai': 'ai',
            'data': 'ai',
            
            # Meme tags
            'memes': 'meme',
            'meme': 'meme',
            'dog-themed': 'meme',
            
            # Infrastructure tags
            'oracle': 'infrastructure',
            'storage': 'infrastructure',
            'infrastructure': 'infrastructure',
            
            # Privacy tags
            'privacy': 'privacy',
            'anonymity': 'privacy'
        }
        
        sector_suggestions = {}
        
        for symbol, data in cache.items():
            tags = [tag.lower().replace(' ', '-') for tag in data.get('tags', [])]
            
            # Find sector based on tags
            detected_sector = None
            for tag in tags:
                if tag in tag_sector_map:
                    detected_sector = tag_sector_map[tag]
                    break
            
            if detected_sector:
                usdt_symbol = f"{symbol}USDT"
                sector_suggestions[usdt_symbol] = detected_sector
        
        print(f"[CMC CACHE] Generated {len(sector_suggestions)} sector suggestions")
        
        # Save suggestions to file for manual review
        suggestions_file = "data/cache/sector_mapping_suggestions.json"
        ensure_cache_directory()
        
        with open(suggestions_file, 'w', encoding='utf-8') as f:
            json.dump(sector_suggestions, f, indent=2, ensure_ascii=False)
        
        print(f"[CMC CACHE] Sector suggestions saved to {suggestions_file}")
        
        return sector_suggestions
        
    except Exception as e:
        print(f"[CMC CACHE] Error updating sector mapping: {e}")
        return {}


if __name__ == "__main__":
    print("=== CoinMarketCap Category Cache Builder ===")
    
    # Check if API key is available
    try:
        api_key = get_cmc_api_key()
        print("✅ CMC API key found")
        
        # Build cache
        cache = build_category_cache()
        
        # Show statistics
        stats = get_cache_statistics()
        print("\n=== Cache Statistics ===")
        print(f"Total tokens: {stats.get('total_tokens', 0)}")
        print(f"Categories: {stats.get('categories', {})}")
        print(f"Top tags: {[f'{tag}({count})' for tag, count in stats.get('top_tags', [])[:5]]}")
        
        # Generate sector mapping suggestions
        print("\n=== Generating Sector Mapping Suggestions ===")
        suggestions = update_sector_mapping_from_cache()
        print(f"Generated {len(suggestions)} suggestions")
        
        # Test integration with existing mappings
        print("\n=== Testing Integration with Existing SECTOR_MAPPING ===")
        integration_report = integrate_with_existing_sector_mapping()
        if integration_report:
            print(f"Integration completed - {integration_report.get('new_mappings', 0)} new mappings added")
        
    except ValueError as e:
        print(f"❌ {e}")
        print("Please set CMC_API_KEY environment variable")
    except Exception as e:
        print(f"❌ Error: {e}")