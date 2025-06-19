"""
Whale Priority Test Data Generator
Tworzy przykładowe dane whale activity dla demonstracji systemu priorytetowania
"""

import json
import os
from datetime import datetime, timedelta, timezone

def create_test_whale_data():
    """
    Tworzy przykładowe dane whale activity dla testowania systemu priorytetowego
    """
    # Ensure directories exist
    os.makedirs("data", exist_ok=True)
    
    now = datetime.now(timezone.utc)
    
    # Test data - various whale activities in last hour
    test_symbols = [
        {
            "symbol": "BTCUSDT",
            "whale_activity": True,
            "minutes_ago": 15,
            "whale_count": 3
        },
        {
            "symbol": "ETHUSDT", 
            "whale_activity": True,
            "minutes_ago": 30,
            "whale_count": 2
        },
        {
            "symbol": "SOLUSDT",
            "whale_activity": True,
            "minutes_ago": 45,
            "whale_count": 1
        },
        {
            "symbol": "ADAUSDT",
            "whale_activity": True,
            "minutes_ago": 5,
            "whale_count": 4
        }
    ]
    
    # Create ppwcs_scores.json with whale activity
    ppwcs_scores = {}
    
    for symbol_data in test_symbols:
        timestamp = now - timedelta(minutes=symbol_data["minutes_ago"])
        
        ppwcs_scores[symbol_data["symbol"]] = {
            "symbol": symbol_data["symbol"],
            "timestamp": timestamp.isoformat(),
            "ppwcs_score": 45,
            "whale_activity": symbol_data["whale_activity"],
            "dex_inflow": False,
            "volume_spike": True,
            "orderbook_anomaly": False
        }
    
    # Save to ppwcs_scores.json
    with open("data/ppwcs_scores.json", 'w', encoding='utf-8') as f:
        json.dump(ppwcs_scores, f, indent=2, ensure_ascii=False)
    
    # Create additional historical entries for extended tracking
    stage2_data = []
    
    for symbol_data in test_symbols:
        # Add multiple entries for each symbol to simulate history
        for i in range(symbol_data["whale_count"]):
            timestamp = now - timedelta(minutes=symbol_data["minutes_ago"] + (i * 10))
            
            stage2_data.append({
                "symbol": symbol_data["symbol"],
                "timestamp": timestamp.isoformat(),
                "whale_activity": "True",
                "dex_inflow": "False",
                "ppwcs_score": 40 + (i * 5),
                "stage2_pass": "True"
            })
    
    # Create CSV header and data
    csv_content = "symbol,timestamp,whale_activity,dex_inflow,ppwcs_score,stage2_pass\n"
    for entry in stage2_data:
        csv_content += f"{entry['symbol']},{entry['timestamp']},{entry['whale_activity']},{entry['dex_inflow']},{entry['ppwcs_score']},{entry['stage2_pass']}\n"
    
    # Save to stage2_stage1.csv
    with open("data/stage2_stage1.csv", 'w', encoding='utf-8') as f:
        f.write(csv_content)
    
    print(f"✅ Created test whale data for {len(test_symbols)} symbols")
    print(f"📊 PPWCS entries: {len(ppwcs_scores)}")
    print(f"📊 CSV entries: {len(stage2_data)}")
    
    return test_symbols

def add_extended_watch_symbols():
    """
    Dodaje symbole z whale activity sprzed 4-6 godzin (under watch)
    """
    now = datetime.now(timezone.utc)
    
    extended_symbols = [
        {
            "symbol": "DOTUSDT",
            "whale_activity": True,
            "hours_ago": 5,
            "whale_count": 2
        },
        {
            "symbol": "LINKUSDT",
            "whale_activity": True, 
            "hours_ago": 4,
            "whale_count": 1
        }
    ]
    
    # Load existing ppwcs_scores.json
    ppwcs_file = "data/ppwcs_scores.json"
    ppwcs_scores = {}
    
    if os.path.exists(ppwcs_file):
        try:
            with open(ppwcs_file, 'r', encoding='utf-8') as f:
                ppwcs_scores = json.load(f)
        except:
            pass
    
    # Add extended watch symbols
    for symbol_data in extended_symbols:
        timestamp = now - timedelta(hours=symbol_data["hours_ago"])
        
        ppwcs_scores[symbol_data["symbol"]] = {
            "symbol": symbol_data["symbol"],
            "timestamp": timestamp.isoformat(),
            "ppwcs_score": 35,
            "whale_activity": symbol_data["whale_activity"],
            "dex_inflow": False,
            "volume_spike": False,
            "orderbook_anomaly": True
        }
    
    # Save updated ppwcs_scores.json
    with open(ppwcs_file, 'w', encoding='utf-8') as f:
        json.dump(ppwcs_scores, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Added {len(extended_symbols)} extended watch symbols")
    
    return extended_symbols

def clear_test_data():
    """
    Czyści testowe dane whale activity
    """
    files_to_clear = [
        "data/ppwcs_scores.json",
        "data/stage2_stage1.csv",
        "data/priority/whale_priority_current.json"
    ]
    
    for file_path in files_to_clear:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🗑️ Cleared {file_path}")
    
    print("✅ Test data cleared")

if __name__ == "__main__":
    print("🧪 Creating whale priority test data...")
    
    # Create main test data
    test_symbols = create_test_whale_data()
    
    # Add extended watch symbols
    extended_symbols = add_extended_watch_symbols()
    
    print("\n🎯 Test Data Summary:")
    print("Recent whale activity (priority tokens):")
    for symbol in test_symbols:
        print(f"  {symbol['symbol']}: {symbol['whale_count']} TX, {symbol['minutes_ago']}min ago")
    
    print("\nExtended watch symbols:")
    for symbol in extended_symbols:
        print(f"  {symbol['symbol']}: {symbol['whale_count']} TX, {symbol['hours_ago']}h ago")
    
    print(f"\n🚀 Run crypto scanner to see whale priority system in action!")