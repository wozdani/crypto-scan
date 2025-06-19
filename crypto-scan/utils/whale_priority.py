"""
Whale Priority Scanner - Pre-Pump 2.0
Inteligentne priorytetowanie tokenów na podstawie wcześniejszej aktywności whale
"""

import json
import os
import csv
from datetime import datetime, timedelta, timezone
from collections import defaultdict

def load_whale_history():
    """
    Ładuje historię whale activity z ppwcs_scores.json i stage2_stage1.csv
    Returns: dict z symbolami i ich whale activity timestamps
    """
    whale_history = defaultdict(list)
    
    # Sprawdź ppwcs_scores.json
    ppwcs_file = "data/ppwcs_scores.json"
    if os.path.exists(ppwcs_file):
        try:
            with open(ppwcs_file, 'r', encoding='utf-8') as f:
                ppwcs_data = json.load(f)
                
            for symbol, data in ppwcs_data.items():
                if isinstance(data, dict) and data.get("whale_activity") is True:
                    timestamp_str = data.get("timestamp")
                    if timestamp_str:
                        try:
                            # Parse timestamp 
                            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            whale_history[symbol].append(dt)
                        except:
                            continue
                            
        except Exception as e:
            print(f"❌ Error loading ppwcs_scores.json: {e}")
    
    # Sprawdź stage2_stage1.csv
    csv_file = "data/stage2_stage1.csv"
    if os.path.exists(csv_file):
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("whale_activity") == "True":
                        symbol = row.get("symbol")
                        timestamp_str = row.get("timestamp")
                        if symbol and timestamp_str:
                            try:
                                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                whale_history[symbol].append(dt)
                            except:
                                continue
                                
        except Exception as e:
            print(f"❌ Error loading stage2_stage1.csv: {e}")
    
    return whale_history

def get_priority_tokens(whale_history, hours_back=1, scan_count=4):
    """
    Identyfikuje tokeny z whale activity w ostatnich N skanach
    
    Args:
        whale_history: dict z historią whale activity
        hours_back: ile godzin wstecz sprawdzać (domyślnie 1h = 4 skany po 15min)
        scan_count: alternatywnie - liczba ostatnich skanów
    
    Returns:
        dict: {symbol: {'last_whale': datetime, 'whale_count': int, 'priority_score': int}}
    """
    now = datetime.now(timezone.utc)
    cutoff_time = now - timedelta(hours=hours_back)
    
    priority_tokens = {}
    
    for symbol, timestamps in whale_history.items():
        # Filtruj ostatnie whale activity
        recent_whales = [ts for ts in timestamps if ts >= cutoff_time]
        
        if recent_whales:
            latest_whale = max(recent_whales)
            whale_count = len(recent_whales)
            
            # Oblicz priority score na podstawie:
            # - liczby whale TX
            # - jak dawno ostatnia aktywność
            minutes_ago = (now - latest_whale).total_seconds() / 60
            recency_score = max(0, 60 - minutes_ago)  # 60 punktów za świeżość
            count_score = min(whale_count * 10, 40)   # max 40 punktów za ilość
            
            priority_score = int(recency_score + count_score)
            
            priority_tokens[symbol] = {
                'last_whale': latest_whale,
                'whale_count': whale_count,
                'priority_score': priority_score,
                'minutes_ago': int(minutes_ago)
            }
    
    return priority_tokens

def detect_whale_patterns(whale_history, symbol):
    """
    Wykrywa wzorce w whale activity (powtarzalne adresy, clustery)
    
    Returns:
        dict: informacje o wzorcach whale
    """
    patterns = {
        "under_watch": False,
        "whale_pattern": None,
        "whale_score_boost": 0
    }
    
    now = datetime.now(timezone.utc)
    timestamps = whale_history.get(symbol, [])
    
    # Sprawdź czy token jest "under watch" (whale activity 4-6h temu)
    extended_cutoff = now - timedelta(hours=6)
    recent_cutoff = now - timedelta(hours=1)
    
    extended_whales = [ts for ts in timestamps if extended_cutoff <= ts < recent_cutoff]
    if extended_whales:
        patterns["under_watch"] = True
        patterns["whale_score_boost"] += 3
    
    # Sprawdź clustery whale TX w ostatniej godzinie
    recent_whales = [ts for ts in timestamps if ts >= recent_cutoff]
    if len(recent_whales) >= 3:
        patterns["whale_pattern"] = "repeat_cluster"
        patterns["whale_score_boost"] += 5
    elif len(recent_whales) >= 2:
        patterns["whale_pattern"] = "double_whale"
        patterns["whale_score_boost"] += 3
    
    return patterns

def prioritize_whale_tokens(symbols):
    """
    Główna funkcja priorytetowania tokenów na podstawie whale activity
    
    Args:
        symbols: lista symboli do przeskanowania
    
    Returns:
        tuple: (priority_symbols, regular_symbols, priority_info)
    """
    print("🐋 Loading whale activity history...")
    
    whale_history = load_whale_history()
    priority_tokens = get_priority_tokens(whale_history)
    
    if not priority_tokens:
        print("📊 No recent whale activity found - proceeding with regular order")
        return symbols, [], {}
    
    # Sortuj tokeny według priority score
    sorted_priorities = sorted(
        priority_tokens.items(), 
        key=lambda x: x[1]['priority_score'], 
        reverse=True
    )
    
    # Podziel symbole na priorytetowe i zwykłe
    priority_symbols = []
    priority_info = {}
    
    for symbol, info in sorted_priorities:
        if symbol in symbols:
            priority_symbols.append(symbol)
            priority_info[symbol] = info
            
            # Dodaj wzorce whale
            patterns = detect_whale_patterns(whale_history, symbol)
            priority_info[symbol].update(patterns)
    
    # Usuń priorytetowe symbole z zwykłej listy
    regular_symbols = [s for s in symbols if s not in priority_symbols]
    
    print(f"🔥 Priority tokens: {len(priority_symbols)}")
    for symbol in priority_symbols[:5]:  # Pokaż top 5
        info = priority_info[symbol]
        print(f"   {symbol}: score={info['priority_score']}, whale={info['minutes_ago']}min ago, count={info['whale_count']}")
    
    if len(priority_symbols) > 5:
        print(f"   ... and {len(priority_symbols) - 5} more")
    
    return priority_symbols + regular_symbols, priority_symbols, priority_info

def save_priority_report(priority_info):
    """
    Zapisuje raport priorytetowania do pliku
    """
    if not priority_info:
        return
        
    os.makedirs("data/priority", exist_ok=True)
    
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority_tokens": len(priority_info),
        "tokens": {}
    }
    
    for symbol, info in priority_info.items():
        report["tokens"][symbol] = {
            "priority_score": info['priority_score'],
            "whale_count": info['whale_count'],
            "minutes_ago": info['minutes_ago'],
            "under_watch": info.get('under_watch', False),
            "whale_pattern": info.get('whale_pattern'),
            "whale_score_boost": info.get('whale_score_boost', 0),
            "last_whale": info['last_whale'].isoformat()
        }
    
    # Zapisz aktualny raport
    with open("data/priority/whale_priority_current.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Dodaj do historii
    date_str = datetime.now(timezone.utc).strftime('%Y%m%d')
    history_file = f"data/priority/whale_priority_{date_str}.json"
    
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except:
            pass
    
    history.append(report)
    
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Priority report saved: {len(priority_info)} tokens prioritized")

def get_whale_boost_for_symbol(symbol, priority_info):
    """
    Zwraca whale boost score dla konkretnego symbolu
    """
    if symbol not in priority_info:
        return 0
        
    info = priority_info[symbol]
    boost = info.get('whale_score_boost', 0)
    
    # Dodatkowy boost za wysoką pozycję w priorytecie
    if info['priority_score'] >= 80:
        boost += 5
    elif info['priority_score'] >= 60:
        boost += 3
    
    return boost