import json
import os

TAG_SCORES = {
    "listing": (10, False),
    "partnership": (10, False),
    "cex_listed": (5, False),
    "airdrop": (3, False),
    "presale": (3, False),
    "burn": (0, False),
    "mint": (0, False),
    "lock": (0, False),
    "drama": (-10, True),
    "unlock": (-10, True),
    "exploit": (-15, True),
    "rug": (-1000, True),
    "delisting": (-1000, True)
}

DEFAULT_SCORE = (0, False)

TAGS_FILE = os.path.join("data", "token_tags.json")

def load_tags():
    """Load token tags from JSON file"""
    try:
        with open(TAGS_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Błąd ładowania token_tags.json: {e}")
        return {}

def detect_stage_minus2_2(symbol):
    """
    Stage -2.2: Advanced pattern recognition
    Detects specific events and news that could impact token price
    
    Returns:
        - tag: event tag string or None
        - score_boost: numerical boost/penalty for PPWCS
        - risk_flag: boolean indicating if this is a risky event
    """
    tags = load_tags()
    tag = tags.get(symbol.upper())
    
    if tag:
        score_boost, risk_flag = TAG_SCORES.get(tag.lower(), DEFAULT_SCORE)
        print(f"📰 {symbol}: wykryto tag '{tag}', boost={score_boost}, risk={risk_flag}")
        return tag, score_boost, risk_flag
    else:
        return None, 0, False

def get_available_tags():
    """Get list of all available event tags and their scores"""
    return TAG_SCORES

def update_token_tag(symbol, tag):
    """Update or add a tag for a specific token"""
    try:
        tags = load_tags()
        tags[symbol.upper()] = tag
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(TAGS_FILE), exist_ok=True)
        
        with open(TAGS_FILE, "w") as f:
            json.dump(tags, f, indent=2)
        
        print(f"✅ Zaktualizowano tag dla {symbol}: {tag}")
        return True
    except Exception as e:
        print(f"❌ Błąd aktualizacji tagu dla {symbol}: {e}")
        return False

def remove_token_tag(symbol):
    """Remove tag for a specific token"""
    try:
        tags = load_tags()
        if symbol.upper() in tags:
            del tags[symbol.upper()]
            
            with open(TAGS_FILE, "w") as f:
                json.dump(tags, f, indent=2)
            
            print(f"✅ Usunięto tag dla {symbol}")
            return True
        else:
            print(f"⚠️ Brak tagu do usunięcia dla {symbol}")
            return False
    except Exception as e:
        print(f"❌ Błąd usuwania tagu dla {symbol}: {e}")
        return False