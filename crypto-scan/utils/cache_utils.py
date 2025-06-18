import os
import time
import json

def should_rebuild_cache(hours=24):
    cache_file = "cache/coingecko_cache.json"
    
    try:
        # Check if file exists
        if not os.path.exists(cache_file):
            print("🚫 Cache file doesn't exist - rebuild needed")
            return True
        
        # Check if file has actual content (not just {})
        with open(cache_file, "r") as f:
            cache_content = json.load(f)
        if not cache_content or len(cache_content) == 0:
            print("🚫 Cache file is empty - rebuild needed")
            return True
        
        # Check age
        stat = os.stat(cache_file)
        modified = stat.st_mtime
        now = time.time()
        age_hours = (now - modified) / 3600
        
        if age_hours > hours:
            print(f"🚫 Cache expired - age: {age_hours:.1f} hours")
            return True
        
        print(f"✅ Cache valid - contains {len(cache_content)} tokens, age: {age_hours:.1f} hours")
        return False
        
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        print(f"🚫 Cache error: {e} - rebuild needed")
        return True
