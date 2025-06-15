import os
import time

def should_rebuild_cache(hours=24):
    try:
        stat = os.stat("cache/coingecko_cache.json")
        modified = stat.st_mtime
        now = time.time()
        return (now - modified) > hours * 3600
    except FileNotFoundError:
        return True
