import json
import os

def load_token_map(filename="token_contract_map.json"):
    """Load token contract mapping from JSON file"""
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Plik {filename} nie znaleziony")
        return {}
    except json.JSONDecodeError:
        print(f"⚠️ Błąd parsowania JSON w {filename}")
        return {}
    except Exception as e:
        print(f"❌ Błąd ładowania mapy tokenów: {e}")
        return {}