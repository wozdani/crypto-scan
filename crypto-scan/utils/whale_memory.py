#!/usr/bin/env python3
"""
Whale Memory System - Pamięć adresów wielorybów
Zapamiętywanie adresów powtarzających się w whale_ping i dex_inflow
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Konfiguracja pamięci wielorybów
WALLET_MEMORY_PATH = "crypto-scan/cache/repeat_wallets.json"
MEMORY_WINDOW_SECONDS = 7 * 24 * 60 * 60  # 7 dni
MIN_REPEAT_COUNT = 3  # Minimum powtórzeń dla uznania za wieloryba

class WhaleMemoryManager:
    """
    Manager pamięci adresów wielorybów
    Przechowuje historię whale_ping i dex_inflow adresów
    """
    
    def __init__(self, memory_path: str = WALLET_MEMORY_PATH):
        self.memory_path = memory_path
        self.memory_window = MEMORY_WINDOW_SECONDS
        self.min_repeat_count = MIN_REPEAT_COUNT
        
        # Zapewnij istnienie katalogu
        os.makedirs(os.path.dirname(memory_path), exist_ok=True)
        
        # Inicjalizuj plik jeśli nie istnieje
        if not os.path.exists(memory_path):
            self._save_memory({})
    
    def _load_memory(self) -> Dict:
        """Załaduj pamięć wielorybów z pliku"""
        try:
            if os.path.exists(self.memory_path):
                with open(self.memory_path, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"[WHALE MEMORY] Error loading memory: {e}")
            return {}
    
    def _save_memory(self, memory: Dict) -> None:
        """Zapisz pamięć wielorybów do pliku"""
        try:
            with open(self.memory_path, "w") as f:
                json.dump(memory, f, indent=2)
        except Exception as e:
            print(f"[WHALE MEMORY] Error saving memory: {e}")
    
    def update_wallet_memory(self, token: str, address: str, timestamp: Optional[int] = None,
                           source: str = "unknown") -> int:
        """
        Aktualizuj pamięć wieloryba
        
        Args:
            token: Symbol tokena (np. BTCUSDT)
            address: Adres wieloryba
            timestamp: Timestamp w sekundach (domyślnie current time)
            source: Źródło sygnału (whale_ping, dex_inflow)
            
        Returns:
            Liczba wystąpień adresu w ostatnich 7 dniach
        """
        if timestamp is None:
            timestamp = int(time.time())
        
        memory = self._load_memory()
        
        # Inicjalizuj strukturę dla tokena jeśli nie istnieje
        if token not in memory:
            memory[token] = {}
        
        token_mem = memory[token]
        
        # Pobierz historię adresu
        if address not in token_mem:
            token_mem[address] = {"timestamps": [], "sources": []}
        
        addr_data = token_mem[address]
        timestamps = addr_data.get("timestamps", [])
        sources = addr_data.get("sources", [])
        
        # Dodaj nowy wpis
        timestamps.append(timestamp)
        sources.append(source)
        
        # Przefiltruj do ostatnich 7 dni
        current_time = int(time.time())
        filtered_timestamps = []
        filtered_sources = []
        
        for ts, src in zip(timestamps, sources):
            if current_time - ts < self.memory_window:
                filtered_timestamps.append(ts)
                filtered_sources.append(src)
        
        # Zapisz przefiltrowane dane
        addr_data["timestamps"] = filtered_timestamps
        addr_data["sources"] = filtered_sources
        token_mem[address] = addr_data
        memory[token] = token_mem
        
        # Zapisz pamięć
        self._save_memory(memory)
        
        print(f"[WHALE MEMORY] {token} {address[:10]}... updated: {len(filtered_timestamps)} entries ({source})")
        
        return len(filtered_timestamps)
    
    def is_repeat_whale(self, token: str, address: str) -> bool:
        """
        Sprawdź czy adres jest powtarzającym się wielorybem
        
        Args:
            token: Symbol tokena
            address: Adres do sprawdzenia
            
        Returns:
            True jeśli adres pojawił się minimum MIN_REPEAT_COUNT razy w ostatnich 7 dniach
        """
        memory = self._load_memory()
        
        token_mem = memory.get(token, {})
        addr_data = token_mem.get(address, {})
        timestamps = addr_data.get("timestamps", [])
        
        # Sprawdź tylko aktualne wpisy (w oknie czasowym)
        current_time = int(time.time())
        recent_count = sum(1 for ts in timestamps if current_time - ts < self.memory_window)
        
        return recent_count >= self.min_repeat_count
    
    def get_repeat_whale_boost(self, token: str, address: str) -> float:
        """
        Oblicz boost score dla powtarzającego się wieloryba
        
        Args:
            token: Symbol tokena
            address: Adres wieloryba
            
        Returns:
            Boost score (0.0 - 1.0)
        """
        if not self.is_repeat_whale(token, address):
            return 0.0
        
        memory = self._load_memory()
        token_mem = memory.get(token, {})
        addr_data = token_mem.get(address, {})
        timestamps = addr_data.get("timestamps", [])
        
        # Oblicz boost na podstawie liczby powtórzeń
        current_time = int(time.time())
        recent_count = sum(1 for ts in timestamps if current_time - ts < self.memory_window)
        
        # Boost formula: min(1.0, (recent_count - 2) * 0.2)
        # 3 wystąpienia = 0.2, 4 = 0.4, 5 = 0.6, 6+ = 1.0
        boost = min(1.0, (recent_count - 2) * 0.2)
        
        print(f"[WHALE MEMORY] {token} {address[:10]}... repeat boost: {boost:.2f} ({recent_count} entries)")
        
        return boost
    
    def get_token_repeat_whales(self, token: str) -> List[Tuple[str, int, List[str]]]:
        """
        Pobierz wszystkich powtarzających się wielorybów dla tokena
        
        Args:
            token: Symbol tokena
            
        Returns:
            Lista tupli (address, count, sources)
        """
        memory = self._load_memory()
        token_mem = memory.get(token, {})
        
        repeat_whales = []
        current_time = int(time.time())
        
        for address, addr_data in token_mem.items():
            timestamps = addr_data.get("timestamps", [])
            sources = addr_data.get("sources", [])
            
            # Filtruj do ostatnich 7 dni
            recent_data = [(ts, src) for ts, src in zip(timestamps, sources) 
                          if current_time - ts < self.memory_window]
            
            if len(recent_data) >= self.min_repeat_count:
                recent_sources = [src for _, src in recent_data]
                repeat_whales.append((address, len(recent_data), recent_sources))
        
        # Sortuj po liczbie wystąpień (malejąco)
        repeat_whales.sort(key=lambda x: x[1], reverse=True)
        
        return repeat_whales
    
    def cleanup_old_entries(self) -> int:
        """
        Wyczyść stare wpisy (starsze niż 7 dni)
        
        Returns:
            Liczba usuniętych wpisów
        """
        memory = self._load_memory()
        current_time = int(time.time())
        cleaned_count = 0
        
        for token in list(memory.keys()):
            token_mem = memory[token]
            
            for address in list(token_mem.keys()):
                addr_data = token_mem[address]
                timestamps = addr_data.get("timestamps", [])
                sources = addr_data.get("sources", [])
                
                # Przefiltruj do aktualnych wpisów
                filtered_data = [(ts, src) for ts, src in zip(timestamps, sources)
                               if current_time - ts < self.memory_window]
                
                if filtered_data:
                    # Zaktualizuj dane
                    new_timestamps, new_sources = zip(*filtered_data)
                    addr_data["timestamps"] = list(new_timestamps)
                    addr_data["sources"] = list(new_sources)
                    token_mem[address] = addr_data
                else:
                    # Usuń adres jeśli brak aktualnych wpisów
                    del token_mem[address]
                    cleaned_count += 1
            
            # Usuń token jeśli brak adresów
            if not token_mem:
                del memory[token]
            else:
                memory[token] = token_mem
        
        self._save_memory(memory)
        
        if cleaned_count > 0:
            print(f"[WHALE MEMORY] Cleaned {cleaned_count} old entries")
        
        return cleaned_count
    
    def get_memory_stats(self) -> Dict:
        """
        Pobierz statystyki pamięci wielorybów
        
        Returns:
            Słownik ze statystykami
        """
        memory = self._load_memory()
        current_time = int(time.time())
        
        stats = {
            "total_tokens": len(memory),
            "total_addresses": 0,
            "repeat_whales": 0,
            "recent_entries": 0,
            "top_tokens": []
        }
        
        token_stats = []
        
        for token, token_mem in memory.items():
            token_addresses = len(token_mem)
            token_repeat_whales = 0
            token_recent_entries = 0
            
            for address, addr_data in token_mem.items():
                timestamps = addr_data.get("timestamps", [])
                recent_count = sum(1 for ts in timestamps if current_time - ts < self.memory_window)
                
                token_recent_entries += recent_count
                
                if recent_count >= self.min_repeat_count:
                    token_repeat_whales += 1
            
            stats["total_addresses"] += token_addresses
            stats["repeat_whales"] += token_repeat_whales
            stats["recent_entries"] += token_recent_entries
            
            if token_addresses > 0:
                token_stats.append({
                    "token": token,
                    "addresses": token_addresses,
                    "repeat_whales": token_repeat_whales,
                    "recent_entries": token_recent_entries
                })
        
        # Top 5 tokenów po liczbie wielorybów
        token_stats.sort(key=lambda x: x["repeat_whales"], reverse=True)
        stats["top_tokens"] = token_stats[:5]
        
        return stats

# Globalna instancja managera
whale_memory_manager = WhaleMemoryManager()

# Funkcje convenience dla łatwego użycia
def update_whale_memory(token: str, address: str, source: str = "unknown") -> int:
    """Aktualizuj pamięć wieloryba"""
    return whale_memory_manager.update_wallet_memory(token, address, source=source)

def is_repeat_whale(token: str, address: str) -> bool:
    """Sprawdź czy adres jest powtarzającym się wielorybem"""
    return whale_memory_manager.is_repeat_whale(token, address)

def get_repeat_whale_boost(token: str, address: str) -> float:
    """Pobierz boost score dla powtarzającego się wieloryba"""
    return whale_memory_manager.get_repeat_whale_boost(token, address)

def cleanup_whale_memory() -> int:
    """Wyczyść starą pamięć wielorybów"""
    return whale_memory_manager.cleanup_old_entries()

def get_whale_memory_stats() -> Dict:
    """Pobierz statystyki pamięci wielorybów"""
    return whale_memory_manager.get_memory_stats()