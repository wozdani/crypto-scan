#!/usr/bin/env python3
"""
🛰️ STAGE 12: Satellite Scanner - Automatyczny skan tokenów bliźniaczych
🎯 Cel: Uruchamianie natychmiastowych skanów na tokenach powiązanych z tokenami generującymi silne stealth alerty

📊 Funkcjonalności:
- Satelitarny skan po wykryciu stealth alertu
- Asynchroniczne skanowanie tokenów bliźniaczych
- Integracja z systemem alertów
- Monitoring wydajności satelitarnego skanu
"""

import asyncio
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sys
import os

# Import modułów stealth engine
from .token_clusters import (
    get_stealth_twins,
    should_trigger_satellite_scan,
    record_satellite_result,
    get_satellite_statistics
)

class SatelliteScanner:
    """
    🛰️ Satelitarny skaner tokenów bliźniaczych
    """
    
    def __init__(self):
        self.scan_queue = asyncio.Queue()
        self.active_scans = set()
        self.scan_results = {}
        self.worker_task = None
        self.is_running = False
        self.lock = threading.Lock()
        
        print("[SATELLITE SCANNER] Initialized satellite scanning system")
    
    async def start_scanner(self):
        """🚀 Uruchom satelitarny skaner"""
        if self.is_running:
            return
        
        self.is_running = True
        self.worker_task = asyncio.create_task(self._worker_loop())
        print("[SATELLITE SCANNER] 🛰️ Started satellite scanner worker")
    
    async def stop_scanner(self):
        """🛑 Zatrzymaj satelitarny skaner"""
        self.is_running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        print("[SATELLITE SCANNER] Stopped satellite scanner")
    
    async def _worker_loop(self):
        """🔄 Główna pętla pracy satelitarnego skanera"""
        while self.is_running:
            try:
                # Pobierz zadanie z kolejki (timeout 5s)
                try:
                    scan_task = await asyncio.wait_for(self.scan_queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue
                
                # Wykonaj satelitarny skan
                await self._execute_satellite_scan(scan_task)
                
                # Oznacz zadanie jako zakończone
                self.scan_queue.task_done()
                
            except Exception as e:
                print(f"[SATELLITE SCANNER ERROR] Worker loop error: {e}")
                await asyncio.sleep(1)
    
    async def _execute_satellite_scan(self, scan_task: Dict):
        """
        🎯 Wykonaj satelitarny skan dla konkretnego zadania
        """
        triggered_by = scan_task["triggered_by"]
        trigger_score = scan_task["trigger_score"]
        twins = scan_task["twins"]
        
        print(f"[SATELLITE SCAN] Processing {len(twins)} twins for {triggered_by} (score: {trigger_score:.2f})")
        
        # Skanuj każdy twin token
        for twin_symbol in twins:
            if twin_symbol in self.active_scans:
                print(f"[SATELLITE SKIP] {twin_symbol} already being scanned")
                continue
            
            # Dodaj do aktywnych skanów
            with self.lock:
                self.active_scans.add(twin_symbol)
            
            try:
                # Wykonaj pojedynczy skan twin tokenu
                result = await self._scan_single_twin(twin_symbol, triggered_by, trigger_score)
                
                # Zapisz wynik
                self.scan_results[twin_symbol] = result
                
                # Zapisz do cache
                record_satellite_result(
                    symbol=twin_symbol,
                    triggered_by=triggered_by,
                    stealth_score=result.get("stealth_score", 0.0),
                    success=result.get("success", False),
                    error_message=result.get("error", None)
                )
                
                print(f"[SATELLITE RESULT] {twin_symbol}: "
                      f"stealth={result.get('stealth_score', 0.0):.2f}, "
                      f"success={result.get('success', False)}")
                
            except Exception as e:
                print(f"[SATELLITE ERROR] Failed to scan {twin_symbol}: {e}")
                record_satellite_result(
                    symbol=twin_symbol,
                    triggered_by=triggered_by,
                    stealth_score=0.0,
                    success=False,
                    error_message=str(e)
                )
            finally:
                # Usuń z aktywnych skanów
                with self.lock:
                    self.active_scans.discard(twin_symbol)
            
            # Krótka przerwa między skanami
            await asyncio.sleep(0.5)
    
    async def _scan_single_twin(self, symbol: str, triggered_by: str, trigger_score: float) -> Dict:
        """
        🔍 Wykonaj skan pojedynczego twin tokenu
        """
        try:
            # Import funkcji skanowania (dynamiczny import aby uniknąć circular imports)
            try:
                from scan_token_async import scan_token_async
                scan_function = scan_token_async
            except ImportError:
                # Fallback - użyj uproszczonej wersji
                return await self._simplified_twin_scan(symbol, triggered_by)
            
            # Wykonaj pełny skan tokenu
            print(f"[SATELLITE SCAN] Starting full scan for {symbol} (triggered by {triggered_by})")
            
            # Przygotuj dane priority dla twin skanu
            priority_info = {
                "satellite_scan": True,
                "triggered_by": triggered_by,
                "trigger_score": trigger_score,
                "scan_type": "twin_satellite"
            }
            
            # Wykonaj skan
            scan_result = await scan_function(symbol, priority_info=priority_info)
            
            if scan_result and isinstance(scan_result, dict):
                stealth_score = scan_result.get("stealth_score", 0.0)
                tjde_score = scan_result.get("tjde_score", 0.0)
                
                # Sprawdź czy skan był udany
                success = stealth_score >= 2.5 or tjde_score >= 0.4
                
                return {
                    "success": success,
                    "stealth_score": stealth_score,
                    "tjde_score": tjde_score,
                    "scan_result": scan_result,
                    "triggered_by": triggered_by,
                    "scan_timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {
                    "success": False,
                    "stealth_score": 0.0,
                    "tjde_score": 0.0,
                    "error": "Invalid scan result",
                    "triggered_by": triggered_by,
                    "scan_timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            print(f"[SATELLITE SCAN ERROR] {symbol}: {e}")
            return {
                "success": False,
                "stealth_score": 0.0,
                "tjde_score": 0.0,
                "error": str(e),
                "triggered_by": triggered_by,
                "scan_timestamp": datetime.utcnow().isoformat()
            }
    
    async def _simplified_twin_scan(self, symbol: str, triggered_by: str) -> Dict:
        """
        📊 Uproszczony skan twin tokenu (fallback)
        """
        try:
            # Import stealth engine do podstawowego skanu
            from .stealth_engine import compute_stealth_score
            
            # Symuluj podstawowe dane rynkowe (w rzeczywistej implementacji pobierz z API)
            mock_market_data = {
                "symbol": symbol,
                "price": 1.0,
                "volume_24h": 1000000,
                "candles_15m": [],
                "orderbook": {"bids": [], "asks": []},
                "dex_inflow": 0
            }
            
            # Oblicz stealth score
            stealth_result = compute_stealth_score(mock_market_data)
            stealth_score = stealth_result.get("score", 0.0)
            
            return {
                "success": stealth_score >= 2.5,
                "stealth_score": stealth_score,
                "tjde_score": 0.0,
                "scan_type": "simplified",
                "triggered_by": triggered_by,
                "scan_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "stealth_score": 0.0,
                "error": f"Simplified scan failed: {e}",
                "triggered_by": triggered_by,
                "scan_timestamp": datetime.utcnow().isoformat()
            }
    
    async def queue_satellite_scan(self, symbol: str, stealth_score: float) -> bool:
        """
        📬 Dodaj satelitarny skan do kolejki
        
        Args:
            symbol: Symbol tokenu, który wygenerował alert
            stealth_score: Score stealth dla tokenu
            
        Returns:
            True jeśli skan został dodany do kolejki
        """
        # Sprawdź czy należy uruchomić satelitarny skan
        if not should_trigger_satellite_scan(symbol, stealth_score):
            return False
        
        # Pobierz tokeny bliźniacze
        twins = get_stealth_twins(symbol)
        if not twins:
            return False
        
        # Przygotuj zadanie satelitarnego skanu
        scan_task = {
            "triggered_by": symbol,
            "trigger_score": stealth_score,
            "twins": twins,
            "queued_at": datetime.utcnow().isoformat()
        }
        
        # Dodaj do kolejki
        await self.scan_queue.put(scan_task)
        
        print(f"[SATELLITE QUEUE] Queued satellite scan for {len(twins)} twins "
              f"(triggered by {symbol}, score: {stealth_score:.2f})")
        
        return True
    
    def get_scanner_status(self) -> Dict:
        """
        📊 Pobierz status satelitarnego skanera
        """
        with self.lock:
            return {
                "is_running": self.is_running,
                "queue_size": self.scan_queue.qsize() if hasattr(self.scan_queue, 'qsize') else 0,
                "active_scans": len(self.active_scans),
                "active_scan_symbols": list(self.active_scans),
                "total_results": len(self.scan_results),
                "recent_results": len([
                    r for r in self.scan_results.values()
                    if (datetime.utcnow() - datetime.fromisoformat(r["scan_timestamp"])).total_seconds() < 3600
                ])
            }
    
    def get_recent_results(self, hours: int = 24) -> List[Dict]:
        """
        📈 Pobierz ostatnie wyniki satelitarnych skanów
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        recent_results = []
        for symbol, result in self.scan_results.items():
            if datetime.fromisoformat(result["scan_timestamp"]) > cutoff:
                recent_results.append({
                    "symbol": symbol,
                    **result
                })
        
        # Sortuj według timestamp (najnowsze pierwsze)
        recent_results.sort(
            key=lambda x: x["scan_timestamp"],
            reverse=True
        )
        
        return recent_results

# Global instance
_satellite_scanner = None

async def get_satellite_scanner() -> SatelliteScanner:
    """Pobierz globalną instancję satelitarnego skanera"""
    global _satellite_scanner
    if _satellite_scanner is None:
        _satellite_scanner = SatelliteScanner()
        await _satellite_scanner.start_scanner()
    return _satellite_scanner

async def handle_stealth_alert(symbol: str, stealth_score: float, 
                              alert_data: Dict = None) -> bool:
    """
    🚨 Handle stealth alert i uruchom satelitarny skan jeśli potrzeba
    
    Args:
        symbol: Symbol tokenu, który wygenerował alert
        stealth_score: Score stealth
        alert_data: Dodatkowe dane alertu
        
    Returns:
        True jeśli satelitarny skan został uruchomiony
    """
    try:
        # Pobierz satelitarny skaner
        scanner = await get_satellite_scanner()
        
        # Sprawdź czy uruchomić satelitarny skan
        satellite_triggered = await scanner.queue_satellite_scan(symbol, stealth_score)
        
        if satellite_triggered:
            print(f"[SATELLITE ALERT] {symbol} triggered satellite scan (score: {stealth_score:.2f})")
            
            # Możesz dodać dodatkową logikę alertu tutaj
            if alert_data:
                print(f"[SATELLITE CONTEXT] Alert data: {alert_data}")
        
        return satellite_triggered
        
    except Exception as e:
        print(f"[SATELLITE ALERT ERROR] Failed to handle stealth alert for {symbol}: {e}")
        return False

async def get_satellite_status() -> Dict:
    """
    📊 Convenience function: Pobierz status satelitarnego skanera
    """
    try:
        scanner = await get_satellite_scanner()
        return scanner.get_scanner_status()
    except Exception as e:
        return {"error": str(e), "is_running": False}

async def get_satellite_results(hours: int = 24) -> List[Dict]:
    """
    📈 Convenience function: Pobierz ostatnie wyniki satelitarne
    """
    try:
        scanner = await get_satellite_scanner()
        return scanner.get_recent_results(hours)
    except Exception as e:
        print(f"[SATELLITE RESULTS ERROR] {e}")
        return []

def cleanup_satellite_scanner():
    """
    🧹 Cleanup function dla satelitarnego skanera
    """
    global _satellite_scanner
    if _satellite_scanner and _satellite_scanner.is_running:
        try:
            asyncio.create_task(_satellite_scanner.stop_scanner())
        except Exception as e:
            print(f"[SATELLITE CLEANUP ERROR] {e}")