#!/usr/bin/env python3
"""
📱 Telegram Alert Manager - Inteligentna kolejka alertów z priority scoring
🎯 Cel: Automatyczne zarządzanie kolejką alertów Telegram z dynamicznym priorytetem

📦 Funkcjonalności:
1. Alert queue management z priority scoring
2. Cykliczne przetwarzanie kolejki alertów
3. Fast-track alerts dla smart money (Stage 7)
4. Dynamic timing na podstawie priority score
5. Integration z alert_router i trigger_alert_system
"""

import time
import threading
import json
import os
from collections import deque
from typing import Dict, List, Optional, Tuple
import asyncio


class TelegramAlertManager:
    """
    📱 Manager kolejki alertów Telegram z priority scoring
    """
    
    def __init__(self, min_delay: int = 30, max_queue_size: int = 100,
                 cache_file: str = "cache/telegram_alert_queue.json"):
        """
        Inicjalizacja Telegram Alert Manager
        
        Args:
            min_delay: Minimalne opóźnienie między alertami (sekundy)
            max_queue_size: Maksymalny rozmiar kolejki
            cache_file: Plik cache dla trwałości kolejki
        """
        
        self.min_delay = min_delay
        self.max_queue_size = max_queue_size
        self.cache_file = cache_file
        
        # Alert queue (deque dla wydajności)
        self.alert_queue = deque(maxlen=max_queue_size)
        
        # Fast-track queue dla natychmiastowych alertów (Stage 7)
        self.fast_track_queue = deque(maxlen=50)
        
        # Processing state
        self.processing_active = False
        self.last_alert_sent = 0
        self.stats = {
            "total_alerts_processed": 0,
            "fast_track_alerts": 0,
            "queue_alerts": 0,
            "alerts_sent_24h": 0,
            "last_reset_24h": time.time()
        }
        
        # Thread safety
        self.queue_lock = threading.Lock()
        
        # Load existing queue
        self._load_queue_cache()
        
        print(f"[TELEGRAM MANAGER] Initialized with min_delay={min_delay}s, max_queue={max_queue_size}")
    
    def _load_queue_cache(self):
        """Wczytaj kolejkę z cache"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    
                # Wczytaj alerts (tylko te nie przetworzone)
                alerts = data.get("alerts", [])
                current_time = time.time()
                
                for alert in alerts:
                    if not alert.get("processed", False):
                        # Sprawdź czy alert nie jest za stary (>24h)
                        if current_time - alert.get("timestamp", 0) < 86400:
                            if alert.get("is_fast_track", False):
                                self.fast_track_queue.append(alert)
                            else:
                                self.alert_queue.append(alert)
                
                # Wczytaj statystyki
                self.stats.update(data.get("stats", {}))
                self.last_alert_sent = data.get("last_alert_sent", 0)
                
                print(f"[TELEGRAM MANAGER] Loaded queue: {len(self.alert_queue)} standard, {len(self.fast_track_queue)} fast-track")
                
        except Exception as e:
            print(f"[TELEGRAM MANAGER] Cache load error: {e}")
    
    def _save_queue_cache(self):
        """Zapisz kolejkę do cache"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            data = {
                "alerts": list(self.alert_queue) + list(self.fast_track_queue),
                "stats": self.stats,
                "last_alert_sent": self.last_alert_sent,
                "last_saved": time.time()
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"[TELEGRAM MANAGER] Cache save error: {e}")
    
    def add_alert(self, alert_data: Dict) -> bool:
        """
        ➕ Dodaj alert do kolejki
        
        Args:
            alert_data: Dane alertu z alert_router
            
        Returns:
            bool: Czy alert został dodany
        """
        
        try:
            with self.queue_lock:
                # Sprawdź czy alert już istnieje (duplikat prevention)
                symbol = alert_data.get("symbol", "")
                existing_symbols = [a.get("symbol", "") for a in self.alert_queue]
                existing_fast_symbols = [a.get("symbol", "") for a in self.fast_track_queue]
                
                if symbol in existing_symbols or symbol in existing_fast_symbols:
                    print(f"[TELEGRAM MANAGER] Duplicate alert skipped: {symbol}")
                    return False
                
                # Dodaj timestamp jeśli nie ma
                if "timestamp" not in alert_data:
                    alert_data["timestamp"] = time.time()
                
                # Fast-track vs normal queue
                if alert_data.get("is_fast_track", False):
                    self.fast_track_queue.append(alert_data)
                    print(f"[TELEGRAM MANAGER] ⚡ Fast-track alert added: {symbol} (priority: {alert_data.get('priority_score', 0):.1f})")
                else:
                    self.alert_queue.append(alert_data)
                    print(f"[TELEGRAM MANAGER] 📥 Standard alert queued: {symbol} (delay: {alert_data.get('delay_seconds', 0)}s)")
                
                # Save to cache
                self._save_queue_cache()
                
                return True
                
        except Exception as e:
            print(f"[TELEGRAM MANAGER] Add alert error: {e}")
            return False
    
    def process_alert_queue(self) -> int:
        """
        🔄 Przetworz kolejkę alertów (jednorazowe wywołanie)
        
        Returns:
            int: Liczba przetworzonych alertów
        """
        
        processed_count = 0
        current_time = time.time()
        
        try:
            with self.queue_lock:
                # 1. Najpierw przetworz fast-track alerts (Stage 7 triggers)
                fast_track_to_process = []
                while self.fast_track_queue:
                    alert = self.fast_track_queue.popleft()
                    if not alert.get("processed", False):
                        fast_track_to_process.append(alert)
                    if len(fast_track_to_process) >= 3:  # Max 3 fast-track na raz
                        break
                
                # 2. Przetworz gotowe standardowe alerty
                standard_to_process = []
                alerts_to_keep = deque()
                
                while self.alert_queue:
                    alert = self.alert_queue.popleft()
                    
                    if alert.get("processed", False):
                        continue  # Skip już przetworzone
                    
                    ready_at = alert.get("ready_at", alert.get("timestamp", 0))
                    
                    if current_time >= ready_at:
                        # Alert gotowy do wysłania
                        standard_to_process.append(alert)
                        if len(standard_to_process) >= 2:  # Max 2 standardowe na raz
                            break
                    else:
                        # Alert nie gotowy - zostaw w kolejce
                        alerts_to_keep.append(alert)
                
                # Przywróć nieopróchnione alerty do kolejki
                self.alert_queue.extend(alerts_to_keep)
                
                # 3. Sortuj standardowe alerty po priority_score (malejąco)
                standard_to_process.sort(key=lambda x: x.get("priority_score", 0), reverse=True)
                
                # 4. Wysyłaj alerty
                all_to_process = fast_track_to_process + standard_to_process
                
                for alert in all_to_process:
                    if self._can_send_alert():
                        success = self._send_telegram_alert(alert)
                        if success:
                            alert["processed"] = True
                            alert["sent_at"] = current_time
                            processed_count += 1
                            self.last_alert_sent = current_time
                            
                            # Update stats
                            self.stats["total_alerts_processed"] += 1
                            if alert.get("is_fast_track", False):
                                self.stats["fast_track_alerts"] += 1
                            else:
                                self.stats["queue_alerts"] += 1
                            
                            # Rate limiting - pauza między alertami
                            if len(all_to_process) > 1:
                                time.sleep(2)  # 2s między alertami
                    else:
                        # Rate limit reached - przywróć alert do kolejki
                        if alert.get("is_fast_track", False):
                            self.fast_track_queue.appendleft(alert)
                        else:
                            self.alert_queue.appendleft(alert)
                        break
                
                # Save cache po przetworzeniu
                if processed_count > 0:
                    self._save_queue_cache()
                
                # Reset 24h stats jeśli potrzeba
                self._reset_24h_stats_if_needed()
            
            if processed_count > 0:
                print(f"[TELEGRAM MANAGER] ✅ Processed {processed_count} alerts")
            
            return processed_count
            
        except Exception as e:
            print(f"[TELEGRAM MANAGER] Process queue error: {e}")
            return 0
    
    def _can_send_alert(self) -> bool:
        """Sprawdź czy można wysłać alert (rate limiting)"""
        current_time = time.time()
        
        # Min delay między alertami
        if current_time - self.last_alert_sent < self.min_delay:
            return False
        
        # Limit alertów na 24h (max 50)
        if self.stats.get("alerts_sent_24h", 0) >= 50:
            return False
        
        return True
    
    def _send_telegram_alert(self, alert_data: Dict) -> bool:
        """
        📤 Wyślij alert na Telegram
        
        Args:
            alert_data: Dane alertu
            
        Returns:
            bool: Czy wysłano pomyślnie
        """
        
        try:
            # 🔔 Sformatuj wiadomość alertu
            symbol = alert_data.get("symbol", "UNKNOWN")
            score = alert_data.get("score", 0)
            priority_score = alert_data.get("priority_score", 0)
            tags = alert_data.get("tags", [])
            trust_score = alert_data.get("trust_score", 0)
            
            # Base alert message
            alert_text = f"🚨 [{symbol}] Score: {score:.2f} | Priority: {priority_score:.2f}"
            
            # Add price info
            price_usd = alert_data.get("price_usd", 0)
            price_change = alert_data.get("price_change_24h", 0)
            if price_usd > 0:
                alert_text += f"\n💰 Price: ${price_usd:.6f} ({price_change:+.1f}%)"
            
            # Add volume info
            volume_24h = alert_data.get("volume_24h", 0)
            if volume_24h > 0:
                alert_text += f"\n📊 Volume: ${volume_24h:,.0f}"
            
            # Add DEX inflow
            dex_inflow = alert_data.get("dex_inflow", 0)
            if dex_inflow > 0:
                alert_text += f"\n🔄 DEX Inflow: ${dex_inflow:,.0f}"
            
            # Add trust score info
            if trust_score > 0:
                alert_text += f"\n🧠 Trust Score: {trust_score:.1%}"
            
            # Add tags
            if tags:
                formatted_tags = " ".join([f"#{tag}" for tag in tags[:5]])  # Max 5 tagów
                alert_text += f"\n🏷️ {formatted_tags}"
            
            # Fast-track indicator
            if alert_data.get("is_fast_track", False):
                fast_track_reason = alert_data.get("fast_track_reason", "High priority")
                alert_text += f"\n⚡ FAST-TRACK: {fast_track_reason}"
            
            # Stage 7 trigger indicator
            if alert_data.get("trigger_detected", False):
                alert_text += f"\n🎯 SMART MONEY DETECTED"
            
            print(f"[TELEGRAM ALERT] Sending: {symbol}")
            print(f"[TELEGRAM ALERT] Message preview: {alert_text[:100]}...")
            
            # TODO: Tutaj dodać prawdziwe wysyłanie na Telegram
            # W tym momencie symulujemy sukces
            
            # Update 24h counter
            self.stats["alerts_sent_24h"] = self.stats.get("alerts_sent_24h", 0) + 1
            
            return True
            
        except Exception as e:
            print(f"[TELEGRAM ALERT] Send error for {alert_data.get('symbol', 'UNKNOWN')}: {e}")
            return False
    
    def _reset_24h_stats_if_needed(self):
        """Reset statystyk 24h jeśli minęła doba"""
        current_time = time.time()
        last_reset = self.stats.get("last_reset_24h", 0)
        
        if current_time - last_reset >= 86400:  # 24h
            self.stats["alerts_sent_24h"] = 0
            self.stats["last_reset_24h"] = current_time
            print("[TELEGRAM MANAGER] 24h stats reset")
    
    def get_queue_status(self) -> Dict:
        """
        📊 Pobierz status kolejki alertów
        
        Returns:
            Dict: Status kolejki
        """
        
        with self.queue_lock:
            current_time = time.time()
            
            # Count ready alerts
            ready_alerts = sum(1 for alert in self.alert_queue 
                             if current_time >= alert.get("ready_at", 0) and not alert.get("processed", False))
            
            # Count pending alerts
            pending_alerts = len(self.alert_queue) - ready_alerts
            
            return {
                "fast_track_queue": len(self.fast_track_queue),
                "standard_queue": len(self.alert_queue),
                "ready_to_send": ready_alerts,
                "pending": pending_alerts,
                "processing_active": self.processing_active,
                "last_alert_sent": self.last_alert_sent,
                "can_send_now": self._can_send_alert(),
                "stats": self.stats.copy()
            }
    
    def cleanup_old_alerts(self, max_age_hours: int = 24):
        """
        🧹 Oczyść stare alerty
        
        Args:
            max_age_hours: Maksymalny wiek alertów w godzinach
        """
        
        try:
            with self.queue_lock:
                current_time = time.time()
                max_age_seconds = max_age_hours * 3600
                
                # Clean standard queue
                cleaned_alerts = deque()
                removed_count = 0
                
                while self.alert_queue:
                    alert = self.alert_queue.popleft()
                    alert_age = current_time - alert.get("timestamp", 0)
                    
                    if alert_age < max_age_seconds and not alert.get("processed", False):
                        cleaned_alerts.append(alert)
                    else:
                        removed_count += 1
                
                self.alert_queue = cleaned_alerts
                
                # Clean fast-track queue
                cleaned_fast_track = deque()
                
                while self.fast_track_queue:
                    alert = self.fast_track_queue.popleft()
                    alert_age = current_time - alert.get("timestamp", 0)
                    
                    if alert_age < max_age_seconds and not alert.get("processed", False):
                        cleaned_fast_track.append(alert)
                    else:
                        removed_count += 1
                
                self.fast_track_queue = cleaned_fast_track
                
                if removed_count > 0:
                    print(f"[TELEGRAM MANAGER] 🧹 Cleaned {removed_count} old alerts")
                    self._save_queue_cache()
                
        except Exception as e:
            print(f"[TELEGRAM MANAGER] Cleanup error: {e}")
    
    def start_processing_loop(self, interval: int = 10):
        """
        🔄 Rozpocznij ciągłe przetwarzanie kolejki alertów
        
        Args:
            interval: Interwał przetwarzania w sekundach
        """
        
        def processing_loop():
            self.processing_active = True
            print(f"[TELEGRAM MANAGER] 🔄 Started processing loop (interval: {interval}s)")
            
            while self.processing_active:
                try:
                    # Process alerts
                    processed = self.process_alert_queue()
                    
                    # Cleanup old alerts co 10 minut
                    if int(time.time()) % 600 == 0:  # Co 10 minut
                        self.cleanup_old_alerts()
                    
                    # Sleep
                    time.sleep(interval)
                    
                except Exception as e:
                    print(f"[TELEGRAM MANAGER] Processing loop error: {e}")
                    time.sleep(interval)
            
            print("[TELEGRAM MANAGER] Processing loop stopped")
        
        # Start in separate thread
        processing_thread = threading.Thread(target=processing_loop, daemon=True)
        processing_thread.start()
    
    def stop_processing_loop(self):
        """⏹️ Zatrzymaj przetwarzanie kolejki"""
        self.processing_active = False
        print("[TELEGRAM MANAGER] Processing loop stopping...")


# Global instance
_telegram_manager = None

def get_telegram_manager() -> TelegramAlertManager:
    """Pobierz globalną instancję Telegram Alert Manager"""
    global _telegram_manager
    if _telegram_manager is None:
        _telegram_manager = TelegramAlertManager()
    return _telegram_manager

def queue_priority_alert(symbol: str, score: float, market_data: Dict,
                        stealth_signals: List[Dict] = None,
                        trust_score: float = 0.0, trigger_detected: bool = False) -> bool:
    """
    🎯 Convenience function: Dodaj alert z priority scoring do kolejki
    
    Args:
        symbol: Symbol tokena
        score: Bazowy score
        market_data: Dane rynkowe
        stealth_signals: Sygnały stealth
        trust_score: Trust score adresów
        trigger_detected: Czy wykryto trigger
        
    Returns:
        bool: Czy alert został dodany
    """
    
    try:
        from .alert_router import route_alert_with_priority
        
        # Generate priority alert data
        alert_data = route_alert_with_priority(
            symbol, score, market_data, stealth_signals, trust_score, trigger_detected
        )
        
        # Add to queue
        manager = get_telegram_manager()
        return manager.add_alert(alert_data)
        
    except Exception as e:
        print(f"[QUEUE PRIORITY ALERT] Error for {symbol}: {e}")
        return False

def get_alert_queue_status() -> Dict:
    """Pobierz status kolejki alertów"""
    manager = get_telegram_manager()
    return manager.get_queue_status()

def process_pending_alerts() -> int:
    """Przetworz oczekujące alerty (jednorazowo)"""
    manager = get_telegram_manager()
    return manager.process_alert_queue()


if __name__ == "__main__":
    # Test Telegram Alert Manager
    print("📱 TELEGRAM ALERT MANAGER TEST")
    print("=" * 50)
    
    # Initialize manager
    manager = TelegramAlertManager()
    
    # Test alert data
    test_alert = {
        "symbol": "TESTUSDT",
        "score": 2.5,
        "priority_score": 4.2,
        "tags": ["trusted", "whale_detected"],
        "is_fast_track": True,
        "fast_track_reason": "Smart money detected",
        "trust_score": 0.85,
        "trigger_detected": True,
        "price_usd": 1.234,
        "volume_24h": 15000000,
        "dex_inflow": 125000
    }
    
    # Add test alert
    success = manager.add_alert(test_alert)
    print(f"✅ Test alert added: {success}")
    
    # Check queue status
    status = manager.get_queue_status()
    print(f"✅ Queue status: {status['fast_track_queue']} fast-track, {status['standard_queue']} standard")
    
    # Process queue
    processed = manager.process_alert_queue()
    print(f"✅ Processed alerts: {processed}")
    
    print("\n🎉 Telegram Alert Manager ready for priority alert processing!")