#!/usr/bin/env python3
"""
Stealth Alert System - Autonomous Pre-Pump Alert System
Wysyła alerty o ukrytych sygnałach pre-pump bez potrzeby wykresów
"""

import json
import os
import time
import hashlib
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

# Stealth Alert Configuration
STEALTH_ALERT_CONFIG = {
    "telegram_enabled": True,
    "file_logging_enabled": True,
    "cooldown_hours": 12,  # 12-godzinna blokada per token (nie per detektor)
    "alert_file": "data/stealth_alerts.json",
    "debug_log_file": "data/alert_debug.json"
}

class StealthAlertManager:
    """Zarządzanie alertami Stealth Engine z 12h blokowaniem per token i filtrem duplikatów"""
    
    def __init__(self):
        self.alert_history = {}
        self.debug_history = []
        self.load_alert_history()
        self.load_debug_history()
    
    def load_alert_history(self):
        """Załaduj historię alertów"""
        try:
            if os.path.exists(STEALTH_ALERT_CONFIG["alert_file"]):
                with open(STEALTH_ALERT_CONFIG["alert_file"], 'r') as f:
                    self.alert_history = json.load(f)
        except Exception as e:
            print(f"[STEALTH ALERT] Error loading alert history: {e}")
            self.alert_history = {}
    
    def load_debug_history(self):
        """Załaduj historię debug alertów"""
        try:
            if os.path.exists(STEALTH_ALERT_CONFIG["debug_log_file"]):
                with open(STEALTH_ALERT_CONFIG["debug_log_file"], 'r') as f:
                    self.debug_history = json.load(f)
                    # Ograniczaj do ostatnich 1000 wpisów
                    if len(self.debug_history) > 1000:
                        self.debug_history = self.debug_history[-1000:]
        except Exception as e:
            print(f"[STEALTH ALERT] Error loading debug history: {e}")
            self.debug_history = []
    
    def save_alert_history(self):
        """Zapisz historię alertów"""
        try:
            os.makedirs(os.path.dirname(STEALTH_ALERT_CONFIG["alert_file"]), exist_ok=True)
            with open(STEALTH_ALERT_CONFIG["alert_file"], 'w') as f:
                json.dump(self.alert_history, f, indent=2)
        except Exception as e:
            print(f"[STEALTH ALERT] Error saving alert history: {e}")
    
    def save_debug_history(self):
        """Zapisz historię debug alertów"""
        try:
            os.makedirs(os.path.dirname(STEALTH_ALERT_CONFIG["debug_log_file"]), exist_ok=True)
            with open(STEALTH_ALERT_CONFIG["debug_log_file"], 'w') as f:
                json.dump(self.debug_history, f, indent=2)
        except Exception as e:
            print(f"[STEALTH ALERT] Error saving debug history: {e}")
    
    def generate_alert_hash(self, symbol: str, stealth_score: float, active_signals: List[str], consensus_decision: str = None) -> str:
        """
        Generuj unikalny hash dla alertu do wykrywania duplikatów
        
        Args:
            symbol: Token symbol
            stealth_score: Score alertu
            active_signals: Lista aktywnych sygnałów
            consensus_decision: Decyzja consensus
            
        Returns:
            Unikalny hash alertu
        """
        # Twórz string reprezentujący alert
        alert_data = {
            "symbol": symbol,
            "score": round(stealth_score, 3),  # Zaokrąglij do 3 miejsc po przecinku
            "signals": sorted(active_signals),  # Sortuj sygnały dla konsystencji
            "consensus": consensus_decision or "NONE"
        }
        
        alert_string = json.dumps(alert_data, sort_keys=True)
        return hashlib.md5(alert_string.encode()).hexdigest()
    
    def log_rejection(self, symbol: str, reason: str, details: dict = None):
        """
        Loguj odrzucony token z powodem
        
        Args:
            symbol: Token symbol
            reason: Powód odrzucenia
            details: Dodatkowe szczegóły
        """
        rejection_entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "reason": reason,
            "details": details or {}
        }
        
        self.debug_history.append(rejection_entry)
        
        # Ograniczaj historię do 1000 ostatnich wpisów
        if len(self.debug_history) > 1000:
            self.debug_history = self.debug_history[-1000:]
        
        # Zapisz od razu
        self.save_debug_history()
        
        # Wyświetl w logach
        print(f"[ALERT REJECTED] {symbol} → {reason}: {details}")
    
    def should_send_alert(self, symbol: str, current_score: float = 0.0, active_signals: List[str] = None, consensus_decision: str = None) -> Tuple[bool, str]:
        """
        Sprawdź czy można wysłać alert (12h blokada per token + anti-duplicate filter)
        
        Args:
            symbol: Token symbol
            current_score: Current stealth score
            active_signals: Lista aktywnych sygnałów
            consensus_decision: Decyzja consensus
            
        Returns:
            (bool, str): (czy_wyslac, powod)
        """
        active_signals = active_signals or []
        
        # 1. Sprawdź 12-godzinną blokadę per token
        if symbol in self.alert_history:
            last_alert_time = self.alert_history[symbol].get("last_alert_time")
            if last_alert_time:
                last_time = datetime.fromisoformat(last_alert_time)
                time_elapsed = datetime.now() - last_time
                cooldown_period = timedelta(hours=STEALTH_ALERT_CONFIG["cooldown_hours"])
                
                if time_elapsed < cooldown_period:
                    remaining_hours = (cooldown_period - time_elapsed).total_seconds() / 3600
                    reason = f"12h token blocking - {remaining_hours:.1f}h remaining"
                    self.log_rejection(symbol, reason, {
                        "score": current_score,
                        "last_alert_time": last_alert_time,
                        "hours_since_last": time_elapsed.total_seconds() / 3600
                    })
                    return False, reason
        
        # 2. Sprawdź anti-duplicate filter
        current_hash = self.generate_alert_hash(symbol, current_score, active_signals, consensus_decision)
        
        if symbol in self.alert_history:
            last_hash = self.alert_history[symbol].get("alert_hash")
            if last_hash == current_hash:
                reason = "duplicate alert detected (identical hash)"
                self.log_rejection(symbol, reason, {
                    "score": current_score,
                    "hash": current_hash,
                    "signals": active_signals,
                    "consensus": consensus_decision
                })
                return False, reason
        
        # 3. Sprawdź podstawowe kryteria (score, consensus)
        if consensus_decision and consensus_decision != "BUY":
            reason = f"consensus decision {consensus_decision} (not BUY)"
            self.log_rejection(symbol, reason, {
                "score": current_score,
                "consensus": consensus_decision,
                "signals_count": len(active_signals)
            })
            return False, reason
        
        if current_score < 0.5:  # Minimalne wymaganie score
            reason = f"score too low ({current_score:.3f} < 0.5)"
            self.log_rejection(symbol, reason, {
                "score": current_score,
                "consensus": consensus_decision,
                "signals": active_signals
            })
            return False, reason
        
        if len(active_signals) == 0:
            reason = "no active signals detected"
            self.log_rejection(symbol, reason, {
                "score": current_score,
                "consensus": consensus_decision
            })
            return False, reason
        
        # Wszystkie sprawdzenia przeszły pomyślnie
        print(f"[ALERT APPROVED] {symbol} → Score: {current_score:.3f}, Signals: {len(active_signals)}, Consensus: {consensus_decision}")
        return True, "alert approved"
    
    def record_alert(self, symbol: str, stealth_score: float, active_signals: List[str], alert_type: str, consensus_decision: str = None):
        """Zapisz wysłany alert z hashem dla anti-duplicate filter"""
        if symbol not in self.alert_history:
            self.alert_history[symbol] = {}
        
        # Generuj hash alertu dla przyszłego sprawdzania duplikatów
        alert_hash = self.generate_alert_hash(symbol, stealth_score, active_signals, consensus_decision)
        
        self.alert_history[symbol].update({
            "last_alert_time": datetime.now().isoformat(),
            "last_score": stealth_score,
            "last_signals": active_signals,
            "last_alert_type": alert_type,
            "last_consensus": consensus_decision,
            "alert_hash": alert_hash,
            "total_alerts": self.alert_history[symbol].get("total_alerts", 0) + 1
        })
        
        self.save_alert_history()
        print(f"[ALERT RECORDED] {symbol} → Hash: {alert_hash[:8]}..., Total alerts: {self.alert_history[symbol]['total_alerts']}")

# Global alert manager
stealth_alert_manager = StealthAlertManager()

async def send_stealth_alert(symbol: str, stealth_score: float, active_signals: List[str], alert_type: str, consensus_decision: str = None, consensus_enabled: bool = False):
    """
    Wysyła alert Stealth Engine z pełną integracją utility modules i sprawdzaniem consensus
    
    Args:
        symbol: Symbol tokena
        stealth_score: Końcowy score stealth
        active_signals: Lista aktywnych sygnałów
        alert_type: Typ alertu (strong_stealth_alert, medium_alert)
        consensus_decision: Decyzja consensus (BUY/HOLD/AVOID)
        consensus_enabled: Czy consensus jest dostępny
    """
    
    # 🔐 CRITICAL CONSENSUS DECISION CHECK FIRST - NAJWAŻNIEJSZE SPRAWDZENIE
    if consensus_enabled and consensus_decision:
        if consensus_decision != "BUY":
            print(f"[STEALTH CONSENSUS BLOCK] {symbol} → Consensus decision {consensus_decision} blocks alert (score={stealth_score:.3f})")
            return  # Blokuj alert jeśli consensus != BUY
        else:
            print(f"[STEALTH CONSENSUS PASS] {symbol} → Consensus decision BUY allows alert (score={stealth_score:.3f})")
    else:
        # Fallback - bez consensus, sprawdź score threshold
        # WYMAGANIE #7: Remove score≥0.7 fallback logic - hard gating only
        print(f"[HARD GATING ONLY] {symbol} → No consensus, score {stealth_score:.3f} - using hard gating requirements only")
        # Removed score >= 0.7 fallback logic - rely on hard gating checks
        return  # Block alert without proper hard gating criteria
    
    # Sprawdź 12h blokadę per token i anti-duplicate filter
    should_send, rejection_reason = stealth_alert_manager.should_send_alert(
        symbol=symbol, 
        current_score=stealth_score, 
        active_signals=active_signals, 
        consensus_decision=consensus_decision
    )
    
    if not should_send:
        print(f"[STEALTH ALERT BLOCKED] {symbol} → {rejection_reason}")
        return
    
    processing_start = time.time()
    
    # 🎯 INTEGRACJA Z UTILITY MODULES
    try:
        # Import utility modules
        from stealth_engine.stealth_labels import save_stealth_label, generate_stealth_label
        from stealth_engine.stealth_debug import log_stealth_debug, stealth_debug_session
        from stealth_engine.stealth_utils import metadata_manager
        
        # Start debug session for comprehensive logging
        with stealth_debug_session(symbol) as session_id:
            
            # Step 1: Generate and save stealth label
            label_filepath = save_stealth_label(symbol, stealth_score, active_signals, alert_type)
            stealth_label = generate_stealth_label(active_signals)
            print(f"[STEALTH ALERT] {symbol} → Label generated: {stealth_label}")
            
            # Step 2: Log detailed debug information
            processing_time = time.time() - processing_start
            log_stealth_debug(
                symbol=symbol,
                stealth_score=stealth_score, 
                active_signals=active_signals,
                signal_details={signal: 1.0 for signal in active_signals},
                processing_time=processing_time
            )
            
            # Step 3: Record in metadata system
            metadata_manager.record_alert(symbol, stealth_score, active_signals, alert_type, processing_time)
            
            # Step 4: Przygotuj wiadomość alertu
            alert_message = format_stealth_alert_message(symbol, stealth_score, active_signals, alert_type)
            
            # Step 5: Wyślij alert przez dostępne kanały
            success = False
            
            # 5a. Telegram Alert
            if STEALTH_ALERT_CONFIG["telegram_enabled"]:
                try:
                    telegram_success = await send_telegram_stealth_alert(alert_message)
                    if telegram_success:
                        success = True
                        print(f"[STEALTH ALERT] {symbol} → Telegram alert sent successfully")
                except Exception as e:
                    print(f"[STEALTH ALERT] {symbol} → Telegram error: {e}")
            
            # 5b. File Logging
            if STEALTH_ALERT_CONFIG["file_logging_enabled"]:
                try:
                    log_stealth_alert_to_file(symbol, stealth_score, active_signals, alert_type, alert_message)
                    success = True
                    print(f"[STEALTH ALERT] {symbol} → Alert logged to file")
                except Exception as e:
                    print(f"[STEALTH ALERT] {symbol} → File logging error: {e}")
            
            # Step 6: Zapisz alert w historii
            if success:
                stealth_alert_manager.record_alert(symbol, stealth_score, active_signals, alert_type, consensus_decision)
                print(f"[STEALTH ALERT] ✅ {symbol} → Complete alert with utilities sent successfully (Label: {stealth_label})")
                
                # STAGE 12 - REMOVED (satellite scanner not requested by user)
                    
            else:
                print(f"[STEALTH ALERT] ❌ {symbol} → Failed to send alert")
                
    except ImportError as e:
        print(f"[STEALTH ALERT] Warning: Utility modules not available ({e}), using basic alert system")
        
        # Fallback to basic alert system
        alert_message = format_stealth_alert_message(symbol, stealth_score, active_signals, alert_type)
        success = False
        
        # Basic telegram and file logging
        if STEALTH_ALERT_CONFIG["telegram_enabled"]:
            try:
                telegram_success = await send_telegram_stealth_alert(alert_message)
                if telegram_success:
                    success = True
            except Exception as e:
                print(f"[STEALTH ALERT] {symbol} → Telegram error: {e}")
        
        if STEALTH_ALERT_CONFIG["file_logging_enabled"]:
            try:
                log_stealth_alert_to_file(symbol, stealth_score, active_signals, alert_type, alert_message)
                success = True
            except Exception as e:
                print(f"[STEALTH ALERT] {symbol} → File logging error: {e}")
        
        if success:
            stealth_alert_manager.record_alert(symbol, stealth_score, active_signals, alert_type, consensus_decision)
            print(f"[STEALTH ALERT] ✅ {symbol} → Basic alert sent successfully")
            
            # STAGE 12 - REMOVED (satellite scanner not requested by user)
                
        else:
            print(f"[STEALTH ALERT] ❌ {symbol} → Failed to send alert")
        
    except Exception as e:
        print(f"[STEALTH ALERT] Error with utility integration: {e}")
        print(f"[STEALTH ALERT] ⚠️ {symbol} → Alert sent with errors")

def format_stealth_alert_message(symbol: str, stealth_score: float, active_signals: List[str], alert_type: str) -> str:
    """Formatuj wiadomość alertu Stealth Engine"""
    
    # Emojis dla różnych typów alertów
    alert_emoji = {
        "strong_stealth_alert": "🚨",
        "medium_alert": "⚠️",
        None: "ℹ️"
    }
    
    emoji = alert_emoji.get(alert_type, "🔍")
    
    # Predykcje na podstawie aktywnych sygnałów
    predictions = generate_stealth_predictions(active_signals)
    
    message = f"""{emoji} **STEALTH ALERT** {emoji}

**Token:** {symbol}
**Stealth Score:** {stealth_score:.3f}
**Alert Type:** {alert_type or 'informational'}

**🔍 Aktywne Sygnały:**
{chr(10).join([f"• {signal.replace('_', ' ').title()}" for signal in active_signals])}

**🎯 Predykcje:**
{chr(10).join([f"• {pred}" for pred in predictions])}

**⏰ Czas:** {datetime.now().strftime('%H:%M:%S')}
**🤖 Stealth Engine v2** - Wykrywanie bez wykresów
"""
    
    return message

def generate_stealth_predictions(active_signals: List[str]) -> List[str]:
    """Generuj predykcje na podstawie aktywnych sygnałów"""
    predictions = []
    
    # Analiza sygnałów orderbook
    orderbook_signals = [s for s in active_signals if 'orderbook' in s or 'bid' in s or 'ask' in s or 'spoofing' in s]
    if orderbook_signals:
        if 'spoofing_detection' in active_signals:
            predictions.append("Wykryto możliwy spoofing - fałszywe zlecenia")
        if 'bid_wall_detection' in active_signals:
            predictions.append("Wykryto bid wall - silne wsparcie")
        if 'ask_wall_detection' in active_signals:
            predictions.append("Wykryto ask wall - silny opór")
        if 'orderbook_imbalance' in active_signals:
            predictions.append("Nierównowaga orderbook - kierunkowa presja")
        if 'bid_ask_spread_tightening' in active_signals:
            predictions.append("Zwężenie spreadu - zwiększona aktywność")
    
    # Analiza sygnałów volume
    volume_signals = [s for s in active_signals if 'volume' in s]
    if volume_signals:
        if 'volume_spike_detection' in active_signals:
            predictions.append("Nagły wzrost wolumenu - zainteresowanie instytucji")
        if 'volume_accumulation' in active_signals:
            predictions.append("Akumulacja wolumenu - stopniowe gromadzenie")
    
    # Analiza sygnałów DEX
    dex_signals = [s for s in active_signals if 'dex' in s or 'whale' in s]
    if dex_signals:
        if 'dex_inflow' in active_signals:
            predictions.append("Napływ do DEX - przygotowanie do ruchu")
        if 'whale_accumulation_pattern' in active_signals:
            predictions.append("Wzorzec akumulacji whale - duże portfele kupują")
    
    # Analiza microstructure
    micro_signals = [s for s in active_signals if 'liquidity' in s or 'microstructure' in s]
    if micro_signals:
        if 'liquidity_absorption' in active_signals:
            predictions.append("Absorpcja płynności - przygotowanie do przełamania")
        if 'hidden_liquidity_detection' in active_signals:
            predictions.append("Ukryta płynność - iceberg orders")
    
    # Domyślna predykcja
    if not predictions:
        predictions.append("Ukryte sygnały pre-pump wykryte")
    
    return predictions

async def send_telegram_stealth_alert(message: str) -> bool:
    """Wyślij alert przez Telegram"""
    try:
        # Import i użyj istniejącego systemu Telegram z crypto-scan
        from utils.telegram_bot import send_telegram_message
        return await send_telegram_message(message)
    except ImportError:
        print("[STEALTH ALERT] Telegram module not available")
        return False
    except Exception as e:
        print(f"[STEALTH ALERT] Telegram error: {e}")
        return False

def log_stealth_alert_to_file(symbol: str, stealth_score: float, active_signals: List[str], alert_type: str, message: str):
    """Zapisz alert do pliku"""
    try:
        alert_entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "stealth_score": stealth_score,
            "active_signals": active_signals,
            "alert_type": alert_type,
            "message": message
        }
        
        # Zapisz do głównego pliku alertów
        alerts_file = "data/stealth_alerts_log.jsonl"
        os.makedirs("data", exist_ok=True)
        
        with open(alerts_file, 'a') as f:
            f.write(json.dumps(alert_entry) + '\n')
        
        # Zapisz także do pliku dziennego
        daily_file = f"data/stealth_alerts_{datetime.now().strftime('%Y%m%d')}.json"
        daily_alerts = []
        
        if os.path.exists(daily_file):
            with open(daily_file, 'r') as f:
                daily_alerts = json.load(f)
        
        daily_alerts.append(alert_entry)
        
        with open(daily_file, 'w') as f:
            json.dump(daily_alerts, f, indent=2)
        
    except Exception as e:
        print(f"[STEALTH ALERT] File logging error: {e}")

def get_stealth_alert_stats() -> dict:
    """Pobierz statystyki alertów Stealth Engine"""
    try:
        stats = {
            "total_symbols_alerted": len(stealth_alert_manager.alert_history),
            "alerts_today": 0,
            "most_active_signals": {},
            "alert_types_distribution": {}
        }
        
        # Zlicz dzisiejsze alerty
        today = datetime.now().strftime('%Y-%m-%d')
        daily_file = f"data/stealth_alerts_{datetime.now().strftime('%Y%m%d')}.json"
        
        if os.path.exists(daily_file):
            with open(daily_file, 'r') as f:
                daily_alerts = json.load(f)
                stats["alerts_today"] = len(daily_alerts)
                
                # Analiza sygnałów
                for alert in daily_alerts:
                    for signal in alert.get("active_signals", []):
                        stats["most_active_signals"][signal] = stats["most_active_signals"].get(signal, 0) + 1
                    
                    alert_type = alert.get("alert_type", "unknown")
                    stats["alert_types_distribution"][alert_type] = stats["alert_types_distribution"].get(alert_type, 0) + 1
        
        return stats
        
    except Exception as e:
        print(f"[STEALTH ALERT] Stats error: {e}")
        return {}

if __name__ == "__main__":
    # Test alertu
    import asyncio
    
    async def test_stealth_alert():
        await send_stealth_alert(
            symbol="BTCUSDT",
            stealth_score=3.5,
            active_signals=["dex_inflow", "bid_ask_spread_tightening", "volume_spike_detection"],
            alert_type="strong_stealth_alert"
        )
    
    asyncio.run(test_stealth_alert())