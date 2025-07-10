#!/usr/bin/env python3
"""
Stealth Alert System - Autonomous Pre-Pump Alert System
Wysyła alerty o ukrytych sygnałach pre-pump bez potrzeby wykresów
"""

import json
import os
import time
from datetime import datetime
from typing import List, Optional

# Stealth Alert Configuration
STEALTH_ALERT_CONFIG = {
    "telegram_enabled": True,
    "file_logging_enabled": True,
    "cooldown_minutes": 15,  # Minimalna przerwa między alertami dla tego samego tokena
    "alert_file": "data/stealth_alerts.json"
}

class StealthAlertManager:
    """Zarządzanie alertami Stealth Engine"""
    
    def __init__(self):
        self.alert_history = {}
        self.load_alert_history()
    
    def load_alert_history(self):
        """Załaduj historię alertów"""
        try:
            if os.path.exists(STEALTH_ALERT_CONFIG["alert_file"]):
                with open(STEALTH_ALERT_CONFIG["alert_file"], 'r') as f:
                    self.alert_history = json.load(f)
        except Exception as e:
            print(f"[STEALTH ALERT] Error loading alert history: {e}")
            self.alert_history = {}
    
    def save_alert_history(self):
        """Zapisz historię alertów"""
        try:
            os.makedirs(os.path.dirname(STEALTH_ALERT_CONFIG["alert_file"]), exist_ok=True)
            with open(STEALTH_ALERT_CONFIG["alert_file"], 'w') as f:
                json.dump(self.alert_history, f, indent=2)
        except Exception as e:
            print(f"[STEALTH ALERT] Error saving alert history: {e}")
    
    def should_send_alert(self, symbol: str) -> bool:
        """Sprawdź czy można wysłać alert (cooldown)"""
        if symbol not in self.alert_history:
            return True
        
        last_alert_time = self.alert_history[symbol].get("last_alert_time")
        if not last_alert_time:
            return True
        
        # Sprawdź cooldown
        from datetime import datetime, timedelta
        last_time = datetime.fromisoformat(last_alert_time)
        cooldown_period = timedelta(minutes=STEALTH_ALERT_CONFIG["cooldown_minutes"])
        
        return datetime.now() - last_time > cooldown_period
    
    def record_alert(self, symbol: str, stealth_score: float, active_signals: List[str], alert_type: str):
        """Zapisz wysłany alert"""
        if symbol not in self.alert_history:
            self.alert_history[symbol] = {}
        
        self.alert_history[symbol].update({
            "last_alert_time": datetime.now().isoformat(),
            "last_score": stealth_score,
            "last_signals": active_signals,
            "last_alert_type": alert_type,
            "total_alerts": self.alert_history[symbol].get("total_alerts", 0) + 1
        })
        
        self.save_alert_history()

# Global alert manager
stealth_alert_manager = StealthAlertManager()

async def send_stealth_alert(symbol: str, stealth_score: float, active_signals: List[str], alert_type: str):
    """
    Wysyła alert Stealth Engine z pełną integracją utility modules
    
    Args:
        symbol: Symbol tokena
        stealth_score: Końcowy score stealth
        active_signals: Lista aktywnych sygnałów
        alert_type: Typ alertu (strong_stealth_alert, medium_alert)
    """
    
    # Sprawdź cooldown
    if not stealth_alert_manager.should_send_alert(symbol):
        print(f"[STEALTH ALERT] {symbol} → Alert w cooldown, pomijam")
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
                stealth_alert_manager.record_alert(symbol, stealth_score, active_signals, alert_type)
                print(f"[STEALTH ALERT] ✅ {symbol} → Complete alert with utilities sent successfully (Label: {stealth_label})")
                
                # 🛰️ STAGE 12: SATELLITE SCANNER INTEGRATION
                try:
                    from stealth_engine.stealth_scanner import handle_stealth_alert_with_satellite
                    
                    # Przygotuj dane alertu dla satelitarnego skanera
                    alert_data = {
                        "active_signals": active_signals,
                        "alert_type": alert_type,
                        "stealth_label": stealth_label,
                        "processing_time": processing_time,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Uruchom satelitarny skan asynchronicznie
                    import asyncio
                    satellite_result = await handle_stealth_alert_with_satellite(
                        symbol=symbol,
                        stealth_score=stealth_score,
                        alert_data=alert_data
                    )
                    
                    if satellite_result.get("satellite_scan_triggered", False):
                        twins = satellite_result.get("satellite_twins", [])
                        print(f"[STAGE 12 SUCCESS] {symbol} → Satellite scan triggered for {len(twins)} twin tokens: {twins}")
                    else:
                        print(f"[STAGE 12 INFO] {symbol} → No satellite scan triggered (threshold/twins not met)")
                        
                except ImportError:
                    print(f"[STAGE 12 WARNING] {symbol} → Satellite scanner module not available")
                except Exception as satellite_error:
                    print(f"[STAGE 12 ERROR] {symbol} → Satellite scan failed: {satellite_error}")
                    
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
            stealth_alert_manager.record_alert(symbol, stealth_score, active_signals, alert_type)
            print(f"[STEALTH ALERT] ✅ {symbol} → Basic alert sent successfully")
            
            # 🛰️ STAGE 12: SATELLITE SCANNER INTEGRATION (fallback mode)
            try:
                from stealth_engine.stealth_scanner import handle_stealth_alert_with_satellite
                
                # Przygotuj podstawowe dane alertu
                alert_data = {
                    "active_signals": active_signals,
                    "alert_type": alert_type,
                    "fallback_mode": True,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Uruchom satelitarny skan asynchronicznie
                import asyncio
                satellite_result = await handle_stealth_alert_with_satellite(
                    symbol=symbol,
                    stealth_score=stealth_score,
                    alert_data=alert_data
                )
                
                if satellite_result.get("satellite_scan_triggered", False):
                    twins = satellite_result.get("satellite_twins", [])
                    print(f"[STAGE 12 FALLBACK] {symbol} → Satellite scan triggered for {len(twins)} twin tokens")
                else:
                    print(f"[STAGE 12 FALLBACK] {symbol} → No satellite scan triggered")
                    
            except ImportError:
                print(f"[STAGE 12 WARNING] {symbol} → Satellite scanner not available in fallback mode")
            except Exception as satellite_error:
                print(f"[STAGE 12 ERROR] {symbol} → Fallback satellite scan failed: {satellite_error}")
                
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