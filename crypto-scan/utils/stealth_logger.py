#!/usr/bin/env python3
"""
Stealth Pre-Pump Engine Logging System
Precyzyjne logowanie dla nowych detektorów, consensus i feedback loop
"""

import json
import os
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class StealthLogger:
    """
    Nowy system logowania dla Stealth Pre-Pump Engine
    """
    
    def __init__(self):
        self.detector_symbols = {
            'whale_ping': '🐳',
            'mastermind_tracing': '🧠', 
            'dex_inflow': '💧',
            'orderbook_anomaly': '📊',
            'whaleclip_vision': '🛰️',
            'consensus_vote': '🎯',
            'feedback_adjust': '🔁',
            'total_score': '🔐'
        }
        
    def get_confidence_level(self, score) -> tuple:
        """Zwróć poziom zagrożenia na podstawie score (obsługuje float lub dict)"""
        # Extract numeric value from score
        numeric_score = self._extract_detector_value(score) if not isinstance(score, (int, float)) else score
        
        if numeric_score > 1.8:
            return "🔥", "[HIGH CONFIDENCE]"
        elif 1.2 < numeric_score <= 1.8:
            return "⚠️", "[MEDIUM CONFIDENCE]"
        else:
            return "🧪", "[LOW SIGNAL]"
    
    def print_top5_stealth_tokens_enhanced(self, top_tokens: List[Dict]) -> None:
        """
        Wyświetl TOP 5 tokenów z nowym formatem logowania
        """
        if not top_tokens:
            print("\n⚠️ TOP 5 STEALTH TOKENS: Brak danych")
            return
            
        print("\n🎯 TOP 5 STEALTH SCORE TOKENS:")
        print("=" * 80)
        
        for i, token_data in enumerate(top_tokens, 1):
            token = token_data.get('token', 'UNKNOWN')
            stealth_score = token_data.get('base_score', 0.0)
            early_score = token_data.get('early_score', stealth_score)
            
            # Extract numeric values for formatting
            stealth_numeric = self._extract_detector_value(stealth_score)
            early_numeric = self._extract_detector_value(early_score)
            dex_numeric = self._extract_detector_value(token_data.get('dex_inflow', 0.0))
            whale_numeric = self._extract_detector_value(token_data.get('whale_ping', 0.0))
            trust_numeric = self._extract_detector_value(token_data.get('trust_boost', 0.0))
            id_numeric = self._extract_detector_value(token_data.get('identity_boost', 0.0))
            
            # Główny nagłówek tokena
            confidence_icon, confidence_text = self.get_confidence_level(stealth_score)
            print(f"{i:2d}. {token:12} | Stealth: {stealth_numeric:6.3f} | Early: {early_numeric:6.3f}")
            print(f"     DEX: {dex_numeric:6.3f} | Whale: {whale_numeric:6.3f} | Trust: {trust_numeric:6.3f} | ID: {id_numeric:6.3f}")
            print(f"     {confidence_icon} {confidence_text}")
            
            # Nowy breakdown detektorów
            self._print_token_breakdown(token_data)
            
            # Sprawdź consensus vote dla alertów
            self._check_consensus_alert(token, token_data)
            
            if i < len(top_tokens):
                print("    " + "-" * 76)
        
        print("=" * 80)
    
    def _print_token_breakdown(self, token_data: Dict) -> None:
        """
        Wyświetl szczegółowy breakdown detektorów z numerical values
        """
        print("🔎 Breakdown:")
        
        # Extract numeric values from detectors (handles dict, bool, float)
        whale_ping = self._extract_detector_value(token_data.get('whale_ping', 0.0))
        mastermind = self._extract_detector_value(token_data.get('mastermind_tracing', 0.0))
        dex_inflow = self._extract_detector_value(token_data.get('dex_inflow', 0.0))
        orderbook = self._extract_detector_value(token_data.get('orderbook_anomaly', 0.0))
        whaleclip = self._extract_detector_value(token_data.get('whaleclip_vision', 0.0))
        
        # Dodaj AI detektory
        diamond_score = self._extract_detector_value(token_data.get('diamond_score', 0.0))
        californium_score = self._extract_detector_value(token_data.get('californium_score', 0.0))
        stealth_engine_score = self._extract_detector_value(token_data.get('stealth_engine_score', token_data.get('base_score', 0.0)))
        
        print(f" - 🐳 whale_ping:         {whale_ping:.2f}")
        print(f" - 🧠 mastermind_tracing:  {mastermind:.2f}")
        print(f" - 💧 dex_inflow:         {dex_inflow:.2f}")
        print(f" - 📊 orderbook_anomaly:   {orderbook:.2f}")
        print(f" - 🛰️ WhaleCLIP (vision):  {whaleclip:.2f}")
        print(f" - 💎 DiamondWhale AI:     {diamond_score:.2f}")
        print(f" - 🌊 CaliforniumWhale:    {californium_score:.2f}")
        print(f" - 🎯 StealthEngine:       {stealth_engine_score:.2f}")
        
        # Pokaż aktywne detektory
        consensus_detectors = token_data.get('consensus_detectors', [])
        if consensus_detectors:
            print(f" - ✅ Active detectors:   {len(consensus_detectors)} ({', '.join(consensus_detectors)})")
        else:
            print(f" - ⚠️  Active detectors:   0 (insufficient data)")
        
        # Consensus decision (używamy nowych pól)
        consensus_decision = token_data.get('consensus_decision', 'UNKNOWN')
        consensus_votes = token_data.get('consensus_votes', [])
        consensus_score = token_data.get('consensus_score', 0.0)
        consensus_confidence = token_data.get('consensus_confidence', 0.0)
        # Handle None values
        if consensus_score is None:
            consensus_score = 0.0
        if consensus_confidence is None:
            consensus_confidence = 0.0
        
        # Policz głosy BUY/HOLD/AVOID
        if isinstance(consensus_votes, list) and consensus_votes:
            # Parse głosy z formatu "DetectorName: VOTE"
            buy_count = 0
            hold_count = 0
            avoid_count = 0
            for vote in consensus_votes:
                if ": BUY" in vote:
                    buy_count += 1
                elif ": HOLD" in vote:
                    hold_count += 1
                elif ": AVOID" in vote:
                    avoid_count += 1
            
            total_count = len(consensus_votes)
            votes_str = f"BUY:{buy_count}, HOLD:{hold_count}, AVOID:{avoid_count}"
            print(f" - 🎯 consensus_decision: {consensus_decision} ({votes_str})")
            # Handle None consensus_score in vote display
            if consensus_score is not None:
                print(f" - 📊 consensus_score:    {consensus_score:.3f} (confidence: {consensus_confidence:.2f})")
            else:
                print(f" - 📊 consensus_score:    0.000 (confidence: {consensus_confidence:.2f})")
        else:
            print(f" - 🎯 consensus_decision: {consensus_decision} (no votes)")
            # Handle None consensus_score
            if consensus_score is not None:
                print(f" - 📊 consensus_score:    {consensus_score:.3f}")
            else:
                print(f" - 📊 consensus_score:    0.000")
        
        # Feedback adjustment
        feedback_adjust = token_data.get('feedback_adjust', 0.0)
        print(f" - 🔁 feedback_adjust:   {feedback_adjust:+5.2f}")
    
    
    
    def _extract_detector_value(self, detector_value) -> float:
        """Extract numeric value from detector result (supports bool, float, or dict)"""
        if isinstance(detector_value, bool):
            return 1.0 if detector_value else 0.0
        elif isinstance(detector_value, (int, float)):
            return float(detector_value)
        elif isinstance(detector_value, dict):
            # Handle dict format like {"active": True, "score": 0.75}
            if "score" in detector_value:
                return float(detector_value["score"])
            elif "active" in detector_value and detector_value["active"]:
                return 1.0
            else:
                return 0.0
        else:
            return 0.0

    def _identify_pattern(self, detector_results: Dict[str, float]) -> str:
        """
        Intelligent pattern identification na podstawie active detectors
        """
        # Convert all values to numeric using _extract_detector_value
        active = []
        for name, score in detector_results.items():
            numeric_score = self._extract_detector_value(score)
            if numeric_score > 0.0:
                active.append(name)
        
        if len(active) >= 3:
            return f"Multi-detector consensus ({'+'.join(active[:3])})"
        elif len(active) == 2:
            return f"{' + '.join(active)} consensus"
        elif len(active) == 1:
            return f"{active[0]} dominant signal"
        else:
            return "Composite stealth pattern"
    
    def _extract_detector_scores(self, stealth_signals: List[Dict]) -> Dict[str, float]:
        """
        Wyciągnij scores z stealth signals
        """
        scores = {}
        
        for signal in stealth_signals:
            signal_name = signal.get('signal_name', '')
            strength = signal.get('strength', 0.0)
            active = signal.get('active', False)
            
            # Mapuj nazwy sygnałów na detektory - PRESERVE WHALE SIGNAL STRENGTH
            if 'whale_ping' in signal_name.lower():
                scores['whale_ping'] = strength  # Always preserve strength from stealth processing
            elif 'dex_inflow' in signal_name.lower():
                scores['dex_inflow'] = strength if active else 0.0
            elif 'spoofing' in signal_name.lower() or 'orderbook' in signal_name.lower():
                scores['orderbook_anomaly'] = strength if active else 0.0
            elif 'mastermind' in signal_name.lower():
                scores['mastermind_tracing'] = strength if active else 0.0
            elif 'whaleclip' in signal_name.lower() or 'vision' in signal_name.lower():
                scores['whaleclip_vision'] = strength if active else 0.0
        
        return scores
    
    def _check_consensus_alert(self, token: str, token_data: Dict) -> None:
        """
        🔐 CRITICAL CONSENSUS VALIDATION - Sprawdź czy RZECZYWIŚCIE jest majority BUY consensus
        """
        # Używamy nowego pola consensus_decision zamiast consensus_vote
        consensus_decision = token_data.get('consensus_decision', 'UNKNOWN')
        consensus_votes = token_data.get('consensus_votes', [])
        consensus_confidence = token_data.get('consensus_confidence', 0.0)
        
        # 🚨 CRITICAL FIX: Sprawdź RZECZYWISTE głosy a nie tylko decision field
        if isinstance(consensus_votes, list) and len(consensus_votes) > 0:
            # Parse głosy z formatu "DetectorName: VOTE"
            buy_count = 0
            hold_count = 0
            avoid_count = 0
            
            for vote in consensus_votes:
                if ": BUY" in str(vote):
                    buy_count += 1
                elif ": HOLD" in str(vote):
                    hold_count += 1
                elif ": AVOID" in str(vote):
                    avoid_count += 1
            
            total_votes = len(consensus_votes)
            # 🔐 CRITICAL FIX: Require 2-vote difference (yes_votes - no_votes >= 2)
            vote_difference = buy_count - avoid_count
            sufficient_consensus = vote_difference >= 2  # Minimum 2-vote difference required
            
            print(f"[CONSENSUS VALIDATION] {token}: BUY={buy_count}, HOLD={hold_count}, AVOID={avoid_count}, total={total_votes}, vote_difference={vote_difference}")
            print(f"[CONSENSUS VALIDATION] {token}: Vote difference: {buy_count} BUY - {avoid_count} AVOID = {vote_difference} (need ≥2 for alert)")
            
            # Alert tylko jeśli jest różnica co najmniej 2 głosów BUY vs AVOID
            if not sufficient_consensus:
                print(f"[CONSENSUS BLOCK] {token}: Insufficient vote difference ({vote_difference} < 2) - BLOCKING alert")
                print(f"[CONSENSUS BLOCK] {token}: Need at least {avoid_count + 2} BUY votes vs {avoid_count} AVOID votes")
                print(f"[CONSENSUS BLOCK] {token}: Votes breakdown: BUY={buy_count}, HOLD={hold_count}, AVOID={avoid_count}")
                return
        elif consensus_decision != 'BUY':
            print(f"[CONSENSUS BLOCK] {token}: Decision {consensus_decision} != BUY - BLOCKING alert")
            return
        
        # Jeśli dotarliśmy tutaj, to mamy valid BUY consensus
        print(f"[CONSENSUS PASS] {token}: Valid BUY consensus confirmed - proceeding with alert")
        total_score = token_data.get('base_score', 0.0)
        
        # Znajdź dominujące źródła
        stealth_signals = token_data.get('stealth_signals', [])
        dominant_sources = []
        
        for signal in stealth_signals:
            if signal.get('active', False) and signal.get('strength', 0.0) > 0.5:
                signal_name = signal.get('signal_name', '')
                if 'mastermind' in signal_name.lower():
                    dominant_sources.append('mastermind_tracing')
                elif 'whaleclip' in signal_name.lower():
                    dominant_sources.append('WhaleCLIP')
                elif 'whale_ping' in signal_name.lower():
                    dominant_sources.append('whale_ping')
                elif 'dex_inflow' in signal_name.lower():
                    dominant_sources.append('dex_inflow')
        
        if not dominant_sources:
            dominant_sources = ['stealth_engine']
        
        # Wygeneruj alert consensus
        print(f"🚨 ALERT [{token}] Consensus BUY | Score: {total_score:.3f}")
        print(f"🧠 Source: {' + '.join(dominant_sources)}")
        
        # FAKTYCZNIE WYŚLIJ DO TELEGRAM
        success = self._send_telegram_alert(token, total_score, token_data)
        if success:
            print("📡 Sent to Telegram ✅")
        else:
            print("📡 Telegram send FAILED ❌")
    
    def _send_telegram_alert(self, token: str, score: float, token_data: Dict) -> bool:
        """
        Faktycznie wyślij alert do Telegram
        """
        try:
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            
            if not bot_token or not chat_id:
                print("❌ Missing Telegram credentials")
                return False
            
            # Zbuduj wiadomość
            whale_ping = self._extract_detector_value(token_data.get('whale_ping', 0.0))
            dex_inflow = self._extract_detector_value(token_data.get('dex_inflow', 0.0))
            diamond_score = self._extract_detector_value(token_data.get('diamond_score', 0.0))
            californium_score = self._extract_detector_value(token_data.get('californium_score', 0.0))
            consensus_decision = token_data.get('consensus_decision', 'UNKNOWN')
            consensus_confidence = token_data.get('consensus_confidence', 0.0)
            
            message = f"""🚨 <b>CRYPTO ALERT</b> 🚨

📊 <b>{token}</b> - Consensus {consensus_decision}
💰 <b>Score:</b> {score:.3f} (confidence: {consensus_confidence:.2f})

🔍 <b>Breakdown:</b>
🐳 Whale Ping: {whale_ping:.2f}
💧 DEX Inflow: {dex_inflow:.2f}
💎 DiamondWhale AI: {diamond_score:.2f}
🌊 CaliforniumWhale: {californium_score:.2f}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
🤖 Powered by Crypto Scanner AI"""

            url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                return True
            else:
                print(f"❌ Telegram API error: {response.status_code}")
                print(f"Response: {response.text[:100]}...")
                return False
                
        except Exception as e:
            print(f"❌ Telegram send error: {e}")
            return False
    
    def log_stealth_analysis_start(self, symbol: str) -> None:
        """Log rozpoczęcia analizy stealth dla tokena"""
        print(f"[STEALTH ENGINE] {symbol} → Analyzing stealth signals...")
    
    def log_stealth_analysis_complete(self, symbol: str, detector_results_or_score, consensus_data_or_signals_count=None) -> None:
        """
        Comprehensive logging dla kompletnej stealth analysis
        Supports both old format (symbol, score, signals_count) and new format (symbol, detector_results, consensus_data)
        """
        # Check if this is the new format (dict) or old format (float)
        if isinstance(detector_results_or_score, dict) and consensus_data_or_signals_count is not None:
            # New format: comprehensive diagnostic logging
            detector_results = detector_results_or_score
            consensus_data = consensus_data_or_signals_count
            
            print(f"\n[STEALTH ANALYSIS] {symbol} - Complete diagnostic breakdown:")
            
            # Calculate total score and active detectors
            total_score = sum(self._extract_detector_value(score) for score in detector_results.values())
            active_detectors = [name for name, score in detector_results.items() 
                              if self._extract_detector_value(score) > 0.0]
            
            # Pattern identification
            pattern = self._identify_pattern(detector_results)
            confidence_icon, confidence_text = self.get_confidence_level(total_score)
            
            print(f"📊 {confidence_icon} Total Score: {total_score:.3f} {confidence_text}")
            print(f"🔍 Pattern: {pattern}")
            print(f"⚡ Active Detectors: {len(active_detectors)}/{len(detector_results)}")
            
            # Individual detector breakdown
            print("🔬 Detector Breakdown:")
            for name, score in detector_results.items():
                numeric_score = self._extract_detector_value(score)
                emoji = self.detector_symbols.get(name, '🔧')
                status = "ACTIVE" if numeric_score > 0.0 else "INACTIVE"
                print(f"  {emoji} {name}: {numeric_score:.3f} ({status})")
            
            # Consensus information
            if consensus_data:
                decision = consensus_data.get('decision', 'UNKNOWN')
                confidence = consensus_data.get('confidence', 0.0)
                votes = consensus_data.get('votes', [])
                print(f"🎯 Consensus: {decision} (confidence: {confidence:.3f})")
                if votes:
                    buy_count = votes.count('BUY')
                    print(f"🗳️  Agent Votes: {votes} → {buy_count}/{len(votes)} BUY")
        else:
            # Old format: simple score logging
            stealth_score = detector_results_or_score
            signals_count = consensus_data_or_signals_count or 0
            confidence_icon, confidence_text = self.get_confidence_level(stealth_score)
            print(f"[STEALTH COMPLETE] {symbol} → Score: {stealth_score:.3f} | Signals: {signals_count} | {confidence_icon} {confidence_text}")
    
    def log_detector_activation(self, symbol: str, detector_name: str, score, active_or_confidence=None) -> None:
        """
        Enhanced logging dla individual detector activation
        Supports both old format (symbol, name, score, active) and new format (symbol, name, score, confidence)
        """
        emoji = self.detector_symbols.get(detector_name, '🔧')
        
        # Extract numeric value from score (supports bool, float, or dict)
        numeric_score = self._extract_detector_value(score)
        
        if active_or_confidence is None or isinstance(active_or_confidence, bool):
            # Old format or simple activation
            active = active_or_confidence if active_or_confidence is not None else (numeric_score > 0.0)
            status = "ACTIVE" if active else "INACTIVE"
            print(f"[DETECTOR] {symbol} → {emoji} {detector_name}: {numeric_score:.3f} ({status})")
        else:
            # New format with confidence
            confidence = active_or_confidence
            confidence_text = f" (confidence: {confidence:.2f})" if confidence else ""
            
            if numeric_score > 0.5:
                print(f"[STEALTH] {emoji} {detector_name} detected pattern{confidence_text} - Score: {numeric_score:.3f}")
            elif numeric_score > 0.0:
                print(f"[STEALTH] {emoji} {detector_name} weak signal{confidence_text} - Score: {numeric_score:.3f}")
    
    def log_consensus_decision(self, symbol: str, consensus_vote: str, votes: str, confidence: float) -> None:
        """Log decyzji consensus system"""
        print(f"[CONSENSUS] {symbol} → 🎯 Vote: {consensus_vote} ({votes}) | Confidence: {confidence:.2f}")
    
    def log_feedback_adjustment(self, symbol: str, original_score: float, adjusted_score: float, adjustment: float) -> None:
        """Log korekty feedback loop"""
        print(f"[FEEDBACK] {symbol} → 🔁 {original_score:.3f} → {adjusted_score:.3f} (adj: {adjustment:+.3f})")

# Singleton instance
stealth_logger = StealthLogger()

def get_stealth_logger() -> StealthLogger:
    """Pobierz singleton instance stealth logger"""
    return stealth_logger

def print_top5_stealth_tokens_enhanced(top_tokens: List[Dict]) -> None:
    """Convenience function dla nowego systemu logowania TOP 5"""
    stealth_logger.print_top5_stealth_tokens_enhanced(top_tokens)

def log_stealth_analysis_start(symbol: str) -> None:
    """Convenience function"""
    stealth_logger.log_stealth_analysis_start(symbol)

def log_stealth_analysis_complete(symbol: str, stealth_score: float, signals_count: int) -> None:
    """Convenience function"""
    stealth_logger.log_stealth_analysis_complete(symbol, stealth_score, signals_count)

def log_detector_activation(symbol: str, detector_name: str, score: float, active: bool) -> None:
    """Convenience function"""
    stealth_logger.log_detector_activation(symbol, detector_name, score, active)

def log_consensus_decision(symbol: str, consensus_vote: str, votes: str, confidence: float) -> None:
    """Convenience function"""
    stealth_logger.log_consensus_decision(symbol, consensus_vote, votes, confidence)

def log_feedback_adjustment(symbol: str, original_score: float, adjusted_score: float, adjustment: float) -> None:
    """Convenience function"""
    stealth_logger.log_feedback_adjustment(symbol, original_score, adjusted_score, adjustment)