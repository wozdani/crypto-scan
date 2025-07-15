#!/usr/bin/env python3
"""
Stealth Pre-Pump Engine Logging System
Precyzyjne logowanie dla nowych detektorów, consensus i feedback loop
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

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
        
    def get_confidence_level(self, score: float) -> tuple:
        """Zwróć poziom zagrożenia na podstawie score"""
        if score > 1.8:
            return "🔥", "[HIGH CONFIDENCE]"
        elif 1.2 < score <= 1.8:
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
            
            # Główny nagłówek tokena
            confidence_icon, confidence_text = self.get_confidence_level(stealth_score)
            print(f"{i:2d}. {token:12} | Stealth: {stealth_score:6.3f} | Early: {early_score:6.3f}")
            print(f"     DEX: {token_data.get('dex_inflow', 0.0):6.3f} | Whale: {token_data.get('whale_ping', 0.0):6.3f} | Trust: {token_data.get('trust_boost', 0.0):6.3f} | ID: {token_data.get('identity_boost', 0.0):6.3f}")
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
        Wyświetl szczegółowy breakdown detektorów
        """
        print("🔎 Breakdown:")
        
        # Pobierz scores z stealth signals
        stealth_signals = token_data.get('stealth_signals', [])
        detector_scores = self._extract_detector_scores(stealth_signals)
        
        # Wyświetl każdy detektor
        print(f" - 🐳 whale_ping:        {detector_scores.get('whale_ping', 0.0):5.2f}")
        print(f" - 🧠 mastermind_tracing: {detector_scores.get('mastermind_tracing', 0.0):5.2f}")
        print(f" - 💧 dex_inflow:        {detector_scores.get('dex_inflow', 0.0):5.2f}")
        print(f" - 📊 orderbook_anomaly:  {detector_scores.get('orderbook_anomaly', 0.0):5.2f}")
        print(f" - 🛰️ WhaleCLIP (vision): {detector_scores.get('whaleclip_vision', 0.0):5.2f}")
        
        # Consensus vote
        consensus_vote = token_data.get('consensus_vote', 'UNKNOWN')
        consensus_count = token_data.get('consensus_count', '0/0')
        print(f" - 🎯 consensus_vote:    {consensus_vote} ({consensus_count})")
        
        # Feedback adjustment
        feedback_adjust = token_data.get('feedback_adjust', 0.0)
        print(f" - 🔁 feedback_adjust:   {feedback_adjust:+5.2f}")
        
        # Total score
        total_score = token_data.get('base_score', 0.0)
        print(f" - 🔐 total_score:       {total_score:5.3f}")
    
    def _extract_detector_scores(self, stealth_signals: List[Dict]) -> Dict[str, float]:
        """
        Wyciągnij scores z stealth signals
        """
        scores = {}
        
        for signal in stealth_signals:
            signal_name = signal.get('signal_name', '')
            strength = signal.get('strength', 0.0)
            active = signal.get('active', False)
            
            # Mapuj nazwy sygnałów na detektory
            if 'whale_ping' in signal_name.lower():
                scores['whale_ping'] = strength if active else 0.0
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
        Sprawdź czy consensus vote == BUY i wygeneruj alert
        """
        consensus_vote = token_data.get('consensus_vote', '')
        
        if consensus_vote == 'BUY':
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
            print("📡 Sent to Telegram")
    
    def log_stealth_analysis_start(self, symbol: str) -> None:
        """Log rozpoczęcia analizy stealth dla tokena"""
        print(f"[STEALTH ENGINE] {symbol} → Analyzing stealth signals...")
    
    def log_stealth_analysis_complete(self, symbol: str, stealth_score: float, signals_count: int) -> None:
        """Log zakończenia analizy stealth"""
        confidence_icon, confidence_text = self.get_confidence_level(stealth_score)
        print(f"[STEALTH COMPLETE] {symbol} → Score: {stealth_score:.3f} | Signals: {signals_count} | {confidence_icon} {confidence_text}")
    
    def log_detector_activation(self, symbol: str, detector_name: str, score: float, active: bool) -> None:
        """Log aktywacji konkretnego detektora"""
        symbol_emoji = self.detector_symbols.get(detector_name, '🔧')
        status = "ACTIVE" if active else "INACTIVE"
        print(f"[DETECTOR] {symbol} → {symbol_emoji} {detector_name}: {score:.3f} ({status})")
    
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