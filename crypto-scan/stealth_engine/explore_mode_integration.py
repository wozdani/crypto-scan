#!/usr/bin/env python3
"""
Explore Mode Integration
Integruje nowy system nazewnictwa explore mode z istniejącym kodem stealth_engine.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from agent_learning.explore_file_manager import ExploreFileManager
    from agent_learning.pump_verification_scheduler import start_pump_verification_scheduler
except ImportError as e:
    print(f"[EXPLORE INTEGRATION WARNING] Import error: {e}")
    ExploreFileManager = None

class ExploreIntegration:
    """
    Integracja między stealth_engine a nowym systemem explore mode.
    """
    
    def __init__(self):
        self.file_manager = ExploreFileManager() if ExploreFileManager else None
        self.enabled = self.file_manager is not None
        
        if not self.enabled:
            print("[EXPLORE INTEGRATION] Warning: ExploreFileManager not available, using legacy mode")
    
    def should_save_to_explore_mode(self, symbol: str, token_data: Dict, score: float) -> bool:
        """
        Sprawdź czy token powinien być zapisany w explore mode.
        WYMAGANIE: Tylko tokeny z score >= 1.0
        
        Args:
            symbol: Symbol tokena
            token_data: Dane tokena
            score: Stealth score
            
        Returns:
            True jeśli powinien być zapisany
        """
        # JEDYNE KRYTERIUM: Score >= 1.0
        if score >= 1.0:
            print(f"[EXPLORE MODE QUALIFIED] {symbol}: Score {score:.3f} >= 1.0 - saving to explore mode")
            return True
        
        print(f"[EXPLORE MODE SKIP] {symbol}: Score {score:.3f} < 1.0 - not saving")
        return False
    
    def save_explore_data_enhanced(self, symbol: str, token_data: Dict, detector_results: Dict, 
                                 consensus_data: Optional[Dict] = None) -> bool:
        """
        Zapisz dane explore mode używając nowego systemu nazewnictwa.
        
        Args:
            symbol: Symbol tokena
            token_data: Dane tokena
            detector_results: Wyniki detektorów
            consensus_data: Dane consensus (opcjonalne)
            
        Returns:
            True jeśli zapisano pomyślnie
        """
        if not self.enabled:
            # Fallback to legacy save
            return self._save_legacy_explore_data(symbol, token_data, detector_results)
        
        try:
            # Przygotuj wzbogacone dane
            enhanced_token_data = self._enhance_token_data(token_data, detector_results, consensus_data)
            
            # Zapisz używając nowego file managera
            filepath = self.file_manager.save_explore_file(symbol, enhanced_token_data, detector_results)
            
            if filepath:
                print(f"[EXPLORE INTEGRATION] Saved enhanced explore data: {os.path.basename(filepath)}")
                return True
            else:
                print(f"[EXPLORE INTEGRATION ERROR] Failed to save explore data for {symbol}")
                return False
                
        except Exception as e:
            print(f"[EXPLORE INTEGRATION ERROR] Exception saving {symbol}: {e}")
            return self._save_legacy_explore_data(symbol, token_data, detector_results)
    
    def _enhance_token_data(self, token_data: Dict, detector_results: Dict, consensus_data: Optional[Dict]) -> Dict:
        """
        Wzbogać token_data o dodatkowe informacje dla explore mode.
        
        Args:
            token_data: Podstawowe dane tokena
            detector_results: Wyniki detektorów
            consensus_data: Dane consensus
            
        Returns:
            Wzbogacone dane tokena
        """
        enhanced_data = token_data.copy()
        
        # Dodaj scores z detector_results
        if "stealth_engine" in detector_results:
            stealth_result = detector_results["stealth_engine"]
            enhanced_data["stealth_score"] = stealth_result.get("score", 0)
            enhanced_data["stealth_confidence"] = stealth_result.get("confidence", 0)
        
        if "diamond_whale" in detector_results:
            diamond_result = detector_results["diamond_whale"]
            enhanced_data["diamond_ai_score"] = diamond_result.get("score", 0)
            enhanced_data["diamond_confidence"] = diamond_result.get("confidence", 0)
        
        if "californium_whale" in detector_results:
            californium_result = detector_results["californium_whale"]
            enhanced_data["californium_ai_score"] = californium_result.get("score", 0)
            enhanced_data["californium_confidence"] = californium_result.get("confidence", 0)
        
        # Dodaj consensus data jeśli dostępne
        if consensus_data:
            enhanced_data["consensus_decision"] = consensus_data.get("decision", "UNKNOWN")
            enhanced_data["consensus_score"] = consensus_data.get("score", 0)
            enhanced_data["consensus_confidence"] = consensus_data.get("confidence", 0)
        
        # Dodaj metadata
        enhanced_data["explore_created_at"] = datetime.now().isoformat()
        enhanced_data["explore_version"] = "v2_enhanced"
        
        return enhanced_data
    
    def _save_legacy_explore_data(self, symbol: str, token_data: Dict, detector_results: Dict) -> bool:
        """
        Fallback do legacy systemu zapisu explore mode.
        
        Args:
            symbol: Symbol tokena
            token_data: Dane tokena
            detector_results: Wyniki detektorów
            
        Returns:
            True jeśli zapisano pomyślnie
        """
        try:
            explore_dir = "crypto-scan/cache/explore_mode"
            os.makedirs(explore_dir, exist_ok=True)
            
            # Legacy format filename
            filename = f"{symbol}_explore.json"
            filepath = os.path.join(explore_dir, filename)
            
            # Przygotuj dane w legacy format
            explore_data = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "mode": "legacy_fallback",
                **token_data,
                "detector_results": detector_results
            }
            
            with open(filepath, 'w') as f:
                json.dump(explore_data, f, indent=2)
            
            print(f"[EXPLORE INTEGRATION] Saved legacy explore data: {filename}")
            return True
            
        except Exception as e:
            print(f"[EXPLORE INTEGRATION ERROR] Legacy save failed for {symbol}: {e}")
            return False
    
    def migrate_existing_files(self) -> Dict[str, int]:
        """
        Migruj istniejące pliki explore mode do nowego formatu.
        
        Returns:
            Statystyki migracji
        """
        if not self.enabled:
            print("[EXPLORE INTEGRATION] Migration skipped - enhanced system not available")
            return {"migrated": 0, "failed": 0, "total_legacy": 0}
        
        try:
            return self.file_manager.migrate_legacy_files()
        except Exception as e:
            print(f"[EXPLORE INTEGRATION ERROR] Migration failed: {e}")
            return {"migrated": 0, "failed": 1, "total_legacy": 0}
    
    def start_verification_scheduler(self):
        """
        Uruchom scheduler weryfikacji pump.
        """
        if not self.enabled:
            print("[EXPLORE INTEGRATION] Scheduler not started - enhanced system not available")
            return
        
        try:
            start_pump_verification_scheduler()
            print("[EXPLORE INTEGRATION] Pump verification scheduler started")
        except Exception as e:
            print(f"[EXPLORE INTEGRATION ERROR] Failed to start scheduler: {e}")
    
    def manual_verification(self) -> Dict:
        """
        Uruchom manualną weryfikację pump.
        
        Returns:
            Wyniki weryfikacji
        """
        if not self.enabled:
            print("[EXPLORE INTEGRATION] Manual verification not available - enhanced system not available")
            return {"error": "enhanced_system_not_available"}
        
        try:
            from agent_learning.pump_verification_scheduler import manual_pump_verification
            return manual_pump_verification()
        except Exception as e:
            print(f"[EXPLORE INTEGRATION ERROR] Manual verification failed: {e}")
            return {"error": str(e)}

# Global integration instance
_integration_instance = None

def get_explore_integration() -> ExploreIntegration:
    """
    Singleton pattern dla integration.
    
    Returns:
        Global integration instance
    """
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = ExploreIntegration()
    return _integration_instance

def save_explore_mode_data(symbol: str, token_data: Dict, detector_results: Dict, 
                          consensus_data: Optional[Dict] = None) -> Optional[str]:
    """
    Public interface do zapisu explore mode data.
    WYMAGANIE: Tylko tokeny z score >= 1.0 są zapisywane
    
    Args:
        symbol: Symbol tokena
        token_data: Dane tokena
        detector_results: Wyniki detektorów
        consensus_data: Dane consensus (opcjonalne)
        
    Returns:
        Filename jeśli zapisano pomyślnie, None jeśli nie
    """
    # Check if token qualifies for explore mode (score >= 1.0)
    integration = get_explore_integration()
    score = token_data.get("stealth_score", 0.0)
    
    if not integration.should_save_to_explore_mode(symbol, token_data, score):
        return None
        
    try:
        # Create explore mode directory
        explore_dir = "crypto-scan/cache/explore_mode"
        os.makedirs(explore_dir, exist_ok=True)
        
        # Simple direct file naming with timestamp for Enhanced Explore Mode
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get detector names for filename
        detector_names = list(detector_results.keys()) if detector_results else ["stealth"]
        detector_str = "_".join(detector_names[:2])  # Max 2 detectors in filename
        
        # Enhanced Explore Mode filename format: TOKEN_YYYYMMDD_HHMMSS_DETECTORS.json
        filename = f"{symbol}_{timestamp}_{detector_str}.json"
        filepath = os.path.join(explore_dir, filename)
        
        # Prepare explore mode data
        explore_data = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "created_at": timestamp,
            "explore_mode": True,
            "version": "enhanced_v2",
            "token_data": token_data,
            "detector_results": detector_results
        }
        
        # Add consensus data if available
        if consensus_data:
            explore_data["consensus_data"] = consensus_data
        
        # Write JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(explore_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"[ENHANCED EXPLORE SAVE SUCCESS] {symbol}: Saved {filename}")
        return filename
        
    except Exception as e:
        print(f"[ENHANCED EXPLORE SAVE ERROR] {symbol}: Failed to save - {e}")
        return False

def initialize_explore_system():
    """
    Inicjalizuj cały enhanced explore system.
    """
    integration = get_explore_integration()
    
    if integration.enabled:
        print("[EXPLORE INTEGRATION] Initializing enhanced explore mode system...")
        
        # Migruj istniejące pliki
        migration_stats = integration.migrate_existing_files()
        if migration_stats["migrated"] > 0:
            print(f"[EXPLORE INTEGRATION] Migrated {migration_stats['migrated']} legacy files")
        
        # Uruchom scheduler
        integration.start_verification_scheduler()
        
        print("[EXPLORE INTEGRATION] Enhanced explore mode system initialized successfully")
    else:
        print("[EXPLORE INTEGRATION] Using legacy explore mode system")

if __name__ == "__main__":
    # Test integration
    integration = ExploreIntegration()
    
    print("Testing integration...")
    test_token_data = {
        "stealth_score": 1.5,
        "whale_ping_strength": 0.8,
        "dex_inflow_usd": 25000,
        "active_signals": ["whale_ping", "dex_inflow", "orderbook_anomaly"],
        "price": 0.1234,
        "volume_24h": 1000000
    }
    
    test_detector_results = {
        "stealth_engine": {"score": 1.5, "confidence": 0.8},
        "diamond_whale": {"score": 0.6, "confidence": 0.9}
    }
    
    success = integration.save_explore_data_enhanced("TESTUSDT", test_token_data, test_detector_results)
    print(f"Save test result: {success}")