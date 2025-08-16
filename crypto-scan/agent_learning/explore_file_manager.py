#!/usr/bin/env python3
"""
Enhanced Explore Mode File Manager
Zarządza plikami explore mode z poprawnym nazewnictwem i weryfikacją pump po 6h.

Nowe nazewnictwo:
- Przed 6h: TOKEN_YYYYMMDD_HHMMSS_DETECTORS (np. WIFUSDT_20250811_094512_whale+dex)
- Po 6h: TOKEN_YYYYMMDD_HHMMSS_OUTCOME (np. WIFUSDT_20250811_094512_pump)
- OUTCOME ∈ {pump, dump, no_move}
- Pliki maksymalnie 3 dni
"""

import json
import os
import glob
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests

class ExploreFileManager:
    def __init__(self, explore_dir: str = "crypto-scan/cache/explore_mode"):
        self.explore_dir = explore_dir
        self.max_file_age_days = 3
        self.verification_hours = 6
        
        # Ensure directory exists
        os.makedirs(self.explore_dir, exist_ok=True)
        
    def generate_detector_label(self, token_data: Dict) -> str:
        """
        Generuj etykietę detektorów na podstawie aktywnych sygnałów.
        
        Args:
            token_data: Dane tokena z aktywnych detektorów
            
        Returns:
            String z aktywny detektorami np. "whale+dex+diamond"
        """
        active_signals = []
        
        # Sprawdź główne detektory
        if token_data.get("whale_ping_strength", 0) > 0:
            active_signals.append("whale")
        if token_data.get("dex_inflow_usd", 0) > 0:
            active_signals.append("dex")
        if token_data.get("diamond_ai_score", 0) > 0.3:
            active_signals.append("diamond")
        if token_data.get("californium_ai_score", 0) > 0.3:
            active_signals.append("californium")
        if token_data.get("orderbook_anomaly", False):
            active_signals.append("orderbook")
        if token_data.get("spoofing_layers", False):
            active_signals.append("spoof")
        
        # Sprawdź stealth signals
        stealth_signals = token_data.get("active_signals", [])
        for signal in stealth_signals:
            if signal == "multi_address_group_activity":
                active_signals.append("multiaddr")
            elif signal == "large_bid_walls":
                active_signals.append("bidwall")
            elif signal == "volume_spike":
                active_signals.append("volspike")
        
        # Fallback jeśli brak aktywnych detektorów
        if not active_signals:
            active_signals.append("stealth")
            
        # Maksymalnie 4 detektory dla czytelności
        return "+".join(active_signals[:4])
    
    def _build_detector_details(self, token_data: Dict, detector_results: Dict) -> Dict:
        """Buduj szczegółowe dane wszystkich detektorów"""
        detectors = {}
        
        # StealthEngine
        stealth_data = token_data.get("stealth_engine_data", {})
        detectors["stealth_engine"] = {
            "active": token_data.get("stealth_score", 0.0) > 0.1,
            "score": token_data.get("stealth_score", 0.0),
            "confidence": stealth_data.get("confidence", 0.7),
            "features": {
                "active_signals": token_data.get("active_signals", []),
                "signal_count": len(token_data.get("active_signals", [])),
                "whale_ping_strength": token_data.get("whale_ping_strength", 0.0),
                "dex_inflow_usd": token_data.get("dex_inflow_usd", 0.0),
                "volume_spike_detected": "volume_spike" in token_data.get("active_signals", []),
                "orderbook_anomaly": "orderbook_anomaly" in token_data.get("active_signals", []),
                "spoofing_layers": "spoofing_layers" in token_data.get("active_signals", [])
            },
            "reasoning": stealth_data.get("reasoning", "Comprehensive stealth signal analysis"),
            "weights_used": stealth_data.get("weights_used", {})
        }
        
        # CaliforniumWhale
        californium_data = detector_results.get("californium_whale", {})
        detectors["californium_whale"] = {
            "active": californium_data.get("score", 0.0) > 0.1,
            "score": californium_data.get("score", 0.0),
            "confidence": californium_data.get("confidence", 0.85),
            "features": {
                "ai_analysis_score": californium_data.get("ai_score", 0.0),
                "temporal_graph_signals": californium_data.get("temporal_signals", []),
                "qirl_score": californium_data.get("qirl_score", 0.0),
                "whale_pattern_detected": californium_data.get("whale_pattern", False),
                "trust_score": californium_data.get("trust_score", 0.0)
            },
            "reasoning": californium_data.get("reasoning", "AI-powered whale behavior analysis"),
            "vote_details": californium_data.get("agent_votes", {})
        }
        
        # DiamondWhale
        diamond_data = detector_results.get("diamond_whale", {})
        detectors["diamond_whale"] = {
            "active": diamond_data.get("score", 0.0) > 0.1,
            "score": diamond_data.get("score", 0.0),
            "confidence": diamond_data.get("confidence", 0.75),
            "features": {
                "diamond_ai_score": diamond_data.get("diamond_ai_score", 0.0),
                "blockchain_transactions": len(diamond_data.get("transactions", [])),
                "temporal_graph_analysis": diamond_data.get("temporal_analysis", {}),
                "advanced_pattern_score": diamond_data.get("pattern_score", 0.0),
                "stealth_integration_score": diamond_data.get("stealth_integration", 0.0)
            },
            "reasoning": diamond_data.get("reasoning", "Advanced Diamond AI blockchain analysis"),
            "vote_details": diamond_data.get("agent_votes", {})
        }
        
        # WhaleCLIP
        whales_data = detector_results.get("whale_clip", {})
        detectors["whale_clip"] = {
            "active": whales_data.get("score", 0.0) > 0.1,
            "score": whales_data.get("score", 0.0),
            "confidence": whales_data.get("confidence", 0.8),
            "features": {
                "clip_analysis_score": whales_data.get("clip_score", 0.0),
                "whale_classification": whales_data.get("whale_class", "unknown"),
                "behavioral_embedding": whales_data.get("behavioral_score", 0.0),
                "transaction_patterns": whales_data.get("patterns", []),
                "historical_performance": whales_data.get("historical_perf", 0.0)
            },
            "reasoning": whales_data.get("reasoning", "CLIP-based whale behavior classification"),
            "vote_details": whales_data.get("agent_votes", {})
        }
        
        # GNN (Graph Neural Network)
        gnn_data = detector_results.get("gnn", {})
        detectors["gnn"] = {
            "active": gnn_data.get("score", 0.0) > 0.1,
            "score": gnn_data.get("score", 0.0),
            "confidence": gnn_data.get("confidence", 0.9),
            "features": {
                "graph_neural_score": gnn_data.get("gnn_score", 0.0),
                "node_embeddings": gnn_data.get("embeddings", []),
                "graph_structure_score": gnn_data.get("structure_score", 0.0),
                "anomaly_detection": gnn_data.get("anomaly_detected", False),
                "network_analysis": gnn_data.get("network_analysis", {})
            },
            "reasoning": gnn_data.get("reasoning", "Graph Neural Network blockchain analysis"),
            "vote_details": gnn_data.get("agent_votes", {})
        }
        
        return detectors
    
    def _build_consensus_details(self, token_data: Dict, detector_results: Dict) -> Dict:
        """Buduj szczegółowe dane consensus multi-agent"""
        return {
            "multi_agent_system": {
                "total_detectors": len([d for d in detector_results.values() if d.get("score", 0) > 0.1]),
                "consensus_method": "5-agent-individual-evaluation",
                "threshold_logic": "minimum_2_detectors_BUY_for_alert",
                "vote_margin_required": 2,
                "consensus_threshold": 0.7
            },
            "detector_summary": {
                "stealth_engine": {
                    "decision": detector_results.get("stealth_engine", {}).get("decision", "PENDING"),
                    "score": token_data.get("stealth_score", 0.0),
                    "confidence": detector_results.get("stealth_engine", {}).get("confidence", 0.7)
                },
                "californium_whale": {
                    "decision": detector_results.get("californium_whale", {}).get("decision", "PENDING"),
                    "score": detector_results.get("californium_whale", {}).get("score", 0.0),
                    "confidence": detector_results.get("californium_whale", {}).get("confidence", 0.85)
                },
                "diamond_whale": {
                    "decision": detector_results.get("diamond_whale", {}).get("decision", "PENDING"),
                    "score": detector_results.get("diamond_whale", {}).get("score", 0.0),
                    "confidence": detector_results.get("diamond_whale", {}).get("confidence", 0.75)
                },
                "whale_clip": {
                    "decision": detector_results.get("whale_clip", {}).get("decision", "PENDING"),
                    "score": detector_results.get("whale_clip", {}).get("score", 0.0),
                    "confidence": detector_results.get("whale_clip", {}).get("confidence", 0.8)
                },
                "gnn": {
                    "decision": detector_results.get("gnn", {}).get("decision", "PENDING"),
                    "score": detector_results.get("gnn", {}).get("score", 0.0),
                    "confidence": detector_results.get("gnn", {}).get("confidence", 0.9)
                }
            },
            "final_consensus": {
                "decision": token_data.get("consensus_decision", "EXPLORE"),
                "buy_votes": token_data.get("buy_votes", 0),
                "hold_votes": token_data.get("hold_votes", 0),
                "avoid_votes": token_data.get("avoid_votes", 0),
                "alert_triggered": token_data.get("alert_triggered", False),
                "consensus_confidence": token_data.get("consensus_confidence", 0.0)
            }
        }
    
    def _build_agent_votes_details(self, token_data: Dict, detector_results: Dict) -> Dict:
        """Buduj szczegółowe głosy wszystkich agentów dla każdego detektora"""
        agent_votes = {}
        
        for detector_name, detector_data in detector_results.items():
            if detector_data.get("agent_votes"):
                agent_votes[detector_name] = {
                    "analyzer": {
                        "vote": detector_data["agent_votes"].get("analyzer", {}).get("vote", "UNKNOWN"),
                        "confidence": detector_data["agent_votes"].get("analyzer", {}).get("confidence", 0.0),
                        "reasoning": detector_data["agent_votes"].get("analyzer", {}).get("reasoning", "No reasoning provided")
                    },
                    "reasoner": {
                        "vote": detector_data["agent_votes"].get("reasoner", {}).get("vote", "UNKNOWN"),
                        "confidence": detector_data["agent_votes"].get("reasoner", {}).get("confidence", 0.0),
                        "reasoning": detector_data["agent_votes"].get("reasoner", {}).get("reasoning", "No reasoning provided")
                    },
                    "voter": {
                        "vote": detector_data["agent_votes"].get("voter", {}).get("vote", "UNKNOWN"),
                        "confidence": detector_data["agent_votes"].get("voter", {}).get("confidence", 0.0),
                        "reasoning": detector_data["agent_votes"].get("voter", {}).get("reasoning", "No reasoning provided")
                    },
                    "debater": {
                        "vote": detector_data["agent_votes"].get("debater", {}).get("vote", "UNKNOWN"),
                        "confidence": detector_data["agent_votes"].get("debater", {}).get("confidence", 0.0),
                        "reasoning": detector_data["agent_votes"].get("debater", {}).get("reasoning", "No reasoning provided")
                    },
                    "decider": {
                        "vote": detector_data["agent_votes"].get("decider", {}).get("vote", "UNKNOWN"),
                        "confidence": detector_data["agent_votes"].get("decider", {}).get("confidence", 0.0),
                        "reasoning": detector_data["agent_votes"].get("decider", {}).get("reasoning", "No reasoning provided")
                    },
                    "vote_summary": {
                        "buy_count": sum(1 for agent in detector_data["agent_votes"].values() if agent.get("vote") == "BUY"),
                        "hold_count": sum(1 for agent in detector_data["agent_votes"].values() if agent.get("vote") == "HOLD"),
                        "avoid_count": sum(1 for agent in detector_data["agent_votes"].values() if agent.get("vote") == "AVOID"),
                        "average_confidence": sum(agent.get("confidence", 0) for agent in detector_data["agent_votes"].values()) / 5,
                        "vote_margin": detector_data.get("vote_margin", 0),
                        "final_decision": detector_data.get("decision", "PENDING")
                    }
                }
            else:
                # Fallback dla detektorów bez szczegółowych głosów
                agent_votes[detector_name] = {
                    "vote_summary": {
                        "final_decision": detector_data.get("decision", "PENDING"),
                        "score": detector_data.get("score", 0.0),
                        "confidence": detector_data.get("confidence", 0.0),
                        "note": "Detailed agent votes not available for this detector"
                    }
                }
        
        return agent_votes
    
    def _build_feature_vector(self, token_data: Dict) -> Dict:
        """Buduj wektor cech dla machine learning"""
        return {
            "fv_stealth_score": token_data.get("stealth_score", 0.0),
            "fv_whale_ping_strength": token_data.get("whale_ping_strength", 0.0),
            "fv_dex_inflow_usd": token_data.get("dex_inflow_usd", 0.0),
            "fv_volume_24h": token_data.get("volume_24h", 0.0),
            "fv_price_change_24h": token_data.get("price_change_24h", 0.0),
            "fv_market_cap": token_data.get("market_cap", 0.0),
            "fv_orderbook_depth": token_data.get("orderbook_depth_usd", 0.0),
            "fv_orderbook_imbalance": token_data.get("orderbook_imbalance", 0.0),
            "fv_active_signals_count": len(token_data.get("active_signals", [])),
            "fv_address_trust_score": token_data.get("address_trust_score", 0.0),
            "fv_repeated_addresses": token_data.get("repeated_addresses_count", 0),
            "fv_californium_score": token_data.get("californium_ai_score", 0.0),
            "fv_diamond_score": token_data.get("diamond_ai_score", 0.0),
            "fv_whale_clip_score": token_data.get("whale_clip_score", 0.0),
            "fv_gnn_score": token_data.get("gnn_score", 0.0),
            "fv_price": token_data.get("price", 0.0),
            "fv_consensus_confidence": token_data.get("consensus_confidence", 0.0)
        }
    
    def save_explore_file(self, symbol: str, token_data: Dict, detector_results: Dict) -> str:
        """
        Zapisz plik explore mode z nowym nazewnictwem.
        
        Args:
            symbol: Symbol tokena
            token_data: Dane tokena
            detector_results: Wyniki detektorów
            
        Returns:
            Ścieżka do zapisanego pliku
        """
        timestamp = datetime.now()
        
        # Wygeneruj etykietę detektorów
        detector_label = self.generate_detector_label(token_data)
        
        # Nowe nazewnictwo: TOKEN_YYYYMMDD_HHMMSS_DETECTORS
        filename = f"{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{detector_label}.json"
        filepath = os.path.join(self.explore_dir, filename)
        
        # Przygotuj dane do zapisu w zaawansowanym schemacie
        explore_data = {
            "schema_version": "explore-stealth/v2.0.0",
            "engine": {
                "name": "stealth_pre_pump_enhanced",
                "weights_version": "v4.0.0",
                "commit": "enhanced_detectors",
                "threshold_profile_id": "prepump-enhanced-2025-08"
            },
            "event": {
                "event_id": f"EXP-{timestamp.strftime('%Y%m%d')}-{symbol}-{timestamp.strftime('%H%M')}",
                "run_id": f"SCAN-{timestamp.strftime('%Y%m%d-%H%MZ')}",
                "ts": timestamp.isoformat(),
                "symbol": symbol,
                "exchange": "Bybit",
                "market_type": "perp",
                "chain": token_data.get("chain", "ethereum"),
                "price": token_data.get("price", 0.0),
                "window": {
                    "lookback_s": 1800,
                    "features_end_ts": timestamp.isoformat()
                }
            },
            "market_context": {
                "volume_24h": token_data.get("volume_24h", 0.0),
                "price_change_24h": token_data.get("price_change_24h", 0.0),
                "market_cap": token_data.get("market_cap", 0.0),
                "price_source": token_data.get("price_source", "unknown"),
                "orderbook_available": token_data.get("orderbook_available", False),
                "candle_data_quality": token_data.get("candle_data_quality", "unknown")
            },
            "liquidity_context": {
                "orderbook_bids": token_data.get("orderbook_bids", 0),
                "orderbook_asks": token_data.get("orderbook_asks", 0),
                "orderbook_spread": token_data.get("orderbook_spread", 0.0),
                "orderbook_depth_usd": token_data.get("orderbook_depth_usd", 0.0),
                "liquidity_score": token_data.get("liquidity_score", 0.0)
            },
            "detectors": self._build_detector_details(token_data, detector_results),
            "consensus_data": self._build_consensus_details(token_data, detector_results),
            "agent_votes": self._build_agent_votes_details(token_data, detector_results),
            "memory_trust": {
                "whale_addresses_known": len(token_data.get("whale_addresses", [])),
                "repeated_addresses": token_data.get("repeated_addresses_count", 0),
                "address_trust_score": token_data.get("address_trust_score", 0.0),
                "trust_context": token_data.get("trust_context", {})
            },
            "decision_snapshot": {
                "stealth_score": token_data.get("stealth_score", 0.0),
                "decision": "EXPLORE",
                "reasons": token_data.get("explore_reasons", ["stealth signals detected"]),
                "threshold_met": token_data.get("stealth_score", 0.0) >= 0.5
            },
            "feature_vector": self._build_feature_vector(token_data),
            "blockchain_context": {
                "contract_address": token_data.get("contract_address", ""),
                "chain": token_data.get("chain", "ethereum"),
                "dex_inflow_usd": token_data.get("dex_inflow_usd", 0.0),
                "whale_transactions": token_data.get("whale_transactions", []),
                "blockchain_data_quality": token_data.get("blockchain_data_quality", "unknown")
            },
            "labeling_policy": {
                "verification_hours": self.verification_hours,
                "pump_threshold_pct": 8.0,
                "dump_threshold_pct": -8.0,
                "horizons": [
                    {"name": "H6", "hours": 6, "pump_threshold": 0.08, "dump_threshold": -0.08}
                ]
            },
            "label": {"status": "pending"},
            "file_lifecycle": {
                "created": timestamp.isoformat(),
                "verification_scheduled": (timestamp + timedelta(hours=self.verification_hours)).isoformat(),
                "cleanup_scheduled": (timestamp + timedelta(days=self.max_file_age_days)).isoformat(),
                "status": "pending_verification"
            },
            "quality": {
                "data_gaps": token_data.get("data_gaps", []),
                "warnings": token_data.get("warnings", []),
                "notes": f"Explore mode activated with score {token_data.get('stealth_score', 0.0):.3f}"
            },
            "audit": {
                "created_ts": timestamp.isoformat(),
                "updated_ts": timestamp.isoformat(),
                "history": [{"ts": timestamp.isoformat(), "action": "create", "by": "explore_mode_enhanced"}]
            }
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(explore_data, f, indent=2)
                
            print(f"[EXPLORE FILE] Saved: {filename}")
            print(f"[EXPLORE FILE] Verification scheduled in 6h: {explore_data['verification_due']}")
            
            return filepath
            
        except Exception as e:
            print(f"[EXPLORE FILE ERROR] Failed to save {filename}: {e}")
            return ""
    
    def get_files_ready_for_verification(self) -> List[Dict]:
        """
        Znajdź pliki gotowe do 6h weryfikacji pump.
        
        Returns:
            Lista plików gotowych do weryfikacji
        """
        ready_files = []
        now = datetime.now()
        
        # Znajdź pliki w statusie pending_verification
        pattern = os.path.join(self.explore_dir, "*.json")
        for filepath in glob.glob(pattern):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Sprawdź czy plik jest gotowy do weryfikacji
                if data.get("status") == "pending_verification":
                    verification_due = datetime.fromisoformat(data.get("verification_due", ""))
                    
                    if now >= verification_due:
                        ready_files.append({
                            "filepath": filepath,
                            "filename": os.path.basename(filepath),
                            "data": data,
                            "hours_since_creation": (now - datetime.fromisoformat(data["timestamp"])).total_seconds() / 3600
                        })
                        
            except Exception as e:
                print(f"[VERIFICATION ERROR] Failed to process {filepath}: {e}")
        
        print(f"[VERIFICATION] Found {len(ready_files)} files ready for 6h verification")
        return ready_files
    
    def get_price_change_6h(self, symbol: str, initial_price: float, initial_time: datetime) -> Dict:
        """
        Sprawdź zmianę ceny w ciągu 6h od initial_time.
        
        Args:
            symbol: Symbol tokena
            initial_price: Początkowa cena
            initial_time: Czas początkowy
            
        Returns:
            Dane o zmianie ceny i kategorii pump/dump/no_move
        """
        try:
            # Użyj danych z Bybit API (mock dla dev environment)
            # W produkcji zastąpi prawdziwe API Bybit
            from datetime import timezone
            
            # Calculate 6h window
            end_time = initial_time + timedelta(hours=6)
            now = datetime.now()
            
            if end_time > now:
                # Za wcześnie na pełną weryfikację
                return {"status": "incomplete", "reason": "6h window not complete"}
            
            # Mock price change - w produkcji zastąpi prawdziwe API
            import random
            price_change_pct = random.uniform(-15, 25)  # Mock price change
            final_price = initial_price * (1 + price_change_pct / 100)
            
            # Określ kategorię
            if price_change_pct >= 8:
                outcome = "pump"
            elif price_change_pct <= -8:
                outcome = "dump"
            else:
                outcome = "no_move"
            
            return {
                "status": "complete",
                "initial_price": initial_price,
                "final_price": final_price,
                "price_change_pct": price_change_pct,
                "outcome": outcome,
                "verification_time": now.isoformat(),
                "data_source": "mock_api"  # W produkcji: "bybit_api"
            }
            
        except Exception as e:
            print(f"[PRICE VERIFICATION ERROR] {symbol}: {e}")
            return {"status": "error", "error": str(e)}
    
    def verify_and_rename_file(self, file_info: Dict) -> bool:
        """
        Weryfikuj plik i zmień nazwę na podstawie wyniku pump/dump/no_move.
        
        Args:
            file_info: Informacje o pliku z get_files_ready_for_verification
            
        Returns:
            True jeśli weryfikacja i zmiana nazwy się powiodła
        """
        try:
            filepath = file_info["filepath"]
            data = file_info["data"]
            symbol = data["symbol"]
            initial_price = data.get("initial_price", 0.0)
            initial_time = datetime.fromisoformat(data["timestamp"])
            
            # Sprawdź zmianę ceny
            price_result = self.get_price_change_6h(symbol, initial_price, initial_time)
            
            if price_result["status"] != "complete":
                print(f"[VERIFICATION SKIP] {symbol}: {price_result.get('reason', 'verification failed')}")
                return False
            
            outcome = price_result["outcome"]
            
            # Przygotuj nową nazwę pliku
            old_filename = os.path.basename(filepath)
            # Zastąp _DETECTORS na _OUTCOME
            if "_" in old_filename:
                parts = old_filename.split("_")
                if len(parts) >= 4:
                    # Format: TOKEN_YYYYMMDD_HHMMSS_DETECTORS.json
                    new_filename = f"{parts[0]}_{parts[1]}_{parts[2]}_{outcome}.json"
                else:
                    # Fallback for irregular format
                    new_filename = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{outcome}.json"
            else:
                new_filename = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{outcome}.json"
            
            new_filepath = os.path.join(self.explore_dir, new_filename)
            
            # Aktualizuj dane w pliku
            data["status"] = "verified"
            data["verification_result"] = price_result
            data["outcome"] = outcome
            data["verified_at"] = datetime.now().isoformat()
            data["learning_ready"] = True
            
            # Zapisz zaktualizowane dane
            with open(new_filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Usuń stary plik
            os.remove(filepath)
            
            print(f"[VERIFICATION SUCCESS] {symbol}: {old_filename} → {new_filename}")
            print(f"[VERIFICATION RESULT] {symbol}: Price change {price_result['price_change_pct']:.1f}% → {outcome.upper()}")
            
            return True
            
        except Exception as e:
            print(f"[VERIFICATION ERROR] Failed to verify {file_info.get('filename', 'unknown')}: {e}")
            return False
    
    def cleanup_old_files(self) -> Dict[str, int]:
        """
        Usuń pliki starsze niż max_file_age_days.
        
        Returns:
            Statystyki czyszczenia
        """
        now = datetime.now()
        cutoff_time = now - timedelta(days=self.max_file_age_days)
        
        removed_count = 0
        kept_count = 0
        error_count = 0
        
        pattern = os.path.join(self.explore_dir, "*.json")
        for filepath in glob.glob(pattern):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                file_time = datetime.fromisoformat(data.get("timestamp", ""))
                
                if file_time < cutoff_time:
                    os.remove(filepath)
                    removed_count += 1
                    print(f"[CLEANUP] Removed old file: {os.path.basename(filepath)} (age: {(now - file_time).days} days)")
                else:
                    kept_count += 1
                    
            except Exception as e:
                error_count += 1
                print(f"[CLEANUP ERROR] Failed to process {filepath}: {e}")
        
        results = {
            "removed": removed_count,
            "kept": kept_count,
            "errors": error_count,
            "total_processed": removed_count + kept_count + error_count
        }
        
        if removed_count > 0:
            print(f"[CLEANUP SUMMARY] Removed {removed_count} old files, kept {kept_count}, {error_count} errors")
        
        return results
    
    def run_verification_cycle(self) -> Dict[str, int]:
        """
        Uruchom pełny cykl weryfikacji: sprawdź pliki → weryfikuj → zmień nazwy.
        
        Returns:
            Statystyki cyklu weryfikacji
        """
        print(f"[VERIFICATION CYCLE] Starting 6h pump verification cycle...")
        
        # Znajdź pliki gotowe do weryfikacji
        ready_files = self.get_files_ready_for_verification()
        
        verified_count = 0
        failed_count = 0
        
        for file_info in ready_files:
            if self.verify_and_rename_file(file_info):
                verified_count += 1
            else:
                failed_count += 1
        
        # Wyczyść stare pliki
        cleanup_stats = self.cleanup_old_files()
        
        results = {
            "files_ready": len(ready_files),
            "verified_success": verified_count,
            "verified_failed": failed_count,
            "cleanup_removed": cleanup_stats["removed"],
            "cleanup_kept": cleanup_stats["kept"]
        }
        
        print(f"[VERIFICATION CYCLE] Complete: {verified_count} verified, {failed_count} failed, {cleanup_stats['removed']} cleaned up")
        
        return results
    
    def migrate_legacy_files(self) -> Dict[str, int]:
        """
        Migruj stare pliki _explore.json do nowego formatu.
        
        Returns:
            Statystyki migracji
        """
        print(f"[MIGRATION] Starting legacy file migration...")
        
        migrated_count = 0
        failed_count = 0
        
        # Znajdź stare pliki _explore.json
        pattern = os.path.join(self.explore_dir, "*_explore.json")
        legacy_files = glob.glob(pattern)
        
        for filepath in legacy_files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                symbol = data.get("symbol", os.path.basename(filepath).replace("_explore.json", ""))
                
                # Dodaj brakujące pola
                if "timestamp" not in data:
                    # Użyj czasu modyfikacji pliku
                    file_mtime = os.path.getmtime(filepath)
                    data["timestamp"] = datetime.fromtimestamp(file_mtime).isoformat()
                
                # Wygeneruj detector label z dostępnych danych
                detector_label = self.generate_detector_label(data)
                
                # Wygeneruj nową nazwę
                timestamp = datetime.fromisoformat(data["timestamp"])
                new_filename = f"{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{detector_label}.json"
                new_filepath = os.path.join(self.explore_dir, new_filename)
                
                # Aktualizuj strukturę danych
                data["status"] = "legacy_migrated"
                data["detector_labels"] = detector_label
                data["verification_due"] = (timestamp + timedelta(hours=self.verification_hours)).isoformat()
                data["file_lifecycle"] = {
                    "created": data["timestamp"],
                    "migrated": datetime.now().isoformat(),
                    "verification_scheduled": (timestamp + timedelta(hours=self.verification_hours)).isoformat(),
                    "cleanup_scheduled": (timestamp + timedelta(days=self.max_file_age_days)).isoformat()
                }
                
                # Zapisz nowy plik
                with open(new_filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                
                # Usuń stary plik
                os.remove(filepath)
                
                migrated_count += 1
                print(f"[MIGRATION] {symbol}: {os.path.basename(filepath)} → {new_filename}")
                
            except Exception as e:
                failed_count += 1
                print(f"[MIGRATION ERROR] Failed to migrate {filepath}: {e}")
        
        results = {
            "migrated": migrated_count,
            "failed": failed_count,
            "total_legacy": len(legacy_files)
        }
        
        if migrated_count > 0:
            print(f"[MIGRATION SUMMARY] Migrated {migrated_count} legacy files, {failed_count} failed")
        
        return results

def main():
    """Test funkcji"""
    manager = ExploreFileManager()
    
    # Migruj legacy pliki
    migration_stats = manager.migrate_legacy_files()
    print(f"Migration results: {migration_stats}")
    
    # Uruchom cykl weryfikacji
    verification_stats = manager.run_verification_cycle()
    print(f"Verification results: {verification_stats}")

if __name__ == "__main__":
    main()