"""
🧠 Priority Learning Memory System - Stealth Engine v2 Etap 11
🎯 Cel: System uczenia się priorytetów na podstawie historycznych sukcesów tokenów

📋 Funkcjonalności:
1. Pamięć tokenów z tagiem stealth_ready → success tracking
2. Adaptacyjny priorytet skanowania na podstawie uczenia
3. Feedback loop z 2h/6h price evaluation
4. Priority bias calculation dla kolejki skanowania
5. Integration z existing stealth_scanner i alert systems
"""

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class LearningEntry:
    """Pojedynczy wpis w pamięci uczenia"""
    timestamp: str
    score: float
    stealth_score: float
    result_success: bool
    price_change_2h: float
    price_change_6h: float
    tags: List[str]
    confidence: float

class PriorityLearningMemory:
    """
    🧠 System uczenia się priorytetów tokenów
    """
    
    def __init__(self, cache_file: str = "cache/priority_learning_memory.json"):
        """
        Inicjalizacja Priority Learning Memory
        
        Args:
            cache_file: Plik cache dla trwałości pamięci
        """
        self.cache_file = cache_file
        self.memory: Dict[str, List[LearningEntry]] = {}
        self.max_entries_per_token = 15  # Trzymaj max 15 rekordów na token
        self.min_entries_for_bias = 3   # Min wpisy do obliczenia bias
        self._load_memory_cache()
    
    def _load_memory_cache(self):
        """Wczytaj pamięć z cache"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    
                # Konwertuj z dict do LearningEntry objects
                for symbol, entries in data.items():
                    self.memory[symbol] = []
                    for entry in entries:
                        # 🔧 STAGE 10 FIX: Handle both dict and string entries
                        if isinstance(entry, dict):
                            # Normal dict entry
                            self.memory[symbol].append(LearningEntry(
                                timestamp=entry.get('timestamp', ''),
                                score=entry.get('score', 0.0),
                                stealth_score=entry.get('stealth_score', 0.0),
                                result_success=entry.get('result_success', False),
                                price_change_2h=entry.get('price_change_2h', 0.0),
                                price_change_6h=entry.get('price_change_6h', 0.0),
                                tags=entry.get('tags', []),
                                confidence=entry.get('confidence', 0.0)
                            ))
                        elif isinstance(entry, str):
                            # String entry - create default LearningEntry
                            print(f"[PRIORITY LEARNING] Converting string entry to dict for {symbol}: {entry[:50]}...")
                            self.memory[symbol].append(LearningEntry(
                                timestamp=datetime.now(timezone.utc).isoformat(),
                                score=0.0,
                                stealth_score=0.0,
                                result_success=False,
                                price_change_2h=0.0,
                                price_change_6h=0.0,
                                tags=["converted_from_string"],
                                confidence=0.0
                            ))
                        else:
                            # Unknown type - skip with warning
                            print(f"[PRIORITY LEARNING WARNING] Skipping unknown entry type {type(entry)} for {symbol}")
                            continue
                        
                print(f"[PRIORITY LEARNING] Loaded memory: {len(self.memory)} tokens")
            else:
                print(f"[PRIORITY LEARNING] Created new memory cache: {self.cache_file}")
                
        except Exception as e:
            print(f"[PRIORITY LEARNING ERROR] Cache load failed: {e}")
            self.memory = {}
    
    def _save_memory_cache(self):
        """Zapisz pamięć do cache"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            # Konwertuj LearningEntry objects do dict
            data = {}
            for symbol, entries in self.memory.items():
                data[symbol] = []
                for entry in entries:
                    data[symbol].append({
                        'timestamp': entry.timestamp,
                        'score': entry.score,
                        'stealth_score': entry.stealth_score,
                        'result_success': entry.result_success,
                        'price_change_2h': entry.price_change_2h,
                        'price_change_6h': entry.price_change_6h,
                        'tags': entry.tags,
                        'confidence': entry.confidence
                    })
            
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"[PRIORITY LEARNING ERROR] Cache save failed: {e}")
    
    def update_learning_memory(self, symbol: str, score: float, stealth_score: float = 0.0,
                              result_success: bool = False, price_change_2h: float = 0.0,
                              price_change_6h: float = 0.0, tags: List[str] = None,
                              confidence: float = 0.0):
        """
        ➕ Dodaj wpis do pamięci uczenia
        
        Args:
            symbol: Symbol tokena
            score: TJDE score
            stealth_score: Stealth Engine score
            result_success: Czy token osiągnął sukces (≥2% wzrost w 2h)
            price_change_2h: Zmiana ceny po 2h (%)
            price_change_6h: Zmiana ceny po 6h (%)
            tags: Lista tagów tokena
            confidence: Confidence level alertu
        """
        try:
            now = datetime.now(timezone.utc)
            
            if symbol not in self.memory:
                self.memory[symbol] = []
            
            # Dodaj nowy wpis
            entry = LearningEntry(
                timestamp=now.isoformat(),
                score=score,
                stealth_score=stealth_score,
                result_success=result_success,
                price_change_2h=price_change_2h,
                price_change_6h=price_change_6h,
                tags=tags or [],
                confidence=confidence
            )
            
            self.memory[symbol].append(entry)
            
            # Trzymaj tylko ostatnie N rekordów
            self.memory[symbol] = self.memory[symbol][-self.max_entries_per_token:]
            
            # Zapisz do cache
            self._save_memory_cache()
            
            print(f"[PRIORITY LEARNING] Updated memory for {symbol}: success={result_success}, "
                  f"2h_change={price_change_2h:.2f}%, entries={len(self.memory[symbol])}")
            
        except Exception as e:
            print(f"[PRIORITY LEARNING ERROR] Update failed for {symbol}: {e}")
    
    def get_token_priority_bias(self, symbol: str) -> float:
        """
        📊 Oblicz priority bias dla tokena na podstawie historii
        
        Args:
            symbol: Symbol tokena
            
        Returns:
            float: Priority bias (0.0-1.0)
        """
        try:
            memory = self.memory.get(symbol, [])
            
            if len(memory) < self.min_entries_for_bias:
                return 0.0  # Brak wystarczających danych
            
            # Oblicz success rate
            total_entries = len(memory)
            successful_entries = sum(1 for entry in memory if entry.result_success)
            success_rate = successful_entries / total_entries
            
            # Oblicz średnią zmianę ceny
            avg_price_change_2h = sum(entry.price_change_2h for entry in memory) / total_entries
            avg_price_change_6h = sum(entry.price_change_6h for entry in memory) / total_entries
            
            # Oblicz bias na podstawie success rate i performance
            bias_base = min(success_rate * 2.0, 1.0)  # Success rate influence
            
            # Bonus za wysokie average returns
            performance_bonus = 0.0
            if avg_price_change_2h > 3.0:  # >3% average 2h return
                performance_bonus += 0.2
            if avg_price_change_6h > 5.0:  # >5% average 6h return
                performance_bonus += 0.3
            
            # Final bias calculation
            final_bias = min(bias_base + performance_bonus, 1.0)
            
            return final_bias
            
        except Exception as e:
            print(f"[PRIORITY LEARNING ERROR] Bias calculation failed for {symbol}: {e}")
            return 0.0
    
    def get_stealth_priority_tokens(self, limit: int = 10) -> List[Tuple[str, float]]:
        """
        🎯 Pobierz tokeny z najwyższym priority bias
        
        Args:
            limit: Maksymalna liczba tokenów
            
        Returns:
            List[Tuple[str, float]]: Lista (symbol, bias) posortowana po bias
        """
        try:
            token_biases = []
            
            for symbol in self.memory.keys():
                bias = self.get_token_priority_bias(symbol)
                if bias > 0.0:
                    token_biases.append((symbol, bias))
            
            # Sortuj po bias (malejąco)
            token_biases.sort(key=lambda x: x[1], reverse=True)
            
            return token_biases[:limit]
            
        except Exception as e:
            print(f"[PRIORITY LEARNING ERROR] Priority tokens failed: {e}")
            return []
    
    def evaluate_stealth_ready_tokens(self, stealth_ready_tokens: List[Dict]) -> List[Dict]:
        """
        🔄 Ewaluuj tokeny z tagiem stealth_ready po czasie
        (Placeholder - integration z existing price evaluation system)
        
        Args:
            stealth_ready_tokens: Lista tokenów z tagiem stealth_ready
            
        Returns:
            List[Dict]: Tokens z price evaluation results
        """
        evaluated_tokens = []
        
        for token_data in stealth_ready_tokens:
            symbol = token_data.get('symbol', '')
            
            # Tu będzie integration z existing price fetching
            # Na razie używamy placeholder logic
            try:
                # Simulated price evaluation (replace with real API call)
                price_change_2h = 0.0  # Will be replaced with real price fetching
                price_change_6h = 0.0  # Will be replaced with real price fetching
                
                result_success = price_change_2h >= 2.0  # ≥2% = sukces
                
                # Update learning memory
                self.update_learning_memory(
                    symbol=symbol,
                    score=token_data.get('tjde_score', 0.0),
                    stealth_score=token_data.get('stealth_score', 0.0),
                    result_success=result_success,
                    price_change_2h=price_change_2h,
                    price_change_6h=price_change_6h,
                    tags=token_data.get('tags', []),
                    confidence=token_data.get('confidence', 0.0)
                )
                
                evaluated_tokens.append({
                    **token_data,
                    'price_change_2h': price_change_2h,
                    'price_change_6h': price_change_6h,
                    'result_success': result_success
                })
                
            except Exception as e:
                print(f"[PRIORITY LEARNING ERROR] Evaluation failed for {symbol}: {e}")
                
        return evaluated_tokens
    
    def get_learning_statistics(self) -> Dict:
        """
        📊 Pobierz statystyki systemu uczenia
        
        Returns:
            Dict: Comprehensive learning statistics
        """
        try:
            total_tokens = len(self.memory)
            total_entries = sum(len(entries) for entries in self.memory.values())
            
            # Success rate statistics
            all_entries = []
            for entries in self.memory.values():
                all_entries.extend(entries)
            
            total_evaluated = len(all_entries)
            successful_entries = sum(1 for entry in all_entries if entry.result_success)
            overall_success_rate = successful_entries / total_evaluated if total_evaluated > 0 else 0.0
            
            # Average performance
            avg_2h_change = sum(entry.price_change_2h for entry in all_entries) / total_evaluated if total_evaluated > 0 else 0.0
            avg_6h_change = sum(entry.price_change_6h for entry in all_entries) / total_evaluated if total_evaluated > 0 else 0.0
            
            # Top performing tokens
            top_tokens = self.get_stealth_priority_tokens(5)
            
            return {
                'total_tokens_tracked': total_tokens,
                'total_entries': total_entries,
                'total_evaluated': total_evaluated,
                'overall_success_rate': round(overall_success_rate, 3),
                'avg_price_change_2h': round(avg_2h_change, 2),
                'avg_price_change_6h': round(avg_6h_change, 2),
                'top_priority_tokens': top_tokens,
                'entries_per_token': round(total_entries / total_tokens, 1) if total_tokens > 0 else 0.0
            }
            
        except Exception as e:
            print(f"[PRIORITY LEARNING ERROR] Statistics failed: {e}")
            return {}
    
    def cleanup_old_entries(self, max_age_days: int = 30):
        """
        🧹 Oczyść stare wpisy z pamięci
        
        Args:
            max_age_days: Maksymalny wiek wpisów w dniach
        """
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)
            removed_count = 0
            
            for symbol in list(self.memory.keys()):
                original_count = len(self.memory[symbol])
                
                # Filtruj stare wpisy
                self.memory[symbol] = [
                    entry for entry in self.memory[symbol]
                    if datetime.fromisoformat(entry.timestamp) > cutoff_date
                ]
                
                removed_count += original_count - len(self.memory[symbol])
                
                # Usuń tokeny bez wpisów
                if not self.memory[symbol]:
                    del self.memory[symbol]
            
            # Zapisz po cleanup
            self._save_memory_cache()
            
            print(f"[PRIORITY LEARNING] Cleanup removed {removed_count} old entries")
            
        except Exception as e:
            print(f"[PRIORITY LEARNING ERROR] Cleanup failed: {e}")

# Global instance
_priority_learning_memory = None

def get_priority_learning_memory() -> PriorityLearningMemory:
    """Pobierz globalną instancję Priority Learning Memory"""
    global _priority_learning_memory
    if _priority_learning_memory is None:
        _priority_learning_memory = PriorityLearningMemory()
    return _priority_learning_memory

def update_stealth_learning(symbol: str, score: float, stealth_score: float = 0.0,
                           result_success: bool = False, price_change_2h: float = 0.0,
                           price_change_6h: float = 0.0, tags: List[str] = None,
                           confidence: float = 0.0):
    """
    🎯 Convenience function: Update learning memory
    """
    memory = get_priority_learning_memory()
    memory.update_learning_memory(
        symbol=symbol,
        score=score,
        stealth_score=stealth_score,
        result_success=result_success,
        price_change_2h=price_change_2h,
        price_change_6h=price_change_6h,
        tags=tags,
        confidence=confidence
    )

def get_token_learning_bias(symbol: str) -> float:
    """
    📊 Convenience function: Get priority bias for token
    """
    memory = get_priority_learning_memory()
    return memory.get_token_priority_bias(symbol)

def get_priority_tokens(limit: int = 10) -> List[Tuple[str, float]]:
    """
    🎯 Convenience function: Get top priority tokens
    """
    memory = get_priority_learning_memory()
    return memory.get_stealth_priority_tokens(limit)

def get_learning_stats() -> Dict:
    """
    📊 Convenience function: Get learning statistics
    """
    try:
        memory = get_priority_learning_memory()
        stats = memory.get_learning_statistics()
        
        # Ensure success_rate is included for backward compatibility
        if "success_rate" not in stats and "overall_success_rate" in stats:
            stats["success_rate"] = stats["overall_success_rate"]
        
        return stats
    except Exception as e:
        print(f"[PRIORITY LEARNING ERROR] Failed to get learning stats: {e}")
        return {
            "total_entries": 0,
            "success_rate": 0.0,
            "overall_success_rate": 0.0,
            "total_tokens_tracked": 0
        }

def evaluate_stealth_tokens(stealth_ready_tokens: List[Dict]) -> List[Dict]:
    """
    🔄 Convenience function: Evaluate stealth ready tokens
    """
    memory = get_priority_learning_memory()
    return memory.evaluate_stealth_ready_tokens(stealth_ready_tokens)