"""
🔍 Stealth Scanner - Enhanced Token Scanning with Priority Learning
🎯 Cel: Integration Priority Learning Memory z main scanning pipeline

📋 Funkcjonalności:
1. Smart token sorting z priority bias calculation
2. Stealth-ready token identification i priority routing
3. Integration z existing async scanner infrastructure
4. Learning feedback loop dla successful predictions
5. Dynamic scanning queue management
"""

import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

from .priority_learning import (
    get_priority_learning_memory, 
    get_token_learning_bias,
    update_stealth_learning,
    get_priority_tokens
)

class StealthScannerManager:
    """
    🔍 Manager skanowania z priority learning integration
    """
    
    def __init__(self):
        """Inicjalizacja Stealth Scanner Manager"""
        self.priority_memory = get_priority_learning_memory()
        self.stealth_ready_threshold = 3.0  # Min stealth score dla stealth_ready tag
        self.learning_enabled = True
        
        print("[STEALTH SCANNER] Initialized with Priority Learning Memory")
    
    def sort_tokens_by_stealth_priority(self, tokens: List[Dict]) -> List[Dict]:
        """
        🎯 Sortuj tokeny według priorytetu z learning bias
        
        Args:
            tokens: Lista tokenów do skanowania
            
        Returns:
            List[Dict]: Tokeny posortowane według priorytetu
        """
        try:
            def calculate_priority_score(token: Dict) -> float:
                """Oblicz priority score dla tokena"""
                symbol = token.get('symbol', '')
                
                # Base priority factors
                stealth_score = token.get('stealth_score', 0.0)
                tjde_score = token.get('tjde_score', 0.0)
                tags = token.get('tags', [])
                
                # Tag-based priorities
                tag_priority = 0.0
                if 'stealth_ready' in tags:
                    tag_priority += 2.0
                if 'smart_money' in tags:
                    tag_priority += 1.5
                if 'trusted' in tags:
                    tag_priority += 1.0
                if 'priority' in tags:
                    tag_priority += 0.5
                
                # Learning bias z historical performance
                learning_bias = get_token_learning_bias(symbol)
                
                # Composite priority calculation
                base_priority = (stealth_score * 0.4 + tjde_score * 0.3 + tag_priority * 0.3)
                final_priority = base_priority + (learning_bias * 2.0)  # Learning bias może dodać do 2.0
                
                return final_priority
            
            # Oblicz priority scores i sortuj
            token_priorities = []
            for token in tokens:
                priority_score = calculate_priority_score(token)
                token_priorities.append((token, priority_score))
            
            # Sortuj po priority score (malejąco)
            token_priorities.sort(key=lambda x: x[1], reverse=True)
            
            # Extract sorted tokens
            sorted_tokens = [token for token, priority in token_priorities]
            
            # Debug info dla top tokens
            if token_priorities:
                top_5 = token_priorities[:5]
                print(f"[STEALTH SCANNER] Top 5 priority tokens:")
                for i, (token, priority) in enumerate(top_5, 1):
                    symbol = token.get('symbol', 'UNKNOWN')
                    learning_bias = get_token_learning_bias(symbol)
                    print(f"   {i}. {symbol}: priority={priority:.3f} (learning_bias={learning_bias:.3f})")
            
            return sorted_tokens
            
        except Exception as e:
            print(f"[STEALTH SCANNER ERROR] Token sorting failed: {e}")
            return tokens  # Return original order on error
    
    def identify_stealth_ready_tokens(self, scan_results: List[Dict]) -> List[Dict]:
        """
        🎯 Identyfikuj tokeny ready dla stealth action
        
        Args:
            scan_results: Wyniki skanowania tokenów
            
        Returns:
            List[Dict]: Tokeny z tagiem stealth_ready
        """
        stealth_ready_tokens = []
        
        try:
            for result in scan_results:
                symbol = result.get('symbol', '')
                stealth_score = result.get('stealth_score', 0.0)
                tags = result.get('tags', [])
                
                # Criteria for stealth_ready tag
                is_stealth_ready = (
                    stealth_score >= self.stealth_ready_threshold and
                    'stealth_ready' in tags
                )
                
                if is_stealth_ready:
                    stealth_ready_tokens.append({
                        **result,
                        'stealth_ready_timestamp': datetime.utcnow().isoformat(),
                        'stealth_ready_score': stealth_score
                    })
                    
                    print(f"[STEALTH READY] {symbol}: score={stealth_score:.3f}, tags={tags}")
            
            print(f"[STEALTH SCANNER] Identified {len(stealth_ready_tokens)} stealth-ready tokens")
            return stealth_ready_tokens
            
        except Exception as e:
            print(f"[STEALTH SCANNER ERROR] Stealth ready identification failed: {e}")
            return []
    
    def get_priority_scanning_queue(self, all_tokens: List[str], limit: int = 100) -> List[str]:
        """
        📋 Pobierz kolejkę skanowania z priorytetem learning
        
        Args:
            all_tokens: Wszystkie dostępne tokeny
            limit: Maksymalna liczba tokenów w kolejce
            
        Returns:
            List[str]: Tokeny posortowane według priorytetu
        """
        try:
            # Pobierz top priority tokens z learning memory
            priority_tokens_with_bias = get_priority_tokens(20)
            priority_symbols = [symbol for symbol, bias in priority_tokens_with_bias]
            
            # Utwórz priority queue
            priority_queue = []
            
            # Najpierw dodaj high-priority tokens z learning memory
            for symbol in priority_symbols:
                if symbol in all_tokens:
                    priority_queue.append(symbol)
            
            # Następnie dodaj pozostałe tokeny
            remaining_tokens = [token for token in all_tokens if token not in priority_queue]
            priority_queue.extend(remaining_tokens)
            
            # Limit queue size
            final_queue = priority_queue[:limit]
            
            if priority_symbols:
                print(f"[STEALTH SCANNER] Priority queue: {len(priority_symbols)} learned priorities, "
                      f"{len(remaining_tokens)} standard tokens")
            
            return final_queue
            
        except Exception as e:
            print(f"[STEALTH SCANNER ERROR] Priority queue creation failed: {e}")
            return all_tokens[:limit]  # Fallback to original order
    
    async def process_stealth_feedback(self, stealth_ready_tokens: List[Dict], 
                                     hours_delay: int = 2) -> int:
        """
        🔄 Process feedback dla stealth ready tokens po określonym czasie
        
        Args:
            stealth_ready_tokens: Tokeny z tagiem stealth_ready
            hours_delay: Opóźnienie w godzinach przed evaluacją
            
        Returns:
            int: Liczba przetworzonych tokenów
        """
        if not self.learning_enabled:
            return 0
            
        processed_count = 0
        
        try:
            # Tu będzie integration z real price evaluation
            # Na razie używamy placeholder logic
            
            for token_data in stealth_ready_tokens:
                symbol = token_data.get('symbol', '')
                
                try:
                    # Placeholder for price evaluation (replace with real API call)
                    # This should fetch actual price changes after hours_delay
                    price_change_2h = 0.0  # Will be implemented with real price fetching
                    price_change_6h = 0.0  # Will be implemented with real price fetching
                    
                    # Determine success based on price performance
                    result_success = price_change_2h >= 2.0  # ≥2% wzrost = sukces
                    
                    # Update learning memory
                    update_stealth_learning(
                        symbol=symbol,
                        score=token_data.get('tjde_score', 0.0),
                        stealth_score=token_data.get('stealth_score', 0.0),
                        result_success=result_success,
                        price_change_2h=price_change_2h,
                        price_change_6h=price_change_6h,
                        tags=token_data.get('tags', []),
                        confidence=token_data.get('confidence', 0.0)
                    )
                    
                    processed_count += 1
                    
                except Exception as e:
                    print(f"[STEALTH FEEDBACK ERROR] Processing failed for {symbol}: {e}")
            
            print(f"[STEALTH FEEDBACK] Processed {processed_count} stealth tokens")
            return processed_count
            
        except Exception as e:
            print(f"[STEALTH FEEDBACK ERROR] Batch processing failed: {e}")
            return 0
    
    def get_stealth_scanner_statistics(self) -> Dict:
        """
        📊 Pobierz statystyki stealth scanner
        
        Returns:
            Dict: Comprehensive scanner statistics
        """
        try:
            learning_stats = self.priority_memory.get_learning_statistics()
            priority_tokens = get_priority_tokens(10)
            
            return {
                'stealth_scanner_status': 'active',
                'learning_enabled': self.learning_enabled,
                'stealth_ready_threshold': self.stealth_ready_threshold,
                'priority_learning_stats': learning_stats,
                'top_priority_tokens': priority_tokens,
                'memory_cache_file': self.priority_memory.cache_file
            }
            
        except Exception as e:
            print(f"[STEALTH SCANNER ERROR] Statistics failed: {e}")
            return {'error': str(e)}

# Global instance
_stealth_scanner_manager = None

def get_stealth_scanner() -> StealthScannerManager:
    """Pobierz globalną instancję Stealth Scanner Manager"""
    global _stealth_scanner_manager
    if _stealth_scanner_manager is None:
        _stealth_scanner_manager = StealthScannerManager()
    return _stealth_scanner_manager

def sort_tokens_by_priority(tokens: List[Dict]) -> List[Dict]:
    """
    🎯 Convenience function: Sort tokens by stealth priority
    """
    scanner = get_stealth_scanner()
    return scanner.sort_tokens_by_stealth_priority(tokens)

def get_priority_scan_queue(all_tokens: List[str], limit: int = 100) -> List[str]:
    """
    📋 Convenience function: Get priority scanning queue
    """
    scanner = get_stealth_scanner()
    return scanner.get_priority_scanning_queue(all_tokens, limit)

def identify_stealth_ready(scan_results: List[Dict]) -> List[Dict]:
    """
    🎯 Convenience function: Identify stealth ready tokens
    """
    scanner = get_stealth_scanner()
    return scanner.identify_stealth_ready_tokens(scan_results)

async def process_stealth_learning_feedback(stealth_tokens: List[Dict], hours: int = 2) -> int:
    """
    🔄 Convenience function: Process stealth learning feedback
    """
    scanner = get_stealth_scanner()
    return await scanner.process_stealth_feedback(stealth_tokens, hours)

def get_scanner_stats() -> Dict:
    """
    📊 Convenience function: Get scanner statistics
    """
    scanner = get_stealth_scanner()
    return scanner.get_stealth_scanner_statistics()