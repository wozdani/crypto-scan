"""
Basic Score Logger - Etap 1
Logowanie wyników tokenów dla samouczenia się progu base_score

Zapisuje dane o skuteczności tokenów, które przeszły wstępny skan (basic engine)
do pliku basic_score_results.jsonl dla późniejszej analizy skuteczności.
"""
import json
import os
from datetime import datetime
from typing import Dict, Optional
import asyncio
import aiohttp


class BasicScoreLogger:
    """
    Logger wyników tokenów dla samouczenia się progu selekcji
    """
    
    def __init__(self):
        self.log_file = "feedback_loop/basic_score_results.jsonl"
        self.ensure_log_file()
    
    def ensure_log_file(self):
        """Zapewnia istnienie pliku logów"""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        if not os.path.exists(self.log_file):
            # Utwórz pusty plik
            with open(self.log_file, 'w') as f:
                pass
    
    def log_basic_score_result(self, symbol: str, basic_score: float, 
                             final_score: float, decision: str, 
                             price_at_scan: float) -> str:
        """
        Loguje wynik tokena do pliku JSONL
        
        Args:
            symbol: Symbol tokena
            basic_score: Wynik z basic engine
            final_score: Finalny wynik TJDE
            decision: Decyzja systemu
            price_at_scan: Cena w momencie skanu
            
        Returns:
            ID wpisu dla późniejszej aktualizacji
        """
        try:
            entry_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            log_entry = {
                "id": entry_id,
                "symbol": symbol,
                "basic_score": basic_score,
                "final_score": final_score,
                "decision": decision,
                "price_at_scan": price_at_scan,
                "timestamp": datetime.now().isoformat(),
                "price_after_6h": None,  # Będzie uzupełnione później
                "result_pct_6h": None,
                "success": None,
                "evaluated": False
            }
            
            # Zapisz do pliku JSONL (każda linia = osobny JSON)
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            print(f"[BASIC SCORE LOG] {symbol}: logged with ID {entry_id}")
            return entry_id
            
        except Exception as e:
            print(f"[BASIC SCORE LOG ERROR] {symbol}: {e}")
            return ""
    
    async def fetch_price_after_6h(self, symbol: str, scan_timestamp: str) -> Optional[float]:
        """
        Pobiera cenę tokena 6 godzin po skanie z Bybit API
        
        Args:
            symbol: Symbol tokena
            scan_timestamp: Timestamp skanu w formacie ISO
            
        Returns:
            Cena po 6 godzinach lub None jeśli błąd
        """
        try:
            from datetime import datetime, timedelta
            
            # Oblicz timestamp 6h po skanie
            scan_time = datetime.fromisoformat(scan_timestamp.replace('Z', '+00:00'))
            target_time = scan_time + timedelta(hours=6)
            target_timestamp_ms = int(target_time.timestamp() * 1000)
            
            # Pobierz dane kline z Bybit API
            async with aiohttp.ClientSession() as session:
                url = "https://api.bybit.com/v5/market/kline"
                params = {
                    "category": "spot",
                    "symbol": symbol,
                    "interval": "60",  # 1h candles
                    "start": target_timestamp_ms - 3600000,  # 1h przed target
                    "end": target_timestamp_ms + 3600000,    # 1h po target
                    "limit": 5
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        klines = data.get("result", {}).get("list", [])
                        
                        if klines:
                            # Znajdź najbliższą świecę do target_time
                            closest_kline = min(klines, 
                                              key=lambda k: abs(int(k[0]) - target_timestamp_ms))
                            price_6h = float(closest_kline[4])  # Close price
                            
                            print(f"[PRICE FETCH] {symbol}: price after 6h = {price_6h}")
                            return price_6h
                        
            return None
            
        except Exception as e:
            print(f"[PRICE FETCH ERROR] {symbol}: {e}")
            return None
    
    async def evaluate_pending_results(self) -> int:
        """
        Ocenia nieewaluowane wpisy - pobiera ceny po 6h i oblicza success
        
        Returns:
            Liczba ocenionych wpisów
        """
        try:
            if not os.path.exists(self.log_file):
                return 0
            
            evaluated_count = 0
            updated_entries = []
            
            # Wczytaj wszystkie wpisy
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                if line.strip():
                    entry = json.loads(line.strip())
                    
                    # Sprawdź czy wymaga ewaluacji
                    if not entry.get('evaluated', False) and entry.get('timestamp'):
                        scan_time = datetime.fromisoformat(entry['timestamp'])
                        time_since_scan = datetime.now() - scan_time
                        
                        # Ewaluuj jeśli minęło ≥6h
                        if time_since_scan.total_seconds() >= 6 * 3600:
                            print(f"[EVALUATION] Evaluating {entry['symbol']} after {time_since_scan}")
                            
                            # Pobierz cenę po 6h
                            price_6h = await self.fetch_price_after_6h(
                                entry['symbol'], 
                                entry['timestamp']
                            )
                            
                            if price_6h is not None:
                                price_at_scan = entry['price_at_scan']
                                result_pct_6h = ((price_6h - price_at_scan) / price_at_scan) * 100
                                
                                # Kryterium success: wzrost ≥2%
                                success = result_pct_6h >= 2.0
                                
                                # Aktualizuj wpis
                                entry['price_after_6h'] = price_6h
                                entry['result_pct_6h'] = round(result_pct_6h, 2)
                                entry['success'] = success
                                entry['evaluated'] = True
                                
                                evaluated_count += 1
                                
                                print(f"[EVALUATION RESULT] {entry['symbol']}: "
                                      f"{result_pct_6h:+.2f}% → {'SUCCESS' if success else 'FAIL'}")
                    
                    updated_entries.append(entry)
            
            # Zapisz zaktualizowane wpisy
            if evaluated_count > 0:
                with open(self.log_file, 'w') as f:
                    for entry in updated_entries:
                        f.write(json.dumps(entry) + '\n')
                
                print(f"[EVALUATION COMPLETE] Evaluated {evaluated_count} entries")
            
            return evaluated_count
            
        except Exception as e:
            print(f"[EVALUATION ERROR] {e}")
            return 0
    
    def get_pending_evaluation_count(self) -> int:
        """
        Zwraca liczbę wpisów oczekujących na ewaluację
        
        Returns:
            Liczba nieewaluowanych wpisów starszych niż 6h
        """
        try:
            if not os.path.exists(self.log_file):
                return 0
            
            pending_count = 0
            current_time = datetime.now()
            
            with open(self.log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line.strip())
                        
                        if not entry.get('evaluated', False) and entry.get('timestamp'):
                            scan_time = datetime.fromisoformat(entry['timestamp'])
                            time_since_scan = current_time - scan_time
                            
                            if time_since_scan.total_seconds() >= 6 * 3600:
                                pending_count += 1
            
            return pending_count
            
        except Exception as e:
            print(f"[PENDING COUNT ERROR] {e}")
            return 0
    
    def get_total_entries_count(self) -> int:
        """Zwraca całkowitą liczbę wpisów w logu"""
        try:
            if not os.path.exists(self.log_file):
                return 0
            
            with open(self.log_file, 'r') as f:
                return len([line for line in f if line.strip()])
                
        except Exception:
            return 0


def log_basic_score_result(symbol: str, basic_score: float, final_score: float, 
                          decision: str, price_at_scan: float) -> str:
    """
    Convenience function dla łatwego logowania wyników
    
    Args:
        symbol: Symbol tokena
        basic_score: Wynik z basic engine  
        final_score: Finalny wynik TJDE
        decision: Decyzja systemu
        price_at_scan: Cena w momencie skanu
        
    Returns:
        ID wpisu
    """
    logger = BasicScoreLogger()
    return logger.log_basic_score_result(symbol, basic_score, final_score, decision, price_at_scan)


async def evaluate_pending_basic_score_results() -> int:
    """
    Convenience function dla ewaluacji nieopracowanych wyników
    
    Returns:
        Liczba ocenionych wpisów
    """
    logger = BasicScoreLogger()
    return await logger.evaluate_pending_results()


def get_basic_score_statistics() -> Dict:
    """
    Pobiera statystyki logowania basic score
    
    Returns:
        Dictionary ze statystykami
    """
    logger = BasicScoreLogger()
    total_entries = logger.get_total_entries_count()
    pending_evaluation = logger.get_pending_evaluation_count()
    
    return {
        "total_entries": total_entries,
        "pending_evaluation": pending_evaluation,
        "evaluated_entries": total_entries - pending_evaluation
    }