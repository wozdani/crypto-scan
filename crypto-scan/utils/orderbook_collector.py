"""
Orderbook Collector - pobieranie danych orderbooku z API Bybit
Integracja z Liquidity Behavior Detector dla zbierania snapshotów co 5 minut
"""

import requests
import json
import time
import hmac
import hashlib
from urllib.parse import urlencode
from datetime import datetime, timezone
from typing import Dict, Optional
from utils.liquidity_behavior import liquidity_analyzer

class BybitOrderbookCollector:
    """Kolektor danych orderbooku z API Bybit"""
    
    def __init__(self):
        # Używamy tych samych kluczy API co w głównym systemie
        import os
        from dotenv import load_dotenv
        
        # Wczytaj zmienne środowiskowe
        load_dotenv()
        
        self.api_key = os.getenv('BYBIT_API_KEY', '')
        self.api_secret = os.getenv('BYBIT_SECRET_KEY', '')
        self.base_url = "https://api.bybit.com"
        
    def _generate_signature(self, params: str, timestamp: str) -> str:
        """Generuje podpis HMAC dla autoryzacji API"""
        if not self.api_secret:
            return ""
            
        param_str = str(timestamp) + self.api_key + "5000" + params
        signature = hmac.new(
            bytes(self.api_secret, "utf-8"),
            param_str.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def get_orderbook(self, symbol: str, limit: int = 25) -> Optional[Dict]:
        """
        Pobiera dane orderbooku dla symbolu z API Bybit
        
        Args:
            symbol: Symbol trading (np. 'BTCUSDT')
            limit: Liczba poziomów orderbooku (max 200)
            
        Returns:
            Dict z danymi orderbooku lub None w przypadku błędu
        """
        try:
            endpoint = "/v5/market/orderbook"
            timestamp = str(int(time.time() * 1000))
            
            params = {
                "category": "linear",
                "symbol": symbol,
                "limit": min(limit, 25)  # Ograniczamy do 25 dla wydajności
            }
            
            param_str = urlencode(params)
            
            headers = {
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-RECV-WINDOW": "5000",
                "Content-Type": "application/json"
            }
            
            # Dodaj podpis jeśli mamy klucze API
            if self.api_key and self.api_secret:
                signature = self._generate_signature(param_str, timestamp)
                headers["X-BAPI-SIGN"] = signature
            
            url = f"{self.base_url}{endpoint}?{param_str}"
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("retCode") == 0:
                    result = data.get("result", {})
                    
                    # Formatuj dane orderbooku dla Liquidity Behavior Detector
                    orderbook_data = {
                        "symbol": symbol,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "bids": result.get("b", []),  # Bybit używa "b" dla bidów
                        "asks": result.get("a", [])   # Bybit używa "a" dla asków
                    }
                    
                    return orderbook_data
                else:
                    print(f"⚠️ API error dla {symbol}: {data.get('retMsg', 'Unknown error')}")
                    return None
            else:
                print(f"⚠️ HTTP error {response.status_code} dla {symbol}")
                return None
                
        except Exception as e:
            print(f"❌ Błąd podczas pobierania orderbooku dla {symbol}: {e}")
            return None
    
    def collect_and_store_snapshot(self, symbol: str) -> bool:
        """
        Pobiera orderbook i zapisuje snapshot dla Liquidity Behavior Detector
        
        Args:
            symbol: Symbol trading
            
        Returns:
            bool: True jeśli snapshot został zapisany pomyślnie
        """
        try:
            # Pobierz dane orderbooku
            orderbook_data = self.get_orderbook(symbol)
            
            if orderbook_data is None:
                return False
            
            # Zapisz snapshot używając Liquidity Behavior Analyzer
            success = liquidity_analyzer.store_orderbook_snapshot(symbol, orderbook_data)
            
            if success:
                print(f"💧 Orderbook snapshot saved for {symbol}")
            else:
                print(f"❌ Failed to save snapshot for {symbol}")
                
            return success
            
        except Exception as e:
            print(f"❌ Błąd podczas zbierania snapshotu dla {symbol}: {e}")
            return False
    
    def batch_collect_snapshots(self, symbols: list) -> Dict[str, bool]:
        """
        Zbiera snapshoty orderbooku dla wielu symboli
        
        Args:
            symbols: Lista symboli trading
            
        Returns:
            Dict: Mapowanie symbol -> success status
        """
        results = {}
        
        for symbol in symbols:
            try:
                success = self.collect_and_store_snapshot(symbol)
                results[symbol] = success
                
                # Rate limiting - pauza między requestami
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Błąd batch collection dla {symbol}: {e}")
                results[symbol] = False
        
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        print(f"💧 Batch orderbook collection: {successful}/{total} successful")
        
        return results

# Global instance dla łatwego importu
orderbook_collector = BybitOrderbookCollector()