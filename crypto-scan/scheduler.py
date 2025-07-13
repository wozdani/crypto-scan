#!/usr/bin/env python3
"""
Cyclic GNN Stealth Engine Scheduler
Uruchamia analizę GNN dla określonych adresów co X minut
"""

import time
import json
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
from stealth_engine_advanced import StealthEngineAdvanced

# Konfiguracja loggera
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GNNScheduler:
    """
    Scheduler dla cyklicznego uruchamiania GNN Stealth Engine
    """
    
    def __init__(self, interval_minutes: int = 5):
        """
        Initialize GNN Scheduler
        
        Args:
            interval_minutes: Interwał między skanami w minutach
        """
        self.interval_seconds = interval_minutes * 60
        self.stealth_engine = StealthEngineAdvanced()
        self.results_file = "cache/scheduler_results.json"
        self.config_file = "cache/scheduler_config.json"
        
        # Lista domyślnych adresów do śledzenia
        self.tracked_addresses = [
            "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",  # Bitfinex whale
            "0xDC76CD25977E0a5Ae17155770273aD58648900D3",  # Example wallet
            "0x8894E0a0c962CB723c1976a4421c95949bE2D4E6",  # Binance hot wallet
            "0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE",  # Binance cold storage
            "0xBE0eB53F46cd790Cd13851d5EFf43D12404d33E8",  # Large holder
        ]
        
        self.load_config()
        logger.info(f"[SCHEDULER] Initialized with {len(self.tracked_addresses)} addresses, interval: {interval_minutes}min")
    
    def load_config(self):
        """Ładuje konfigurację z pliku"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.tracked_addresses = config.get('tracked_addresses', self.tracked_addresses)
                    logger.info(f"[CONFIG] Loaded {len(self.tracked_addresses)} addresses from config")
        except Exception as e:
            logger.warning(f"[CONFIG] Failed to load config: {e}, using defaults")
    
    def save_config(self):
        """Zapisuje konfigurację do pliku"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            config = {
                'tracked_addresses': self.tracked_addresses,
                'last_update': datetime.now().isoformat()
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"[CONFIG] Saved configuration with {len(self.tracked_addresses)} addresses")
        except Exception as e:
            logger.error(f"[CONFIG] Failed to save config: {e}")
    
    def add_address(self, address: str, chain: str = 'ethereum'):
        """
        Dodaje nowy adres do śledzenia
        
        Args:
            address: Adres portfela do śledzenia
            chain: Sieć blockchain (ethereum, bsc, polygon)
        """
        if address not in self.tracked_addresses:
            self.tracked_addresses.append(address)
            self.save_config()
            logger.info(f"[SCHEDULER] Added address {address} ({chain}) to tracking list")
        else:
            logger.info(f"[SCHEDULER] Address {address} already tracked")
    
    def remove_address(self, address: str):
        """
        Usuwa adres z listy śledzenia
        
        Args:
            address: Adres do usunięcia
        """
        if address in self.tracked_addresses:
            self.tracked_addresses.remove(address)
            self.save_config()
            logger.info(f"[SCHEDULER] Removed address {address} from tracking list")
        else:
            logger.warning(f"[SCHEDULER] Address {address} not found in tracking list")
    
    def scan_single_address(self, address: str, chain: str = 'ethereum') -> Dict[str, Any]:
        """
        Skanuje pojedynczy adres
        
        Args:
            address: Adres do analizy
            chain: Sieć blockchain
            
        Returns:
            Wyniki analizy GNN
        """
        try:
            logger.info(f"[SCAN] Starting GNN analysis for {address[:10]}... on {chain}")
            
            result = self.stealth_engine.run_stealth_engine_for_address(
                address=address,
                chain=chain
            )
            
            # Dodaj metadata
            result['scan_timestamp'] = datetime.now().isoformat()
            result['scheduler_scan'] = True
            
            logger.info(f"[SCAN] Completed {address[:10]}... - Alert sent: {result.get('alert_sent', False)}")
            return result
            
        except Exception as e:
            logger.error(f"[SCAN ERROR] {address[:10]}... failed: {e}")
            return {
                'address': address,
                'chain': chain,
                'error': str(e),
                'scan_timestamp': datetime.now().isoformat(),
                'status': 'failed'
            }
    
    def run_batch_scan(self) -> List[Dict[str, Any]]:
        """
        Uruchamia skan wszystkich śledzonych adresów
        
        Returns:
            Lista wyników skanów
        """
        logger.info(f"[BATCH SCAN] Starting scan of {len(self.tracked_addresses)} addresses")
        
        batch_results = []
        successful_scans = 0
        alerts_sent = 0
        
        for i, address in enumerate(self.tracked_addresses, 1):
            logger.info(f"[BATCH] Processing {i}/{len(self.tracked_addresses)}: {address[:10]}...")
            
            result = self.scan_single_address(address)
            batch_results.append(result)
            
            if result.get('status') != 'failed':
                successful_scans += 1
                if result.get('alert_sent', False):
                    alerts_sent += 1
            
            # Krótka przerwa między skanami
            time.sleep(2)
        
        # Zapisz wyniki
        self.save_batch_results(batch_results)
        
        logger.info(f"[BATCH COMPLETE] {successful_scans}/{len(self.tracked_addresses)} successful, "
                   f"{alerts_sent} alerts sent")
        
        return batch_results
    
    def save_batch_results(self, results: List[Dict[str, Any]]):
        """Zapisuje wyniki batch skanu"""
        try:
            os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
            
            # Załaduj poprzednie wyniki
            all_results = []
            if os.path.exists(self.results_file):
                with open(self.results_file, 'r') as f:
                    all_results = json.load(f)
            
            # Dodaj nowe wyniki
            batch_data = {
                'batch_timestamp': datetime.now().isoformat(),
                'addresses_scanned': len(results),
                'results': results
            }
            all_results.append(batch_data)
            
            # Zachowaj tylko ostatnie 100 batch skanów
            all_results = all_results[-100:]
            
            with open(self.results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
                
            logger.info(f"[RESULTS] Saved batch results to {self.results_file}")
            
        except Exception as e:
            logger.error(f"[RESULTS] Failed to save results: {e}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Pobiera ostatnie alerty z określonego okresu
        
        Args:
            hours: Liczba godzin wstecz
            
        Returns:
            Lista alertów
        """
        try:
            if not os.path.exists(self.results_file):
                return []
            
            with open(self.results_file, 'r') as f:
                all_results = json.load(f)
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_alerts = []
            
            for batch in all_results:
                batch_time = datetime.fromisoformat(batch['batch_timestamp'])
                if batch_time >= cutoff_time:
                    for result in batch['results']:
                        if result.get('alert_sent', False):
                            recent_alerts.append(result)
            
            return recent_alerts
            
        except Exception as e:
            logger.error(f"[ALERTS] Failed to get recent alerts: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Pobiera statystyki schedulera"""
        try:
            stats = {
                'tracked_addresses': len(self.tracked_addresses),
                'interval_minutes': self.interval_seconds // 60,
                'recent_alerts_24h': len(self.get_recent_alerts(24)),
                'stealth_engine_stats': self.stealth_engine.get_system_stats()
            }
            
            if os.path.exists(self.results_file):
                with open(self.results_file, 'r') as f:
                    all_results = json.load(f)
                    stats['total_batches'] = len(all_results)
                    if all_results:
                        stats['last_scan'] = all_results[-1]['batch_timestamp']
            
            return stats
            
        except Exception as e:
            logger.error(f"[STATS] Failed to get stats: {e}")
            return {'error': str(e)}
    
    def run_scheduled_scan(self):
        """
        Główna pętla schedulera - uruchamia skany w określonych interwałach
        """
        logger.info(f"[SCHEDULER] Starting scheduled scans every {self.interval_seconds // 60} minutes")
        logger.info(f"[SCHEDULER] Tracking {len(self.tracked_addresses)} addresses")
        
        try:
            while True:
                start_time = datetime.now()
                logger.info(f"[SCHEDULER] Starting batch scan at {start_time.strftime('%H:%M:%S')}")
                
                # Uruchom batch scan
                results = self.run_batch_scan()
                
                # Statystyki
                successful = len([r for r in results if r.get('status') != 'failed'])
                alerts = len([r for r in results if r.get('alert_sent', False)])
                
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"[SCHEDULER] Batch completed in {duration:.1f}s - "
                           f"{successful}/{len(results)} successful, {alerts} alerts")
                
                # Czekaj do następnego skanu
                logger.info(f"[SCHEDULER] Sleeping {self.interval_seconds // 60} minutes until next scan...")
                time.sleep(self.interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("[SCHEDULER] Stopped by user")
        except Exception as e:
            logger.error(f"[SCHEDULER] Fatal error: {e}")
            raise

def test_scheduler():
    """Test schedulera z pojedynczym skanem"""
    print("\n🧪 Testing GNN Scheduler...")
    
    scheduler = GNNScheduler(interval_minutes=1)  # 1 minuta dla testu
    
    # Test dodawania adresu
    scheduler.add_address("0x1234567890123456789012345678901234567890")
    
    # Test pojedynczego skanu
    print("🔍 Running single address scan...")
    result = scheduler.scan_single_address("0x8894E0a0c962CB723c1976a4421c95949bE2D4E6")
    print(f"✅ Single scan result: {result.get('status', 'unknown')}")
    
    # Test batch skanu (tylko 2 adresy dla testu)
    scheduler.tracked_addresses = scheduler.tracked_addresses[:2]
    print("📦 Running batch scan...")
    batch_results = scheduler.run_batch_scan()
    print(f"✅ Batch scan: {len(batch_results)} addresses processed")
    
    # Test statystyk
    stats = scheduler.get_stats()
    print(f"📊 Scheduler stats: {stats}")
    
    print("🎉 Scheduler test completed!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_scheduler()
    else:
        # Uruchom scheduler z domyślnym interwałem 5 minut
        scheduler = GNNScheduler(interval_minutes=5)
        scheduler.run_scheduled_scan()