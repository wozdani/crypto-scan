#!/usr/bin/env python3
"""
Check GNN Scheduler Status
"""

import json
import os
from datetime import datetime

def check_scheduler_status():
    """Sprawdź status schedulera"""
    print("📊 GNN Scheduler Status Check")
    print("=" * 50)
    
    # Config file
    config_file = "cache/scheduler_config.json"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"✅ Tracked Addresses: {len(config.get('tracked_addresses', []))}")
        print(f"📅 Last Update: {config.get('last_update', 'Unknown')}")
        
        print("\n🔍 Tracked Addresses:")
        for i, addr in enumerate(config.get('tracked_addresses', []), 1):
            print(f"  {i}. {addr[:10]}...{addr[-8:]}")
    else:
        print("❌ Config file not found")
    
    # Results file
    results_file = "cache/scheduler_results.json"
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print(f"\n📈 Total Batches: {len(results)}")
        
        if results:
            last_batch = results[-1]
            print(f"📅 Last Scan: {last_batch.get('batch_timestamp', 'Unknown')}")
            print(f"🔍 Addresses Scanned: {last_batch.get('addresses_scanned', 0)}")
            
            # Count alerts
            alerts = sum(1 for r in last_batch.get('results', []) if r.get('alert_sent', False))
            print(f"🚨 Alerts Sent: {alerts}")
            
            # Recent success rate
            successful = sum(1 for r in last_batch.get('results', []) if r.get('status') != 'failed')
            total = len(last_batch.get('results', []))
            if total > 0:
                success_rate = (successful / total) * 100
                print(f"✅ Success Rate: {success_rate:.1f}%")
    else:
        print("❌ Results file not found")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    check_scheduler_status()