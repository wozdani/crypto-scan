# integration/scanner_patch.py
"""
Patch file showing how to integrate multi-agent consensus into main scanner
Apply these changes to your main scanning logic
"""

# EXAMPLE INTEGRATION POINT IN MAIN SCANNER:

def process_token_with_consensus(symbol, market_data, detector_results):
    """
    Example integration showing where to add multi-agent consensus
    
    This should be called after all detector data is collected but before alert decisions
    """
    
    # 1. Prepare payload from existing scanner data
    payload = {
        "symbol": symbol,
        "detector_breakdown": {
            "stealth_engine": detector_results.get("stealth_engine", {}),
            "californium_whale": detector_results.get("californium_whale", {}),
            "diamond_whale": detector_results.get("diamond_whale", {}),
            "whale_clip": detector_results.get("whale_clip", {}),
            "gnn_detector": detector_results.get("gnn_detector", {}),
            # Add other detectors as available
        },
        "meta": {
            "price": market_data.get("price", 0),
            "volume_24h": market_data.get("volume_24h", 0),
            "price_change_24h": market_data.get("price_change_24h", 0),
            "market_cap": market_data.get("market_cap", 0),
            "spread": market_data.get("spread", 0),
        },
        "trust": {
            "whale_addresses": market_data.get("whale_addresses", []),
            "trust_score": market_data.get("trust_score", 0.5),
            "dex_inflow": market_data.get("dex_inflow", 0),
        },
        "history": {
            "recent_pumps": market_data.get("recent_pumps", []),
            "volume_pattern": market_data.get("volume_pattern", "normal"),
            "price_trend": market_data.get("price_trend", "stable"),
        },
        "perf": {
            "detector_precision": market_data.get("detector_precision", {}),
            "false_positive_rate": market_data.get("false_positive_rate", {}),
            "avg_lag_mins": market_data.get("avg_lag_mins", 30),
        }
    }
    
    # 2. Run multi-agent consensus
    from integration.multi_agent_integration import multi_agent_scanner
    
    consensus_result = multi_agent_scanner.process_token_consensus(
        symbol=symbol,
        detector_breakdown=payload["detector_breakdown"],
        meta_dict=payload["meta"],
        trust_dict=payload["trust"],
        history_dict=payload["history"],
        perf_dict=payload["perf"]
    )
    
    # 3. Use consensus result for alert decision
    if consensus_result:
        should_alert = multi_agent_scanner.should_trigger_alert(
            consensus_result, 
            min_confidence=0.7  # Configurable threshold
        )
        
        if should_alert:
            alert_message = multi_agent_scanner.format_alert_message(symbol, consensus_result)
            
            # Send alert through existing alert system
            # send_telegram_alert(alert_message)
            # save_alert_to_database(symbol, consensus_result)
            
            print(f"[CONSENSUS ALERT] {symbol}")
            print(alert_message)
            
            return {
                "action": "ALERT",
                "confidence": max(consensus_result.final_probs.values()),
                "message": alert_message,
                "consensus_data": consensus_result
            }
        else:
            dominant_action = max(consensus_result.final_probs, key=consensus_result.final_probs.get)
            confidence = consensus_result.final_probs[dominant_action]
            
            print(f"[CONSENSUS] {symbol}: {dominant_action} ({confidence:.3f}) - No alert")
            
            return {
                "action": dominant_action,
                "confidence": confidence,
                "message": f"Consensus: {dominant_action}",
                "consensus_data": consensus_result
            }
    else:
        print(f"[CONSENSUS] {symbol}: Failed to get consensus - falling back to legacy logic")
        return {"action": "FALLBACK", "confidence": 0.0, "message": "Consensus failed"}

# EXAMPLE USAGE IN MAIN SCANNING LOOP:
"""
for symbol in active_tokens:
    # Existing detector logic...
    detector_results = run_all_detectors(symbol)
    market_data = get_market_data(symbol)
    
    # NEW: Multi-agent consensus integration
    consensus_result = process_token_with_consensus(symbol, market_data, detector_results)
    
    if consensus_result["action"] == "ALERT":
        # Handle alert
        handle_consensus_alert(symbol, consensus_result)
    elif consensus_result["action"] == "FALLBACK":
        # Use legacy alert logic
        handle_legacy_alert_logic(symbol, detector_results)
    else:
        # Log non-alert consensus decision
        log_consensus_decision(symbol, consensus_result)
"""