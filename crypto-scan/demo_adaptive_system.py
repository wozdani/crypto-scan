#!/usr/bin/env python3
"""
Demo Adaptive System - Demonstracja AdaptiveTraderScore bez żywych danych

Pokazuje działanie systemu adaptacyjnego z przykładowymi danymi
"""

import os
import sys
import json
from datetime import datetime, timezone

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.adaptive_weights import AdaptiveWeightEngine
from utils.context_modifiers import apply_contextual_modifiers, get_market_context


def create_sample_features() -> dict:
    """Tworzy przykładowe cechy dla testów"""
    return {
        "trend_strength": 0.75,
        "pullback_quality": 0.60,
        "support_reaction": 0.55,
        "liquidity_pattern_score": 0.45,
        "psych_score": 0.30,  # Inverted: low = good
        "htf_supportive_score": 0.80,
        "market_phase_modifier": 0.15
    }


def demo_adaptive_learning():
    """Demo procesu uczenia adaptacyjnego"""
    print("🧠 DEMO: Adaptive Learning Process")
    print("=" * 60)
    
    # Create fresh engine for demo
    engine = AdaptiveWeightEngine(memory_size=50)
    engine.reset_memory()
    
    # Scenario 1: Strong trend signals work well
    print("\n📚 Phase 1: Learning from strong trend signals...")
    strong_trend_features = {
        "trend_strength": 0.9,
        "pullback_quality": 0.7,
        "support_reaction": 0.6,
        "liquidity_pattern_score": 0.5,
        "psych_score": 0.2,  # Low = good
        "htf_supportive_score": 0.8,
        "market_phase_modifier": 0.1
    }
    
    # Add successful examples with strong trend
    for i in range(8):
        features = strong_trend_features.copy()
        features["trend_strength"] += (i * 0.01)  # Small variations
        engine.add_example(features, True, f"STRONG_TREND_{i}")
    
    # Scenario 2: Weak support reactions fail
    print("📚 Phase 2: Learning from weak support failures...")
    weak_support_features = {
        "trend_strength": 0.5,
        "pullback_quality": 0.4,
        "support_reaction": 0.1,  # Very weak
        "liquidity_pattern_score": 0.3,
        "psych_score": 0.8,  # High manipulation
        "htf_supportive_score": 0.3,
        "market_phase_modifier": 0.0
    }
    
    # Add failed examples with weak support
    for i in range(5):
        features = weak_support_features.copy()
        features["support_reaction"] += (i * 0.02)
        engine.add_example(features, False, f"WEAK_SUPPORT_{i}")
    
    # Scenario 3: High psychology manipulation fails
    print("📚 Phase 3: Learning from manipulation patterns...")
    manipulation_features = {
        "trend_strength": 0.6,
        "pullback_quality": 0.5,
        "support_reaction": 0.7,
        "liquidity_pattern_score": 0.4,
        "psych_score": 0.9,  # Very high manipulation
        "htf_supportive_score": 0.5,
        "market_phase_modifier": 0.1
    }
    
    # Add failed examples with high manipulation
    for i in range(4):
        features = manipulation_features.copy()
        features["psych_score"] -= (i * 0.05)
        engine.add_example(features, False, f"MANIPULATION_{i}")
    
    # Show learning progress
    stats = engine.get_performance_stats()
    print(f"\n📊 Learning Results:")
    print(f"Total examples: {stats['total_examples']}")
    print(f"Success rate: {stats['success_rate']:.1%}")
    
    # Compare weights
    adaptive_weights = engine.compute_weights()
    default_weights = {
        "trend_strength": 0.25,
        "pullback_quality": 0.20,
        "support_reaction": 0.15,
        "liquidity_pattern_score": 0.10,
        "psych_score": 0.10,
        "htf_supportive_score": 0.10,
        "market_phase_modifier": 0.10
    }
    
    print(f"\n⚖️ Weight Evolution:")
    print(f"{'Feature':<25} {'Default':<8} {'Learned':<8} {'Change':<8}")
    print("-" * 50)
    
    for feature in default_weights:
        default_w = default_weights[feature]
        learned_w = adaptive_weights.get(feature, 0.0)
        change = ((learned_w - default_w) / default_w * 100) if default_w != 0 else 0
        
        indicator = "📈" if change > 5 else "📉" if change < -5 else "➡️"
        print(f"{feature:<25} {default_w:<8.3f} {learned_w:<8.3f} {change:+5.1f}% {indicator}")
    
    return engine


def demo_contextual_modifiers():
    """Demo modyfikatorów kontekstowych"""
    print(f"\n🌍 DEMO: Contextual Modifiers")
    print("=" * 60)
    
    base_features = create_sample_features()
    
    contexts = [
        {
            "name": "London Retest",
            "context": {
                "market_phase": "retest-confirmation",
                "session": "london",
                "btc_global_trend": "strong_up"
            }
        },
        {
            "name": "NY Breakout",
            "context": {
                "market_phase": "breakout-continuation",
                "session": "ny",
                "volatility_level": "high"
            }
        },
        {
            "name": "Asia Bear Market",
            "context": {
                "session": "asia",
                "btc_global_trend": "strong_down",
                "volatility_level": "high"
            }
        }
    ]
    
    for scenario in contexts:
        print(f"\n📊 Scenario: {scenario['name']}")
        print(f"Context: {scenario['context']}")
        
        modified = apply_contextual_modifiers(base_features, scenario['context'])
        
        print("Key changes:")
        for feature in base_features:
            original = base_features[feature]
            new_val = modified[feature]
            if abs(new_val - original) > 0.05:
                change_pct = ((new_val - original) / original * 100)
                print(f"  {feature}: {original:.3f} → {new_val:.3f} ({change_pct:+.1f}%)")


def demo_full_scoring_system():
    """Demo pełnego systemu scoringu"""
    print(f"\n🎯 DEMO: Complete Scoring System")
    print("=" * 60)
    
    # Get trained engine from previous demo
    engine = demo_adaptive_learning()
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Strong Setup",
            "features": {
                "trend_strength": 0.85,
                "pullback_quality": 0.70,
                "support_reaction": 0.75,
                "liquidity_pattern_score": 0.60,
                "psych_score": 0.25,
                "htf_supportive_score": 0.80,
                "market_phase_modifier": 0.15
            },
            "context": {
                "market_phase": "pre-breakout",
                "session": "london",
                "btc_global_trend": "strong_up"
            }
        },
        {
            "name": "Weak Setup",
            "features": {
                "trend_strength": 0.30,
                "pullback_quality": 0.25,
                "support_reaction": 0.20,
                "liquidity_pattern_score": 0.15,
                "psych_score": 0.80,
                "htf_supportive_score": 0.25,
                "market_phase_modifier": 0.05
            },
            "context": {
                "market_phase": "exhaustion-pullback",
                "session": "asia",
                "btc_global_trend": "strong_down"
            }
        }
    ]
    
    adaptive_weights = engine.compute_weights()
    
    for scenario in test_scenarios:
        print(f"\n🧪 Testing: {scenario['name']}")
        
        # Original features
        features = scenario['features']
        
        # Apply contextual modifications
        modified_features = apply_contextual_modifiers(features, scenario['context'])
        
        # Calculate scores
        static_score = sum(features[k] * 0.143 for k in features)  # Equal weights
        adaptive_score = sum(modified_features[k] * adaptive_weights.get(k, 0.0) for k in modified_features)
        
        # Decision logic
        if adaptive_score >= 0.7:
            decision = "JOIN"
        elif adaptive_score >= 0.45:
            decision = "CONSIDER"
        else:
            decision = "AVOID"
        
        print(f"  Static score: {static_score:.3f}")
        print(f"  Adaptive score: {adaptive_score:.3f}")
        print(f"  Decision: {decision}")
        print(f"  Context: {scenario['context']['session']} session, {scenario['context'].get('market_phase', 'unknown')} phase")


def save_demo_results(engine: AdaptiveWeightEngine):
    """Zapisz wyniki demo"""
    try:
        os.makedirs("logs", exist_ok=True)
        
        analysis = engine.export_weights_analysis("logs/demo_adaptive_analysis.json")
        
        demo_summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "demo_type": "adaptive_scoring_system",
            "engine_stats": engine.get_performance_stats(),
            "learned_weights": engine.compute_weights(),
            "summary": "Successful demonstration of adaptive learning and contextual modifications"
        }
        
        with open("logs/demo_adaptive_summary.json", "w", encoding="utf-8") as f:
            json.dump(demo_summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Demo results saved to logs/demo_adaptive_*.json")
        
    except Exception as e:
        print(f"Failed to save demo results: {e}")


def main():
    """Main demo function"""
    print("🚀 ADAPTIVE SCORING SYSTEM DEMONSTRATION")
    print("=" * 70)
    print("This demo shows how the system learns and adapts without live data")
    
    try:
        # Demo 1: Adaptive learning
        engine = demo_adaptive_learning()
        
        # Demo 2: Contextual modifiers
        demo_contextual_modifiers()
        
        # Demo 3: Full system
        demo_full_scoring_system()
        
        # Save results
        save_demo_results(engine)
        
        print(f"\n✅ ADAPTIVE SYSTEM DEMONSTRATION COMPLETE")
        print("Key features demonstrated:")
        print("• Adaptive weight learning from historical performance")
        print("• Contextual feature modifications based on market conditions")
        print("• Dynamic scoring that evolves with experience")
        print("• Integration ready for live trading system")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()