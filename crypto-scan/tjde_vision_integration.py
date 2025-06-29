"""
TJDE-Vision Integration Module
Integrates CLIP confidence with TJDE scoring for enhanced decision making
"""

from typing import Dict, Tuple, Optional
import os
from vision_ai_pipeline import EnhancedCLIPPredictor, integrate_clip_with_tjde


def enhance_tjde_with_vision_ai(symbol: str, tjde_result: Dict, 
                               chart_path: Optional[str] = None) -> Dict:
    """
    Enhance TJDE decision with Vision-AI CLIP analysis
    
    Args:
        symbol: Trading symbol
        tjde_result: Original TJDE result dictionary
        chart_path: Path to chart image (optional)
        
    Returns:
        Enhanced TJDE result with Vision-AI integration
    """
    try:
        original_score = tjde_result.get('final_score', 0.0)
        current_phase = tjde_result.get('market_phase', 'unknown')
        
        # Find chart if not provided
        if not chart_path:
            chart_path = find_latest_chart(symbol)
        
        if not chart_path or not os.path.exists(chart_path):
            print(f"[VISION-AI] {symbol}: No chart available for analysis")
            return tjde_result
        
        # Integrate CLIP analysis
        enhanced_score, clip_info = integrate_clip_with_tjde(
            original_score, current_phase, chart_path
        )
        
        # Update TJDE result
        enhanced_result = tjde_result.copy()
        enhanced_result.update({
            'final_score': enhanced_score,
            'vision_ai_enhancement': enhanced_score - original_score,
            'clip_phase': clip_info.get('clip_phase', 'N/A'),
            'clip_confidence': clip_info.get('clip_confidence', 0.0),
            'phase_match': clip_info.get('phase_match', False),
            'enhanced_by_vision_ai': True
        })
        
        # Update decision based on enhanced score
        if enhanced_score >= 0.85 and enhanced_result.get('decision', '') != 'join_trend':
            enhanced_result['decision'] = 'join_trend'
            enhanced_result['decision_upgraded_by_vision_ai'] = True
            print(f"[VISION-AI] {symbol}: Decision upgraded to JOIN_TREND (score: {enhanced_score:.3f})")
        
        return enhanced_result
        
    except Exception as e:
        print(f"[VISION-AI ERROR] {symbol}: {e}")
        return tjde_result


def find_latest_chart(symbol: str) -> Optional[str]:
    """Find the latest chart for a symbol"""
    search_paths = [
        "training_data/charts",
        "charts", 
        "exports",
        "training_data/charts"
    ]
    
    latest_chart = None
    latest_time = 0
    
    for path in search_paths:
        if not os.path.exists(path):
            continue
            
        for filename in os.listdir(path):
            if filename.startswith(symbol) and filename.endswith('.png'):
                file_path = os.path.join(path, filename)
                file_time = os.path.getmtime(file_path)
                
                if file_time > latest_time:
                    latest_time = file_time
                    latest_chart = file_path
    
    return latest_chart


def generate_vision_ai_alert_message(symbol: str, enhanced_result: Dict) -> str:
    """Generate enhanced alert message with Vision-AI information"""
    
    base_message = f"🎯 {symbol} - Enhanced TJDE Alert\n\n"
    
    # Core metrics
    base_message += f"📊 TJDE Score: {enhanced_result.get('final_score', 0):.3f}\n"
    base_message += f"🎲 Decision: {enhanced_result.get('decision', 'unknown').upper()}\n"
    base_message += f"📈 Phase: {enhanced_result.get('market_phase', 'unknown')}\n\n"
    
    # Vision-AI enhancement
    if enhanced_result.get('enhanced_by_vision_ai', False):
        clip_phase = enhanced_result.get('clip_phase', 'N/A')
        clip_confidence = enhanced_result.get('clip_confidence', 0.0)
        enhancement = enhanced_result.get('vision_ai_enhancement', 0.0)
        phase_match = enhanced_result.get('phase_match', False)
        
        base_message += f"👁️ VISION-AI ANALYSIS:\n"
        base_message += f"🎨 CLIP Phase: {clip_phase}\n"
        base_message += f"🎯 Confidence: {clip_confidence:.3f}\n"
        base_message += f"⚡ Score Enhancement: {enhancement:+.3f}\n"
        base_message += f"✅ Phase Match: {'Yes' if phase_match else 'No'}\n"
        
        if enhanced_result.get('decision_upgraded_by_vision_ai', False):
            base_message += f"🚀 Decision UPGRADED by Vision-AI\n"
    
    return base_message


def run_vision_ai_integration_test():
    """Test Vision-AI integration with sample data"""
    print("[VISION-AI TEST] Testing TJDE-Vision integration...")
    
    # Sample TJDE result
    sample_tjde = {
        'symbol': 'BTCUSDT',
        'final_score': 0.75,
        'market_phase': 'trend-following',
        'decision': 'consider_entry',
        'trend_strength': 0.8
    }
    
    # Test enhancement
    enhanced = enhance_tjde_with_vision_ai('BTCUSDT', sample_tjde)
    
    print(f"Original score: {sample_tjde['final_score']}")
    print(f"Enhanced score: {enhanced.get('final_score', 0)}")
    print(f"Vision-AI enhancement: {enhanced.get('vision_ai_enhancement', 0):+.3f}")
    
    # Test alert message
    alert_msg = generate_vision_ai_alert_message('BTCUSDT', enhanced)
    print(f"\nSample alert message:\n{alert_msg}")
    
    print("[VISION-AI TEST] Integration test complete")


if __name__ == "__main__":
    run_vision_ai_integration_test()