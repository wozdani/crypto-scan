"""
GPT Chart Analysis and Setup Label Extraction
Analyzes TradingView screenshots to identify trading setups and generate meaningful labels for CLIP training
"""

import os
import re
import json
import base64
from typing import Dict, Optional, Tuple
from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def log_warning(category, exception=None, additional_info=None):
    """Centralized warning logging"""
    message = f"[{category}]"
    if additional_info:
        message += f" {additional_info}"
    if exception:
        message += f" - {str(exception)}"
    print(message)

def encode_image_to_base64(image_path: str) -> Optional[str]:
    """
    Encode image to base64 for GPT Vision API
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64 encoded image string or None if failed
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        log_warning("IMAGE ENCODING ERROR", e, f"Failed to encode {image_path}")
        return None

def extract_setup_label_from_commentary(gpt_commentary: str) -> str:
    """
    Extract clean setup label from GPT commentary
    
    Args:
        gpt_commentary: Raw GPT analysis text
        
    Returns:
        Clean setup label suitable for filename
    """
    if not gpt_commentary:
        return "unknown_setup"
    
    # Common trading setup patterns - SYNCHRONIZED WITH GPT PROMPT
    setup_patterns = [
        # TREND PATTERNS (exact match with prompt)
        (r'\bpullback_in_trend\b', 'pullback_in_trend'),
        (r'\btrend_continuation\b', 'trend_continuation'),
        (r'\bmomentum_follow\b', 'momentum_follow'),
        
        # BREAKOUT PATTERNS (exact match with prompt)  
        (r'\bbreakout_pattern\b', 'breakout_pattern'),
        (r'\bresistance_break\b', 'resistance_break'),
        (r'\bsupport_break\b', 'support_break'),
        
        # REVERSAL PATTERNS (exact match with prompt)
        (r'\breversal_pattern\b', 'reversal_pattern'),
        (r'\bdouble_top\b', 'double_top'),
        (r'\bhead_shoulders\b', 'head_shoulders'),
        
        # CONSOLIDATION PATTERNS (exact match with prompt)
        (r'\brange_trading\b', 'range_trading'),
        (r'\bconsolidation_squeeze\b', 'consolidation_squeeze'),
        (r'\btriangle_pattern\b', 'triangle_pattern'),
        
        # FALLBACK: Legacy patterns for backward compatibility
        (r'\b(?:trend\s*)?pullback(?:\s+(?:in|to|from)\s+\w+)?', 'pullback_in_trend'),
        (r'\b(?:trend\s*)?continuation(?:\s+pattern)?', 'trend_continuation'),
        (r'\bbreakout(?:\s+(?:above|below|from)\s+\w+)?', 'breakout_pattern'),
        (r'\bresistance\s+break', 'resistance_break'),
        (r'\bsupport\s+break', 'support_break'),
        (r'\breversal(?:\s+pattern)?', 'reversal_pattern'),
        (r'\brange(?:\s+(?:bound|trading))?', 'range_trading'),
        (r'\bconsolidation', 'consolidation_squeeze'),
        (r'\baccumulation', 'accumulation_zone'),
        
        # Support/Resistance
        (r'\bsupport\s+(?:test|bounce|hold)', 'support_test'),
        (r'\bresistance\s+(?:test|rejection)', 'resistance_test'),
        (r'\bkey\s+level', 'key_level_test'),
        (r'\bpivot\s+(?:point|level)', 'pivot_level'),
        
        # Squeeze patterns
        (r'\bsqueeze(?:\s+pattern)?', 'squeeze_pattern'),
        (r'\bcompression', 'price_compression'),
        (r'\btight\s+range', 'tight_range'),
        
        # Special patterns
        (r'\bflag(?:\s+pattern)?', 'flag_pattern'),
        (r'\bpennant', 'pennant_pattern'),
        (r'\btriangle', 'triangle_pattern'),
        (r'\bwedge', 'wedge_pattern'),
        (r'\bfakeout', 'fakeout_pattern'),
        (r'\bfalse\s+break', 'false_breakout'),
        
        # Volume patterns
        (r'\bvolume\s+(?:spike|surge)', 'volume_spike'),
        (r'\blow\s+volume', 'low_volume_move'),
        (r'\bhigh\s+volume', 'high_volume_move'),
        
        # No clear pattern
        (r'\bno\s+(?:clear|obvious)\s+pattern', 'no_clear_pattern'),
        (r'\bnoise', 'market_noise'),
        (r'\bchoppy', 'choppy_action'),
    ]
    
    # Clean and lowercase commentary for pattern matching
    clean_text = gpt_commentary.lower().strip()
    
    # Try to find matching patterns
    for pattern, label in setup_patterns:
        if re.search(pattern, clean_text):
            return label
    
    # Fallback: extract first few meaningful words
    words = re.findall(r'\b[a-z]+\b', clean_text)
    meaningful_words = [w for w in words if len(w) > 3 and w not in ['this', 'that', 'with', 'from', 'the']]
    
    if meaningful_words:
        return '_'.join(meaningful_words[:2])
    
    return "unknown_setup"

def analyze_chart_with_gpt(image_path: str, symbol: str, tjde_score: float) -> Dict:
    """
    Analyze TradingView chart using GPT-4o Vision to identify trading setup
    
    Args:
        image_path: Path to TradingView screenshot
        symbol: Trading symbol
        tjde_score: TJDE score for context
        
    Returns:
        Dictionary with GPT analysis, setup label, and metadata
    """
    if not openai_client:
        log_warning("GPT ANALYSIS SKIP", None, "OpenAI API key not available")
        return {
            'success': False,
            'gpt_commentary': 'GPT analysis unavailable - no API key',
            'setup_label': 'unknown_setup',
            'error': 'no_api_key'
        }
    
    if not os.path.exists(image_path):
        log_warning("GPT ANALYSIS ERROR", None, f"Chart file not found: {image_path}")
        return {
            'success': False,
            'gpt_commentary': 'Chart file not found',
            'setup_label': 'unknown_setup',
            'error': 'file_not_found'
        }
    
    try:
        # Encode image to base64
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            return {
                'success': False,
                'gpt_commentary': 'Failed to encode image',
                'setup_label': 'unknown_setup',
                'error': 'encoding_failed'
            }
        
        # GPT prompt for trading setup analysis - Enhanced for consistency
        prompt = f"""Analyze this TradingView chart for {symbol} (TJDE score: {tjde_score:.3f}).

You must identify the PRIMARY setup pattern using ONLY ONE of these exact terms:

TREND PATTERNS:
- pullback_in_trend (price pulling back within an established trend)
- trend_continuation (trend resuming after brief pause)
- momentum_follow (strong directional momentum)

BREAKOUT PATTERNS:
- breakout_pattern (price breaking key level with volume)
- resistance_break (breaking above resistance)
- support_break (breaking below support)

REVERSAL PATTERNS:
- reversal_pattern (clear trend change signal)
- double_top (or double_bottom reversal)
- head_shoulders (reversal formation)

CONSOLIDATION PATTERNS:
- range_trading (sideways movement between levels)
- consolidation_squeeze (tight range compression)
- triangle_pattern (converging price action)

Choose the SINGLE most dominant pattern. If unclear, use "no_clear_pattern".

Required format:
SETUP: [exact term from above list]
ANALYSIS: [2-3 sentence technical explanation why you chose this specific setup]

Be consistent - the SETUP field must match your analysis description exactly."""

        # Using GPT-4o for reliable crypto chart analysis capabilities
        # using latest OpenAI model for superior pattern recognition
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=1.0,
            max_completion_tokens=300
        )
        
        gpt_analysis = response.choices[0].message.content
        
        # Extract setup label from analysis
        setup_label = extract_setup_label_from_commentary(gpt_analysis or "")
        
        print(f"[GPT CHART ANALYSIS] {symbol}: {setup_label}")
        if gpt_analysis:
            print(f"[GPT COMMENTARY] {gpt_analysis[:100]}...")
        else:
            print(f"[GPT COMMENTARY] No analysis content returned")
        
        # Check for label consistency using the new consistency checker
        label_conflict = False
        consistency_check = {"has_conflict": False}
        
        try:
            from utils.gpt_label_consistency import check_gpt_label_consistency
            
            # FIXED: Compare two different label sources
            # 1. setup_label comes from extract_setup_label_from_commentary() 
            # 2. Extract SETUP: field directly from GPT response
            
            # Extract SETUP: field from GPT response (direct field)
            setup_field_pattern = re.search(r'\*\*SETUP:\*\*\s*([^\n*]+)', gpt_analysis or "", re.IGNORECASE)
            if not setup_field_pattern:
                setup_field_pattern = re.search(r'SETUP:\s*([^\n]+)', gpt_analysis or "", re.IGNORECASE)
            
            setup_field_text = setup_field_pattern.group(1).strip() if setup_field_pattern else ""
            
            # Compare setup_label (from commentary extraction) vs setup_field_text (from direct parsing)
            if setup_label and setup_field_text and setup_label != setup_field_text:
                print(f"[CONFLICT DEBUG] {symbol}: setup_label='{setup_label}' vs setup_field='{setup_field_text}'")
                
                consistency_check = check_gpt_label_consistency(
                    gpt_analysis=setup_label,  # From extract_setup_label_from_commentary()
                    gpt_commentary=setup_field_text,  # From **SETUP:** field
                    symbol=symbol
                )
            else:
                # No conflict if labels are identical or one is missing
                consistency_check = {"has_conflict": False, "reason": "identical_or_missing"}
            
            # If critical conflict detected, flag for review
            if consistency_check.get('has_conflict') and consistency_check.get('severity') == 'critical':
                label_conflict = True
                log_warning("GPT LABEL CONFLICT", None, 
                          f"{symbol}: {consistency_check['description']} - CRITICAL CONFLICT DETECTED")
                print(f"[LABEL CONFLICT] {symbol}: {consistency_check.get('conflict_type', 'unknown')} - {consistency_check.get('severity', 'unknown')}")
            elif consistency_check.get('has_conflict'):
                print(f"[LABEL CHECK] {symbol}: Minor inconsistency - {consistency_check.get('description', 'no details')}")
        
        except Exception as e:
            log_warning("CONSISTENCY CHECK ERROR", e, f"Failed to check label consistency for {symbol}")
            consistency_check = {"has_conflict": False, "error": str(e)}
        
        return {
            'success': True,
            'gpt_commentary': gpt_analysis,
            'setup_label': setup_label,
            'symbol': symbol,
            'tjde_score': tjde_score,
            'image_path': image_path,
            'model': 'gpt-4o',
            'tokens_used': response.usage.total_tokens if response.usage else None,
            'label_conflict': label_conflict,
            'consistency_check': consistency_check
        }
        
    except Exception as e:
        log_warning("GPT ANALYSIS ERROR", e, f"Failed to analyze {symbol}")
        return {
            'success': False,
            'gpt_commentary': f'GPT analysis failed: {str(e)}',
            'setup_label': 'analysis_failed',
            'error': str(e)
        }

def rename_chart_with_setup_label(
    original_path: str, 
    setup_label: str, 
    gpt_analysis: Dict
) -> Optional[str]:
    """
    Rename chart file to include setup label and update metadata
    
    Args:
        original_path: Original chart file path
        setup_label: Setup label from GPT analysis
        gpt_analysis: Complete GPT analysis result
        
    Returns:
        New file path or None if failed
    """
    try:
        if not os.path.exists(original_path):
            log_warning("RENAME ERROR", None, f"Original file not found: {original_path}")
            return None
        
        # Parse original filename to extract components
        # Expected format: SYMBOL_EXCHANGE_score-XXX_YYYYMMDD_HHMM.png
        original_filename = os.path.basename(original_path)
        name_parts = original_filename.replace('.png', '').split('_')
        
        if len(name_parts) >= 5:
            symbol = name_parts[0]
            exchange = name_parts[1] 
            score_part = name_parts[2]  # score-XXX
            date_part = name_parts[3]   # YYYYMMDD
            time_part = name_parts[4]   # HHMM
            
            # Create new filename with setup label
            new_filename = f"{symbol}_{exchange}_{setup_label}_{score_part}_{date_part}_{time_part}.png"
        else:
            # Fallback for unexpected filename format
            base_name = original_filename.replace('.png', '')
            new_filename = f"{base_name}_{setup_label}.png"
        
        # Ensure new filename is filesystem-safe
        new_filename = re.sub(r'[^a-zA-Z0-9_\-.]', '_', new_filename)
        
        # Create new path
        directory = os.path.dirname(original_path)
        new_path = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(original_path, new_path)
        print(f"[CHART RENAME] {os.path.basename(original_path)} → {new_filename}")
        
        # Update metadata file if it exists
        original_metadata = original_path.replace('.png', '_metadata.json')
        if os.path.exists(original_metadata):
            new_metadata = new_path.replace('.png', '_metadata.json')
            
            try:
                # Load existing metadata
                with open(original_metadata, 'r') as f:
                    metadata = json.load(f)
                
                # Add GPT analysis data including consistency check results
                metadata.update({
                    'gpt_analysis': gpt_analysis.get('gpt_commentary', ''),
                    'setup_label': setup_label,
                    'gpt_model': gpt_analysis.get('model', 'gpt-4o'),
                    'gpt_tokens': gpt_analysis.get('tokens_used'),
                    'original_filename': original_filename,
                    'renamed_for_clip_training': True,
                    'label_conflict': gpt_analysis.get('label_conflict', False),
                    'consistency_check': gpt_analysis.get('consistency_check', {}),
                    'needs_review': gpt_analysis.get('label_conflict', False)
                })
                
                # Save updated metadata
                with open(new_metadata, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Remove old metadata file
                os.remove(original_metadata)
                print(f"[METADATA UPDATE] GPT analysis added to {os.path.basename(new_metadata)}")
                
            except Exception as meta_e:
                log_warning("METADATA UPDATE ERROR", meta_e, f"Failed to update metadata for {new_filename}")
        
        return new_path
        
    except Exception as e:
        log_warning("CHART RENAME ERROR", e, f"Failed to rename {original_path}")
        return None

def analyze_and_label_chart(image_path: str, symbol: str, tjde_score: float) -> Optional[str]:
    """
    Complete pipeline: analyze chart with GPT and rename with setup label
    
    Args:
        image_path: Path to TradingView screenshot
        symbol: Trading symbol  
        tjde_score: TJDE score
        
    Returns:
        New file path with setup label or None if failed
    """
    print(f"[GPT LABELING] Starting analysis for {symbol}...")
    
    # Step 1: Analyze chart with GPT
    gpt_result = analyze_chart_with_gpt(image_path, symbol, tjde_score)
    
    if not gpt_result.get('success'):
        log_warning("GPT LABELING FAILED", None, f"Analysis failed for {symbol}: {gpt_result.get('error')}")
        return image_path  # Return original path
    
    # Step 2: Rename chart with setup label
    new_path = rename_chart_with_setup_label(
        image_path, 
        gpt_result['setup_label'], 
        gpt_result
    )
    
    if new_path:
        print(f"[GPT LABELING SUCCESS] {symbol} → {gpt_result['setup_label']}")
        return new_path
    else:
        log_warning("GPT LABELING ERROR", None, f"Rename failed for {symbol}")
        return image_path  # Return original path

def test_gpt_analysis():
    """Test GPT chart analysis with sample data"""
    print("🧪 Testing GPT Chart Analysis...")
    
    # Find a sample chart to test
    chart_dir = "training_data/charts"
    if os.path.exists(chart_dir):
        for filename in os.listdir(chart_dir):
            if filename.endswith('.png') and 'BINANCE' in filename:
                test_path = os.path.join(chart_dir, filename)
                symbol = filename.split('_')[0]
                
                print(f"🔍 Testing with: {filename}")
                result = analyze_chart_with_gpt(test_path, symbol, 0.65)
                
                if result['success']:
                    print(f"✅ GPT Analysis successful:")
                    print(f"   Setup: {result['setup_label']}")
                    print(f"   Commentary: {result['gpt_commentary'][:150]}...")
                else:
                    print(f"❌ GPT Analysis failed: {result.get('error')}")
                
                return result
    
    print("⚠️ No test charts found")
    return None

if __name__ == "__main__":
    test_gpt_analysis()