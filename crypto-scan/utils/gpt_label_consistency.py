#!/usr/bin/env python3
"""
GPT Label Consistency Detector & Corrector
Detects and resolves inconsistencies between GPT_CHART_ANALYSIS and GPT_COMMENTARY setup labels
"""

import re
import json
import logging
from typing import Dict, Tuple, Optional, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class GPTLabelConsistencyChecker:
    """Detects and corrects inconsistencies in GPT-generated chart labels"""
    
    def __init__(self):
        """Initialize consistency checker with pattern definitions"""
        
        # Define mutually exclusive setup categories
        self.setup_categories = {
            "trend_movement": {
                "pullback_in_trend", "pullback_in_uptrend", "pullback_in_downtrend",
                "trend_pullback_reacted", "trend_continuation", "trend_following",
                "pullback_to_moving_average", "retracement_pattern"
            },
            "breakout_movement": {
                "breakout_pattern", "breakout_continuation", "breakout_setup",
                "resistance_breakout", "support_breakout", "volume_breakout",
                "momentum_breakout", "trend_breakout_with_strong_volume"
            },
            "consolidation": {
                "range_trading", "range_consolidation", "consolidation_squeeze",
                "sideways_movement", "accumulation_pattern", "distribution_pattern",
                "squeeze_before_breakout", "compression_pattern"
            },
            "reversal": {
                "reversal_pattern", "trend_reversal", "double_top", "double_bottom",
                "head_and_shoulders", "inverse_head_shoulders", "support_bounce",
                "resistance_rejection", "fakeout_on_resistance"
            }
        }
        
        # Keywords that indicate specific movements
        self.movement_keywords = {
            "pullback": ["pullback", "korekta", "correction", "retracement", "cofka", "reakcja"],
            "breakout": ["breakout", "wybicie", "breakthrough", "momentum", "wybija", "przebicie"],
            "consolidation": ["consolidation", "konsolidacja", "range", "sideways", "squeeze", "compression"],
            "reversal": ["reversal", "odwrócenie", "reversal", "bounce", "rejection", "odbicie"]
        }
        
        # Setup characteristics for validation
        self.setup_characteristics = {
            "pullback_in_trend": {
                "movement": "correction",
                "volume": ["normal", "decreasing"],
                "entry": "on_retracement",
                "location": "within_trend",
                "momentum": "counter_trend"
            },
            "breakout_pattern": {
                "movement": "breakout",
                "volume": ["increasing", "high"],
                "entry": "on_momentum", 
                "location": "at_resistance",
                "momentum": "strong_directional"
            },
            "trend_continuation": {
                "movement": "continuation",
                "volume": ["steady", "increasing"],
                "entry": "trend_following",
                "location": "trend_direction",
                "momentum": "aligned_with_trend"
            },
            "range_trading": {
                "movement": "sideways",
                "volume": ["varied", "decreasing"],
                "entry": "at_levels",
                "location": "between_support_resistance",
                "momentum": "weak_directional"
            }
        }
        
        # Conflict severity levels
        self.conflict_severity = {
            "critical": ["pullback_in_trend vs breakout_pattern", "consolidation vs breakout"],
            "moderate": ["trend_continuation vs pullback", "range_trading vs trend_following"],
            "minor": ["breakout_pattern vs breakout_continuation", "pullback vs retracement"]
        }

    def extract_labels_from_gpt_output(self, gpt_analysis: str, gpt_commentary: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract setup labels from GPT analysis and commentary
        
        Args:
            gpt_analysis: Raw GPT chart analysis text
            gpt_commentary: GPT commentary with setup information
            
        Returns:
            Tuple of (analysis_label, commentary_label)
        """
        analysis_label = None
        commentary_label = None
        
        try:
            # Extract from GPT_CHART_ANALYSIS
            if gpt_analysis:
                # Look for direct label after colon
                analysis_match = re.search(r'(?:ANALYSIS|GPT_CHART_ANALYSIS):\s*([a-z_]+)', gpt_analysis, re.IGNORECASE)
                if analysis_match:
                    analysis_label = analysis_match.group(1).lower().strip()
                else:
                    # Look for pattern in first line
                    first_line = gpt_analysis.split('\n')[0].lower()
                    for category_setups in self.setup_categories.values():
                        for setup in category_setups:
                            if setup in first_line:
                                analysis_label = setup
                                break
                        if analysis_label:
                            break
            
            # Extract from GPT_COMMENTARY
            if gpt_commentary:
                # Look for SETUP: pattern
                setup_match = re.search(r'SETUP:\s*([a-z_]+)', gpt_commentary, re.IGNORECASE)
                if setup_match:
                    commentary_label = setup_match.group(1).lower().strip()
                else:
                    # Look for setup keywords in commentary
                    commentary_lower = gpt_commentary.lower()
                    for category_setups in self.setup_categories.values():
                        for setup in category_setups:
                            if setup in commentary_lower:
                                commentary_label = setup
                                break
                        if commentary_label:
                            break
            
            return analysis_label, commentary_label
            
        except Exception as e:
            logger.error(f"Error extracting GPT labels: {e}")
            return None, None

    def detect_label_conflict(self, analysis_label: str, commentary_label: str) -> Dict:
        """
        Detect conflicts between analysis and commentary labels
        
        Args:
            analysis_label: Label from GPT chart analysis
            commentary_label: Label from GPT commentary
            
        Returns:
            Dictionary with conflict detection results
        """
        if not analysis_label or not commentary_label:
            return {
                "has_conflict": False,
                "conflict_type": "missing_label",
                "severity": "minor",
                "description": "One or both labels missing"
            }
        
        # Normalize labels
        analysis_label = analysis_label.lower().strip()
        commentary_label = commentary_label.lower().strip()
        
        # If labels are identical, no conflict
        if analysis_label == commentary_label:
            return {
                "has_conflict": False,
                "conflict_type": "none",
                "severity": "none",
                "description": "Labels are identical"
            }
        
        # Find categories for each label
        analysis_category = None
        commentary_category = None
        
        for category, setups in self.setup_categories.items():
            if analysis_label in setups:
                analysis_category = category
            if commentary_label in setups:
                commentary_category = category
        
        # Check for category conflicts
        if analysis_category and commentary_category:
            if analysis_category != commentary_category:
                # Different categories = critical conflict
                conflict_description = f"{analysis_label} ({analysis_category}) vs {commentary_label} ({commentary_category})"
                severity = "critical"
                
                # Check for specific critical conflicts
                if any(conflict in conflict_description for conflict in self.conflict_severity["critical"]):
                    severity = "critical"
                elif any(conflict in conflict_description for conflict in self.conflict_severity["moderate"]):
                    severity = "moderate"
                else:
                    severity = "minor"
                
                return {
                    "has_conflict": True,
                    "conflict_type": "category_mismatch",
                    "severity": severity,
                    "description": conflict_description,
                    "analysis_category": analysis_category,
                    "commentary_category": commentary_category,
                    "suggested_resolution": self._suggest_resolution(analysis_label, commentary_label)
                }
            else:
                # Same category but different specific setups = minor conflict
                return {
                    "has_conflict": True,
                    "conflict_type": "setup_variation",
                    "severity": "minor",
                    "description": f"Same category ({analysis_category}) but different setups: {analysis_label} vs {commentary_label}",
                    "analysis_category": analysis_category,
                    "commentary_category": commentary_category,
                    "suggested_resolution": commentary_label  # Prefer commentary label for minor conflicts
                }
        
        # Unknown labels - classify as moderate conflict
        return {
            "has_conflict": True,
            "conflict_type": "unknown_labels",
            "severity": "moderate", 
            "description": f"Unknown label classification: {analysis_label} vs {commentary_label}",
            "suggested_resolution": commentary_label  # Default to commentary
        }

    def _suggest_resolution(self, analysis_label: str, commentary_label: str) -> str:
        """Suggest which label should be used for resolution"""
        
        # Priority rules for conflict resolution
        priority_order = [
            "breakout_pattern",
            "pullback_in_trend", 
            "trend_continuation",
            "range_trading",
            "consolidation_squeeze",
            "reversal_pattern"
        ]
        
        # Check if either label is in priority list
        for priority_label in priority_order:
            if priority_label == analysis_label:
                return analysis_label
            if priority_label == commentary_label:
                return commentary_label
        
        # Default: prefer commentary label (more detailed analysis)
        return commentary_label

    def generate_correction_prompt(self, symbol: str, analysis_label: str, commentary_label: str, conflict_info: Dict) -> str:
        """
        Generate GPT prompt for label correction
        
        Args:
            symbol: Trading symbol
            analysis_label: Original analysis label
            commentary_label: Original commentary label
            conflict_info: Conflict detection results
            
        Returns:
            Correction prompt for GPT
        """
        if conflict_info["severity"] == "critical":
            prompt = f"""
LABEL CONSISTENCY CHECK for {symbol}:

DETECTED CONFLICT: {conflict_info['description']}
Severity: {conflict_info['severity'].upper()}

Original labels:
- Chart Analysis: {analysis_label}
- Commentary Setup: {commentary_label}

These represent fundamentally different market movements:
• Pullback = correction within existing trend (entry on retracement)
• Breakout = momentum move through resistance (entry on strength)

Please review the chart and provide ONE consistent label that accurately describes:
1. The primary market action shown
2. The entry point type (retracement vs momentum)
3. The volume characteristics visible

Respond with: CORRECTED_SETUP: [single_consistent_label]
"""
        else:
            prompt = f"""
LABEL REFINEMENT for {symbol}:

Minor inconsistency detected: {conflict_info['description']}

Original labels:
- Chart Analysis: {analysis_label} 
- Commentary Setup: {commentary_label}

Please confirm which label better describes the chart setup or suggest a unified label.

Respond with: REFINED_SETUP: [best_label]
"""
        
        return prompt.strip()

    def check_metadata_consistency(self, metadata_file: Path) -> Dict:
        """
        Check consistency in existing metadata file
        
        Args:
            metadata_file: Path to metadata JSON file
            
        Returns:
            Consistency check results
        """
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            gpt_analysis = metadata.get('gpt_analysis', '')
            setup_label = metadata.get('setup_label', '')
            
            # Extract labels from text
            analysis_label, commentary_label = self.extract_labels_from_gpt_output(gpt_analysis, gpt_analysis)
            
            # Use setup_label as commentary if available
            if setup_label:
                commentary_label = setup_label
            
            if analysis_label and commentary_label:
                conflict_info = self.detect_label_conflict(analysis_label, commentary_label)
                conflict_info["metadata_file"] = str(metadata_file)
                conflict_info["symbol"] = metadata.get('symbol', 'unknown')
                conflict_info["analysis_label"] = analysis_label
                conflict_info["commentary_label"] = commentary_label
                
                return conflict_info
            else:
                return {
                    "has_conflict": False,
                    "conflict_type": "insufficient_data",
                    "metadata_file": str(metadata_file),
                    "symbol": metadata.get('symbol', 'unknown')
                }
                
        except Exception as e:
            logger.error(f"Error checking metadata consistency: {e}")
            return {
                "has_conflict": False,
                "conflict_type": "error",
                "error": str(e),
                "metadata_file": str(metadata_file)
            }

    def scan_training_data_consistency(self, limit: int = 50) -> List[Dict]:
        """
        Scan recent training data for label inconsistencies
        
        Args:
            limit: Maximum number of files to check
            
        Returns:
            List of conflict detection results
        """
        conflicts = []
        
        try:
            # Get recent metadata files
            charts_dir = Path("training_data/charts")
            if not charts_dir.exists():
                logger.warning("Training data charts directory not found")
                return conflicts
            
            metadata_files = sorted(charts_dir.glob("*_metadata.json"))[-limit:]
            
            logger.info(f"Scanning {len(metadata_files)} metadata files for label consistency")
            
            for metadata_file in metadata_files:
                conflict_info = self.check_metadata_consistency(metadata_file)
                
                if conflict_info.get("has_conflict"):
                    conflicts.append(conflict_info)
                    logger.warning(f"Label conflict detected in {metadata_file.name}: {conflict_info['description']}")
            
            return conflicts
            
        except Exception as e:
            logger.error(f"Error scanning training data consistency: {e}")
            return conflicts

    def apply_consistency_fix(self, metadata_file: Path, corrected_label: str) -> bool:
        """
        Apply consistency fix to metadata file
        
        Args:
            metadata_file: Path to metadata file
            corrected_label: Corrected setup label
            
        Returns:
            Success status
        """
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Update labels
            metadata['setup_label'] = corrected_label
            metadata['label_corrected'] = True
            metadata['label_correction_time'] = datetime.now().isoformat()
            metadata['original_setup_label'] = metadata.get('setup_label', 'unknown')
            
            # Write updated metadata
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Applied label correction to {metadata_file.name}: {corrected_label}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying consistency fix: {e}")
            return False

def check_gpt_label_consistency(gpt_analysis: str, gpt_commentary: str, symbol: str = "unknown") -> Dict:
    """
    Convenience function to check label consistency
    
    Args:
        gpt_analysis: GPT chart analysis text
        gpt_commentary: GPT commentary text  
        symbol: Trading symbol
        
    Returns:
        Consistency check results
    """
    checker = GPTLabelConsistencyChecker()
    analysis_label, commentary_label = checker.extract_labels_from_gpt_output(gpt_analysis, gpt_commentary)
    
    if analysis_label and commentary_label:
        conflict_info = checker.detect_label_conflict(analysis_label, commentary_label)
        conflict_info["symbol"] = symbol
        conflict_info["analysis_label"] = analysis_label
        conflict_info["commentary_label"] = commentary_label
        
        if conflict_info["has_conflict"]:
            conflict_info["correction_prompt"] = checker.generate_correction_prompt(
                symbol, analysis_label, commentary_label, conflict_info
            )
        
        return conflict_info
    else:
        return {
            "has_conflict": False,
            "conflict_type": "extraction_failed",
            "symbol": symbol,
            "analysis_label": analysis_label,
            "commentary_label": commentary_label
        }

def main():
    """Test consistency checker"""
    checker = GPTLabelConsistencyChecker()
    
    # Test case from user example
    gpt_analysis = "GPT_CHART_ANALYSIS: pullback_in_trend"
    gpt_commentary = "SETUP: breakout_pattern"
    
    result = check_gpt_label_consistency(gpt_analysis, gpt_commentary, "TESTUSDT")
    
    print("=== GPT LABEL CONSISTENCY TEST ===")
    print(f"Analysis Label: {result.get('analysis_label')}")
    print(f"Commentary Label: {result.get('commentary_label')}")
    print(f"Has Conflict: {result.get('has_conflict')}")
    print(f"Conflict Type: {result.get('conflict_type')}")
    print(f"Severity: {result.get('severity')}")
    print(f"Description: {result.get('description')}")
    
    if result.get('correction_prompt'):
        print("\n=== CORRECTION PROMPT ===")
        print(result['correction_prompt'])
    
    # Scan training data
    conflicts = checker.scan_training_data_consistency(limit=20)
    print(f"\n=== TRAINING DATA SCAN ===")
    print(f"Found {len(conflicts)} conflicts in recent training data")
    
    for conflict in conflicts[:3]:  # Show first 3
        print(f"• {conflict['symbol']}: {conflict['description']} ({conflict['severity']})")

if __name__ == "__main__":
    main()