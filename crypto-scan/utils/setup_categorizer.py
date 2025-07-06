"""
Setup Type Categorization System
Organizes trading setups into MAIN, NEUTRAL, and NEGATIVE categories
for enhanced Vision-AI training data organization
"""

import os
import json
import shutil
from typing import Dict, List, Optional
from datetime import datetime

class SetupCategorizer:
    """
    Categorizes trading setups and organizes training data accordingly
    """
    
    def __init__(self):
        self.setup_categories = {
            "MAIN": [
                "breakout_pattern",
                "momentum_follow", 
                "reversal_pattern",
                "trend_continuation",
                "volume_backed_breakout",
                "bullish_momentum",
                "bearish_momentum",
                "impulse",
                "expansion"
            ],
            "NEUTRAL": [
                "range_trading",
                "consolidation", 
                "accumulation",
                "range",
                "range-accumulation",
                "compression",
                "pullback",
                "pullback-in-trend",
                "pullback-quality"
            ],
            "NEGATIVE": [
                "fake_breakout",
                "exhaustion_pattern",
                "exhaustion-pullback", 
                "distribution",
                "bear_trend",
                "setup_analysis",
                "unknown",
                "error",
                "insufficient_data",
                "no_clear_pattern",
                "chaos"
            ]
        }
        
        self.base_dirs = {
            "MAIN": "training_data/main",
            "NEUTRAL": "training_data/neutral", 
            "NEGATIVE": "training_data/negative",
            "UNCATEGORIZED": "training_data/uncategorized"
        }
        
        # Ensure directories exist
        for dir_path in self.base_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def categorize_setup(self, setup_label: str) -> str:
        """
        Categorize a setup label into MAIN, NEUTRAL, or NEGATIVE
        
        Args:
            setup_label: The setup label from GPT analysis
            
        Returns:
            Category: MAIN, NEUTRAL, NEGATIVE, or UNCATEGORIZED
        """
        if not setup_label:
            return "UNCATEGORIZED"
        
        setup_lower = setup_label.lower().strip()
        
        for category, setups in self.setup_categories.items():
            if setup_lower in [s.lower() for s in setups]:
                return category
        
        # Check for partial matches (but be more restrictive)
        for category, setups in self.setup_categories.items():
            for setup in setups:
                if len(setup_lower) > 3 and (setup.lower() in setup_lower or setup_lower in setup.lower()):
                    return category
        
        return "UNCATEGORIZED"
    
    def organize_training_chart(self, chart_path: str, metadata_path: str = None) -> Optional[str]:
        """
        Move training chart to appropriate category folder
        
        Args:
            chart_path: Path to the chart file
            metadata_path: Path to metadata file (optional)
            
        Returns:
            New chart path or None if failed
        """
        try:
            if not os.path.exists(chart_path):
                print(f"❌ [CATEGORIZER] Chart not found: {chart_path}")
                return None
            
            # Extract setup from filename or metadata
            setup_label = self._extract_setup_from_path(chart_path)
            
            if not setup_label and metadata_path and os.path.exists(metadata_path):
                setup_label = self._extract_setup_from_metadata(metadata_path)
            
            if not setup_label:
                print(f"⚠️  [CATEGORIZER] No setup label found for: {chart_path}")
                category = "UNCATEGORIZED"
            else:
                category = self.categorize_setup(setup_label)
            
            # Create destination path
            filename = os.path.basename(chart_path)
            dest_dir = self.base_dirs[category]
            dest_path = os.path.join(dest_dir, filename)
            
            # Move chart file
            shutil.move(chart_path, dest_path)
            print(f"📁 [CATEGORIZER] {filename}: {setup_label} → {category}")
            
            # Move metadata if exists
            if metadata_path and os.path.exists(metadata_path):
                metadata_filename = os.path.basename(metadata_path)
                dest_metadata = os.path.join(dest_dir, metadata_filename)
                shutil.move(metadata_path, dest_metadata)
            
            # Update category statistics
            self._update_category_stats(category, setup_label)
            
            return dest_path
            
        except Exception as e:
            print(f"❌ [CATEGORIZER] Error organizing {chart_path}: {e}")
            return None
    
    def _extract_setup_from_path(self, chart_path: str) -> Optional[str]:
        """Extract setup label from chart filename"""
        try:
            filename = os.path.basename(chart_path)
            # Pattern: SYMBOL_EXCHANGE_SETUP_score-XXX_timestamp.png
            parts = filename.replace(".png", "").split("_")
            
            if len(parts) >= 5:
                # Find setup part (should be between exchange and score)
                for i, part in enumerate(parts):
                    if part.startswith("score-"):
                        if i > 0:
                            return parts[i-1]
            
            return None
            
        except Exception as e:
            print(f"[CATEGORIZER] Error extracting setup from path: {e}")
            return None
    
    def _extract_setup_from_metadata(self, metadata_path: str) -> Optional[str]:
        """Extract setup label from metadata file"""
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return metadata.get("setup_label") or metadata.get("setup_field")
            
        except Exception as e:
            print(f"[CATEGORIZER] Error reading metadata: {e}")
            return None
    
    def _update_category_stats(self, category: str, setup_label: str):
        """Update statistics for categorized charts"""
        try:
            stats_file = "data/category_stats.json"
            
            # Load existing stats
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
            else:
                stats = {"categories": {}, "setups": {}, "last_updated": None}
            
            # Update category count
            if category not in stats["categories"]:
                stats["categories"][category] = 0
            stats["categories"][category] += 1
            
            # Update setup count
            if setup_label:
                if setup_label not in stats["setups"]:
                    stats["setups"][setup_label] = {"count": 0, "category": category}
                stats["setups"][setup_label]["count"] += 1
            
            # Update timestamp
            stats["last_updated"] = datetime.now().isoformat()
            
            # Save updated stats
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
                
        except Exception as e:
            print(f"[CATEGORIZER] Error updating stats: {e}")
    
    def get_category_distribution(self) -> Dict:
        """Get current distribution of charts across categories"""
        try:
            distribution = {}
            
            for category, dir_path in self.base_dirs.items():
                if os.path.exists(dir_path):
                    chart_count = len([f for f in os.listdir(dir_path) if f.endswith('.png')])
                    distribution[category] = chart_count
                else:
                    distribution[category] = 0
            
            return distribution
            
        except Exception as e:
            print(f"[CATEGORIZER] Error getting distribution: {e}")
            return {}
    
    def organize_existing_charts(self, source_dir: str = "training_data/charts") -> Dict[str, int]:
        """
        Organize all existing charts in source directory
        
        Args:
            source_dir: Directory containing charts to organize
            
        Returns:
            Dictionary with organization results
        """
        results = {"organized": 0, "failed": 0, "categories": {}}
        
        try:
            if not os.path.exists(source_dir):
                print(f"❌ [CATEGORIZER] Source directory not found: {source_dir}")
                return results
            
            chart_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]
            print(f"📁 [CATEGORIZER] Organizing {len(chart_files)} existing charts...")
            
            for chart_file in chart_files:
                chart_path = os.path.join(source_dir, chart_file)
                metadata_path = chart_path.replace('.png', '_metadata.json')
                
                new_path = self.organize_training_chart(chart_path, metadata_path)
                
                if new_path:
                    results["organized"] += 1
                    # Determine category from new path
                    for category, dir_path in self.base_dirs.items():
                        if new_path.startswith(dir_path):
                            results["categories"][category] = results["categories"].get(category, 0) + 1
                            break
                else:
                    results["failed"] += 1
            
            print(f"✅ [CATEGORIZER] Organization complete: {results['organized']} organized, {results['failed']} failed")
            return results
            
        except Exception as e:
            print(f"❌ [CATEGORIZER] Error organizing existing charts: {e}")
            return results

# Global categorizer instance
setup_categorizer = SetupCategorizer()

def categorize_setup_label(setup_label: str) -> str:
    """Convenience function for setup categorization"""
    return setup_categorizer.categorize_setup(setup_label)

def organize_chart_by_setup(chart_path: str, metadata_path: str = None) -> Optional[str]:
    """Convenience function for chart organization"""
    return setup_categorizer.organize_training_chart(chart_path, metadata_path)

def get_setup_distribution() -> Dict:
    """Convenience function for getting category distribution"""
    return setup_categorizer.get_category_distribution()

def organize_all_existing_charts() -> Dict[str, int]:
    """Convenience function for organizing all existing charts"""
    return setup_categorizer.organize_existing_charts()