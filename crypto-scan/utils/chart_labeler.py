"""
Intelligent Chart Labeling System using OpenAI Vision API
Automatically classifies chart patterns for Computer Vision training
"""

import os
import json
import base64
from datetime import datetime, timezone
from typing import Dict, Optional, List
from pathlib import Path

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI not available. Install with: pip install openai")


class ChartLabeler:
    """OpenAI Vision-based chart pattern labeler"""
    
    def __init__(self):
        self.client = None
        if OPENAI_AVAILABLE:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                print("⚠️ OPENAI_API_KEY not found in environment")
        
        # Supported chart pattern classes
        self.pattern_classes = [
            "breakout_with_pullback",
            "clean_pullback_in_trend", 
            "trend_exhaustion",
            "fake_breakout",
            "range_accumulation",
            "pump_after_sweep",
            "choppy_consolidation"
        ]
        
        # Dataset file for storing labeled data
        self.dataset_file = Path("data/training_dataset.jsonl")
        self.dataset_file.parent.mkdir(parents=True, exist_ok=True)
    
    def encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """Encode image file to base64 string"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"[CHART LABELER] ❌ Failed to encode image {image_path}: {e}")
            return None
    
    def create_vision_prompt(self, features: Dict) -> str:
        """Create GPT Vision prompt with market features"""
        
        # Format features for display
        features_text = []
        for key, value in features.items():
            if isinstance(value, bool):
                features_text.append(f"- {key}: {value}")
            elif isinstance(value, float):
                features_text.append(f"- {key}: {value:.2f}")
            else:
                features_text.append(f"- {key}: {value}")
        
        features_str = "\n".join(features_text)
        classes_str = "\n".join([f"- {cls}" for cls in self.pattern_classes])
        
        prompt = f"""🖼️ Załączony obraz wykresu: SYMBOL (15m)

📊 Dodatkowe dane:
{features_str}

📌 Instrukcja:
Odpowiedz tylko jedną etykietą, która najlepiej opisuje sytuację na wykresie. Masz do wyboru jedną z poniższych klas:

{classes_str}

🔁 Odpowiedz tylko i wyłącznie nazwą klasy. Bez żadnych dodatków."""

        return prompt
    
    def label_chart_image(self, image_path: str, features: Dict) -> str:
        """
        Wysyła wykres z dodatkowymi danymi rynkowymi do OpenAI Vision i odbiera jedną nazwę klasy
        
        Args:
            image_path: Ścieżka do lokalnego pliku PNG wykresu
            features: Dict z danymi dodatkowymi (trend_strength, pullback_quality, etc.)
            
        Returns:
            Nazwa klasy wzorca lub 'unknown' jeśli błąd
        """
        try:
            if not OPENAI_AVAILABLE or not self.client:
                print("[CHART LABELER] ❌ OpenAI client not available")
                return "unknown"
            
            if not os.path.exists(image_path):
                print(f"[CHART LABELER] ❌ Image file not found: {image_path}")
                return "unknown"
            
            print(f"[CHART LABELER] 🔍 Analyzing chart: {os.path.basename(image_path)}")
            
            # Encode image to base64
            base64_image = self.encode_image_to_base64(image_path)
            if not base64_image:
                return "unknown"
            
            # Create prompt with features
            prompt = self.create_vision_prompt(features)
            
            # Call OpenAI Vision API with GPT-5
            response = self.client.chat.completions.create(
                model="gpt-5",  # Upgraded to GPT-5 for enhanced chart pattern recognition capabilities
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                temperature=1.0,
                max_completion_tokens=20  # Short response expected
            )
            
            # Extract label from response
            label = response.choices[0].message.content.strip().lower()
            
            # Validate label against known classes
            if label in self.pattern_classes:
                print(f"[CHART LABELER] ✅ Classified as: {label}")
                return label
            else:
                # Try to find closest match
                for pattern_class in self.pattern_classes:
                    if pattern_class.replace("_", "").replace("-", "") in label.replace("_", "").replace("-", ""):
                        print(f"[CHART LABELER] ✅ Matched to: {pattern_class} (from: {label})")
                        return pattern_class
                
                print(f"[CHART LABELER] ⚠️ Unknown label returned: {label}, using 'choppy_consolidation'")
                return "choppy_consolidation"  # Default fallback
            
        except Exception as e:
            print(f"[CHART LABELER] ❌ Vision API error: {e}")
            return "unknown"
    
    def save_labeled_data(
        self, 
        symbol: str, 
        label: str, 
        features: Dict, 
        image_path: str
    ) -> bool:
        """
        Save labeled data to JSONL training dataset
        
        Args:
            symbol: Trading symbol
            label: Pattern classification
            features: Market features used
            image_path: Path to chart image
            
        Returns:
            True if saved successfully
        """
        try:
            data_entry = {
                "symbol": symbol,
                "label": label,
                "features": features,
                "image": os.path.basename(image_path),
                "image_path": str(image_path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "labeler": "openai_vision_gpt4o"
            }
            
            # Append to JSONL file
            with open(self.dataset_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data_entry, ensure_ascii=False) + '\n')
            
            print(f"[CHART LABELER] 💾 Saved labeled data for {symbol}: {label}")
            return True
            
        except Exception as e:
            print(f"[CHART LABELER] ❌ Failed to save data: {e}")
            return False
    
    def label_and_save(self, image_path: str, features: Dict, symbol: str = None) -> str:
        """
        Complete workflow: label chart and save to dataset
        
        Args:
            image_path: Path to chart image
            features: Market features
            symbol: Trading symbol (extracted from filename if None)
            
        Returns:
            Classification label
        """
        try:
            # Extract symbol from filename if not provided
            if not symbol:
                filename = os.path.basename(image_path)
                symbol = filename.split('_')[0] if '_' in filename else "UNKNOWN"
            
            # Get classification from Vision API
            label = self.label_chart_image(image_path, features)
            
            # Save to dataset if classification successful
            if label != "unknown":
                self.save_labeled_data(symbol, label, features, image_path)
            
            return label
            
        except Exception as e:
            print(f"[CHART LABELER] ❌ Complete workflow failed: {e}")
            return "unknown"
    
    def batch_label_charts(self, chart_directory: str, features_list: List[Dict] = None) -> Dict:
        """
        Label multiple charts in a directory
        
        Args:
            chart_directory: Directory containing chart images
            features_list: Optional list of features for each chart
            
        Returns:
            Dict with labeling results
        """
        try:
            chart_dir = Path(chart_directory)
            if not chart_dir.exists():
                print(f"[CHART LABELER] ❌ Directory not found: {chart_directory}")
                return {}
            
            # Find all PNG images
            chart_files = list(chart_dir.glob("*.png"))
            
            if not chart_files:
                print(f"[CHART LABELER] ⚠️ No PNG files found in {chart_directory}")
                return {}
            
            print(f"[CHART LABELER] 📊 Processing {len(chart_files)} charts...")
            
            results = {}
            
            for i, chart_file in enumerate(chart_files):
                try:
                    # Use provided features or create default ones
                    if features_list and i < len(features_list):
                        features = features_list[i]
                    else:
                        # Create default features based on filename pattern detection
                        features = self._extract_features_from_filename(chart_file.name)
                    
                    label = self.label_and_save(str(chart_file), features)
                    results[chart_file.name] = {
                        "label": label,
                        "features": features,
                        "status": "success" if label != "unknown" else "failed"
                    }
                    
                    print(f"[BATCH] {i+1}/{len(chart_files)}: {chart_file.name} → {label}")
                    
                except Exception as e:
                    print(f"[BATCH] ❌ Failed to process {chart_file.name}: {e}")
                    results[chart_file.name] = {"status": "error", "error": str(e)}
            
            return results
            
        except Exception as e:
            print(f"[CHART LABELER] ❌ Batch processing failed: {e}")
            return {}
    
    def _extract_features_from_filename(self, filename: str) -> Dict:
        """Extract features from filename patterns"""
        features = {
            "trend_strength": 0.6,
            "pullback_quality": 0.5,
            "liquidity_score": 0.7,
            "htf_trend_match": True,
            "phase": "unknown"
        }
        
        # Detect pattern from filename
        filename_lower = filename.lower()
        if "breakout" in filename_lower:
            features.update({
                "trend_strength": 0.8,
                "phase": "breakout-continuation"
            })
        elif "pullback" in filename_lower:
            features.update({
                "pullback_quality": 0.9,
                "phase": "pullback-setup"
            })
        elif "exhaustion" in filename_lower:
            features.update({
                "trend_strength": 0.3,
                "phase": "trend-exhaustion"
            })
        
        return features
    
    def get_dataset_stats(self) -> Dict:
        """Get statistics about the labeled dataset"""
        try:
            if not self.dataset_file.exists():
                return {"total_entries": 0, "labels": {}}
            
            label_counts = {}
            total_entries = 0
            
            with open(self.dataset_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            label = entry.get("label", "unknown")
                            label_counts[label] = label_counts.get(label, 0) + 1
                            total_entries += 1
                        except json.JSONDecodeError:
                            continue
            
            return {
                "total_entries": total_entries,
                "labels": label_counts,
                "dataset_file": str(self.dataset_file)
            }
            
        except Exception as e:
            print(f"[CHART LABELER] ❌ Failed to get stats: {e}")
            return {"error": str(e)}


# Global instance
chart_labeler = ChartLabeler()


def label_chart_image(image_path: str, features: Dict) -> str:
    """
    Main function for chart labeling
    
    Args:
        image_path: Ścieżka do lokalnego pliku PNG wykresu
        features: Dict z danymi dodatkowymi
        
    Returns:
        Nazwa klasy wzorca
    """
    return chart_labeler.label_chart_image(image_path, features)