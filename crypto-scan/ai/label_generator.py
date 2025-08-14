"""
AI Label Generator for Vision Training
Generates labels for chart images using GPT or local analysis
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


class VisionLabelGenerator:
    """Generates training labels for chart images"""
    
    def __init__(self):
        self.client = None
        if OPENAI_AVAILABLE:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = OpenAI(api_key=api_key)
        
        # Define label categories
        self.setup_types = [
            "breakout_with_pullback",
            "exhaustion",
            "fakeout", 
            "range_accumulation",
            "trend_continuation",
            "reversal_pattern",
            "support_retest",
            "resistance_rejection"
        ]
        
        self.phase_types = [
            "breakout-continuation",
            "exhaustion-pullback",
            "consolidation-range",
            "trend-acceleration", 
            "reversal-formation",
            "support-bounce",
            "resistance-test"
        ]
    
    def generate_label_gpt(self, image_path: str, context_features: Dict) -> Dict:
        """
        Generate labels using GPT Vision API
        
        Args:
            image_path: Path to chart image
            context_features: Market context and scoring data
            
        Returns:
            Dict with setup_type, phase_type, and confidence
        """
        try:
            if not self.client:
                return self._generate_local_label(image_path, context_features)
            
            if not os.path.exists(image_path):
                return {"error": "Image file not found"}
            
            # Encode image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Create context string
            context_str = self._format_context(context_features)
            
            # GPT prompt for labeling
            prompt = f"""Na podstawie wykresu i kontekstu scoringu:
{context_str}

Podaj tylko w formacie JSON:
{{
    "phase_type": "jedna z: {', '.join(self.phase_types)}",
    "setup_type": "jedna z: {', '.join(self.setup_types)}",
    "confidence": 0.0-1.0
}}

Bez wyjaśnień."""
            
            # Call GPT-4o Vision for reliable chart analysis
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Using GPT-4o for reliable chart pattern recognition capabilities
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
                max_completion_tokens=150,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            # Validate and clean response
            cleaned_result = self._validate_labels(result)
            cleaned_result["timestamp"] = datetime.now(timezone.utc).isoformat()
            cleaned_result["labeler"] = "gpt-4o-vision"
            
            print(f"[LABEL GEN] GPT labeled: {cleaned_result['phase_type']} / {cleaned_result['setup_type']} ({cleaned_result['confidence']:.2f})")
            return cleaned_result
            
        except Exception as e:
            print(f"[LABEL GEN] GPT labeling failed: {e}")
            return self._generate_local_label(image_path, context_features)
    
    def _format_context(self, features: Dict) -> str:
        """Format context features for GPT prompt"""
        context_lines = []
        for key, value in features.items():
            if isinstance(value, float):
                context_lines.append(f"- {key}: {value:.3f}")
            else:
                context_lines.append(f"- {key}: {value}")
        return "\n".join(context_lines)
    
    def _validate_labels(self, result: Dict) -> Dict:
        """Validate and clean GPT response"""
        cleaned = {
            "phase_type": "consolidation-range",  # default
            "setup_type": "range_accumulation",   # default
            "confidence": 0.5                     # default
        }
        
        # Validate phase_type
        if result.get("phase_type") in self.phase_types:
            cleaned["phase_type"] = result["phase_type"]
        
        # Validate setup_type
        if result.get("setup_type") in self.setup_types:
            cleaned["setup_type"] = result["setup_type"]
        
        # Validate confidence
        try:
            conf = float(result.get("confidence", 0.5))
            cleaned["confidence"] = max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            cleaned["confidence"] = 0.5
        
        return cleaned
    
    def _generate_local_label(self, image_path: str, context_features: Dict) -> Dict:
        """
        Generate labels using local heuristics when GPT is not available
        
        Args:
            image_path: Path to chart image
            context_features: Market context
            
        Returns:
            Local label estimation
        """
        try:
            # Extract symbol from path for pattern detection
            filename = os.path.basename(image_path)
            
            # Analyze context features
            trend_strength = context_features.get("trend_strength", 0.5)
            pullback_quality = context_features.get("pullback_quality", 0.5)
            phase_score = context_features.get("phase_score", 0.5)
            
            # Heuristic labeling based on features
            if trend_strength > 0.7 and phase_score > 0.7:
                phase_type = "trend-acceleration"
                setup_type = "trend_continuation"
                confidence = 0.7
            elif pullback_quality > 0.8 and trend_strength > 0.6:
                phase_type = "breakout-continuation"
                setup_type = "breakout_with_pullback"
                confidence = 0.75
            elif trend_strength < 0.4:
                phase_type = "consolidation-range"
                setup_type = "range_accumulation"
                confidence = 0.6
            else:
                phase_type = "consolidation-range"
                setup_type = "range_accumulation"
                confidence = 0.5
            
            result = {
                "phase_type": phase_type,
                "setup_type": setup_type,
                "confidence": confidence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "labeler": "local-heuristic"
            }
            
            print(f"[LABEL GEN] Local label: {phase_type} / {setup_type} ({confidence:.2f})")
            return result
            
        except Exception as e:
            print(f"[LABEL GEN] Local labeling failed: {e}")
            return {
                "phase_type": "consolidation-range",
                "setup_type": "range_accumulation", 
                "confidence": 0.3,
                "error": str(e)
            }
    
    def generate_batch_labels(
        self, 
        image_paths: List[str], 
        context_list: List[Dict]
    ) -> List[Dict]:
        """Generate labels for multiple images"""
        results = []
        
        for i, image_path in enumerate(image_paths):
            context = context_list[i] if i < len(context_list) else {}
            label = self.generate_label_gpt(image_path, context)
            results.append({
                "image_path": image_path,
                "labels": label
            })
        
        return results
    
    def save_labels_jsonl(self, labels_data: List[Dict], output_file: str = "labels.jsonl"):
        """Save labels to JSONL format for training"""
        try:
            with open(output_file, 'w') as f:
                for entry in labels_data:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            print(f"[LABEL GEN] Saved {len(labels_data)} labels to {output_file}")
            return True
            
        except Exception as e:
            print(f"[LABEL GEN] Failed to save labels: {e}")
            return False


# Global instance
label_generator = VisionLabelGenerator()


def generate_label_gpt(image_path: str, context_features: Dict) -> Dict:
    """
    Main function for label generation
    
    Args:
        image_path: Path to chart image
        context_features: Market context and scoring
        
    Returns:
        Labels with setup_type, phase_type, confidence
    """
    return label_generator.generate_label_gpt(image_path, context_features)


def parse_labels(response: str) -> Dict:
    """Parse GPT response into structured labels"""
    try:
        # Try to parse as JSON first
        return json.loads(response)
    except json.JSONDecodeError:
        # Fallback parsing for non-JSON responses
        lines = response.strip().split('\n')
        result = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip('- ').strip()
                value = value.strip()
                
                if key.lower() in ['phase_type', 'setup_type']:
                    result[key.lower()] = value
                elif key.lower() == 'confidence':
                    try:
                        result['confidence'] = float(value)
                    except ValueError:
                        result['confidence'] = 0.5
        
        return result