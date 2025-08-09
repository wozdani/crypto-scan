#!/usr/bin/env python3
"""
GPT Chart Labeler for Vision-AI Training
Generates setup descriptions using GPT for training data
"""

import os
import sys
import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class GPTChartLabeler:
    """Generates setup descriptions for charts using GPT"""
    
    def __init__(self):
        self.client = None
        if OPENAI_AVAILABLE:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = OpenAI(api_key=api_key)
        
        # Setup categories for consistent labeling
        self.setup_patterns = [
            "breakout-continuation",
            "retest-on-support", 
            "volume-backed-move",
            "pullback-in-trend",
            "exhaustion-pattern",
            "fake-breakout",
            "range-accumulation",
            "support-rejection",
            "resistance-break",
            "trend-reversal"
        ]
        
        self.market_phases = [
            "trending-up",
            "trending-down", 
            "consolidation",
            "breakout-phase",
            "pullback-phase",
            "exhaustion-phase"
        ]
    
    def generate_chart_description(
        self, 
        image_path: str, 
        context_data: Dict = None
    ) -> Optional[str]:
        """
        Generate setup description for chart using GPT Vision
        
        Args:
            image_path: Path to chart image
            context_data: Additional context from TJDE analysis
            
        Returns:
            Setup description string or None if failed
        """
        try:
            if not self.client:
                return self._generate_fallback_description(context_data)
            
            if not os.path.exists(image_path):
                print(f"[GPT LABELER] Image not found: {image_path}")
                return None
            
            # Encode image to base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Create context string
            context_str = self._format_context_data(context_data) if context_data else ""
            
            # GPT prompt for setup description
            prompt = f"""Analyze this cryptocurrency chart and provide a concise setup description.

Available setup patterns: {' | '.join(self.setup_patterns)}
Available market phases: {' | '.join(self.market_phases)}

{context_str}

Provide response in format: [setup-pattern] | [market-phase] | [additional-context]

Example: breakout-continuation | trending-up | volume-backed

Keep it concise and technical. Focus on the most obvious pattern visible on the chart."""
            
            # Call GPT Vision API
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
                max_completion_tokens=100
            )
            
            # Extract and clean description
            description = response.choices[0].message.content.strip()
            
            # Validate format
            cleaned_description = self._validate_and_clean_description(description)
            
            print(f"[GPT LABELER] Generated: {cleaned_description}")
            return cleaned_description
            
        except Exception as e:
            print(f"[GPT LABELER] GPT labeling failed: {e}")
            return self._generate_fallback_description(context_data)
    
    def _format_context_data(self, context_data: Dict) -> str:
        """Format context data for GPT prompt"""
        if not context_data:
            return ""
        
        context_lines = ["Context from technical analysis:"]
        
        # Add relevant TJDE data
        if "final_score" in context_data:
            context_lines.append(f"- Final score: {context_data['final_score']:.2f}")
        
        if "decision" in context_data:
            context_lines.append(f"- Decision: {context_data['decision']}")
        
        if "market_structure" in context_data:
            ms = context_data["market_structure"]
            if "trend_direction" in ms:
                context_lines.append(f"- Trend: {ms['trend_direction']}")
        
        if "candle_behavior" in context_data:
            cb = context_data["candle_behavior"]
            if "pattern_strength" in cb:
                context_lines.append(f"- Pattern strength: {cb['pattern_strength']:.2f}")
        
        return "\n".join(context_lines)
    
    def _validate_and_clean_description(self, description: str) -> str:
        """Validate and clean GPT description"""
        try:
            # Remove extra text and keep main description
            lines = description.split('\n')
            main_desc = lines[0].strip()
            
            # Clean common GPT prefixes
            prefixes_to_remove = [
                "Based on the chart analysis:",
                "The chart shows:",
                "This appears to be:",
                "Looking at the chart:"
            ]
            
            for prefix in prefixes_to_remove:
                if main_desc.startswith(prefix):
                    main_desc = main_desc[len(prefix):].strip()
            
            # Ensure it follows the pattern format
            if " | " not in main_desc:
                # Try to format it properly
                parts = main_desc.split()
                if len(parts) >= 2:
                    setup = parts[0].replace("_", "-")
                    phase = parts[1].replace("_", "-") if len(parts) > 1 else "trending-up"
                    context = " ".join(parts[2:]) if len(parts) > 2 else "technical-setup"
                    main_desc = f"{setup} | {phase} | {context}"
            
            return main_desc
            
        except Exception as e:
            print(f"[GPT LABELER] Description cleaning failed: {e}")
            return description
    
    def _generate_fallback_description(self, context_data: Dict = None) -> str:
        """Generate fallback description when GPT is not available"""
        try:
            if not context_data:
                return "consolidation | trending-up | technical-setup"
            
            # Analyze context to generate reasonable description
            final_score = context_data.get("final_score", 0.5)
            decision = context_data.get("decision", "wait")
            
            if final_score > 0.7:
                if "trend" in decision or "join" in decision:
                    return "breakout-continuation | trending-up | high-conviction"
                else:
                    return "volume-backed-move | trending-up | strong-signal"
            elif final_score > 0.5:
                return "pullback-in-trend | consolidation | medium-conviction"
            else:
                return "range-accumulation | consolidation | wait-setup"
                
        except Exception as e:
            print(f"[GPT LABELER] Fallback generation failed: {e}")
            return "consolidation | trending-up | technical-setup"
    
    def save_chart_label(
        self, 
        chart_path: str, 
        description: str,
        context_data: Dict = None
    ) -> Optional[str]:
        """
        Save chart description to corresponding text file
        
        Args:
            chart_path: Path to chart image
            description: Setup description
            context_data: Additional context data
            
        Returns:
            Path to saved text file or None if failed
        """
        try:
            # Create text file path (same name, .txt extension)
            chart_file = Path(chart_path)
            text_file = chart_file.with_suffix('.txt')
            
            # Prepare content
            content_lines = [description]
            
            # Add context data as comments
            if context_data:
                content_lines.append("")
                content_lines.append("# Context data:")
                
                for key, value in context_data.items():
                    if isinstance(value, (int, float, str, bool)):
                        content_lines.append(f"# {key}: {value}")
            
            # Add metadata
            content_lines.append("")
            content_lines.append(f"# Generated: {datetime.now().isoformat()}")
            content_lines.append(f"# Chart: {chart_file.name}")
            
            # Write to file
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(content_lines))
            
            print(f"[GPT LABELER] Saved label: {text_file.name}")
            return str(text_file)
            
        except Exception as e:
            print(f"[GPT LABELER] Failed to save label: {e}")
            return None
    
    def label_chart_with_context(
        self, 
        chart_path: str, 
        context_data: Dict = None
    ) -> Dict:
        """
        Complete labeling workflow: generate description and save
        
        Args:
            chart_path: Path to chart image
            context_data: TJDE analysis context
            
        Returns:
            Labeling results
        """
        try:
            # Generate description
            description = self.generate_chart_description(chart_path, context_data)
            
            if not description:
                return {"error": "Failed to generate description"}
            
            # Save label
            text_path = self.save_chart_label(chart_path, description, context_data)
            
            if not text_path:
                return {"error": "Failed to save label"}
            
            result = {
                "chart_path": chart_path,
                "text_path": text_path,
                "description": description,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_labeling_stats(self, charts_dir: str = "data/charts") -> Dict:
        """Get statistics about chart labeling"""
        try:
            chart_path = Path(charts_dir)
            
            if not chart_path.exists():
                return {"total_charts": 0, "labeled_charts": 0}
            
            chart_files = list(chart_path.glob("*.png"))
            text_files = list(chart_path.glob("*.txt"))
            
            # Count matching pairs
            labeled_count = 0
            for chart_file in chart_files:
                text_file = chart_file.with_suffix('.txt')
                if text_file.exists():
                    labeled_count += 1
            
            return {
                "total_charts": len(chart_files),
                "total_labels": len(text_files),
                "labeled_charts": labeled_count,
                "labeling_rate": (labeled_count / len(chart_files)) if chart_files else 0.0,
                "charts_dir": str(chart_path)
            }
            
        except Exception as e:
            return {"error": str(e)}


# Global labeler instance
gpt_labeler = GPTChartLabeler()


def label_chart(chart_path: str, context_data: Dict = None) -> Dict:
    """
    Main function for chart labeling
    
    Args:
        chart_path: Path to chart image
        context_data: TJDE analysis context
        
    Returns:
        Labeling results
    """
    return gpt_labeler.label_chart_with_context(chart_path, context_data)


def main():
    """Test GPT chart labeling"""
    print("🤖 GPT Chart Labeler Test")
    print("=" * 30)
    
    # Check if GPT is available
    if not gpt_labeler.client:
        print("⚠️ GPT not available, using fallback mode")
    else:
        print("✅ GPT client initialized")
    
    # Test fallback description
    test_context = {
        "final_score": 0.75,
        "decision": "join_trend",
        "symbol": "TESTUSDT"
    }
    
    fallback_desc = gpt_labeler._generate_fallback_description(test_context)
    print(f"📝 Fallback description: {fallback_desc}")
    
    # Show labeling statistics
    stats = gpt_labeler.get_labeling_stats()
    print(f"\n📊 Labeling Statistics:")
    print(f"  Total charts: {stats.get('total_charts', 0)}")
    print(f"  Labeled charts: {stats.get('labeled_charts', 0)}")
    print(f"  Labeling rate: {stats.get('labeling_rate', 0):.1%}")


if __name__ == "__main__":
    main()