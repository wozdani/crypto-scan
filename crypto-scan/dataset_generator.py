"""
Vision-AI Dataset Generator
Creates centralized JSONL datasets from training_data/charts/ PNG+JSON pairs
"""

import os
import json
from typing import List, Dict, Optional
from datetime import datetime


def generate_dataset_jsonl(charts_folder: str = "training_data/charts", output_path: str = "training_dataset.jsonl") -> int:
    """
    Generate centralized JSONL dataset from training charts and metadata
    
    Args:
        charts_folder: Directory containing PNG+JSON training pairs
        output_path: Output JSONL file path
        
    Returns:
        Number of training examples generated
    """
    if not os.path.exists(charts_folder):
        print(f"[DATASET] Charts folder not found: {charts_folder}")
        return 0
    
    lines = []
    processed_count = 0
    error_count = 0
    
    print(f"[DATASET] Scanning {charts_folder} for training pairs...")
    
    # Process all PNG files in training_data/charts directory
    for fname in sorted(os.listdir(charts_folder)):
        if not fname.endswith(".png"):
            continue
            
        processed_count += 1
        base_name = fname.replace(".png", "")
        json_path = os.path.join(charts_folder, base_name + ".json")
        image_path = os.path.join(charts_folder, fname)
        
        # Skip if corresponding JSON metadata doesn't exist
        if not os.path.exists(json_path):
            print(f"[DATASET] Missing metadata: {json_path}")
            error_count += 1
            continue
        
        try:
            # Load metadata from JSON file
            with open(json_path, "r") as f:
                meta = json.load(f)
            
            # Load CLIP prediction if available
            clip_prediction = None
            clip_json_path = json_path.replace(".json", "_clip.json")
            if os.path.exists(clip_json_path):
                try:
                    with open(clip_json_path, "r") as clip_f:
                        clip_data = json.load(clip_f)
                        clip_prediction = {
                            "decision": clip_data.get("decision"),
                            "confidence": clip_data.get("confidence", 0.0)
                        }
                except Exception as e:
                    print(f"[DATASET] Failed to load CLIP data for {fname}: {e}")
            
            # Create enhanced training example entry
            training_example = {
                "image_path": image_path,
                "symbol": meta.get("symbol"),
                "alerts": meta.get("alerts", []),
                "phase": meta.get("phase"),
                "setup": meta.get("setup"),
                "score": meta.get("score"),
                "decision": meta.get("decision"),
                "timestamp": meta.get("timestamp"),
                "chart_type": meta.get("chart_type", "vision_ai_training"),
                "multi_alert": meta.get("multi_alert", False),
                "alert_count": meta.get("alert_count", 0),
                "clip_prediction": clip_prediction,
                "was_correct": None,  # To be filled by feedback loop
                "alert_outcome": None  # To be filled by feedback loop
            }
            
            # Add to dataset
            lines.append(json.dumps(training_example))
            
        except Exception as e:
            print(f"[DATASET ERROR] Failed to process {json_path}: {e}")
            error_count += 1
            continue
    
    # Write JSONL dataset file
    try:
        with open(output_path, "w") as out_file:
            for line in lines:
                out_file.write(line + "\n")
        
        print(f"[DATASET SUCCESS] Generated: {output_path}")
        print(f"[DATASET STATS] {len(lines)} examples, {error_count} errors, {processed_count} files processed")
        
        return len(lines)
        
    except Exception as e:
        print(f"[DATASET ERROR] Failed to write {output_path}: {e}")
        return 0


def analyze_dataset_distribution(jsonl_path: str = "training_dataset.jsonl") -> Dict:
    """
    Analyze the distribution of training data for Vision-AI models
    
    Args:
        jsonl_path: Path to JSONL dataset file
        
    Returns:
        Distribution statistics
    """
    if not os.path.exists(jsonl_path):
        print(f"[ANALYSIS] Dataset not found: {jsonl_path}")
        return {}
    
    stats = {
        "total_examples": 0,
        "phases": {},
        "decisions": {},
        "setups": {},
        "symbols": {},
        "multi_alert_ratio": 0,
        "score_distribution": {"high": 0, "medium": 0, "low": 0}
    }
    
    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                    
                data = json.loads(line)
                stats["total_examples"] += 1
                
                # Count distributions
                phase = data.get("phase", "unknown")
                stats["phases"][phase] = stats["phases"].get(phase, 0) + 1
                
                decision = data.get("decision", "unknown")
                stats["decisions"][decision] = stats["decisions"].get(decision, 0) + 1
                
                setup = data.get("setup", "unknown")
                stats["setups"][setup] = stats["setups"].get(setup, 0) + 1
                
                symbol = data.get("symbol", "unknown")
                stats["symbols"][symbol] = stats["symbols"].get(symbol, 0) + 1
                
                # Multi-alert analysis
                if data.get("multi_alert", False):
                    stats["multi_alert_ratio"] += 1
                
                # Score distribution
                score = data.get("score", 0)
                if score >= 0.7:
                    stats["score_distribution"]["high"] += 1
                elif score >= 0.5:
                    stats["score_distribution"]["medium"] += 1
                else:
                    stats["score_distribution"]["low"] += 1
        
        # Calculate ratios
        if stats["total_examples"] > 0:
            stats["multi_alert_ratio"] = stats["multi_alert_ratio"] / stats["total_examples"]
        
        return stats
        
    except Exception as e:
        print(f"[ANALYSIS ERROR] Failed to analyze {jsonl_path}: {e}")
        return {}


def validate_dataset_quality(jsonl_path: str = "training_dataset.jsonl") -> bool:
    """
    Validate the quality and completeness of training dataset
    
    Args:
        jsonl_path: Path to JSONL dataset file
        
    Returns:
        True if dataset meets quality standards
    """
    if not os.path.exists(jsonl_path):
        print(f"[VALIDATION] Dataset not found: {jsonl_path}")
        return False
    
    issues = []
    valid_examples = 0
    total_examples = 0
    
    required_fields = ["image_path", "symbol", "phase", "decision", "score"]
    
    try:
        with open(jsonl_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                total_examples += 1
                
                try:
                    data = json.loads(line)
                    
                    # Check required fields
                    missing_fields = [field for field in required_fields if not data.get(field)]
                    if missing_fields:
                        issues.append(f"Line {line_num}: Missing fields {missing_fields}")
                        continue
                    
                    # Validate image file exists
                    if not os.path.exists(data["image_path"]):
                        issues.append(f"Line {line_num}: Image file not found {data['image_path']}")
                        continue
                    
                    # Validate score range
                    score = data.get("score", 0)
                    if not (0 <= score <= 1):
                        issues.append(f"Line {line_num}: Invalid score {score}")
                        continue
                    
                    valid_examples += 1
                    
                except json.JSONDecodeError:
                    issues.append(f"Line {line_num}: Invalid JSON format")
                    continue
        
        # Report validation results
        quality_ratio = valid_examples / total_examples if total_examples > 0 else 0
        
        print(f"[VALIDATION] Dataset quality: {quality_ratio:.2%}")
        print(f"[VALIDATION] Valid examples: {valid_examples}/{total_examples}")
        
        if issues:
            print(f"[VALIDATION] Found {len(issues)} issues:")
            for issue in issues[:10]:  # Show first 10 issues
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more issues")
        
        return quality_ratio >= 0.9  # 90% quality threshold
        
    except Exception as e:
        print(f"[VALIDATION ERROR] Failed to validate {jsonl_path}: {e}")
        return False


def main():
    """Test dataset generation and analysis"""
    print("=== VISION-AI DATASET GENERATOR ===")
    
    # Generate dataset
    examples_count = generate_dataset_jsonl()
    
    if examples_count > 0:
        # Analyze distribution
        print("\n=== DATASET ANALYSIS ===")
        stats = analyze_dataset_distribution()
        
        if stats:
            print(f"Total examples: {stats['total_examples']}")
            print(f"Phases: {dict(list(stats['phases'].items())[:5])}")
            print(f"Decisions: {dict(list(stats['decisions'].items())[:5])}")
            print(f"Multi-alert ratio: {stats['multi_alert_ratio']:.2%}")
        
        # Validate quality
        print("\n=== DATASET VALIDATION ===")
        is_valid = validate_dataset_quality()
        print(f"Dataset validation: {'PASSED' if is_valid else 'FAILED'}")


if __name__ == "__main__":
    main()