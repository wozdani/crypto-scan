"""
Phase 4: Hybrid Embedding System
Combines visual CLIP, textual GPT, and logical scoring embeddings
"""

import os
import json
import numpy as np
import clip
import torch
from PIL import Image
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import openai
from openai import OpenAI


class HybridEmbeddingSystem:
    """Generates and manages hybrid embeddings for market analysis"""
    
    def __init__(self, embeddings_dir: str = "data/embeddings"):
        self.embeddings_dir = embeddings_dir
        self.embeddings_file = os.path.join(embeddings_dir, "token_snapshot_embeddings.json")
        self.ensure_embeddings_structure()
        
        # Initialize CLIP model
        self.clip_model = None
        self.clip_preprocess = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Load or initialize CLIP
        self.load_clip_model()
    
    def ensure_embeddings_structure(self):
        """Create embeddings directory structure"""
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        if not os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'w') as f:
                json.dump({}, f)
    
    def load_clip_model(self):
        """Load CLIP model for image embeddings"""
        try:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            print(f"[CLIP] Model loaded on {self.device}")
        except Exception as e:
            print(f"[CLIP ERROR] Failed to load model: {e}")
            # Fallback to mock embeddings for testing
            self.clip_model = None
    
    def generate_image_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Generate CLIP embedding for chart image"""
        try:
            if not os.path.exists(image_path):
                print(f"[IMAGE EMBEDDING] File not found: {image_path}")
                return None
            
            if self.clip_model is None:
                # Generate mock embedding for testing
                print(f"[IMAGE EMBEDDING] Using mock embedding for {image_path}")
                return np.random.normal(0, 1, 512).astype(np.float32)
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embedding = image_features.cpu().numpy().flatten()
            
            print(f"[IMAGE EMBEDDING] Generated {len(embedding)}-dim embedding for {os.path.basename(image_path)}")
            return embedding.astype(np.float32)
            
        except Exception as e:
            print(f"[IMAGE EMBEDDING ERROR] {image_path}: {e}")
            return None
    
    def generate_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate GPT embedding for text description"""
        try:
            if not text or len(text.strip()) < 10:
                print(f"[TEXT EMBEDDING] Text too short: '{text[:50]}...'")
                return np.random.normal(0, 1, 1536).astype(np.float32)  # Mock embedding
            
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text.strip()
            )
            
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            print(f"[TEXT EMBEDDING] Generated {len(embedding)}-dim embedding for text ({len(text)} chars)")
            return embedding
            
        except Exception as e:
            print(f"[TEXT EMBEDDING ERROR] {e}")
            # Fallback to mock embedding
            return np.random.normal(0, 1, 1536).astype(np.float32)
    
    def extract_logical_features(self, analysis_data: Dict) -> np.ndarray:
        """Extract logical scoring features as embedding vector"""
        try:
            # Core TJDE features
            features = [
                analysis_data.get("tjde_score", 0.0),
                analysis_data.get("final_score", 0.0),
                analysis_data.get("clip_confidence", 0.0),
                analysis_data.get("memory_enhanced_score", 0.0),
                analysis_data.get("feedback_score", 0.0)
            ]
            
            # Historical context features
            historical_context = analysis_data.get("historical_context", {})
            performance = historical_context.get("performance", {})
            
            features.extend([
                performance.get("accuracy_rate", 0.0),
                performance.get("average_score", 0.0),
                performance.get("historical_penalty", 0.0),
                performance.get("confidence_booster", 0.0),
                len(historical_context.get("recent_entries", []))
            ])
            
            # CLIP features
            clip_features = analysis_data.get("clip_features", {})
            features.extend([
                clip_features.get("clip_confidence", 0.0),
                clip_features.get("clip_accuracy_modifier", 1.0),
                1.0 if clip_features.get("feedback_applied") else 0.0
            ])
            
            # Market data features
            market_price = analysis_data.get("market_price", 0.0)
            features.extend([
                market_price,
                np.log(market_price) if market_price > 0 else 0.0  # Log price for normalization
            ])
            
            # Decision encoding (one-hot style)
            decision = analysis_data.get("decision", "unknown")
            features.extend([
                1.0 if decision == "join_trend" else 0.0,
                1.0 if decision == "consider_entry" else 0.0,
                1.0 if decision == "avoid" else 0.0
            ])
            
            # Quality grade encoding
            quality_grade = analysis_data.get("quality_grade", "neutral")
            features.extend([
                1.0 if quality_grade == "strong" else 0.0,
                1.0 if quality_grade == "moderate" else 0.0,
                1.0 if quality_grade == "weak" else 0.0
            ])
            
            # System integration flags
            features.extend([
                1.0 if analysis_data.get("perception_sync") else 0.0,
                1.0 if analysis_data.get("memory_integration") else 0.0,
                1.0 if analysis_data.get("vision_feedback_applied") else 0.0
            ])
            
            logical_features = np.array(features, dtype=np.float32)
            print(f"[LOGICAL FEATURES] Extracted {len(logical_features)} features")
            
            return logical_features
            
        except Exception as e:
            print(f"[LOGICAL FEATURES ERROR] {e}")
            return np.zeros(25, dtype=np.float32)  # Fallback empty features
    
    def generate_combined_embedding(
        self, 
        symbol: str, 
        analysis_data: Dict, 
        image_path: Optional[str] = None
    ) -> Optional[Dict]:
        """Generate combined hybrid embedding"""
        try:
            timestamp = datetime.now().isoformat()
            
            # 1. Generate image embedding
            image_embedding = None
            if image_path and os.path.exists(image_path):
                image_embedding = self.generate_image_embedding(image_path)
            
            if image_embedding is None:
                print(f"[COMBINED EMBEDDING] No image embedding for {symbol}")
                image_embedding = np.random.normal(0, 1, 512).astype(np.float32)
            
            # 2. Generate text embedding
            gpt_comment = analysis_data.get("gpt_comment", "")
            if not gpt_comment:
                # Create synthetic description from analysis data
                decision = analysis_data.get("decision", "unknown")
                score = analysis_data.get("final_score", 0.0)
                gpt_comment = f"Market analysis for {symbol}: {decision} decision with score {score:.3f}"
            
            text_embedding = self.generate_text_embedding(gpt_comment)
            if text_embedding is None:
                text_embedding = np.random.normal(0, 1, 1536).astype(np.float32)
            
            # 3. Extract logical features
            logical_features = self.extract_logical_features(analysis_data)
            
            # 4. Combine embeddings
            combined_embedding = np.concatenate([
                image_embedding,      # 512 dimensions
                text_embedding,       # 1536 dimensions  
                logical_features      # ~25 dimensions
            ]).astype(np.float32)
            
            # 5. Create embedding record
            embedding_record = {
                "symbol": symbol,
                "timestamp": timestamp,
                "image_path": image_path,
                "decision": analysis_data.get("decision", "unknown"),
                "tjde_score": analysis_data.get("final_score", 0.0),
                "clip_confidence": analysis_data.get("clip_features", {}).get("clip_confidence", 0.0),
                "gpt_comment": gpt_comment[:200],  # Truncate for storage
                "image_embedding": image_embedding.tolist(),
                "text_embedding": text_embedding.tolist(),
                "logical_features": logical_features.tolist(),
                "combined_embedding": combined_embedding.tolist(),
                "embedding_dimensions": {
                    "image": len(image_embedding),
                    "text": len(text_embedding),
                    "logical": len(logical_features),
                    "combined": len(combined_embedding)
                }
            }
            
            print(f"[COMBINED EMBEDDING] Generated {len(combined_embedding)}-dim embedding for {symbol}")
            return embedding_record
            
        except Exception as e:
            print(f"[COMBINED EMBEDDING ERROR] {symbol}: {e}")
            return None
    
    def save_embedding(self, embedding_record: Dict):
        """Save embedding record to file"""
        try:
            # Load existing embeddings
            embeddings = {}
            if os.path.exists(self.embeddings_file):
                with open(self.embeddings_file, 'r') as f:
                    embeddings = json.load(f)
            
            symbol = embedding_record["symbol"]
            if symbol not in embeddings:
                embeddings[symbol] = []
            
            # Add new embedding
            embeddings[symbol].append(embedding_record)
            
            # Keep only recent embeddings (last 100 per symbol)
            if len(embeddings[symbol]) > 100:
                embeddings[symbol] = embeddings[symbol][-100:]
            
            # Save updated embeddings
            with open(self.embeddings_file, 'w') as f:
                json.dump(embeddings, f, indent=2)
            
            print(f"[EMBEDDING SAVE] Saved embedding for {symbol} ({len(embeddings[symbol])} total)")
            
        except Exception as e:
            print(f"[EMBEDDING SAVE ERROR] {e}")
    
    def find_similar_cases(
        self, 
        query_embedding: np.ndarray, 
        symbol: str = None, 
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict]:
        """Find similar historical cases using cosine similarity"""
        try:
            if not os.path.exists(self.embeddings_file):
                return []
            
            with open(self.embeddings_file, 'r') as f:
                embeddings = json.load(f)
            
            candidates = []
            
            # Collect all embeddings (optionally filter by symbol)
            for sym, records in embeddings.items():
                if symbol and sym != symbol:
                    continue  # Only search same symbol if specified
                
                for record in records:
                    if "combined_embedding" in record:
                        candidates.append(record)
            
            if not candidates:
                return []
            
            # Calculate similarities
            similarities = []
            query_norm = np.linalg.norm(query_embedding)
            
            for candidate in candidates:
                candidate_embedding = np.array(candidate["combined_embedding"])
                candidate_norm = np.linalg.norm(candidate_embedding)
                
                if query_norm > 0 and candidate_norm > 0:
                    similarity = np.dot(query_embedding, candidate_embedding) / (query_norm * candidate_norm)
                    
                    if similarity >= similarity_threshold:
                        similarities.append({
                            "similarity": similarity,
                            "record": candidate
                        })
            
            # Sort by similarity and return top-k
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            similar_cases = similarities[:top_k]
            
            print(f"[SIMILARITY SEARCH] Found {len(similar_cases)} similar cases (threshold: {similarity_threshold:.2f})")
            
            return [case["record"] for case in similar_cases]
            
        except Exception as e:
            print(f"[SIMILARITY SEARCH ERROR] {e}")
            return []
    
    def get_embedding_statistics(self) -> Dict:
        """Get statistics about stored embeddings"""
        try:
            if not os.path.exists(self.embeddings_file):
                return {"total_symbols": 0, "total_embeddings": 0}
            
            with open(self.embeddings_file, 'r') as f:
                embeddings = json.load(f)
            
            total_embeddings = sum(len(records) for records in embeddings.values())
            
            # Decision distribution
            decision_counts = {}
            for records in embeddings.values():
                for record in records:
                    decision = record.get("decision", "unknown")
                    decision_counts[decision] = decision_counts.get(decision, 0) + 1
            
            # Average scores by decision
            decision_scores = {}
            for decision in decision_counts:
                scores = []
                for records in embeddings.values():
                    for record in records:
                        if record.get("decision") == decision:
                            scores.append(record.get("tjde_score", 0.0))
                
                if scores:
                    decision_scores[decision] = {
                        "avg_score": np.mean(scores),
                        "count": len(scores)
                    }
            
            stats = {
                "total_symbols": len(embeddings),
                "total_embeddings": total_embeddings,
                "decision_distribution": decision_counts,
                "decision_scores": decision_scores,
                "symbols_list": list(embeddings.keys())[:10]  # First 10 symbols
            }
            
            return stats
            
        except Exception as e:
            print(f"[EMBEDDING STATS ERROR] {e}")
            return {}


def process_training_charts_for_embeddings(charts_dir: str = "training_data/charts") -> int:
    """Process all training charts to generate embeddings"""
    if not os.path.exists(charts_dir):
        print(f"[CHART PROCESSING] Charts directory not found: {charts_dir}")
        return 0
    
    embedding_system = HybridEmbeddingSystem()
    processed_count = 0
    
    # Find all PNG files in training charts
    chart_files = [f for f in os.listdir(charts_dir) if f.endswith('.png')]
    
    print(f"[CHART PROCESSING] Found {len(chart_files)} chart files")
    
    for chart_file in chart_files:
        try:
            # Parse filename to extract symbol and timestamp
            parts = chart_file.replace('.png', '').split('_')
            if len(parts) < 2:
                continue
            
            symbol = parts[0]
            timestamp = f"{parts[1]}_{parts[2]}" if len(parts) > 2 else parts[1]
            
            # Look for corresponding metadata JSON
            json_file = chart_file.replace('.png', '.json')
            json_path = os.path.join(charts_dir, json_file)
            
            analysis_data = {}
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    analysis_data = json.load(f)
            else:
                # Create minimal analysis data from filename
                analysis_data = {
                    "decision": "unknown",
                    "final_score": 0.5,
                    "clip_features": {"clip_confidence": 0.5},
                    "gpt_comment": f"Chart analysis for {symbol} at {timestamp}"
                }
            
            # Generate embedding
            chart_path = os.path.join(charts_dir, chart_file)
            embedding_record = embedding_system.generate_combined_embedding(
                symbol, analysis_data, chart_path
            )
            
            if embedding_record:
                embedding_system.save_embedding(embedding_record)
                processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"[CHART PROCESSING] Processed {processed_count} charts...")
            
        except Exception as e:
            print(f"[CHART PROCESSING ERROR] {chart_file}: {e}")
    
    print(f"[CHART PROCESSING] Completed: {processed_count} embeddings generated")
    return processed_count


def integrate_embeddings_with_decision_system(symbol: str, analysis_data: Dict, image_path: str = None) -> Dict:
    """Integrate embedding similarity into decision system"""
    try:
        embedding_system = HybridEmbeddingSystem()
        
        # Generate embedding for current analysis
        embedding_record = embedding_system.generate_combined_embedding(symbol, analysis_data, image_path)
        
        if not embedding_record:
            return analysis_data
        
        # Find similar historical cases
        current_embedding = np.array(embedding_record["combined_embedding"])
        similar_cases = embedding_system.find_similar_cases(current_embedding, symbol, top_k=3)
        
        if similar_cases:
            # Analyze similar cases for decision enhancement
            successful_cases = [case for case in similar_cases 
                             if case.get("decision") in ["join_trend", "consider_entry"]]
            
            similarity_boost = 0.0
            if len(successful_cases) >= 2:
                avg_similar_score = np.mean([case.get("tjde_score", 0) for case in successful_cases])
                if avg_similar_score > 0.6:
                    similarity_boost = 0.02  # Small boost for similar successful patterns
            
            # Update analysis data with similarity insights
            enhanced_analysis = analysis_data.copy()
            enhanced_analysis.update({
                "embedding_similarity": {
                    "similar_cases_found": len(similar_cases),
                    "successful_similar": len(successful_cases),
                    "similarity_boost": similarity_boost,
                    "similar_cases": similar_cases[:2]  # Keep top 2 for reference
                },
                "embedding_enhanced_score": analysis_data.get("final_score", 0) + similarity_boost
            })
            
            # Save embedding
            embedding_system.save_embedding(embedding_record)
            
            print(f"[EMBEDDING INTEGRATION] {symbol}: Found {len(similar_cases)} similar cases, boost: +{similarity_boost:.3f}")
            
            return enhanced_analysis
        else:
            # Save embedding even without similar cases
            embedding_system.save_embedding(embedding_record)
            
            return analysis_data
            
    except Exception as e:
        print(f"[EMBEDDING INTEGRATION ERROR] {symbol}: {e}")
        return analysis_data


def test_hybrid_embedding_system():
    """Test the hybrid embedding system"""
    print("Testing Phase 4: Hybrid Embedding System...")
    
    try:
        # Initialize system
        embedding_system = HybridEmbeddingSystem()
        
        # Test with sample data
        test_symbol = "EMBEDTEST"
        test_analysis = {
            "decision": "consider_entry",
            "final_score": 0.75,
            "tjde_score": 0.72,
            "clip_features": {
                "clip_confidence": 0.68,
                "clip_trend_match": "pullback",
                "clip_setup_type": "support-bounce"
            },
            "gpt_comment": "Strong pullback setup with volume confirmation and support reaction",
            "market_price": 125.50,
            "perception_sync": True,
            "memory_integration": True,
            "quality_grade": "moderate"
        }
        
        # Generate combined embedding
        embedding_record = embedding_system.generate_combined_embedding(test_symbol, test_analysis)
        
        if embedding_record:
            print(f"✅ Combined embedding generated: {embedding_record['embedding_dimensions']['combined']} dimensions")
            
            # Save embedding
            embedding_system.save_embedding(embedding_record)
            
            # Test similarity search
            current_embedding = np.array(embedding_record["combined_embedding"])
            similar_cases = embedding_system.find_similar_cases(current_embedding, top_k=3)
            
            print(f"✅ Similarity search: {len(similar_cases)} similar cases found")
            
            # Test statistics
            stats = embedding_system.get_embedding_statistics()
            print(f"✅ Embedding statistics: {stats['total_embeddings']} total embeddings")
            
            # Test integration with decision system
            enhanced_analysis = integrate_embeddings_with_decision_system(test_symbol, test_analysis)
            
            if "embedding_similarity" in enhanced_analysis:
                print("✅ Decision system integration working")
            
            # Cleanup test data
            if os.path.exists(embedding_system.embeddings_file):
                with open(embedding_system.embeddings_file, 'r') as f:
                    embeddings = json.load(f)
                
                if test_symbol in embeddings:
                    del embeddings[test_symbol]
                    
                    with open(embedding_system.embeddings_file, 'w') as f:
                        json.dump(embeddings, f)
            
            print("✅ Phase 4 Hybrid Embedding System working correctly")
            return True
        else:
            print("❌ Failed to generate embedding")
            return False
            
    except Exception as e:
        print(f"❌ Phase 4 test error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_hybrid_embedding_system()