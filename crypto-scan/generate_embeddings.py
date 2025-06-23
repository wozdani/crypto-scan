"""
Combined Embedding Generator for Vision-AI System
Łączy embeddingi CLIP, TJDE scoring i GPT commentary w unified vectors
"""

import os
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
import glob

from ai import get_clip_image_embedding
from utils.gpt_embedding import get_gpt_text_embedding
from utils.score_embedding import embed_score_vector

logger = logging.getLogger(__name__)

class CombinedEmbeddingGenerator:
    """Generator kombinowanych embeddingów dla systemu Vision-AI"""
    
    def __init__(self):
        """Initialize combined embedding generator"""
        self.embeddings_dir = Path("data/embeddings")
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Expected dimensions
        self.clip_image_dim = 512  # CLIP ViT-B/32 image features
        self.gpt_text_dim = 1536   # OpenAI text-embedding-3-small
        self.score_dim = 12        # TJDE scoring features
        
        self.total_dim = self.clip_image_dim + self.gpt_text_dim + self.score_dim
        
        logger.info(f"Combined embedding generator initialized (total dim: {self.total_dim})")
    
    def find_chart_image(self, symbol: str) -> Optional[str]:
        """
        Find most recent chart image for symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Path to chart image or None
        """
        # Check multiple chart locations
        chart_locations = [
            f"charts/{symbol}_*.png",
            f"exports/{symbol}_*.png",
            f"data/charts/{symbol}_*.png",
            f"training_data/clip/{symbol}_*.png",
            f"data/vision_ai/train_data/charts/{symbol}_*.png"
        ]
        
        for pattern in chart_locations:
            matches = glob.glob(pattern)
            if matches:
                # Return most recent by filename timestamp
                latest_chart = sorted(matches, reverse=True)[0]
                return latest_chart
        
        return None
    
    def get_gpt_comment(self, symbol: str) -> Optional[str]:
        """
        Get GPT comment for symbol from various sources
        
        Args:
            symbol: Trading symbol
            
        Returns:
            GPT comment text or None
        """
        # Try to get from session history
        try:
            history_file = Path("logs/auto_label_session_history.json")
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                
                # Find most recent entry for symbol
                for entry in reversed(history):
                    if entry.get("symbol") == symbol and entry.get("label"):
                        return entry["label"]
        except Exception:
            pass
        
        # Try to get from CLIP predictions
        try:
            from utils.clip_prediction_loader import load_clip_prediction
            prediction = load_clip_prediction(symbol)
            if prediction:
                return prediction
        except Exception:
            pass
        
        return None
    
    def generate_combined_embedding(
        self, 
        symbol: str, 
        image_path: Optional[str] = None,
        score_dict: Optional[Dict] = None, 
        gpt_comment: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Generate combined embedding from CLIP + TJDE + GPT
        
        Args:
            symbol: Trading symbol
            image_path: Path to chart image (auto-detected if None)
            score_dict: TJDE scoring results
            gpt_comment: GPT commentary text
            
        Returns:
            Combined embedding vector or None if failed
        """
        try:
            # Auto-detect image if not provided
            if image_path is None:
                image_path = self.find_chart_image(symbol)
            
            # Auto-detect GPT comment if not provided
            if gpt_comment is None:
                gpt_comment = self.get_gpt_comment(symbol)
            
            # Generate individual embeddings
            image_embed = None
            if image_path and os.path.exists(image_path):
                try:
                    clip_embedding = get_clip_image_embedding(image_path)
                    if clip_embedding is not None:
                        # Flatten and convert to numpy
                        image_embed = clip_embedding.flatten().numpy().astype(np.float32)
                        
                        # Ensure correct dimension
                        if len(image_embed) != self.clip_image_dim:
                            # Resize to expected dimension
                            if len(image_embed) > self.clip_image_dim:
                                image_embed = image_embed[:self.clip_image_dim]
                            else:
                                # Pad with zeros
                                padded = np.zeros(self.clip_image_dim, dtype=np.float32)
                                padded[:len(image_embed)] = image_embed
                                image_embed = padded
                except Exception as e:
                    logger.error(f"Error generating CLIP embedding for {symbol}: {e}")
            
            if image_embed is None:
                logger.warning(f"No CLIP embedding available for {symbol}")
                image_embed = np.zeros(self.clip_image_dim, dtype=np.float32)
            
            # Generate GPT text embedding
            text_embed = None
            if gpt_comment:
                text_embed = get_gpt_text_embedding(gpt_comment)
                if text_embed is not None and len(text_embed) != self.gpt_text_dim:
                    # Resize to expected dimension
                    if len(text_embed) > self.gpt_text_dim:
                        text_embed = text_embed[:self.gpt_text_dim]
                    else:
                        # Pad with zeros
                        padded = np.zeros(self.gpt_text_dim, dtype=np.float32)
                        padded[:len(text_embed)] = text_embed
                        text_embed = padded
            
            if text_embed is None:
                logger.warning(f"No GPT embedding available for {symbol}")
                text_embed = np.zeros(self.gpt_text_dim, dtype=np.float32)
            
            # Generate score embedding
            score_embed = None
            if score_dict:
                score_embed = embed_score_vector(score_dict)
                if len(score_embed) != self.score_dim:
                    # Resize to expected dimension
                    if len(score_embed) > self.score_dim:
                        score_embed = score_embed[:self.score_dim]
                    else:
                        # Pad with zeros
                        padded = np.zeros(self.score_dim, dtype=np.float32)
                        padded[:len(score_embed)] = score_embed
                        score_embed = padded
            
            if score_embed is None:
                logger.warning(f"No score embedding available for {symbol}")
                score_embed = np.zeros(self.score_dim, dtype=np.float32)
            
            # Combine embeddings
            combined = np.concatenate([image_embed, text_embed, score_embed])
            
            logger.info(f"Generated combined embedding for {symbol}: {combined.shape}")
            return combined
            
        except Exception as e:
            logger.error(f"Error generating combined embedding for {symbol}: {e}")
            return None
    
    def save_embedding(self, symbol: str, embedding: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """
        Save embedding to file with metadata
        
        Args:
            symbol: Trading symbol
            embedding: Combined embedding vector
            metadata: Optional metadata dictionary
            
        Returns:
            True if saved successfully
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save embedding
            embedding_file = self.embeddings_dir / f"{symbol}_{timestamp}.npy"
            np.save(embedding_file, embedding)
            
            # Save metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                "symbol": symbol,
                "timestamp": timestamp,
                "embedding_shape": embedding.shape,
                "embedding_dimension": len(embedding),
                "clip_dim": self.clip_image_dim,
                "gpt_dim": self.gpt_text_dim,
                "score_dim": self.score_dim,
                "generation_time": datetime.now().isoformat()
            })
            
            metadata_file = self.embeddings_dir / f"{symbol}_{timestamp}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved embedding for {symbol} to {embedding_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving embedding for {symbol}: {e}")
            return False
    
    def generate_and_save_embedding(
        self, 
        symbol: str, 
        image_path: Optional[str] = None,
        score_dict: Optional[Dict] = None, 
        gpt_comment: Optional[str] = None
    ) -> bool:
        """
        Generate and save combined embedding
        
        Args:
            symbol: Trading symbol
            image_path: Path to chart image
            score_dict: TJDE scoring results
            gpt_comment: GPT commentary
            
        Returns:
            True if successful
        """
        # Generate embedding
        embedding = self.generate_combined_embedding(symbol, image_path, score_dict, gpt_comment)
        
        if embedding is None:
            return False
        
        # Prepare metadata
        metadata = {
            "has_image": image_path is not None and os.path.exists(image_path),
            "has_score": score_dict is not None,
            "has_gpt_comment": gpt_comment is not None,
            "image_path": image_path,
            "gpt_comment": gpt_comment[:100] + "..." if gpt_comment and len(gpt_comment) > 100 else gpt_comment,
            "score_summary": {
                "final_score": score_dict.get("final_score", 0) if score_dict else 0,
                "decision": score_dict.get("decision", "unknown") if score_dict else "unknown"
            }
        }
        
        # Save embedding
        return self.save_embedding(symbol, embedding, metadata)
    
    def process_top_symbols(self, symbols: List[str], tjde_results: Optional[Dict] = None) -> Dict[str, bool]:
        """
        Process embeddings for top symbols from scan
        
        Args:
            symbols: List of symbols to process
            tjde_results: Optional TJDE results dictionary
            
        Returns:
            Dictionary of symbol -> success status
        """
        results = {}
        
        for symbol in symbols:
            try:
                # Get TJDE result for this symbol
                score_dict = None
                if tjde_results and symbol in tjde_results:
                    score_dict = tjde_results[symbol]
                
                # Generate and save embedding
                success = self.generate_and_save_embedding(symbol, score_dict=score_dict)
                results[symbol] = success
                
                if success:
                    print(f"[EMBEDDING] Generated embedding for {symbol}")
                else:
                    print(f"[EMBEDDING] Failed to generate embedding for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error processing embedding for {symbol}: {e}")
                results[symbol] = False
        
        successful = sum(1 for success in results.values() if success)
        print(f"[EMBEDDING] Generated {successful}/{len(symbols)} embeddings successfully")
        
        return results
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about saved embeddings"""
        try:
            embedding_files = list(self.embeddings_dir.glob("*.npy"))
            metadata_files = list(self.embeddings_dir.glob("*_metadata.json"))
            
            stats = {
                "total_embeddings": len(embedding_files),
                "total_metadata": len(metadata_files),
                "symbols": set(),
                "latest_embedding": None,
                "oldest_embedding": None
            }
            
            # Extract symbols and timestamps
            for emb_file in embedding_files:
                try:
                    parts = emb_file.stem.split('_')
                    if len(parts) >= 2:
                        symbol = parts[0]
                        stats["symbols"].add(symbol)
                        
                        file_time = datetime.fromtimestamp(emb_file.stat().st_mtime)
                        if stats["latest_embedding"] is None or file_time > stats["latest_embedding"]:
                            stats["latest_embedding"] = file_time
                        if stats["oldest_embedding"] is None or file_time < stats["oldest_embedding"]:
                            stats["oldest_embedding"] = file_time
                except Exception:
                    continue
            
            stats["unique_symbols"] = len(stats["symbols"])
            stats["symbols"] = list(stats["symbols"])
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting embedding stats: {e}")
            return {"error": str(e)}


# Global instance
_global_embedding_generator = None

def get_embedding_generator() -> CombinedEmbeddingGenerator:
    """Get global combined embedding generator instance"""
    global _global_embedding_generator
    if _global_embedding_generator is None:
        _global_embedding_generator = CombinedEmbeddingGenerator()
    return _global_embedding_generator

def generate_combined_embedding(symbol: str, image_path: str, score_dict: Dict, gpt_comment: str) -> Optional[np.ndarray]:
    """
    Convenience function to generate combined embedding
    
    Args:
        symbol: Trading symbol
        image_path: Path to chart image
        score_dict: TJDE scoring results
        gpt_comment: GPT commentary
        
    Returns:
        Combined embedding vector
    """
    generator = get_embedding_generator()
    return generator.generate_combined_embedding(symbol, image_path, score_dict, gpt_comment)

def main():
    """Test combined embedding generation"""
    print("Testing Combined Embedding Generator")
    print("=" * 50)
    
    generator = CombinedEmbeddingGenerator()
    
    # Get embedding statistics
    stats = generator.get_embedding_stats()
    print(f"Embedding Statistics:")
    for key, value in stats.items():
        if key not in ["latest_embedding", "oldest_embedding"]:
            print(f"   {key}: {value}")
    
    # Test embedding generation
    test_symbols = ["BTCUSDT", "ETHUSDT"]
    
    print(f"\nTesting embedding generation:")
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}:")
        
        # Find chart
        chart_path = generator.find_chart_image(symbol)
        if chart_path:
            print(f"   Chart found: {Path(chart_path).name}")
        else:
            print(f"   Chart: Not found")
        
        # Get GPT comment
        gpt_comment = generator.get_gpt_comment(symbol)
        if gpt_comment:
            print(f"   GPT comment: {gpt_comment[:50]}...")
        else:
            print(f"   GPT comment: Not found")
        
        # Mock TJDE score
        mock_score = {
            "final_score": 0.685,
            "decision": "consider_entry",
            "score_breakdown": {
                "trend_strength": 0.75,
                "pullback_quality": 0.68,
                "support_reaction": 0.72,
                "liquidity_pattern_score": 0.65,
                "psych_score": 0.80,
                "htf_supportive_score": 0.55,
                "market_phase_modifier": 0.05
            }
        }
        
        # Generate embedding
        success = generator.generate_and_save_embedding(symbol, chart_path, mock_score, gpt_comment)
        
        if success:
            print(f"   ✅ Embedding generated and saved")
        else:
            print(f"   ❌ Embedding generation failed")
    
    # Show updated stats
    updated_stats = generator.get_embedding_stats()
    print(f"\nUpdated Statistics:")
    print(f"   Total embeddings: {updated_stats.get('total_embeddings', 0)}")
    print(f"   Unique symbols: {updated_stats.get('unique_symbols', 0)}")
    
    print(f"\n✅ Combined embedding test completed")

if __name__ == "__main__":
    main()