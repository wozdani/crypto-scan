"""
Background Chart Generation Worker
Separates TradingView chart generation from main scanning pipeline
to improve performance and achieve <15s scan target
"""

import asyncio
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional
from queue import Queue
import threading

from tradingview_robust import RobustTradingViewGenerator
from vision.gpt_labeler import GPTLabeler

class ChartGenerationWorker:
    """
    Background worker for handling TradingView chart generation
    Runs independently from main scanning pipeline
    """
    
    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self.chart_queue = Queue()
        self.is_running = False
        self.worker_thread = None
        self.stats = {
            "charts_generated": 0,
            "charts_failed": 0,
            "total_time": 0.0,
            "avg_time_per_chart": 0.0
        }
        
        # Initialize generators
        self.tv_generator = RobustTradingViewGenerator()
        self.gpt_labeler = GPTLabeler()
    
    def start_worker(self):
        """Start background chart generation worker"""
        if self.is_running:
            print("[CHART WORKER] Already running")
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print("🎨 [CHART WORKER] Background chart generation started")
    
    def stop_worker(self):
        """Stop background chart generation worker"""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        print("🛑 [CHART WORKER] Background chart generation stopped")
    
    def queue_chart_generation(self, symbol: str, tjde_score: float, 
                              tjde_decision: str, priority: int = 1) -> str:
        """
        Queue a token for chart generation
        
        Args:
            symbol: Token symbol
            tjde_score: TJDE score
            tjde_decision: TJDE decision
            priority: Priority level (1=high, 2=normal, 3=low)
            
        Returns:
            Queue ID for tracking
        """
        queue_id = f"{symbol}_{int(time.time())}"
        
        chart_task = {
            "queue_id": queue_id,
            "symbol": symbol,
            "tjde_score": tjde_score,
            "tjde_decision": tjde_decision,
            "priority": priority,
            "queued_at": datetime.now().isoformat(),
            "status": "queued"
        }
        
        self.chart_queue.put((priority, chart_task))
        print(f"📋 [CHART QUEUE] {symbol}: Queued for generation (priority: {priority})")
        
        return queue_id
    
    def _worker_loop(self):
        """Main worker loop for processing chart generation queue"""
        print("[CHART WORKER] Worker loop started")
        
        while self.is_running:
            try:
                # Check for tasks with timeout
                try:
                    priority, task = self.chart_queue.get(timeout=1.0)
                except:
                    continue
                
                # Process the chart generation task
                start_time = time.time()
                success = self._generate_chart_task(task)
                end_time = time.time()
                
                # Update statistics
                generation_time = end_time - start_time
                self.stats["total_time"] += generation_time
                
                if success:
                    self.stats["charts_generated"] += 1
                    print(f"✅ [CHART WORKER] {task['symbol']}: Generated in {generation_time:.1f}s")
                else:
                    self.stats["charts_failed"] += 1
                    print(f"❌ [CHART WORKER] {task['symbol']}: Failed after {generation_time:.1f}s")
                
                # Update average time
                total_charts = self.stats["charts_generated"] + self.stats["charts_failed"]
                if total_charts > 0:
                    self.stats["avg_time_per_chart"] = self.stats["total_time"] / total_charts
                
                self.chart_queue.task_done()
                
            except Exception as e:
                print(f"[CHART WORKER ERROR] {e}")
                continue
    
    def _generate_chart_task(self, task: Dict) -> bool:
        """
        Generate chart for a specific task
        
        Args:
            task: Chart generation task dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            symbol = task["symbol"]
            tjde_score = task["tjde_score"]
            tjde_decision = task["tjde_decision"]
            
            print(f"🎨 [CHART WORKER] Starting generation for {symbol} (score: {tjde_score:.3f})")
            
            # Generate TradingView chart
            chart_path = self.tv_generator.generate_tradingview_chart(
                symbol=symbol,
                score=tjde_score,
                save_dir="training_data/charts"
            )
            
            if not chart_path or "TRADINGVIEW_FAILED" in chart_path:
                print(f"❌ [CHART WORKER] {symbol}: TradingView generation failed")
                return False
            
            # Generate GPT labeling
            try:
                gpt_result = self.gpt_labeler.analyze_and_label_chart(
                    chart_path=chart_path,
                    symbol=symbol
                )
                
                if gpt_result and gpt_result.get("setup_label"):
                    # Rename chart with GPT label
                    base_name = os.path.basename(chart_path)
                    dir_name = os.path.dirname(chart_path)
                    name_parts = base_name.split("_")
                    
                    if len(name_parts) >= 3:
                        # Insert GPT label
                        new_name = f"{name_parts[0]}_{name_parts[1]}_{gpt_result['setup_label']}_{name_parts[2]}"
                        new_path = os.path.join(dir_name, new_name)
                        
                        os.rename(chart_path, new_path)
                        print(f"🏷️  [CHART WORKER] {symbol}: Labeled as {gpt_result['setup_label']}")
                        
                        # Update metadata
                        metadata_path = chart_path.replace(".png", "_metadata.json")
                        if os.path.exists(metadata_path):
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            
                            metadata.update(gpt_result)
                            
                            new_metadata_path = new_path.replace(".png", "_metadata.json")
                            with open(new_metadata_path, 'w') as f:
                                json.dump(metadata, f, indent=2)
                            
                            # Remove old metadata
                            os.remove(metadata_path)
                
            except Exception as e:
                print(f"⚠️  [CHART WORKER] {symbol}: GPT labeling failed: {e}")
                # Chart generation still succeeded even if labeling failed
            
            return True
            
        except Exception as e:
            print(f"❌ [CHART WORKER] {task['symbol']}: Generation error: {e}")
            return False
    
    def get_queue_status(self) -> Dict:
        """Get current queue status and statistics"""
        return {
            "queue_size": self.chart_queue.qsize(),
            "is_running": self.is_running,
            "stats": self.stats.copy()
        }
    
    def queue_top5_charts(self, top5_tokens: List[Dict]) -> List[str]:
        """
        Queue TOP 5 tokens for chart generation
        
        Args:
            top5_tokens: List of TOP 5 token dictionaries
            
        Returns:
            List of queue IDs
        """
        queue_ids = []
        
        for i, token_data in enumerate(top5_tokens):
            symbol = token_data.get("symbol")
            tjde_score = token_data.get("tjde_score", 0.0)
            tjde_decision = token_data.get("tjde_decision", "unknown")
            
            # Higher ranking = higher priority (lower number)
            priority = i + 1
            
            queue_id = self.queue_chart_generation(
                symbol=symbol,
                tjde_score=tjde_score,
                tjde_decision=tjde_decision,
                priority=priority
            )
            
            queue_ids.append(queue_id)
        
        print(f"📊 [CHART WORKER] Queued {len(top5_tokens)} TOP 5 tokens for generation")
        return queue_ids

# Global worker instance
chart_worker = ChartGenerationWorker()

def start_chart_worker():
    """Start the global chart generation worker"""
    chart_worker.start_worker()

def stop_chart_worker():
    """Stop the global chart generation worker"""
    chart_worker.stop_worker()

def queue_chart_for_generation(symbol: str, tjde_score: float, 
                              tjde_decision: str, priority: int = 1) -> str:
    """Queue a chart for background generation"""
    return chart_worker.queue_chart_generation(symbol, tjde_score, tjde_decision, priority)

def queue_top5_for_generation(top5_tokens: List[Dict]) -> List[str]:
    """Queue TOP 5 tokens for background generation"""
    return chart_worker.queue_top5_charts(top5_tokens)

def get_chart_worker_status() -> Dict:
    """Get current chart worker status"""
    return chart_worker.get_queue_status()