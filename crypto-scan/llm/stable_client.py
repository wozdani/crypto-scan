#!/usr/bin/env python3
"""
Stable LLM Client with JSON-only responses, retry logic, and cost control
"""

import json
import logging
import time
from typing import Dict, Any, Optional, List
from openai import OpenAI
import os
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from consensus.contracts import AgentOpinion
from contracts.agent_contracts import validate_agent_response_json, AgentResponse

logger = logging.getLogger(__name__)

class CostTracker:
    """Tracks OpenAI API costs and usage"""
    
    def __init__(self):
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self.calls_count = 0
        self.daily_limit_usd = 50.0  # Configurable daily limit
        
    def add_usage(self, prompt_tokens: int, completion_tokens: int, model: str = "gpt-4o"):
        """Add usage statistics"""
        self.calls_count += 1
        self.total_tokens += prompt_tokens + completion_tokens
        
        # GPT-4o pricing (as of Aug 2025)
        prompt_cost = prompt_tokens * 0.005 / 1000  # $0.005 per 1K input tokens
        completion_cost = completion_tokens * 0.015 / 1000  # $0.015 per 1K output tokens
        call_cost = prompt_cost + completion_cost
        
        self.total_cost_usd += call_cost
        
        logger.debug(f"API call cost: ${call_cost:.4f} (tokens: {prompt_tokens}+{completion_tokens})")
        
        if self.total_cost_usd > self.daily_limit_usd:
            logger.warning(f"Daily cost limit exceeded: ${self.total_cost_usd:.2f} > ${self.daily_limit_usd}")
            
        return call_cost
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        return {
            "total_calls": self.calls_count,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "avg_cost_per_call": round(self.total_cost_usd / max(1, self.calls_count), 4),
            "daily_limit_usd": self.daily_limit_usd,
            "limit_utilization": round(self.total_cost_usd / self.daily_limit_usd * 100, 1)
        }

class StableLLMClient:
    """Stable OpenAI client with JSON-only responses and robust error handling"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.cost_tracker = CostTracker()
        self.model = "gpt-4o"
        self.max_tokens = 800
        self.temperature = 0.3  # Lower temperature for more consistent JSON
        
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((Exception,))
    )
    def chat_json_only(self, system_prompt: str, user_data: Dict[str, Any], 
                       agent_name: str = "unknown") -> AgentResponse:
        """
        Make OpenAI API call with JSON-only response
        Returns validated AgentResponse object
        """
        start_time = time.time()
        
        try:
            messages = [
                {
                    "role": "system", 
                    "content": f"{system_prompt}\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown, no text before/after JSON."
                },
                {
                    "role": "user",
                    "content": json.dumps(user_data, indent=2)
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"},  # Force JSON response
                timeout=30
            )
            
            # Track costs
            usage = response.usage
            call_cost = self.cost_tracker.add_usage(
                usage.prompt_tokens, 
                usage.completion_tokens,
                self.model
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            
            # Clean JSON (remove markdown blocks if present)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # Parse and validate JSON
            try:
                data = json.loads(content)
                agent_response = validate_agent_response_json(data)
                
                # Add metadata
                processing_time = int((time.time() - start_time) * 1000)
                
                logger.info(f"[{agent_name}] API success: {processing_time}ms, ${call_cost:.4f}, "
                           f"tokens: {usage.prompt_tokens}+{usage.completion_tokens}")
                
                return agent_response
                
            except json.JSONDecodeError as e:
                logger.error(f"[{agent_name}] JSON parse error: {e}, content: {content[:100]}")
                raise ValueError(f"Invalid JSON response: {e}")
                
        except Exception as e:
            logger.error(f"[{agent_name}] API call failed: {e}")
            
            # Return fallback response
            return AgentResponse(
                action_probs={"BUY": 0.2, "HOLD": 0.5, "AVOID": 0.2, "ABSTAIN": 0.1},
                uncertainty={"epistemic": 0.9, "aleatoric": 0.6},
                evidence=[
                    {"name": "api_failure", "direction": "neutral", "strength": 0.0},
                    {"name": "fallback_response", "direction": "neutral", "strength": 0.3},
                    {"name": "error_recovery", "direction": "con", "strength": 0.8}
                ],
                rationale=f"API call failed: {str(e)[:80]}",
                calibration_hint={"reliability": 0.2, "expected_ttft_mins": 45}
            )
    
    def get_cost_stats(self) -> Dict[str, Any]:
        """Get current cost and usage statistics"""
        return self.cost_tracker.get_stats()
    
    def reset_daily_costs(self):
        """Reset daily cost tracking (call at midnight)"""
        self.cost_tracker = CostTracker()
        logger.info("Daily cost tracking reset")

# Global instance
stable_client = StableLLMClient()