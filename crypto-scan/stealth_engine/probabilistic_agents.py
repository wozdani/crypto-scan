"""
Probabilistic Multi-Agent System for Crypto Signal Analysis
Uses soft reasoning, uncertainty quantification, and probabilistic consensus.
No hard thresholds - all decisions based on LLM reasoning and evidence.
"""

import json
import math
import openai
import os
from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TokenMeta:
    def __init__(self, price: float, volume_24h: float, spread_bps: float, 
                 liquidity_tier: str, is_perp: bool, exchange: str, 
                 funding_apr: float = 0.0, oi_change: float = 0.0, news_flag: bool = False):
        self.price = price
        self.volume_24h = volume_24h
        self.spread_bps = spread_bps
        self.liquidity_tier = liquidity_tier
        self.is_perp = is_perp
        self.exchange = exchange
        self.funding_apr = funding_apr
        self.oi_change = oi_change
        self.news_flag = news_flag

class TrustProfile:
    def __init__(self, trusted_addresses_share: float, recurring_wallets_7d: int, 
                 smart_money_score: float):
        self.trusted_addresses_share = trusted_addresses_share
        self.recurring_wallets_7d = recurring_wallets_7d
        self.smart_money_score = smart_money_score

class TokenHistory:
    def __init__(self, events_72h: List[str], repeats_24h: int, 
                 cooldown_active: bool, last_alert_outcome: str):
        self.events_72h = events_72h
        self.repeats_24h = repeats_24h
        self.cooldown_active = cooldown_active
        self.last_alert_outcome = last_alert_outcome

class DetectorPerfStats:
    def __init__(self, precision_7d: float, tp_rate: float, fp_rate: float, 
                 avg_lag_mins: float):
        self.precision_7d = precision_7d
        self.tp_rate = tp_rate
        self.fp_rate = fp_rate
        self.avg_lag_mins = avg_lag_mins

class ProbabilisticMultiAgentSystem:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.agent_reliability = {
            "analyzer": 0.8,
            "reasoner": 0.75,
            "voter": 0.85,
            "debater": 0.7
        }
        
        # System prompts for each agent
        self.prompts = {
            "analyzer": self._get_analyzer_prompt(),
            "reasoner": self._get_reasoner_prompt(),
            "voter": self._get_voter_prompt(),
            "debater": self._get_debater_prompt()
        }

    def _get_analyzer_prompt(self) -> str:
        return """Jesteś Analyzerem. Nie używaj sztywnych reguł. Zamiast tego:

- Zidentyfikuj wzorce współwystępowania (np. whale+dex+orderbook) jako dowody pro lub kontra
- Oceniaj spójność: czy sygnały wskazują na tę samą hipotezę (akumulacja → pre-pump)?
- Zgłoś action_probs przez miękkie ważenie dowodów (no hard thresholds)
- Oszacuj epistemic/aleatoric niepewność na podstawie konflitku/chaosu danych

Heurystyka (opisowa, nie reguła):
- Traktuj kombinacje jako silniejsze dowody, ale jeśli istnieją kontrdowody (np. news-only), osłab część siły
- Zawsze raportuj co najmniej 5 pozycji evidence (mogą być neutral)

CRITICAL: Return EXACTLY this JSON format, nothing else:
{
  "action_probs": {"BUY": 0.3, "HOLD": 0.4, "AVOID": 0.2, "ABSTAIN": 0.1},
  "uncertainty": {"epistemic": 0.3, "aleatoric": 0.2},
  "evidence": [
    {"name": "whale_dex_correlation", "direction": "pro", "strength": 0.7},
    {"name": "orderbook_anomaly", "direction": "pro", "strength": 0.5},
    {"name": "volume_consistency", "direction": "neutral", "strength": 0.3},
    {"name": "news_counter_signal", "direction": "con", "strength": 0.4},
    {"name": "pattern_coherence", "direction": "pro", "strength": 0.6}
  ],
  "rationale": "Moderate signals with some coherence but mixed evidence quality.",
  "calibration_hint": {"reliability": 0.8, "expected_ttft_mins": 25}
}

Ensure probabilities sum to 1.0. Always include at least 5 evidence items."""

    def _get_reasoner_prompt(self) -> str:
        return """Jesteś agentem REASONER. Oceniaj sekwencyjne rozumowanie i wystarczalność dowodów w czasie bez twardych progów.

Oceniaj:
- Spójność temporalną (czy dowody eskalują czy zanikają?)
- Recykling adresów i rytm (powtarzalność w 24h/72h)
- Wzorce sekwencyjne bez reguł typu '≥N zdarzeń'

Używaj miękkich metryk:
- Ocena spójności temporalnej (opisowa, 0..1)
- Siła powtarzalności (opisowa, 0..1)
- Kara za konflikty (opisowa, 0..1)

Konwertuj to na action_probs + uncertainty.

Wskazówki (opisowe, nie if-then):
- Spójne wzorce temporalne → zwiększ 'BUY' miękko
- Asymetryczne wzorce (piki + brak kontynuacji) → zwiększ 'HOLD'/'ABSTAIN'
- Tylko-newsy → podnieś aleatoric uncertainty

CRITICAL: Return EXACTLY this JSON format, nothing else:
{
  "action_probs": {"BUY": 0.25, "HOLD": 0.45, "AVOID": 0.2, "ABSTAIN": 0.1},
  "uncertainty": {"epistemic": 0.4, "aleatoric": 0.3},
  "evidence": [
    {"name": "temporal_coherence", "direction": "pro", "strength": 0.6},
    {"name": "address_recycling", "direction": "neutral", "strength": 0.4},
    {"name": "pattern_consistency", "direction": "con", "strength": 0.3},
    {"name": "sequence_escalation", "direction": "pro", "strength": 0.5},
    {"name": "rhythm_analysis", "direction": "neutral", "strength": 0.4}
  ],
  "rationale": "Temporal patterns show moderate consistency with some recycling activity.",
  "calibration_hint": {"reliability": 0.75, "expected_ttft_mins": 30}
}

Ensure probabilities sum to 1.0. Focus on time-based evidence patterns."""

    def _get_voter_prompt(self) -> str:
        return """Jesteś agentem VOTER. Kalibrujesz decyzje względem rzeczywistej, niedawnej wydajności bez progów.

Używaj miękkich wag dla:
- Detektorów z wyższą precision_7d i niższą fp_rate
- Jeśli avg_lag_mins jest wysokie, przesuń prawdopodobieństwo w kierunku 'HOLD'
- Brak progów - operuj proporcjami i miękkimi karami

W calibration_hint.reliability podaj rolling VOTER reliability (0.6-0.9) dla późniejszej meta-kalibracji.

Skup się na statystycznym ugruntowaniu, a nie na rozpoznawaniu wzorców.

CRITICAL: Return EXACTLY this JSON format, nothing else:
{
  "action_probs": {"BUY": 0.2, "HOLD": 0.5, "AVOID": 0.2, "ABSTAIN": 0.1},
  "uncertainty": {"epistemic": 0.35, "aleatoric": 0.25},
  "evidence": [
    {"name": "detector_precision", "direction": "pro", "strength": 0.7},
    {"name": "false_positive_rate", "direction": "con", "strength": 0.4},
    {"name": "lag_compensation", "direction": "neutral", "strength": 0.5},
    {"name": "performance_trend", "direction": "pro", "strength": 0.6},
    {"name": "statistical_confidence", "direction": "neutral", "strength": 0.5}
  ],
  "rationale": "Performance calibration suggests moderate confidence with lag considerations.",
  "calibration_hint": {"reliability": 0.85, "expected_ttft_mins": 20}
}

Ensure probabilities sum to 1.0. Include detector names with weights in evidence."""

    def _get_debater_prompt(self) -> str:
        return """Jesteś agentem DEBATER. Twórz wyraźne argumenty trade-off.

Proces:
1. Utwórz pary argumentów (pro/con) ze wszystkich danych wejściowych
2. Przypisz siłę (0..1) każdemu argumentowi opisowo, bez progów
3. Zastosuj miękkie balansowanie: dominacja = średnia(pros) - średnia(cons) (opisowo)
4. Konwertuj na action_probs + uncertainty

Żadnych binarnych decyzji - wszystko to miękkie równoważenie dowodów.

Skup się na:
- Wyraźnych argumentach pro vs con
- Analizie trade-off
- Ważeniu siły bez twardych granic

CRITICAL: Return EXACTLY this JSON format, nothing else:
{
  "action_probs": {"BUY": 0.3, "HOLD": 0.4, "AVOID": 0.2, "ABSTAIN": 0.1},
  "uncertainty": {"epistemic": 0.3, "aleatoric": 0.3},
  "evidence": [
    {"name": "pro_arguments", "direction": "pro", "strength": 0.6},
    {"name": "con_arguments", "direction": "con", "strength": 0.4},
    {"name": "argument_balance", "direction": "neutral", "strength": 0.5},
    {"name": "trade_off_analysis", "direction": "neutral", "strength": 0.7},
    {"name": "evidence_weight", "direction": "pro", "strength": 0.5}
  ],
  "rationale": "Pro arguments slightly outweigh cons but with significant uncertainty.",
  "calibration_hint": {"reliability": 0.7, "expected_ttft_mins": 35}
}

Ensure probabilities sum to 1.0. Focus on balanced argumentation without hard rules."""

    def _openai_chat_json(self, system_prompt: str, user_data: Dict) -> Dict[str, Any]:
        """Make OpenAI API call and parse JSON response"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_data)}
                ],
                temperature=0.1,
                max_tokens=2048
            )
            
            content = response.choices[0].message.content
            if content:
                content = content.strip()
                # Extract JSON from response
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                parsed = json.loads(content)
                # Validate and normalize the response
                return self._normalize_agent_response(parsed)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            # Return default probabilistic response
            return {
                "action_probs": {"BUY": 0.2, "HOLD": 0.5, "AVOID": 0.2, "ABSTAIN": 0.1},
                "uncertainty": {"epistemic": 0.8, "aleatoric": 0.5},
                "evidence": [{"name": "api_error", "direction": "neutral", "strength": 0.0}],
                "rationale": f"API error: {str(e)[:100]}",
                "calibration_hint": {"reliability": 0.3, "expected_ttft_mins": 30}
            }
    
    def _normalize_agent_response(self, response: Any) -> Dict[str, Any]:
        """Normalize agent response to expected format"""
        if not isinstance(response, dict):
            # Convert non-dict responses to default format
            return {
                "action_probs": {"BUY": 0.2, "HOLD": 0.5, "AVOID": 0.2, "ABSTAIN": 0.1},
                "uncertainty": {"epistemic": 0.5, "aleatoric": 0.3},
                "evidence": [{"name": "normalized_response", "direction": "neutral", "strength": 0.5}],
                "rationale": "Response normalized from non-standard format",
                "calibration_hint": {"reliability": 0.5, "expected_ttft_mins": 30}
            }
        
        # Ensure action_probs exists and is properly formatted
        action_probs = response.get("action_probs", {})
        if not isinstance(action_probs, dict) or not action_probs:
            # Try to extract from different possible formats
            if "buy" in response:
                action_probs = {
                    "BUY": float(response.get("buy", 0.2)),
                    "HOLD": float(response.get("hold", 0.5)),
                    "AVOID": float(response.get("sell", 0.2)),
                    "ABSTAIN": float(response.get("abstain", 0.1))
                }
            elif "BUY" in response:
                action_probs = {
                    "BUY": float(response.get("BUY", 0.2)),
                    "HOLD": float(response.get("HOLD", 0.5)),
                    "AVOID": float(response.get("AVOID", 0.2)),
                    "ABSTAIN": float(response.get("ABSTAIN", 0.1))
                }
            else:
                action_probs = {"BUY": 0.2, "HOLD": 0.5, "AVOID": 0.2, "ABSTAIN": 0.1}
        
        # Normalize probabilities to sum to 1
        total = sum(action_probs.values())
        if total > 0:
            action_probs = {k: v/total for k, v in action_probs.items()}
        
        return {
            "action_probs": action_probs,
            "uncertainty": response.get("uncertainty", {"epistemic": 0.3, "aleatoric": 0.3}),
            "evidence": response.get("evidence", [{"name": "soft_reasoning", "direction": "neutral", "strength": 0.5}]),
            "rationale": response.get("rationale", "Probabilistic analysis completed"),
            "calibration_hint": response.get("calibration_hint", {"reliability": 0.6, "expected_ttft_mins": 30})
        }

    def run_analyzer(self, inputs: Dict) -> Dict:
        """Run ANALYZER agent with soft reasoning"""
        print(f"[PROBABILISTIC] Running ANALYZER with soft reasoning...")
        return self._openai_chat_json(self.prompts["analyzer"], inputs)

    def run_reasoner(self, inputs: Dict) -> Dict:
        """Run REASONER agent with temporal analysis"""
        print(f"[PROBABILISTIC] Running REASONER with temporal analysis...")
        return self._openai_chat_json(self.prompts["reasoner"], inputs)

    def run_voter(self, inputs: Dict) -> Dict:
        """Run VOTER agent with performance calibration"""
        print(f"[PROBABILISTIC] Running VOTER with performance calibration...")
        return self._openai_chat_json(self.prompts["voter"], inputs)

    def run_debater(self, inputs: Dict) -> Dict:
        """Run DEBATER agent with pro/con analysis"""
        print(f"[PROBABILISTIC] Running DEBATER with pro/con analysis...")
        return self._openai_chat_json(self.prompts["debater"], inputs)

    def decider_aggregate(self, opinions: List[Dict]) -> Dict:
        """
        Soft aggregation using Bradley-Terry/Luce model instead of majority vote
        """
        print(f"[PROBABILISTIC] DECIDER aggregating {len(opinions)} agent opinions...")
        
        A = ["BUY", "HOLD", "AVOID", "ABSTAIN"]
        eps = 1e-6
        scores = {a: 0.0 for a in A}
        
        total_weight = 0.0
        epistemic_sum = 0.0
        aleatoric_sum = 0.0
        all_evidence = []
        
        for op in opinions:
            # Handle different response formats gracefully
            if isinstance(op, dict):
                p = op.get("action_probs", {})
                # If no action_probs, try to infer from other fields
                if not p:
                    # Some agents return different formats - normalize them
                    if "buy" in op and "hold" in op:
                        p = {"BUY": op.get("buy", 0.2), "HOLD": op.get("hold", 0.5), 
                             "AVOID": op.get("sell", 0.2), "ABSTAIN": op.get("abstain", 0.1)}
                    elif "BUY" in op:
                        p = {"BUY": op.get("BUY", 0.2), "HOLD": op.get("HOLD", 0.5), 
                             "AVOID": op.get("AVOID", 0.2), "ABSTAIN": op.get("ABSTAIN", 0.1)}
                    else:
                        # Default probabilistic distribution
                        p = {"BUY": 0.2, "HOLD": 0.5, "AVOID": 0.2, "ABSTAIN": 0.1}
                
                rel = 0.6
                if isinstance(op.get("calibration_hint"), dict):
                    rel = op.get("calibration_hint", {}).get("reliability", 0.6)
                
                uncertainty = op.get("uncertainty", {})
                if not isinstance(uncertainty, dict):
                    uncertainty = {"epistemic": 0.3, "aleatoric": 0.3}
                epi = uncertainty.get("epistemic", 0.3)
                ale = uncertainty.get("aleatoric", 0.3)
            else:
                # Fallback for non-dict responses
                p = {"BUY": 0.2, "HOLD": 0.5, "AVOID": 0.2, "ABSTAIN": 0.1}
                rel = 0.5
                epi = 0.5
                ale = 0.3
            
            # Weight = reliability * (1 - epistemic_uncertainty)
            weight = rel * (1.0 - epi)
            total_weight += weight
            
            # Accumulate uncertainties
            epistemic_sum += epi
            aleatoric_sum += ale
            
            # Collect evidence safely
            evidence = op.get("evidence", []) if isinstance(op, dict) else []
            if isinstance(evidence, list):
                all_evidence.extend(evidence)
            else:
                all_evidence.append({"name": "unknown_evidence", "strength": 0.5})
            
            # Soft aggregation: weighted log-probabilities
            for a in A:
                prob = max(p.get(a, 0.0), eps)
                scores[a] += weight * math.log(prob)
        
        # Softmax normalization
        if total_weight > 0:
            for a in A:
                scores[a] /= total_weight
                
        score_values = list(scores.values())
        m = max(score_values) if score_values else 0
        exps = {a: math.exp(scores[a] - m) for a in A}
        Z = sum(exps.values())
        
        if Z > 0:
            final_probs = {a: exps[a] / Z for a in A}
        else:
            final_probs = {a: 0.25 for a in A}  # Uniform fallback
        
        # Soft entropy-based ABSTAIN boost (no hard threshold)
        H = -sum(p * math.log(max(p, eps)) for p in final_probs.values()) / math.log(len(A))
        if H > 0.75:
            # Soft shift toward ABSTAIN for high uncertainty
            abstain_boost = 0.07 * (H - 0.75) / 0.25
            final_probs["ABSTAIN"] = min(1.0, final_probs["ABSTAIN"] + abstain_boost)
            
            # Renormalize
            total = sum(final_probs.values())
            if total > 0:
                final_probs = {a: p / total for a, p in final_probs.items()}
        
        # Global uncertainties
        num_agents = max(len(opinions), 1)
        global_epistemic = epistemic_sum / num_agents
        global_aleatoric = aleatoric_sum / num_agents
        
        # Extract top evidence safely
        try:
            evidence_by_strength = sorted([e for e in all_evidence if isinstance(e, dict)], 
                                        key=lambda x: x.get("strength", 0), 
                                        reverse=True)
            top_evidence = [e.get("name", "unknown") for e in evidence_by_strength[:3]]
            if not top_evidence:
                top_evidence = ["probabilistic_analysis", "soft_reasoning", "consensus"]
        except Exception:
            top_evidence = ["analysis_completed", "consensus_reached"]
        
        # Generate rationale based on final probabilities
        max_action = max(final_probs, key=final_probs.get)
        max_prob = final_probs[max_action]
        
        if max_prob < 0.4:
            rationale = f"High uncertainty across agents (entropy={H:.2f}), leaning toward {max_action}"
        else:
            rationale = f"Consensus toward {max_action} (p={max_prob:.2f}) based on {top_evidence[:2]}"
        
        result = {
            "final_probs": final_probs,
            "top_evidence": top_evidence,
            "uncertainty_global": {
                "epistemic": global_epistemic,
                "aleatoric": global_aleatoric
            },
            "rationale": rationale,
            "entropy": H,
            "dominant_action": max_action,
            "confidence": max_prob
        }
        
        print(f"[PROBABILISTIC] DECIDER result: {max_action} ({max_prob:.2f}), entropy={H:.2f}")
        return result

    def probabilistic_consensus(self, detector_breakdown: Dict[str, float], 
                              meta: TokenMeta, trust: TrustProfile, 
                              history: TokenHistory, perf: DetectorPerfStats) -> Dict:
        """
        Main probabilistic consensus function
        """
        symbol = meta.__dict__.get('symbol', 'UNKNOWN')
        print(f"[PROBABILISTIC] Starting consensus for {symbol}")
        
        # Prepare input data for agents
        inputs = {
            "detector_breakdown": detector_breakdown,
            "meta": meta.__dict__,
            "trust": trust.__dict__,
            "history": history.__dict__,
            "perf": perf.__dict__
        }
        
        # Run all agents
        opinions = []
        
        try:
            analyzer_result = self.run_analyzer(inputs)
            opinions.append(analyzer_result)
            print(f"[PROBABILISTIC] ANALYZER: {analyzer_result.get('action_probs', {})}")
        except Exception as e:
            print(f"[PROBABILISTIC] ANALYZER failed: {e}")
        
        try:
            reasoner_result = self.run_reasoner(inputs)
            opinions.append(reasoner_result)
            print(f"[PROBABILISTIC] REASONER: {reasoner_result.get('action_probs', {})}")
        except Exception as e:
            print(f"[PROBABILISTIC] REASONER failed: {e}")
            
        try:
            voter_result = self.run_voter(inputs)
            opinions.append(voter_result)
            print(f"[PROBABILISTIC] VOTER: {voter_result.get('action_probs', {})}")
        except Exception as e:
            print(f"[PROBABILISTIC] VOTER failed: {e}")
            
        try:
            debater_result = self.run_debater(inputs)
            opinions.append(debater_result)
            print(f"[PROBABILISTIC] DEBATER: {debater_result.get('action_probs', {})}")
        except Exception as e:
            print(f"[PROBABILISTIC] DEBATER failed: {e}")
        
        if not opinions:
            print(f"[PROBABILISTIC] No agent opinions available - fallback")
            return {
                "final_probs": {"BUY": 0.2, "HOLD": 0.5, "AVOID": 0.2, "ABSTAIN": 0.1},
                "top_evidence": ["no_data"],
                "uncertainty_global": {"epistemic": 0.9, "aleatoric": 0.5},
                "rationale": "All agents failed - insufficient data",
                "entropy": 1.0,
                "dominant_action": "ABSTAIN",
                "confidence": 0.1
            }
        
        # Aggregate opinions using soft consensus
        final_result = self.decider_aggregate(opinions)
        
        print(f"[PROBABILISTIC] Final consensus: {final_result['dominant_action']} "
              f"(confidence: {final_result['confidence']:.2f})")
        
        return final_result

    def update_agent_reliability(self, agent: str, outcome: str, target_prob: float, actual_prob: float):
        """
        Update agent reliability based on outcome (EMA smoothing, no thresholds)
        """
        if agent not in self.agent_reliability:
            return
            
        # Simple EMA update based on calibration
        alpha = 0.1  # Smoothing factor
        error = abs(target_prob - actual_prob)
        new_reliability = (1 - alpha) * self.agent_reliability[agent] + alpha * (1 - error)
        self.agent_reliability[agent] = max(0.3, min(1.0, new_reliability))
        
        print(f"[PROBABILISTIC] Updated {agent} reliability: {self.agent_reliability[agent]:.3f}")