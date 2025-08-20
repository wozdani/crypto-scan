# consensus/decider.py
import math
from typing import List
from .contracts import AgentOpinion, FinalDecision, Uncertainty

ACTIONS = ["BUY","HOLD","AVOID","ABSTAIN"]

def aggregate(opinions: List[AgentOpinion]) -> FinalDecision:
    eps = 1e-6
    scores = {a: 0.0 for a in ACTIONS}
    weights = []
    for op in opinions:
        rel = op.calibration_hint.reliability
        epi = op.uncertainty.epistemic
        w = max(0.0, rel * (1.0 - epi))
        weights.append(w)
        for a in ACTIONS:
            p = max(op.action_probs.get(a, 0.0), eps)
            scores[a] += w * math.log(p)
    m = max(scores.values())
    exps = {a: math.exp(scores[a]-m) for a in ACTIONS}
    Z = sum(exps.values()) or 1.0
    final_probs = {a: exps[a]/Z for a in ACTIONS}

    # global uncertainty = średnia
    if not weights: weights = [1.0]
    avg_epi = sum(op.uncertainty.epistemic for op in opinions)/len(opinions)
    avg_ale = sum(op.uncertainty.aleatoric for op in opinions)/len(opinions)

    # lekki shift w stronę ABSTAIN przy wysokiej entropii (miękko)
    H = -sum(p*math.log(max(p,eps)) for p in final_probs.values())/math.log(len(ACTIONS))
    if H > 0.75:
        final_probs["ABSTAIN"] += 0.07 * (H-0.75)/0.25
        S = sum(final_probs.values())
        for a in ACTIONS: final_probs[a] /= S

    # top evidence (z 2-3 agentów o najwyższej wadze)
    top = []
    top_ops = sorted(zip(opinions, weights), key=lambda x:x[1], reverse=True)[:3]
    for op,_ in top_ops:
        for ev in op.evidence[:2]:
            tag = f"{ev.direction}:{ev.name}"
            if tag not in top:
                top.append(tag)
            if len(top) >= 5: break
        if len(top) >= 5: break

    return FinalDecision(
        final_probs=final_probs,
        uncertainty_global=Uncertainty(epistemic=avg_epi, aleatoric=avg_ale),
        top_evidence=top,
        rationale="Miękka agregacja rozkładów agentów z wagami reliability*(1-epistemic)."
    )