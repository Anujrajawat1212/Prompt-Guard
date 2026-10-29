"""
Suspicion Score Aggregator & Decision Engine.
Fuses signals from the Rule Detector, ML Classifier, and Llama Semantic Evaluator
into an aggregated risk metric, determining whether prompts are SAFE, MALICIOUS, or UNCERTAIN.
"""
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    WEIGHT_RULE,
    WEIGHT_ML,
    WEIGHT_LLM,
    SAFE_THRESHOLD,
    MALICIOUS_THRESHOLD,
    UNCERTAIN_POLICY,
)
from detection.rule_detector import RuleDetectionResult
from detection.ml_detector import MLDetectionResult
from detection.llama_detector import LlamaDetectionResult


@dataclass
class AggregatedSecurityDecision:
    final_risk_score: float
    classification: str          # "SAFE" | "MALICIOUS" | "UNCERTAIN"
    decision: str                # "ALLOWED" | "BLOCKED" | "WARNED"
    attack_category: str
    reason: str
    rule_score: float
    ml_score: float
    llm_score: float
    confidence: float
    rule_result: RuleDetectionResult
    ml_result: MLDetectionResult
    llm_result: LlamaDetectionResult
    total_detection_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_risk_score": self.final_risk_score,
            "classification": self.classification,
            "decision": self.decision,
            "attack_category": self.attack_category,
            "reason": self.reason,
            "rule_score": self.rule_score,
            "ml_score": self.ml_score,
            "llm_score": self.llm_score,
            "confidence": self.confidence,
            "total_detection_latency_ms": self.total_detection_latency_ms,
            "rule_details": self.rule_result.to_dict(),
            "ml_details": self.ml_result.to_dict(),
            "llm_details": self.llm_result.to_dict(),
        }


class SuspicionScoreAggregator:
    """Combines multi-modal detection signals and produces actionable security decisions."""

    def __init__(
        self,
        w_rule: float = WEIGHT_RULE,
        w_ml: float = WEIGHT_ML,
        w_llm: float = WEIGHT_LLM,
        safe_threshold: float = SAFE_THRESHOLD,
        malicious_threshold: float = MALICIOUS_THRESHOLD,
        uncertain_policy: str = UNCERTAIN_POLICY,
    ):
        self.w_rule = w_rule
        self.w_ml = w_ml
        self.w_llm = w_llm
        self.safe_threshold = safe_threshold
        self.malicious_threshold = malicious_threshold
        self.uncertain_policy = uncertain_policy

    def aggregate(
        self,
        rule_res: RuleDetectionResult,
        ml_res: MLDetectionResult,
        llm_res: LlamaDetectionResult,
    ) -> AggregatedSecurityDecision:
        """Aggregates scores and enforces 3-way decision thresholds."""
        start_time = time.perf_counter()

        # Weighted score fusion
        # Note: If high-severity rule patterns or severe delimiter injections are triggered, apply floor guarantee
        fused_score = (
            self.w_rule * rule_res.score
            + self.w_ml * ml_res.probability
            + self.w_llm * llm_res.score
        )

        # Fail-safe boost: If 2 out of 3 detectors strongly trigger, elevate risk score
        strong_votes = sum([
            rule_res.score >= 0.70,
            ml_res.probability >= 0.70,
            llm_res.score >= 0.70,
        ])
        if strong_votes >= 2:
            fused_score = max(fused_score, 0.85)

        # Educational safeguard: If user is asking an educational question and ML/Llama agree it is safe
        if rule_res.is_educational_inquiry and ml_res.probability < 0.40 and llm_res.classification == "SAFE":
            fused_score = min(fused_score, 0.20)

        final_risk_score = round(min(1.0, max(0.0, fused_score)), 4)

        # 3-Way Classification
        if final_risk_score < self.safe_threshold:
            classification = "SAFE"
            decision = "ALLOWED"
        elif final_risk_score > self.malicious_threshold:
            classification = "MALICIOUS"
            decision = "BLOCKED"
        else:
            classification = "UNCERTAIN"
            decision = "BLOCKED" if self.uncertain_policy == "BLOCK" else "WARNED"

        # Resolve primary attack category
        if classification == "SAFE":
            attack_category = "false_positive_trap" if rule_res.is_educational_inquiry else "benign"
        else:
            if rule_res.primary_category and rule_res.primary_category not in ["benign", "false_positive_trap"]:
                attack_category = rule_res.primary_category
            elif llm_res.attack_category and llm_res.attack_category != "benign":
                attack_category = llm_res.attack_category
            else:
                attack_category = "instruction_override"

        # Build Explainable Reason
        reasons = []
        if rule_res.matched_patterns:
            reasons.append(f"Matched rule patterns: {', '.join(rule_res.matched_patterns[:2])}")
        if ml_res.probability >= 0.50:
            reasons.append(f"ML semantic risk ({ml_res.probability:.2f})")
        if llm_res.reason and llm_res.classification == "MALICIOUS":
            reasons.append(llm_res.reason)

        if not reasons:
            if classification == "SAFE":
                reason = "No prompt injection indicators detected; prompt verified safe."
            elif classification == "UNCERTAIN":
                reason = f"Borderline suspicion score ({final_risk_score:.2f}) under {self.uncertain_policy} policy."
            else:
                reason = "Aggregated security risk score exceeded safe thresholds."
        else:
            reason = " | ".join(reasons)

        # Calculate overall confidence
        confidence = round((rule_res.score + ml_res.confidence + llm_res.confidence) / 3.0, 4)
        total_latency = (time.perf_counter() - start_time) * 1000 + rule_res.latency_ms + ml_res.latency_ms + llm_res.latency_ms

        return AggregatedSecurityDecision(
            final_risk_score=final_risk_score,
            classification=classification,
            decision=decision,
            attack_category=attack_category,
            reason=reason,
            rule_score=rule_res.score,
            ml_score=ml_res.probability,
            llm_score=llm_res.score,
            confidence=confidence,
            rule_result=rule_res,
            ml_result=ml_res,
            llm_result=llm_res,
            total_detection_latency_ms=round(total_latency, 2),
        )
