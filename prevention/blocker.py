"""
Security Prevention Blocker.
Guarantees the critical security boundary: malicious requests are intercepted
and halted before any tokens or parameters are transmitted to the response generation LLM.
"""
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection.scorer import AggregatedSecurityDecision


@dataclass
class PreventionResult:
    is_blocked: bool
    action_taken: str          # "BLOCKED" | "ALLOWED" | "WARNED"
    block_reason: str
    security_event: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_blocked": self.is_blocked,
            "action_taken": self.action_taken,
            "block_reason": self.block_reason,
            "security_event": self.security_event,
        }


class SecurityBlocker:
    """Enforces blocking and prevention policies prior to LLM generator dispatch."""

    def evaluate(self, decision: AggregatedSecurityDecision) -> PreventionResult:
        """Determines if the prompt must be halted or forwarded."""
        if decision.decision == "BLOCKED":
            return PreventionResult(
                is_blocked=True,
                action_taken="BLOCKED",
                block_reason=decision.reason,
                security_event=decision.to_dict(),
            )
        elif decision.decision == "WARNED":
            return PreventionResult(
                is_blocked=False,
                action_taken="WARNED",
                block_reason=decision.reason,
                security_event=decision.to_dict(),
            )
        else:
            return PreventionResult(
                is_blocked=False,
                action_taken="ALLOWED",
                block_reason="",
                security_event=decision.to_dict(),
            )
