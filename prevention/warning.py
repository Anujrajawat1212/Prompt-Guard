"""
Security Warning Formatter.
Generates explainable, professional, and transparent security notifications
when an untrusted input is flagged or blocked.
"""
from typing import Dict, Any
from config import ATTACK_CATEGORIES


class SecurityWarningFormatter:
    """Formats explainable security alert messages for users and security logs."""

    @staticmethod
    def format_block_message(security_event: Dict[str, Any]) -> str:
        """Formats an informative markdown warning explaining why the prompt was blocked."""
        category = security_event.get("attack_category", "instruction_override")
        category_desc = ATTACK_CATEGORIES.get(category, "Direct prompt injection attempt")
        risk_score = security_event.get("final_risk_score", 1.0)
        reason = security_event.get("reason", "Malicious prompt pattern detected.")

        return f"""🛡️ **Security Alert: Prompt Injection Blocked**

Your prompt was intercepted by the **Prompt Injection Prevention Layer** and was **not forwarded** to the AI response generator.

* **Detected Attack Category:** `{category.upper()}`
* **Category Description:** {category_desc}
* **Calculated Risk Score:** `{risk_score:.2f}` / 1.00
* **Detection Reason:** {reason}

> *Note: This security event has been recorded in the local SQLite audit log.*
"""

    @staticmethod
    def format_uncertain_warning(security_event: Dict[str, Any]) -> str:
        """Formats a warning for borderline or uncertain prompts."""
        risk_score = security_event.get("final_risk_score", 0.5)
        reason = security_event.get("reason", "Ambiguous prompt structure")

        return f"""⚠️ **Security Warning: Elevated Suspicion Detected**

Your query exhibited structural or semantic patterns resembling prompt injection (Risk Score: `{risk_score:.2f}`). 
* **Reason:** {reason}
* **Action:** Proceeding with heightened security constraints.
"""
