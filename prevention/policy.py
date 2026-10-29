"""
Security Policy Engine.
Configurable policy checks for maximum prompt length, rate limiting,
and strict safety enforcement for uncertain prompts.
"""
import time
from typing import Dict, Any, Tuple, Optional


class SecurityPolicyEngine:
    """Enforces policy constraints and input safety envelopes."""

    def __init__(
        self,
        max_prompt_chars: int = 4000,
        enable_rate_limiting: bool = False,
        max_requests_per_minute: int = 60,
    ):
        self.max_prompt_chars = max_prompt_chars
        self.enable_rate_limiting = enable_rate_limiting
        self.max_requests_per_minute = max_requests_per_minute
        self._request_history = []

    def check_input_policy(self, raw_prompt: str) -> Tuple[bool, Optional[str]]:
        """Validates that incoming prompts comply with system envelope limits."""
        if raw_prompt is None or not raw_prompt.strip():
            return False, "Input prompt cannot be empty."

        if len(raw_prompt) > self.max_prompt_chars:
            return False, f"Prompt length ({len(raw_prompt)} chars) exceeds policy limit of {self.max_prompt_chars} chars."

        if self.enable_rate_limiting:
            now = time.time()
            self._request_history = [t for t in self._request_history if now - t < 60]
            if len(self._request_history) >= self.max_requests_per_minute:
                return False, "Rate limit exceeded. Please wait before submitting more queries."
            self._request_history.append(now)

        return True, None
