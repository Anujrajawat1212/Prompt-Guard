"""
Input Preprocessing Module.
Normalizes text, standardizes encodings, and extracts obfuscated or hidden payloads
without modifying the audited original prompt.
"""
import re
import base64
import binascii
import urllib.parse
import unicodedata
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class PreprocessedResult:
    original_prompt: str
    normalized_prompt: str
    sanitized_for_detection: str
    decoded_payloads: List[str] = field(default_factory=list)
    is_empty: bool = False
    length: int = 0
    has_homoglyphs: bool = False
    has_excessive_whitespace: bool = False
    has_prompt_delimiters: bool = False
    delimiter_matches: List[str] = field(default_factory=list)
    encoding_flags: List[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "normalized_prompt": self.normalized_prompt,
            "sanitized_for_detection": self.sanitized_for_detection,
            "decoded_payloads": self.decoded_payloads,
            "is_empty": self.is_empty,
            "length": self.length,
            "has_homoglyphs": self.has_homoglyphs,
            "has_excessive_whitespace": self.has_excessive_whitespace,
            "has_prompt_delimiters": self.has_prompt_delimiters,
            "delimiter_matches": self.delimiter_matches,
            "encoding_flags": self.encoding_flags,
            "latency_ms": self.latency_ms,
        }


class PreprocessingPipeline:
    """Preprocesses raw prompts for security evaluation."""

    DELIMITER_PATTERNS = [
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"\[INST\]",
        r"\[/INST\]",
        r"<<SYS>>",
        r"<</SYS>>",
        r"###\s*System:?",
        r"---\s*(?:END OF )?SYSTEM(?: PROMPT)?\s*---",
        r"BEGIN_ADMIN_BLOCK",
        r"END_ADMIN_BLOCK",
        r"\[\[SYSTEM\s*(?:COMMAND|OVERRIDE)?\]\]",
        r"</system_prompt>",
        r"<system_prompt>",
    ]

    def __init__(self):
        self._compiled_delimiters = [
            re.compile(p, re.IGNORECASE) for p in self.DELIMITER_PATTERNS
        ]

    def process(self, raw_prompt: str) -> PreprocessedResult:
        """Executes the complete preprocessing pipeline."""
        start_time = time.perf_counter()

        if raw_prompt is None:
            raw_prompt = ""

        original = str(raw_prompt)
        length = len(original)

        if not original.strip():
            latency = (time.perf_counter() - start_time) * 1000
            return PreprocessedResult(
                original_prompt=original,
                normalized_prompt="",
                sanitized_for_detection="",
                is_empty=True,
                length=0,
                latency_ms=latency,
            )

        # 1. Unicode Normalization (NFKC to resolve compatibility characters and canonical equivalences)
        unicode_normalized = unicodedata.normalize("NFKC", original)
        has_homoglyphs = unicode_normalized != original

        # 2. Check for delimiter injection
        delimiter_matches = []
        for pattern in self._compiled_delimiters:
            matches = pattern.findall(unicode_normalized)
            if matches:
                delimiter_matches.extend(matches)
        has_prompt_delimiters = len(delimiter_matches) > 0

        # 3. Excessive whitespace check & whitespace normalization
        has_excessive_whitespace = bool(
            re.search(r"\n{3,}", unicode_normalized)
            or re.search(r" {4,}", unicode_normalized)
            or re.search(r"(?:[a-zA-Z] ){5,}[a-zA-Z]", unicode_normalized)
        )

        # Standard whitespace collapsed text
        normalized = re.sub(r"\s+", " ", unicode_normalized).strip()

        # Handle spaced characters e.g. "I g n o r e" -> "Ignore"
        de_spaced = re.sub(r"(?<=\b[a-zA-Z])\s(?=[a-zA-Z]\b)", "", normalized)

        # 4. Obfuscation checks (Base64, Hex, URL, Rot13)
        decoded_payloads = []
        encoding_flags = []

        # URL Decoding
        if "%" in normalized:
            try:
                url_decoded = urllib.parse.unquote(normalized)
                if url_decoded != normalized and any(
                    c.isalnum() for c in url_decoded
                ):
                    decoded_payloads.append(url_decoded)
                    encoding_flags.append("URL_ENCODED")
            except Exception:
                pass

        # Base64 Extraction
        b64_candidates = re.findall(
            r"(?:[A-Za-z0-9+/]{4}){3,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?",
            normalized,
        )
        for cand in b64_candidates:
            if len(cand) >= 12:
                try:
                    decoded_bytes = base64.b64decode(cand, validate=True)
                    decoded_str = decoded_bytes.decode("utf-8", errors="ignore")
                    # Check if decoded string is meaningful printable text
                    if len(decoded_str) >= 4 and sum(c.isalnum() or c.isspace() for c in decoded_str) / len(decoded_str) > 0.8:
                        decoded_payloads.append(decoded_str)
                        encoding_flags.append("BASE64")
                except Exception:
                    pass

        # Hex Extraction
        hex_candidates = re.findall(r"\b[0-9a-fA-F]{16,}\b", normalized)
        for hex_cand in hex_candidates:
            try:
                hex_bytes = bytes.fromhex(hex_cand)
                hex_str = hex_bytes.decode("utf-8", errors="ignore")
                if len(hex_str) >= 4 and sum(c.isalnum() or c.isspace() for c in hex_str) / len(hex_str) > 0.8:
                    decoded_payloads.append(hex_str)
                    encoding_flags.append("HEX")
            except Exception:
                pass

        # Rot13 Scanner for obvious rot13 payloads
        if "rot13" in normalized.lower():
            try:
                import codecs
                rot_decoded = codecs.decode(normalized, "rot_13")
                decoded_payloads.append(rot_decoded)
                encoding_flags.append("ROT13")
            except Exception:
                pass

        # 5. Build sanitized detection representation
        detection_parts = [normalized]
        if de_spaced != normalized:
            detection_parts.append(de_spaced)
        for payload in decoded_payloads:
            detection_parts.append(payload)

        sanitized_for_detection = " || ".join(list(dict.fromkeys(detection_parts)))

        latency = (time.perf_counter() - start_time) * 1000

        return PreprocessedResult(
            original_prompt=original,
            normalized_prompt=normalized,
            sanitized_for_detection=sanitized_for_detection,
            decoded_payloads=list(set(decoded_payloads)),
            is_empty=False,
            length=length,
            has_homoglyphs=has_homoglyphs,
            has_excessive_whitespace=has_excessive_whitespace,
            has_prompt_delimiters=has_prompt_delimiters,
            delimiter_matches=delimiter_matches,
            encoding_flags=list(set(encoding_flags)),
            latency_ms=latency,
        )
