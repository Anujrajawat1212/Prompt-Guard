"""
Ollama HTTP REST Client.
Manages communication with the locally running Ollama instance, model health checks,
semantic detection requests, and streaming/non-streaming response generation.
"""
import json
import time
import requests
from typing import Dict, Any, List, Optional, Tuple, Generator

from config import (
    OLLAMA_BASE_URL,
    DEFAULT_MODEL,
    DETECTOR_MODEL,
    OLLAMA_REQUEST_TIMEOUT,
    GENERATOR_SYSTEM_PROMPT,
)


class OllamaClient:
    """Client for local Ollama API communication."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.timeout = OLLAMA_REQUEST_TIMEOUT
        self._last_health_check_time = 0
        self._is_online = False
        self._cached_models = []

    def check_health(self, force: bool = False) -> Tuple[bool, str, List[str]]:
        """Checks if Ollama service is reachable and lists available models."""
        now = time.time()
        if not force and (now - self._last_health_check_time < 5):
            return self._is_online, "Cached health status", self._cached_models

        self._last_health_check_time = now
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2.5)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name") for m in data.get("models", [])]
                self._is_online = True
                self._cached_models = models
                return True, "Ollama service is running locally.", models
            self._is_online = False
            self._cached_models = []
            return False, f"Ollama returned HTTP status {resp.status_code}", []
        except Exception:
            self._is_online = False
            self._cached_models = []
            return False, "Ollama service is not running or unreachable at localhost:11434.", []

    def stream_chat_response(
        self,
        messages: List[Dict[str, str]],
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """Streams tokens from local Llama 3.2 model in real-time."""
        is_up, _, _ = self.check_health()
        if not is_up:
            yield "⚠️ **Ollama Offline**: Local Ollama server is not reachable at `http://localhost:11434`."
            return

        formatted_messages = []
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            formatted_messages.append({"role": "system", "content": GENERATOR_SYSTEM_PROMPT})
        formatted_messages.extend(messages)

        payload = {
            "model": model,
            "messages": formatted_messages,
            "stream": True,
            "options": {
                "temperature": temperature,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                for line in resp.iter_lines():
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
            else:
                yield f"⚠️ Generation error (HTTP {resp.status_code}): {resp.text}"
        except requests.exceptions.Timeout:
            yield "⚠️ Generation timed out. The local LLM took too long to respond."
        except Exception as e:
            yield f"⚠️ Generation error: {str(e)}"

    def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Sends chat conversation turns to Llama 3.2 model for complete response generation."""
        start_time = time.perf_counter()
        
        is_up, status_msg, _ = self.check_health()
        if not is_up:
            latency = (time.perf_counter() - start_time) * 1000
            return {
                "success": False,
                "response": "⚠️ **Ollama Offline**: Local Ollama server is not running at `http://localhost:11434`. Please start Ollama.",
                "model": model,
                "latency_ms": latency,
                "error": "Ollama connection refused",
            }

        formatted_messages = []
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            formatted_messages.append({"role": "system", "content": GENERATOR_SYSTEM_PROMPT})
        formatted_messages.extend(messages)

        payload = {
            "model": model,
            "messages": formatted_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            latency = (time.perf_counter() - start_time) * 1000

            if resp.status_code == 200:
                res_data = resp.json()
                content = res_data.get("message", {}).get("content", "")
                return {
                    "success": True,
                    "response": content,
                    "model": model,
                    "latency_ms": latency,
                    "error": None,
                }
            else:
                return {
                    "success": False,
                    "response": f"Ollama HTTP error {resp.status_code}",
                    "model": model,
                    "latency_ms": latency,
                    "error": f"Ollama HTTP error {resp.status_code}: {resp.text}",
                }
        except requests.exceptions.Timeout:
            latency = (time.perf_counter() - start_time) * 1000
            return {
                "success": False,
                "response": "⚠️ Generation timed out. The local LLM took too long to compute.",
                "model": model,
                "latency_ms": latency,
                "error": "Request timed out",
            }
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            return {
                "success": False,
                "response": f"Generation failed: {str(e)}",
                "model": model,
                "latency_ms": latency,
                "error": str(e),
            }

    def generate_raw_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: str = DETECTOR_MODEL,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Generates completion for structured evaluation (e.g. LLM-based detection)."""
        start_time = time.perf_counter()
        
        is_up, _, _ = self.check_health()
        if not is_up:
            latency = (time.perf_counter() - start_time) * 1000
            return {
                "success": False,
                "response": "",
                "latency_ms": latency,
                "error": "Ollama service unavailable (offline)",
            }

        payload = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temperature,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30,  # 30s timeout for classifier
            )
            latency = (time.perf_counter() - start_time) * 1000

            if resp.status_code == 200:
                res_data = resp.json()
                response_text = res_data.get("response", "")
                return {
                    "success": True,
                    "response": response_text,
                    "latency_ms": latency,
                    "error": None,
                }
            else:
                return {
                    "success": False,
                    "response": "",
                    "latency_ms": latency,
                    "error": f"Ollama HTTP error {resp.status_code}",
                }
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            return {
                "success": False,
                "response": "",
                "latency_ms": latency,
                "error": str(e),
            }


# Singleton client instance
ollama_client = OllamaClient()
