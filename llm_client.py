"""
Lightweight OpenAI-compatible LLM client for the Agentic Chunker.

Zero external dependencies beyond the standard library (like the
doc-sanitiser client.py pattern). Uses ``urllib.request`` and ``json``
only. Supports:
  - Chat completions via POST /v1/chat/completions
  - Model auto-detection via GET /v1/models
  - Retry with exponential backoff (429) and flat retry (5xx)
  - ``<think>`` tag stripping (Qwen3 safety net)
  - ``/no_think`` directive appending (configurable)
  - Markdown fence stripping from responses
"""

import json
import logging
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agentic-chunker.llm")


# ===================================================================
# Pure helpers
# ===================================================================

def strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` blocks from LLM output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def strip_markdown_fences(text: str) -> str:
    """Remove wrapping markdown code fences from LLM output."""
    stripped = text.strip()
    match = re.match(
        r"^```(?:\w+)?\s*\n(.*?)\n\s*```\s*$",
        stripped,
        flags=re.DOTALL,
    )
    if match:
        return match.group(1)
    return text


# ===================================================================
# Model detection
# ===================================================================

def detect_model(base_url: str, api_key: Optional[str] = None) -> str:
    """Hit ``/v1/models`` and return the first served model ID.

    Raises:
        RuntimeError: If the endpoint is unreachable or returns no models.
    """
    url = f"{base_url}/models"
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data", [])
            if models:
                return models[0]["id"]
            raise RuntimeError("No models returned from /v1/models")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp else ""
        raise RuntimeError(f"Model detection failed: HTTP {exc.code} — {body}")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Model detection failed: {exc.reason}")


# ===================================================================
# LLMClient
# ===================================================================

class LLMClient:
    """OpenAI-compatible LLM client — stdlib only.

    Usage::

        client = LLMClient(
            base_url="http://host:8000/v1",
            api_key="key",
            model="Qwen/Qwen3-235B-A22B",
        )
        answer = client.generate("What colour is the sky?")
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        model: str = "auto",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 120,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        no_think: bool = True,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.no_think = no_think

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Send a single user prompt and return the assistant response.

        Args:
            prompt: User message content.
            **kwargs: Optional overrides — ``temperature``, ``max_tokens``.

        Returns:
            Assistant response text (think-tags and fences stripped).
        """
        messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]

        # Append /no_think to last user message if enabled
        if self.no_think:
            last_msg = messages[-1]
            if not last_msg["content"].rstrip().endswith("/no_think"):
                last_msg["content"] = last_msg["content"] + " /no_think"

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        raw = self._request("/chat/completions", payload)
        content = raw["choices"][0]["message"]["content"]

        # Post-processing safety nets
        content = strip_think_tags(content)
        content = strip_markdown_fences(content)

        return content.strip()

    def _request(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """HTTP POST with retry logic."""
        url = f"{self.base_url}{path}"

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = json.dumps(payload).encode("utf-8")
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))

            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8") if exc.fp else ""

                if exc.code == 429:
                    wait = (2 ** attempt) * self.retry_base_delay
                    logger.warning("Rate limited (429), retrying in %.1fs", wait)
                    time.sleep(wait)
                    last_error = RuntimeError(f"Rate limited: HTTP 429 — {body}")
                elif exc.code >= 500:
                    logger.warning("Server error (%d), retrying", exc.code)
                    time.sleep(self.retry_base_delay)
                    last_error = RuntimeError(f"Server error: HTTP {exc.code} — {body}")
                else:
                    raise RuntimeError(
                        f"API error: HTTP {exc.code} — {body}"
                    )

            except urllib.error.URLError as exc:
                logger.warning("Connection error: %s, retrying", exc.reason)
                time.sleep(self.retry_base_delay)
                last_error = RuntimeError(f"Connection error: {exc.reason}")

            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON response: {exc}")

        if last_error:
            raise last_error
        raise RuntimeError("Max retries exceeded")
