"""LLM clients used as the Generator (M_gen) in Step A.

Two backends are supported and share the same callable contract:
    client(prompt: str) -> str

- GeminiClient   : Google Gemini API (default; needs GOOGLE_API_KEY)
- OllamaClient   : Local Ollama server (default model gemma3:4b)
"""

from __future__ import annotations

import os
import time
from typing import Protocol


class LLMClient(Protocol):
    def __call__(self, prompt: str) -> str: ...


class GeminiClient:
    """Calls Google Gemini via the official `google-genai` SDK."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.3,
        max_retries: int = 3,
    ):
        from google import genai
        from google.genai import types

        api_key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "Gemini backend requires GOOGLE_API_KEY (or GEMINI_API_KEY) "
                "in the environment or a .env file."
            )

        self._genai_types = types
        self.client = genai.Client(api_key=api_key)
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.temperature = temperature
        self.max_retries = max_retries

    def __call__(self, prompt: str) -> str:
        cfg = self._genai_types.GenerateContentConfig(
            temperature=self.temperature,
            response_mime_type="application/json",
        )
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=cfg,
                )
                return (resp.text or "").strip()
            except Exception as exc:
                last_err = exc
                time.sleep(2**attempt)
        raise RuntimeError(
            f"Gemini API failed after {self.max_retries} attempts: {last_err}"
        )


class OllamaClient:
    """Calls a local Ollama server."""

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        temperature: float = 0.3,
    ):
        import ollama

        host = host or os.environ.get("OLLAMA_HOST")
        self.client = ollama.Client(host=host) if host else ollama.Client()
        self.model = model or os.environ.get("OLLAMA_MODEL", "gemma3:4b")
        self.temperature = temperature

    def __call__(self, prompt: str) -> str:
        resp = self.client.generate(
            model=self.model,
            prompt=prompt,
            format="json",
            options={"temperature": self.temperature},
        )
        return (resp.get("response") or "").strip()


def get_client(
    backend: str = "gemini",
    model: str | None = None,
) -> LLMClient:
    """Factory: returns a configured client for the named backend."""
    backend = backend.lower()
    if backend == "gemini":
        return GeminiClient(model=model)
    if backend == "ollama":
        return OllamaClient(model=model)
    raise ValueError(f"Unknown backend {backend!r}. Use 'gemini' or 'ollama'.")
