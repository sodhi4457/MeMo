"""LLM clients shared by Step A (Generator) and Step C (Executive).

Two backends are supported and share the same callable contract:
    client(prompt: str) -> str

- GeminiClient   : Google Gemini API (default; needs GOOGLE_API_KEY)
- OllamaClient   : Local Ollama server (default model gemma3:4b)

Both clients accept ``json_mode`` (default True for Step A / Generator,
False for Step C / Executive).  When False, the response_mime_type /
format constraint is dropped so the model can return free-form text.
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
        json_mode: bool = True,
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
        self.json_mode = json_mode

    def __call__(self, prompt: str) -> str:
        cfg_kwargs: dict = {"temperature": self.temperature}
        if self.json_mode:
            cfg_kwargs["response_mime_type"] = "application/json"
        cfg = self._genai_types.GenerateContentConfig(**cfg_kwargs)
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
        json_mode: bool = True,
    ):
        import ollama

        host = host or os.environ.get("OLLAMA_HOST")
        self.client = ollama.Client(host=host) if host else ollama.Client()
        self.model = model or os.environ.get("OLLAMA_MODEL", "gemma3:4b")
        self.temperature = temperature
        self.json_mode = json_mode

    def __call__(self, prompt: str) -> str:
        kwargs: dict = {
            "model": self.model,
            "prompt": prompt,
            "options": {"temperature": self.temperature},
        }
        if self.json_mode:
            kwargs["format"] = "json"
        resp = self.client.generate(**kwargs)
        return (resp.get("response") or "").strip()


def get_client(
    backend: str = "gemini",
    model: str | None = None,
    json_mode: bool = True,
) -> LLMClient:
    """Factory: returns a configured client for the named backend.

    Args:
        json_mode: Force JSON output format (True for Step A Generator,
                   False for Step C Executive which needs free-form text).
    """
    backend = backend.lower()
    if backend == "gemini":
        return GeminiClient(model=model, json_mode=json_mode)
    if backend == "ollama":
        return OllamaClient(model=model, json_mode=json_mode)
    raise ValueError(f"Unknown backend {backend!r}. Use 'gemini' or 'ollama'.")
