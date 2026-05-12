"""Backend protocol + Mock / Groq / Anthropic adapters for judge LLMs."""

from __future__ import annotations

import os
import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class JudgeBackend(Protocol):
    model: str

    def complete(self, prompt: str, system: str | None = None) -> str: ...


class MockJudgeBackend:
    """Deterministic judge for tests. Accepts a ``handler`` callable for full
    control or a ``verdict_map`` keyed by substring."""

    def __init__(self, model: str = "mock", handler=None, verdict_map: dict[str, str] | None = None) -> None:
        self.model = model
        self.handler = handler
        self.verdict_map = verdict_map or {}
        self.calls: list[tuple[str, str | None]] = []

    def complete(self, prompt: str, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        if self.handler is not None:
            return self.handler(prompt, system)
        for needle, ans in self.verdict_map.items():
            if needle in prompt:
                return ans
        return ""


class GroqJudgeBackend:
    def __init__(self, model: str | None = None, api_key: str | None = None, min_interval_secs: float = 2.1) -> None:
        from groq import Groq  # lazy

        self.model = model or os.environ.get("JUDGE_GROQ_MODEL", "llama-3.1-8b-instant")
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self.min_interval_secs = min_interval_secs
        self._last_t = 0.0

    def _wait(self) -> None:
        now = time.monotonic()
        delta = now - self._last_t
        if delta < self.min_interval_secs:
            time.sleep(self.min_interval_secs - delta)
        self._last_t = time.monotonic()

    def complete(self, prompt: str, system: str | None = None) -> str:
        from groq import RateLimitError

        backoff = 4.0
        for attempt in range(5):
            self._wait()
            try:
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=80, temperature=0.0
                )
                return resp.choices[0].message.content or ""
            except RateLimitError:
                if attempt == 4:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
        raise RuntimeError("unreachable")


class AnthropicJudgeBackend:
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        from anthropic import Anthropic  # lazy

        self.model = model or os.environ.get("JUDGE_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, prompt: str, system: str | None = None) -> str:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": 80,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
