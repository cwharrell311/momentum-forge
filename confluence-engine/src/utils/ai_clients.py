"""
AI provider clients for the Command Center.

Wraps Anthropic (Claude) and OpenAI (GPT/Codex) APIs behind a uniform
interface so the router can call whichever provider is best for the task.

Both clients track usage (token counts, errors) the same way FMPClient
tracks API quota — visible on the system status page.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import anthropic
import openai

log = logging.getLogger("confluence.ai")


@dataclass
class AIResponse:
    """Uniform response from any AI provider."""

    provider: str          # "claude" or "openai"
    model: str             # Actual model used
    content: str           # The response text
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    error: str | None = None


class ClaudeClient:
    """
    Anthropic Claude API client.

    Best for: reasoning about trade setups, explaining confluence scores,
    risk analysis, narrative summaries, multi-step analysis.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model
        self._client: anthropic.AsyncAnthropic | None = None
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0
        self._error_count = 0

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _ensure_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> AIResponse:
        """Send a prompt to Claude and return the response."""
        if not self.is_configured:
            return AIResponse(
                provider="claude", model=self.model, content="",
                error="ANTHROPIC_API_KEY not configured",
            )

        client = self._ensure_client()
        start = time.monotonic()

        try:
            kwargs: dict = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                kwargs["system"] = system

            msg = await client.messages.create(**kwargs)

            self._call_count += 1
            self._total_input_tokens += msg.usage.input_tokens
            self._total_output_tokens += msg.usage.output_tokens

            content = ""
            for block in msg.content:
                if block.type == "text":
                    content += block.text

            return AIResponse(
                provider="claude",
                model=self.model,
                content=content,
                input_tokens=msg.usage.input_tokens,
                output_tokens=msg.usage.output_tokens,
                latency_ms=int((time.monotonic() - start) * 1000),
            )
        except Exception as e:
            self._error_count += 1
            log.error("Claude API error: %s", e)
            return AIResponse(
                provider="claude", model=self.model, content="",
                error=str(e),
                latency_ms=int((time.monotonic() - start) * 1000),
            )

    @property
    def usage_stats(self) -> dict:
        return {
            "provider": "claude",
            "model": self.model,
            "configured": self.is_configured,
            "calls": self._call_count,
            "errors": self._error_count,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
        }

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None


class OpenAIClient:
    """
    OpenAI API client (GPT-4o / Codex).

    Best for: code generation, structured data extraction, function
    calling, quick factual lookups, quantitative calculations.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        self._client: openai.AsyncOpenAI | None = None
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0
        self._error_count = 0

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _ensure_client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            self._client = openai.AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> AIResponse:
        """Send a prompt to OpenAI and return the response."""
        if not self.is_configured:
            return AIResponse(
                provider="openai", model=self.model, content="",
                error="OPENAI_API_KEY not configured",
            )

        client = self._ensure_client()
        start = time.monotonic()

        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            resp = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            self._call_count += 1
            usage = resp.usage
            if usage:
                self._total_input_tokens += usage.prompt_tokens
                self._total_output_tokens += usage.completion_tokens

            content = resp.choices[0].message.content or ""

            return AIResponse(
                provider="openai",
                model=self.model,
                content=content,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                latency_ms=int((time.monotonic() - start) * 1000),
            )
        except Exception as e:
            self._error_count += 1
            log.error("OpenAI API error: %s", e)
            return AIResponse(
                provider="openai", model=self.model, content="",
                error=str(e),
                latency_ms=int((time.monotonic() - start) * 1000),
            )

    @property
    def usage_stats(self) -> dict:
        return {
            "provider": "openai",
            "model": self.model,
            "configured": self.is_configured,
            "calls": self._call_count,
            "errors": self._error_count,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
        }

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
