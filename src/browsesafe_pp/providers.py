"""One OpenAI-compatible Chat Completions client shared by every model-backed component.

Configuration comes from arguments or the environment:

* ``BROWSESAFE_MODEL_ENDPOINT``: API base URL, for example ``https://api.openai.com/v1``
  (``/chat/completions`` is appended unless the URL already ends with it).
* ``BROWSESAFE_API_KEY``: bearer token, sent as ``Authorization: Bearer ...`` when set.
* ``BROWSESAFE_MODEL``: default model name.

Requires ``httpx`` (``pip install 'browsesafe-pp[eval]'``). Requests are never retried
implicitly; failures raise :class:`ProviderError` carrying the usage that was incurred.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from .contracts import Usage

ENDPOINT_ENV = "BROWSESAFE_MODEL_ENDPOINT"
API_KEY_ENV = "BROWSESAFE_API_KEY"
MODEL_ENV = "BROWSESAFE_MODEL"


class ProviderError(RuntimeError):
    """A failed or unusable completion. ``category`` names the failure; ``usage`` is what it cost."""

    def __init__(self, message: str, usage: Usage | None = None, *, category: str = "provider_error") -> None:
        super().__init__(message)
        self.usage = usage
        self.category = category


@dataclass(frozen=True)
class ToolCall:
    id: str | None
    name: str
    arguments: str


@dataclass(frozen=True)
class ChatResponse:
    text: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str | None
    usage: Usage
    raw: Mapping[str, Any] = field(repr=False, default_factory=dict)


def _url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    return endpoint if endpoint.endswith("/chat/completions") else endpoint + "/chat/completions"


class OpenAICompatible:
    """Async client for any endpoint that implements the OpenAI Chat Completions API.

    Args:
        endpoint: API base URL; defaults to ``$BROWSESAFE_MODEL_ENDPOINT``.
        model: model name; defaults to ``$BROWSESAFE_MODEL``.
        api_key: bearer token; defaults to ``$BROWSESAFE_API_KEY`` (optional for local servers).
        timeout: per-request timeout in seconds.
        token_limit_field: ``"max_completion_tokens"`` (default) or ``"max_tokens"`` for older servers.
        prices_per_million: optional ``(input, output)`` USD prices used to estimate cost when the
            provider does not report it.
        extra_body: extra request fields (for example ``{"reasoning_effort": "low"}``).
        transport: optional ``httpx`` transport, for tests or custom networking.
    """

    def __init__(self, endpoint: str | None = None, model: str | None = None, api_key: str | None = None, *,
                 timeout: float = 120, token_limit_field: str = "max_completion_tokens",
                 prices_per_million: tuple[float, float] | None = None,
                 extra_body: Mapping[str, Any] | None = None, transport: Any = None) -> None:
        endpoint = endpoint or os.environ.get(ENDPOINT_ENV)
        model = model or os.environ.get(MODEL_ENV)
        if not endpoint:
            raise ValueError(f"No model endpoint: pass endpoint= or set {ENDPOINT_ENV}")
        if not model:
            raise ValueError(f"No model name: pass model= or set {MODEL_ENV}")
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
            raise ValueError("endpoint must be an HTTP(S) URL without embedded credentials")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if token_limit_field not in {"max_completion_tokens", "max_tokens"}:
            raise ValueError("token_limit_field must be max_completion_tokens or max_tokens")
        if prices_per_million is not None and (len(prices_per_million) != 2 or any(
                type(x) not in {int, float} or not math.isfinite(x) or x < 0 for x in prices_per_million)):
            raise ValueError("prices_per_million must contain nonnegative input and output prices")
        extra = dict(extra_body or {})
        if set(extra) & {"model", "messages", "stream", "max_tokens", "max_completion_tokens"}:
            raise ValueError("extra_body cannot replace model, messages, stream or token limits")
        self.url = _url(endpoint)
        self.model = model
        self._api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
        self.timeout = timeout
        self.token_limit_field = token_limit_field
        self.prices = prices_per_million
        self.extra_body = extra
        self._transport = transport
        self._clients: dict[int, Any] = {}

    @property
    def configuration(self) -> dict[str, Any]:
        """Recordable settings; the API key is never included."""
        parsed = urlsplit(self.url)
        return {"adapter": "openai_chat_completions", "model": self.model,
                "endpoint": f"{parsed.scheme}://{parsed.netloc}{parsed.path}",
                "token_limit_field": self.token_limit_field, "prices_per_million": self.prices,
                "extra_body": self.extra_body, "retry_policy": "none"}

    def _client(self):
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError("Model providers need httpx: pip install 'browsesafe-pp[eval]'") from exc
        key = id(asyncio.get_running_loop())
        client = self._clients.get(key)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(timeout=self.timeout, follow_redirects=False, transport=self._transport)
            self._clients[key] = client
        return client

    async def aclose(self) -> None:
        """Close pooled connections opened in the running event loop."""
        client = self._clients.pop(id(asyncio.get_running_loop()), None)
        if client is not None:
            await client.aclose()

    def _usage(self, result: Mapping[str, Any]) -> Usage:
        reported = result.get("usage") or {}
        tokens_in, tokens_out = reported.get("prompt_tokens"), reported.get("completion_tokens")
        tokens_in = tokens_in if type(tokens_in) is int and tokens_in >= 0 else None
        tokens_out = tokens_out if type(tokens_out) is int and tokens_out >= 0 else None
        cost = reported.get("cost_usd", reported.get("cost"))
        cost = float(cost) if type(cost) in {int, float} and math.isfinite(cost) and cost >= 0 else None
        basis = "provider_reported" if cost is not None else None
        if cost is None and self.prices and tokens_in is not None and tokens_out is not None:
            cost = (tokens_in * self.prices[0] + tokens_out * self.prices[1]) / 1_000_000
            basis = "configured_token_rates"
        return Usage(1, tokens_in, tokens_out, cost, cost_basis=basis)

    async def chat(self, messages: Sequence[Mapping[str, Any]], *, max_tokens: int = 1024,
                   temperature: float | None = None, tools: Sequence[Mapping[str, Any]] | None = None,
                   tool_choice: Any = None, response_format: Mapping[str, Any] | None = None,
                   seed: int | None = None) -> ChatResponse:
        """Send one Chat Completions request and return its text, tool calls and usage."""
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        body: dict[str, Any] = {"model": self.model, "messages": list(messages),
                                self.token_limit_field: max_tokens, "stream": False, **self.extra_body}
        for name, value in (("temperature", temperature), ("tools", list(tools) if tools else None),
                            ("tool_choice", tool_choice), ("response_format", response_format),
                            ("seed", seed)):
            if value is not None:
                body[name] = value
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        usage = Usage(1, None, None, None)
        category = "request_error"
        try:
            response = await self._client().post(self.url, json=body, headers=headers)
            category = "http_error"
            response.raise_for_status()
            category = "invalid_response"
            result = response.json()
            usage = self._usage(result)
            choice = result["choices"][0]
            message = choice.get("message") or {}
            finish = choice.get("finish_reason")
            if finish == "length":
                raise ProviderError("Response reached the output token limit", usage, category="truncated")
            if finish == "content_filter" or message.get("refusal"):
                raise ProviderError("Provider refused the request", usage, category="refusal")
            calls = tuple(ToolCall(call.get("id"), call["function"]["name"], call["function"].get("arguments") or "")
                          for call in (message.get("tool_calls") or []))
            return ChatResponse(message.get("content") or "", calls, finish, usage, result)
        except asyncio.CancelledError as exc:
            exc.usage = usage  # type: ignore[attr-defined]
            raise
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Model request failed: {type(exc).__name__}", usage, category=category) from None


def decode_json(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating a surrounding Markdown code fence."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:])
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


__all__ = ["API_KEY_ENV", "ENDPOINT_ENV", "MODEL_ENV", "ChatResponse", "OpenAICompatible", "ProviderError",
           "ToolCall", "decode_json"]
