from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import litellm
from litellm.caching.caching import Cache, LiteLLMCacheType

from app.config import KeyEntry, Settings

logger = logging.getLogger(__name__)

SUPPORTED_SIMULATED_STATUSES = {429, 500, 502, 503}


class AllProvidersUnavailableError(Exception):
    """Raised when every configured API key fails to produce a completion."""


@dataclass(frozen=True)
class GatewayResult:
    reply: str
    resolution_source: str  # "LIVE_INFERENCE" | "LLM_CACHE"
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: Optional[str]
    fallback_attempts: int


def parse_simulate_failure_header(header_value: Optional[str]) -> Dict[str, int]:
    """Parse X-Simulate-Failure header like 'key_1=429,key_2=500' or '*=429'."""
    if not header_value:
        return {}

    parsed: Dict[str, int] = {}
    for raw_pair in header_value.split(","):
        pair = raw_pair.strip()
        if not pair or "=" not in pair:
            continue
        key_part, status_part = pair.split("=", 1)
        key_id = key_part.strip()
        try:
            status_code = int(status_part.strip())
        except ValueError:
            continue
        if key_id and status_code in SUPPORTED_SIMULATED_STATUSES:
            parsed[key_id] = status_code
    return parsed


def build_simulated_exception(status_code: int, key_id: str, model: str) -> Exception:
    provider = model.split("/", 1)[0] if "/" in model else "openai"
    message = f"Simulated upstream HTTP {status_code} for {key_id}"
    if status_code == 429:
        return litellm.RateLimitError(
            message=message,
            llm_provider=provider,
            model=model,
        )
    if status_code == 500:
        return litellm.InternalServerError(
            message=message,
            llm_provider=provider,
            model=model,
        )
    return litellm.ServiceUnavailableError(
        message=message,
        llm_provider=provider,
        model=model,
    )


def is_failover_error(exc: Exception) -> bool:
    """Return True if an exception represents a 429, quota error, 5xx, or transient network issue."""
    if isinstance(
        exc,
        (
            litellm.RateLimitError,
            litellm.InternalServerError,
            litellm.ServiceUnavailableError,
            litellm.BadGatewayError,
            litellm.APIConnectionError,
            litellm.Timeout,
        ),
    ):
        return True

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and (status_code == 429 or 500 <= status_code < 600):
        return True

    lowered = str(exc).lower()
    quota_markers = (
        "429",
        "rate limit",
        "ratelimit",
        "quota",
        "resource_exhausted",
        "insufficient_quota",
        "too many requests",
        "service unavailable",
        "bad gateway",
        "internal server error",
        "overloaded",
    )
    return any(marker in lowered for marker in quota_markers)


def _extract_usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return 0, 0

    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        return int(prompt_tokens), int(completion_tokens)

    prompt_tokens = getattr(usage, "prompt_tokens", None)
    if prompt_tokens is None:
        prompt_tokens = getattr(usage, "input_tokens", 0)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if completion_tokens is None:
        completion_tokens = getattr(usage, "output_tokens", 0)
    return int(prompt_tokens or 0), int(completion_tokens or 0)


def _extract_reply(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        return ""
    first = choices[0]
    message = getattr(first, "message", None)
    if message is None and isinstance(first, dict):
        message = first.get("message")
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    return (content or "").strip()


def _is_cache_hit(response: Any) -> bool:
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict) and hidden.get("cache_hit") is True:
        return True
    if isinstance(response, dict):
        hidden_dict = response.get("_hidden_params")
        if isinstance(hidden_dict, dict) and hidden_dict.get("cache_hit") is True:
            return True
    return False


class LLMGateway:
    def __init__(
        self,
        settings: Settings,
        completion_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.settings = settings
        self.completion_fn = completion_fn or litellm.completion
        self.cache_enabled = True

        # Suppress noisy telemetry/logging
        litellm.telemetry = False

        if settings.cache_type == "disk":
            litellm.cache = Cache(
                type=LiteLLMCacheType.DISK,
                disk_cache_dir=settings.cache_dir,
                ttl=settings.cache_ttl_seconds,
            )
        else:
            litellm.cache = Cache(
                type=LiteLLMCacheType.LOCAL,
                ttl=settings.cache_ttl_seconds,
            )

    def generate(
        self,
        messages: List[Dict[str, str]],
        simulate_failure_header: Optional[str] = None,
    ) -> GatewayResult:
        if not self.settings.key_pool:
            raise AllProvidersUnavailableError("No API keys configured")

        simulated_map: Dict[str, int] = {}
        if self.settings.enable_failure_simulation and simulate_failure_header:
            simulated_map = parse_simulate_failure_header(simulate_failure_header)

        simulation_active = len(simulated_map) > 0
        start = time.perf_counter()
        fallback_attempts = 0

        for entry in self.settings.key_pool:
            try:
                simulated_status = simulated_map.get("*") or simulated_map.get(entry.key_id)
                if simulated_status is not None:
                    raise build_simulated_exception(
                        status_code=simulated_status,
                        key_id=entry.key_id,
                        model=entry.model,
                    )

                kwargs: Dict[str, Any] = {
                    "model": entry.model,
                    "messages": messages,
                    "api_key": entry.api_key,
                    "timeout": 30,
                }
                if simulation_active:
                    kwargs["cache"] = {"no-cache": True, "no-store": True}
                else:
                    kwargs["caching"] = True

                response = self.completion_fn(**kwargs)
                reply = _extract_reply(response)
                if not reply:
                    raise litellm.APIError(
                        message="Empty response content from model",
                        llm_provider=entry.model.split("/", 1)[0],
                        model=entry.model,
                        status_code=502,
                    )

                latency_ms = max(0, int(round((time.perf_counter() - start) * 1000)))
                input_tokens, output_tokens = _extract_usage(response)
                cache_hit = (not simulation_active) and _is_cache_hit(response)
                resolved_model = getattr(response, "model", None) or entry.model

                # Preserve the configured LiteLLM provider-prefixed model name (e.g. groq/openai/gpt-oss-20b)
                provider_prefix = entry.model.split("/", 1)[0] if "/" in entry.model else ""
                if provider_prefix and not str(resolved_model).startswith(f"{provider_prefix}/"):
                    resolved_model = entry.model

                if cache_hit:
                    return GatewayResult(
                        reply=reply,
                        resolution_source="LLM_CACHE",
                        model=str(resolved_model),
                        latency_ms=latency_ms,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        key_id=None,
                        fallback_attempts=0,
                    )

                return GatewayResult(
                    reply=reply,
                    resolution_source="LIVE_INFERENCE",
                    model=str(resolved_model),
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    key_id=entry.key_id,
                    fallback_attempts=fallback_attempts,
                )

            except Exception as exc:
                if is_failover_error(exc):
                    fallback_attempts += 1
                    logger.warning(
                        "Key %s failed (%s); failing over to next key",
                        entry.key_id,
                        type(exc).__name__,
                    )
                    continue
                logger.warning(
                    "Key %s encountered non-failover error (%s); attempting next key",
                    entry.key_id,
                    type(exc).__name__,
                )
                fallback_attempts += 1
                continue

        raise AllProvidersUnavailableError("all providers unavailable")
