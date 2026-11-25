from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Dict, List

import litellm
import pytest
from fastapi.testclient import TestClient

from app.config import KeyEntry, Settings
from app.llm_gateway import LLMGateway
from app.main import create_app


def make_fake_response(
    content: str,
    model: str = "gemini/gemini-2.0-flash",
    prompt_tokens: int = 12,
    completion_tokens: int = 6,
    cache_hit: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        _hidden_params={"cache_hit": cache_hit},
    )


def build_test_settings(
    db_path: str,
    enable_failure_simulation: bool = True,
    keys: List[KeyEntry] | None = None,
) -> Settings:
    key_pool = (
        keys
        if keys is not None
        else [
            KeyEntry(key_id="key_1", api_key="secret-real-key-one", model="gemini/gemini-2.0-flash"),
            KeyEntry(key_id="key_2", api_key="secret-real-key-two", model="gemini/gemini-2.0-flash"),
        ]
    )
    return Settings(
        key_pool=key_pool,
        default_model="gemini/gemini-2.0-flash",
        database_url=f"sqlite:///{db_path}",
        allowed_origins=["http://localhost:3000"],
        enable_failure_simulation=enable_failure_simulation,
        default_system_prompt="You are a helpful assistant.",
        cache_type="local",
        cache_ttl_seconds=300,
        cache_dir="./.test_cache",
    )


class FakeLiteLLMBackend:
    """Simulates LiteLLM completion + caching behavior deterministically without network."""

    def __init__(self, fail_keys: Dict[str, Exception] | None = None) -> None:
        self.fail_keys = fail_keys or {}
        self.cache: Dict[str, SimpleNamespace] = {}
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        model = kwargs["model"]
        messages = kwargs["messages"]
        api_key = kwargs["api_key"]
        cache_opts = kwargs.get("cache") or {}
        no_cache = bool(cache_opts.get("no-cache"))
        no_store = bool(cache_opts.get("no-store"))

        cache_key = f"{model}::{repr(messages)}"

        if not no_cache and cache_key in self.cache:
            cached = self.cache[cache_key]
            return make_fake_response(
                content=cached.choices[0].message.content,
                model=cached.model,
                prompt_tokens=cached.usage.prompt_tokens,
                completion_tokens=cached.usage.completion_tokens,
                cache_hit=True,
            )

        self.calls.append({"model": model, "messages": messages, "api_key": api_key})

        if api_key in self.fail_keys:
            raise self.fail_keys[api_key]

        last_user = messages[-1]["content"]
        reply = f"Echo[{len(messages)}]: {last_user}"
        prompt_tokens = sum(len(m["content"].split()) for m in messages) + 5
        completion_tokens = len(reply.split())

        resp = make_fake_response(
            content=reply,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_hit=False,
        )
        if not no_store:
            self.cache[cache_key] = resp
        return resp


def test_health_ok_and_missing_keys(tmp_path):
    db_file = str(tmp_path / "test_health.db")
    settings = build_test_settings(db_file)
    app = create_app(settings=settings, gateway_override=LLMGateway(settings, completion_fn=FakeLiteLLMBackend()))
    client = TestClient(app)

    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "keys_configured": 2, "cache_enabled": True}

    no_key_settings = build_test_settings(db_file, keys=[])
    app_no_keys = create_app(
        settings=no_key_settings,
        gateway_override=LLMGateway(no_key_settings, completion_fn=FakeLiteLLMBackend()),
    )
    res_503 = TestClient(app_no_keys).get("/health")
    assert res_503.status_code == 503
    assert "detail" in res_503.json()


def test_validation_errors_and_no_history_written(tmp_path):
    db_file = str(tmp_path / "test_validation.db")
    settings = build_test_settings(db_file)
    app = create_app(settings=settings, gateway_override=LLMGateway(settings, completion_fn=FakeLiteLLMBackend()))
    client = TestClient(app)

    # Empty/whitespace message -> 422
    r1 = client.post("/chat", json={"session_id": "s-val", "message": "   \n\t "})
    assert r1.status_code == 422
    assert "detail" in r1.json()

    # Over 4000 chars -> 422
    r2 = client.post("/chat", json={"session_id": "s-val", "message": "a" * 4001})
    assert r2.status_code == 422
    assert "detail" in r2.json()

    # Empty session_id -> 422
    r3 = client.post("/chat", json={"session_id": "", "message": "hello"})
    assert r3.status_code == 422

    # Malformed JSON -> 422
    r4 = client.post(
        "/chat",
        data="{not-json",
        headers={"Content-Type": "application/json"},
    )
    assert r4.status_code == 422
    assert "detail" in r4.json()

    # Failed 422 requests must not create session history
    hist = client.get("/sessions/s-val/messages")
    assert hist.status_code == 404
    assert hist.json() == {"detail": "session not found"}

    # Whitespace-padded message whose trimmed length <= 4000 succeeds
    r5 = client.post("/chat", json={"session_id": "s-val", "message": "  " + ("b" * 4000) + "  "})
    assert r5.status_code == 200


def test_multiturn_history_and_session_isolation(tmp_path):
    db_file = str(tmp_path / "test_multiturn.db")
    settings = build_test_settings(db_file)
    fake_backend = FakeLiteLLMBackend()
    app = create_app(settings=settings, gateway_override=LLMGateway(settings, completion_fn=fake_backend))
    client = TestClient(app)

    res1 = client.post(
        "/chat",
        json={"session_id": "s-alpha", "message": "My name is Alice.", "system_prompt": "Be concise."},
    )
    assert res1.status_code == 200
    body1 = res1.json()
    assert body1["session_id"] == "s-alpha"
    assert body1["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert body1["metadata"]["key_id"] == "key_1"

    res2 = client.post("/chat", json={"session_id": "s-alpha", "message": "What is my name?"})
    assert res2.status_code == 200
    # Second call should receive system + turn1 user + turn1 assistant + turn2 user = 4 messages
    assert len(fake_backend.calls[-1]["messages"]) == 4
    assert fake_backend.calls[-1]["messages"][0] == {"role": "system", "content": "Be concise."}

    # Separate session does not share s-alpha history
    res_beta = client.post("/chat", json={"session_id": "s-beta", "message": "Hello"})
    assert res_beta.status_code == 200
    assert len(fake_backend.calls[-1]["messages"]) == 2  # default system + new message

    # History endpoint returns oldest first without system prompt
    hist = client.get("/sessions/s-alpha/messages")
    assert hist.status_code == 200
    hist_data = hist.json()
    assert hist_data["session_id"] == "s-alpha"
    assert [m["role"] for m in hist_data["messages"]] == ["user", "assistant", "user", "assistant"]
    assert all(m["role"] != "system" for m in hist_data["messages"])
    assert hist_data["messages"][1]["metadata"]["input_tokens"] > 0

    # Verify persistence across a simulated app restart against same SQLite DB
    app_restarted = create_app(
        settings=settings,
        gateway_override=LLMGateway(settings, completion_fn=fake_backend),
    )
    hist_after_restart = TestClient(app_restarted).get("/sessions/s-alpha/messages")
    assert hist_after_restart.status_code == 200
    assert len(hist_after_restart.json()["messages"]) == 4


def test_caching_behavior_across_sessions(tmp_path):
    db_file = str(tmp_path / "test_cache.db")
    settings = build_test_settings(db_file)
    fake_backend = FakeLiteLLMBackend()
    app = create_app(settings=settings, gateway_override=LLMGateway(settings, completion_fn=fake_backend))
    client = TestClient(app)

    r1 = client.post("/chat", json={"session_id": "s-c1", "message": "What is the capital of France?"})
    assert r1.status_code == 200
    m1 = r1.json()["metadata"]
    assert m1["resolution_source"] == "LIVE_INFERENCE"
    assert m1["key_id"] == "key_1"
    assert m1["fallback_attempts"] == 0

    # Identical first message in a new session -> LLM_CACHE
    r2 = client.post("/chat", json={"session_id": "s-c2", "message": "What is the capital of France?"})
    assert r2.status_code == 200
    assert r2.json()["reply"] == r1.json()["reply"]
    m2 = r2.json()["metadata"]
    assert m2["resolution_source"] == "LLM_CACHE"
    assert m2["key_id"] is None
    assert m2["fallback_attempts"] == 0
    assert m2["input_tokens"] == m1["input_tokens"]
    assert m2["output_tokens"] == m1["output_tokens"]

    # Different message -> LIVE_INFERENCE
    r3 = client.post("/chat", json={"session_id": "s-c3", "message": "What is the capital of Spain?"})
    assert r3.status_code == 200
    assert r3.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"

    # Same message with different system prompt -> LIVE_INFERENCE (cache key includes system prompt)
    r4 = client.post(
        "/chat",
        json={
            "session_id": "s-c4",
            "message": "What is the capital of France?",
            "system_prompt": "Reply in French only.",
        },
    )
    assert r4.status_code == 200
    assert r4.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_failover_real_error_and_simulation_hook(tmp_path):
    db_file = str(tmp_path / "test_failover.db")
    settings = build_test_settings(db_file, enable_failure_simulation=True)

    # 1. Real 429 on key_1 -> fails over to key_2
    backend_real_fail = FakeLiteLLMBackend(
        fail_keys={
            "secret-real-key-one": litellm.RateLimitError(
                message="429 rate limit", llm_provider="gemini", model="gemini/gemini-2.0-flash"
            )
        }
    )
    app_real = create_app(
        settings=settings,
        gateway_override=LLMGateway(settings, completion_fn=backend_real_fail),
    )
    r_real = TestClient(app_real).post("/chat", json={"session_id": "s-f1", "message": "Hi"})
    assert r_real.status_code == 200
    meta_real = r_real.json()["metadata"]
    assert meta_real["key_id"] == "key_2"
    assert meta_real["fallback_attempts"] == 1
    assert "secret-real-key" not in r_real.text

    # 2. X-Simulate-Failure hook tests
    client = TestClient(
        create_app(settings=settings, gateway_override=LLMGateway(settings, completion_fn=FakeLiteLLMBackend()))
    )

    r_sim_429 = client.post(
        "/chat",
        json={"session_id": "s-sim1", "message": "Test 429"},
        headers={"X-Simulate-Failure": "key_1=429"},
    )
    assert r_sim_429.status_code == 200
    assert r_sim_429.json()["metadata"]["key_id"] == "key_2"
    assert r_sim_429.json()["metadata"]["fallback_attempts"] == 1

    r_sim_500 = client.post(
        "/chat",
        json={"session_id": "s-sim2", "message": "Test 500"},
        headers={"X-Simulate-Failure": "key_1=500"},
    )
    assert r_sim_500.status_code == 200
    assert r_sim_500.json()["metadata"]["key_id"] == "key_2"
    assert r_sim_500.json()["metadata"]["fallback_attempts"] == 1

    # *=429 -> 503 and nothing saved to SQLite
    r_all_fail = client.post(
        "/chat",
        json={"session_id": "s-allfail", "message": "Should fail"},
        headers={"X-Simulate-Failure": "*=429"},
    )
    assert r_all_fail.status_code == 503
    assert r_all_fail.json() == {"detail": "all providers unavailable"}
    assert client.get("/sessions/s-allfail/messages").status_code == 404

    # 3. Hook disabled when ENABLE_FAILURE_SIMULATION=false
    disabled_settings = build_test_settings(db_file, enable_failure_simulation=False)
    disabled_client = TestClient(
        create_app(
            settings=disabled_settings,
            gateway_override=LLMGateway(disabled_settings, completion_fn=FakeLiteLLMBackend()),
        )
    )
    r_ignored = disabled_client.post(
        "/chat",
        json={"session_id": "s-disabled", "message": "Hello"},
        headers={"X-Simulate-Failure": "*=429"},
    )
    assert r_ignored.status_code == 200
    assert r_ignored.json()["metadata"]["key_id"] == "key_1"


def test_real_litellm_cache_integration(tmp_path):
    """Verify real litellm.completion + litellm.Cache (using mock_response for first turn)."""
    db_file = str(tmp_path / "test_litellm_real_cache.db")
    settings = build_test_settings(db_file)
    gateway = LLMGateway(settings)  # Uses real litellm.completion + litellm.cache
    app = create_app(settings=settings, gateway_override=gateway)
    client = TestClient(app)

    # First call uses litellm.completion with mock_response to populate real litellm.cache
    messages = [
        {"role": "system", "content": settings.default_system_prompt},
        {"role": "user", "content": "Real cache integration prompt"},
    ]
    litellm.completion(
        model="gemini/gemini-2.0-flash",
        messages=messages,
        api_key="secret-real-key-one",
        mock_response="Paris is the capital.",
        caching=True,
    )

    # Now calling POST /chat with that exact prompt must hit real litellm.cache without network
    res = client.post(
        "/chat",
        json={"session_id": "s-real-cache", "message": "Real cache integration prompt"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["reply"] == "Paris is the capital."
    assert data["metadata"]["resolution_source"] == "LLM_CACHE"
    assert data["metadata"]["key_id"] is None
    assert data["metadata"]["fallback_attempts"] == 0
