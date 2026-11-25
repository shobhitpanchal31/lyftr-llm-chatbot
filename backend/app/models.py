from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    keys_configured: int
    cache_enabled: bool


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str
    system_prompt: Optional[str] = None

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("session_id must not be empty")
        if len(value) > 128:
            raise ValueError("session_id must be at most 128 characters")
        return value

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("message must not be empty")
        if len(trimmed) > 4000:
            raise ValueError("message must be at most 4000 characters after trimming whitespace")
        return trimmed


class ChatMetadata(BaseModel):
    resolution_source: Literal["LIVE_INFERENCE", "LLM_CACHE"]
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: Optional[str]
    fallback_attempts: int


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    metadata: ChatMetadata


class SessionMessageItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    metadata: Optional[ChatMetadata] = None


class SessionMessagesResponse(BaseModel):
    session_id: str
    messages: List[SessionMessageItem]
