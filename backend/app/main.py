from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings, load_settings
from app.db import Database
from app.llm_gateway import AllProvidersUnavailableError, LLMGateway
from app.models import (
    ChatMetadata,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SessionMessagesResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app(
    settings: Optional[Settings] = None,
    gateway_override: Optional[LLMGateway] = None,
) -> FastAPI:
    cfg = settings or load_settings()
    db = Database(cfg.database_url)
    db.init_schema()
    gateway = gateway_override or LLMGateway(cfg)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        db.init_schema()
        yield

    app = FastAPI(
        title="Lyftr LLM Gateway Chatbot",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    app.state.settings = cfg
    app.state.db = db
    app.state.gateway = gateway

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": jsonable_encoder(exc.errors())},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled exception: %s", type(exc).__name__)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "internal server error"},
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        if not db.check_health():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="database unavailable",
            )
        if not cfg.key_pool:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="no API keys configured",
            )
        return HealthResponse(
            status="ok",
            keys_configured=len(cfg.key_pool),
            cache_enabled=gateway.cache_enabled,
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(
        payload: ChatRequest,
        x_simulate_failure: Optional[str] = Header(default=None, alias="X-Simulate-Failure"),
    ) -> ChatResponse:
        history = db.get_session_messages(payload.session_id)
        stored_system_prompt = db.get_session_system_prompt(payload.session_id)

        if payload.system_prompt is not None:
            cleaned_prompt = payload.system_prompt.strip()
            effective_system_prompt = cleaned_prompt if cleaned_prompt else cfg.default_system_prompt
            system_prompt_to_persist = cleaned_prompt if cleaned_prompt else None
        elif stored_system_prompt:
            effective_system_prompt = stored_system_prompt
            system_prompt_to_persist = None
        else:
            effective_system_prompt = cfg.default_system_prompt
            system_prompt_to_persist = None

        llm_messages: List[Dict[str, str]] = []
        if effective_system_prompt:
            llm_messages.append({"role": "system", "content": effective_system_prompt})

        for item in history:
            llm_messages.append({"role": item["role"], "content": item["content"]})

        llm_messages.append({"role": "user", "content": payload.message})

        try:
            result = gateway.generate(
                messages=llm_messages,
                simulate_failure_header=x_simulate_failure,
            )
        except AllProvidersUnavailableError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="all providers unavailable",
            )

        metadata = ChatMetadata(
            resolution_source=result.resolution_source,
            model=result.model,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            key_id=result.key_id,
            fallback_attempts=result.fallback_attempts,
        )

        db.save_turn(
            session_id=payload.session_id,
            user_content=payload.message,
            assistant_content=result.reply,
            assistant_metadata=metadata.model_dump(),
            system_prompt=system_prompt_to_persist,
        )

        return ChatResponse(
            session_id=payload.session_id,
            reply=result.reply,
            metadata=metadata,
        )

    @app.get("/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
    def get_session_messages(session_id: str) -> SessionMessagesResponse:
        messages = db.get_session_messages(session_id)
        if not messages:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="session not found",
            )
        return SessionMessagesResponse(session_id=session_id, messages=messages)

    return app


app = create_app()
