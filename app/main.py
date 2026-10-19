"""HTTP API: POST /chat and GET /health (Section 9)."""
from __future__ import annotations

import logging
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Callable

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings, load_settings
from .db import repository as repo
from .ingestion.pipeline import KnowledgeBase, build_knowledge_base
from .logging_setup import log_event, setup_logging
from .orchestration.agent import ChatService
from .orchestration.llm import LLMProvider, build_provider
from .orchestration.sessions import SessionOwnershipError, SessionStore
from .web.routes import build_web_router

logger = logging.getLogger(__name__)

MAX_MESSAGE_CHARS = 8000
STATIC_DIR = Path(__file__).parent / "web" / "static"


class AppState:
    def __init__(self) -> None:
        self.ready = threading.Event()
        self.service: ChatService | None = None
        self.sessions = SessionStore()


def create_app(settings: Settings | None = None, kb: KnowledgeBase | None = None,
               provider_factory: Callable[[], LLMProvider] | None = None) -> FastAPI:
    settings = settings or load_settings()
    setup_logging(settings.log_level, settings.log_file)
    state = AppState()
    factory = provider_factory or (lambda: build_provider(settings))

    def start(prebuilt: KnowledgeBase | None) -> None:
        try:
            knowledge = prebuilt or build_knowledge_base(settings)
            state.service = ChatService(settings, knowledge, factory)
            try:
                _ = state.service.provider  # fail fast in the logs if the LLM is misconfigured
            except Exception as exc:
                log_event(logger, "llm_provider_init_failed", logging.ERROR, error=type(exc).__name__)
            state.ready.set()
            log_event(logger, "service_ready")
        except Exception as exc:
            log_event(logger, "startup_failed", logging.ERROR, error=type(exc).__name__)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if kb is None:
            # Ingest in the background so /health can answer 503 while it runs.
            threading.Thread(target=start, args=(None,), name="ingestion", daemon=True).start()
        yield

    if kb is not None:
        start(kb)
    app = FastAPI(title="Veltrix PeopleOps Assistant", lifespan=lifespan)
    app.state.app_state = state
    app.include_router(build_web_router(settings, state))
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health():
        if state.ready.is_set():
            return {"status": "ok"}
        return JSONResponse(status_code=503, content={"status": "starting"})

    @app.post("/chat")
    async def chat(request: Request):
        employee_id = (request.headers.get("X-Employee-Id") or "").strip()
        if not employee_id:
            return JSONResponse(status_code=401, content={"detail": "X-Employee-Id header is required."})
        with repo.session(settings.db_path) as conn:
            requester = repo.get_employee(conn, employee_id)
        if requester is None:
            return JSONResponse(status_code=401, content={"detail": "Unknown employee."})

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"detail": "Request body must be valid JSON."})
        if not isinstance(body, dict):
            return JSONResponse(status_code=400, content={"detail": "Request body must be a JSON object."})
        session_id, message = body.get("session_id"), body.get("message")
        if not isinstance(session_id, str) or not session_id.strip():
            return JSONResponse(status_code=400, content={"detail": "session_id must be a non-empty string."})
        if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE_CHARS:
            return JSONResponse(status_code=400, content={"detail": "message must be a non-empty string."})

        if not state.ready.is_set() or state.service is None:
            return JSONResponse(status_code=503, content={"detail": "The service is still starting."})
        try:
            session = state.sessions.get_or_create(session_id, requester.employee_id)
        except SessionOwnershipError:
            return JSONResponse(status_code=403, content={"detail": "This session belongs to another employee."})

        request_id = str(uuid.uuid4())
        result = await run_in_threadpool(state.service.handle, request_id=request_id, session_id=session_id,
                                         session=session, requester=requester, message=message.strip())
        return JSONResponse(content=result)

    return app


def get_app() -> FastAPI:
    """Uvicorn factory entry point: `uvicorn app.main:get_app --factory`."""
    return create_app()
