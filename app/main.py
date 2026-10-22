"""FastAPI service.

POST /v1/transcriptions                  upload audio  -> 202 + job id
GET  /v1/transcriptions/{id}             status, progress, error
GET  /v1/transcriptions/{id}/result      ?format=json|srt|vtt|txt
GET  /health                             liveness

Run:  uvicorn app.main:app --port 8000      (docs at /docs)
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import __version__
from .audio import SUPPORTED_EXTENSIONS
from .config import Settings
from .exporters import FORMATS, MEDIA_TYPES
from .jobs import JobManager, JobStore
from .pipeline import TranscriptionPipeline
from .storage import LocalStorage
from .transcriber import Transcriber, build_transcriber

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

UPLOAD_READ_SIZE = 1024 * 1024  # stream uploads to disk 1 MiB at a time


class JobResponse(BaseModel):
    id: str
    status: str
    filename: str
    attempts: int
    progress: float
    duration_sec: float | None = None
    language: str | None = None
    error: str | None = None
    created_at: float
    updated_at: float
    links: dict[str, str]


def _to_response(job: dict) -> JobResponse:
    base = f"/v1/transcriptions/{job['id']}"
    links = {"self": base}
    if job["status"] == "completed":
        links.update({fmt: f"{base}/result?format={fmt}" for fmt in FORMATS})
    total = job["total_chunks"]
    progress = 1.0 if job["status"] == "completed" else (
        round(job["done_chunks"] / total, 3) if total else 0.0
    )
    return JobResponse(
        id=job["id"],
        status=job["status"],
        filename=job["filename"],
        attempts=job["attempts"],
        progress=progress,
        duration_sec=job["duration_sec"],
        language=job["language"],
        error=job["error"],
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        links=links,
    )


def create_app(settings: Settings | None = None, transcriber: Transcriber | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        storage = LocalStorage(settings.data_dir)
        store = JobStore(settings.data_dir / "jobs.db")
        engine = transcriber or build_transcriber(settings)
        manager = JobManager(settings, storage, store, TranscriptionPipeline(engine, settings))
        app.state.settings = settings
        app.state.storage = storage
        app.state.store = store
        app.state.manager = manager
        app.state.engine_name = engine.name
        manager.recover()  # pick up anything left over from a previous run
        yield
        manager.shutdown()

    app = FastAPI(title="Transcription Pipeline API", version=__version__, lifespan=lifespan)

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "engine": request.app.state.engine_name, "version": __version__}

    @app.post("/v1/transcriptions", response_model=JobResponse, status_code=202)
    async def create_transcription(
        request: Request,
        response: Response,
        file: UploadFile = File(...),
        language: str | None = Form(default=None),
    ):
        state = request.app.state
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{suffix or 'none'}'. "
                       f"Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            )

        job_id = uuid4().hex
        input_file = f"original{suffix}"
        state.storage.prepare(job_id)
        dest = state.storage.input_path(job_id, input_file)
        max_bytes = state.settings.max_upload_mb * 1024 * 1024

        size = 0
        try:
            with dest.open("wb") as out:
                while True:
                    block = await file.read(UPLOAD_READ_SIZE)
                    if not block:
                        break
                    size += len(block)
                    if size > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File exceeds the {state.settings.max_upload_mb} MB limit",
                        )
                    out.write(block)
            if size == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
        except HTTPException:
            state.storage.delete_job(job_id)
            raise
        except Exception:
            state.storage.delete_job(job_id)
            raise

        job = state.store.create(
            job_id, file.filename or input_file, input_file, (language or "").strip() or None
        )
        state.manager.submit(job_id)
        response.headers["Location"] = f"/v1/transcriptions/{job_id}"
        return _to_response(job)

    @app.get("/v1/transcriptions/{job_id}", response_model=JobResponse)
    def get_transcription(job_id: str, request: Request):
        job = request.app.state.store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Transcription not found")
        return _to_response(job)

    @app.get("/v1/transcriptions/{job_id}/result")
    def get_result(
        job_id: str,
        request: Request,
        format: Literal["json", "srt", "vtt", "txt"] = "json",
    ):
        state = request.app.state
        job = state.store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Transcription not found")
        if job["status"] == "failed":
            raise HTTPException(status_code=422, detail=f"Transcription failed: {job['error']}")
        if job["status"] != "completed":
            raise HTTPException(
                status_code=409,
                detail=f"Transcription is {job['status']}; try again shortly",
                headers={"Retry-After": "5"},
            )
        path = state.storage.output_path(job_id, format)
        if not path.exists():
            raise HTTPException(status_code=500, detail="Result file is missing")
        return FileResponse(path, media_type=MEDIA_TYPES[format])

    return app


app = create_app()
