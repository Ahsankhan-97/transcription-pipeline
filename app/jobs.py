"""Job state (SQLite) and the background runner with retries and recovery.

Why SQLite + a thread pool instead of Celery/Redis here? Zero infrastructure
for a take-home, but the shape is the same as a production queue:
  * state lives in a database, not in memory  -> survives restarts
  * a worker pulls a job id and processes it  -> easy to move to SQS/Celery
  * failures are classified retryable / permanent
  * retries use exponential back-off and resume from per-chunk checkpoints
  * on startup, unfinished jobs are re-queued (crash recovery)
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from enum import Enum
from pathlib import Path

from .config import Settings
from .exporters import write_all
from .pipeline import TranscriptionPipeline
from .storage import LocalStorage

log = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    input_file    TEXT NOT NULL,
    language_hint TEXT,
    status        TEXT NOT NULL,
    attempts      INTEGER NOT NULL DEFAULT 0,
    error         TEXT,
    language      TEXT,
    duration_sec  REAL,
    done_chunks   INTEGER NOT NULL DEFAULT 0,
    total_chunks  INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""

_UPDATABLE = {"status", "attempts", "error", "language", "duration_sec",
              "done_chunks", "total_chunks"}


class JobStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")  # readers do not block the writer
            conn.executescript(_SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def create(self, job_id: str, filename: str, input_file: str,
               language_hint: str | None) -> dict:
        now = time.time()
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO jobs (id, filename, input_file, language_hint, status, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (job_id, filename, input_file, language_hint,
                 JobStatus.QUEUED.value, now, now),
            )
        return self.get(job_id)

    def get(self, job_id: str) -> dict | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def update(self, job_id: str, **fields) -> None:
        unknown = set(fields) - _UPDATABLE
        if unknown:
            raise ValueError(f"Cannot update column(s): {sorted(unknown)}")
        if isinstance(fields.get("status"), JobStatus):
            fields["status"] = fields["status"].value
        fields["updated_at"] = time.time()
        assignments = ", ".join(f"{col} = ?" for col in fields)  # column names are whitelisted
        with closing(self._connect()) as conn, conn:
            conn.execute(f"UPDATE jobs SET {assignments} WHERE id = ?",
                         (*fields.values(), job_id))

    def list_incomplete(self) -> list[str]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id FROM jobs WHERE status IN (?, ?) ORDER BY created_at",
                (JobStatus.QUEUED.value, JobStatus.PROCESSING.value),
            ).fetchall()
        return [r["id"] for r in rows]


class JobManager:
    def __init__(self, settings: Settings, storage: LocalStorage,
                 store: JobStore, pipeline: TranscriptionPipeline):
        self.settings = settings
        self.storage = storage
        self.store = store
        self.pipeline = pipeline
        self._executor = ThreadPoolExecutor(
            max_workers=settings.workers, thread_name_prefix="transcribe"
        )
        self._closed = False

    # -- public API ------------------------------------------------------
    def submit(self, job_id: str) -> None:
        if self._closed:
            return
        try:
            self._executor.submit(self._process, job_id)
        except RuntimeError:  # executor already shut down
            log.warning("Executor closed; job %s stays queued for next start", job_id)

    def recover(self) -> int:
        """Re-queue jobs that were queued or mid-flight when the process stopped."""
        ids = self.store.list_incomplete()
        for job_id in ids:
            self.store.update(job_id, status=JobStatus.QUEUED)
            self.submit(job_id)
        if ids:
            log.info("Recovered %d unfinished job(s)", len(ids))
        return len(ids)

    def shutdown(self) -> None:
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)

    # -- worker ----------------------------------------------------------
    def _process(self, job_id: str) -> None:
        try:
            self._process_inner(job_id)
        except Exception:  # last-resort guard: never let a worker thread die silently
            log.exception("Unexpected error while handling job %s", job_id)
            self._fail(job_id, "Internal error (see server logs)")

    def _process_inner(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None or job["status"] in (JobStatus.COMPLETED.value, JobStatus.FAILED.value):
            return
        if job["attempts"] >= self.settings.max_attempts:
            # Only reachable after crashes mid-attempt: stops a poison file
            # from crashing the service over and over.
            self._fail(job_id, "Gave up: maximum attempts reached (possible crash loop)")
            return

        attempt = job["attempts"] + 1
        self.store.update(job_id, status=JobStatus.PROCESSING, attempts=attempt, error=None)
        log.info("Job %s: attempt %d/%d", job_id, attempt, self.settings.max_attempts)

        try:
            result = self.pipeline.run(
                self.storage.input_path(job_id, job["input_file"]),
                self.storage.work_dir(job_id),
                language=job["language_hint"],
                on_progress=lambda done, total: self.store.update(
                    job_id, done_chunks=done, total_chunks=total
                ),
            )
            write_all(result, self.storage.output_dir(job_id))
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            retryable = getattr(exc, "retryable", True)
            if retryable and attempt < self.settings.max_attempts:
                delay = self.settings.retry_base_delay_sec * (2 ** (attempt - 1))
                log.warning("Job %s failed (%s). Retrying in %.1fs", job_id, message, delay)
                self.store.update(job_id, status=JobStatus.QUEUED, error=message)
                timer = threading.Timer(delay, self.submit, args=(job_id,))
                timer.daemon = True
                timer.start()
            else:
                log.error("Job %s failed permanently: %s", job_id, message)
                self._fail(job_id, message)
            return

        self.store.update(
            job_id,
            status=JobStatus.COMPLETED,
            language=result.language,
            duration_sec=result.duration_sec,
            error=None,
        )
        if not self.settings.keep_work_files:
            self.storage.remove_work_dir(job_id)
        log.info("Job %s completed in %.1fs", job_id, result.processing_time_sec)

    def _fail(self, job_id: str, message: str) -> None:
        self.store.update(job_id, status=JobStatus.FAILED, error=message)
