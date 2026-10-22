"""Runtime configuration, read once from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default):
    return os.environ.get(name, str(default))


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")

    # Speech-to-text engine
    transcriber_backend: str = "faster-whisper"  # "faster-whisper" or "mock"
    whisper_model: str = "base"  # tiny / base / small / medium / large-v3
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"  # int8 is the sweet spot on CPU

    # Upload limits
    max_upload_mb: int = 500

    # Long-audio handling: files longer than the threshold are split into
    # ~target-second chunks, preferring cut points inside silences.
    chunk_threshold_sec: float = 900.0
    chunk_target_sec: float = 600.0
    chunk_search_window_sec: float = 60.0
    chunk_min_sec: float = 30.0
    silence_noise_db: float = -35.0
    silence_min_sec: float = 0.5

    # Background job execution
    workers: int = 1  # transcription is CPU bound; 1 worker per model instance
    max_attempts: int = 3
    retry_base_delay_sec: float = 2.0
    keep_work_files: bool = False

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.chunk_target_sec <= self.chunk_min_sec:
            raise ValueError("chunk_target_sec must be larger than chunk_min_sec")

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            data_dir=Path(_env("DATA_DIR", d.data_dir)),
            transcriber_backend=_env("TRANSCRIBER_BACKEND", d.transcriber_backend),
            whisper_model=_env("WHISPER_MODEL", d.whisper_model),
            whisper_device=_env("WHISPER_DEVICE", d.whisper_device),
            whisper_compute_type=_env("WHISPER_COMPUTE_TYPE", d.whisper_compute_type),
            max_upload_mb=int(_env("MAX_UPLOAD_MB", d.max_upload_mb)),
            chunk_threshold_sec=float(_env("CHUNK_THRESHOLD_SEC", d.chunk_threshold_sec)),
            chunk_target_sec=float(_env("CHUNK_TARGET_SEC", d.chunk_target_sec)),
            chunk_search_window_sec=float(
                _env("CHUNK_SEARCH_WINDOW_SEC", d.chunk_search_window_sec)
            ),
            chunk_min_sec=float(_env("CHUNK_MIN_SEC", d.chunk_min_sec)),
            silence_noise_db=float(_env("SILENCE_NOISE_DB", d.silence_noise_db)),
            silence_min_sec=float(_env("SILENCE_MIN_SEC", d.silence_min_sec)),
            workers=int(_env("WORKERS", d.workers)),
            max_attempts=int(_env("MAX_ATTEMPTS", d.max_attempts)),
            retry_base_delay_sec=float(_env("RETRY_BASE_DELAY_SEC", d.retry_base_delay_sec)),
            keep_work_files=_env("KEEP_WORK_FILES", "0").lower() in {"1", "true", "yes"},
        )
