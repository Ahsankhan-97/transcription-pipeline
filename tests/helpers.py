"""Shared test helpers (plain functions, so no fixture magic to explain)."""
from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from app.config import Settings
from scripts.make_sample_audio import generate

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def make_settings(tmp_path: Path, **overrides) -> Settings:
    base = Settings(
        data_dir=tmp_path / "data",
        transcriber_backend="mock",
        retry_base_delay_sec=0.05,  # keep retry tests fast
        chunk_threshold_sec=20.0,
        chunk_target_sec=10.0,
        chunk_search_window_sec=4.0,
        chunk_min_sec=3.0,
    )
    return replace(base, **overrides)


def make_tone(path: Path, seconds: float, tone_sec: float = 8.0, silence_sec: float = 1.0) -> Path:
    return generate(path, seconds, tone_sec=tone_sec, silence_sec=silence_sec)


def convert(src: Path, dst: Path, *extra: str) -> bool:
    """Re-encode ``src`` with ffmpeg. Returns False if this ffmpeg build lacks the encoder."""
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), *extra, str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def wait_for_status(store, job_id: str, wanted: set[str], timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = store.get(job_id)
        if job and job["status"] in wanted:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached {wanted}; last={store.get(job_id)}")
