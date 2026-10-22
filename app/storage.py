"""Local-disk storage layout.

    data/
      jobs.db                      job metadata (SQLite)
      jobs/<job_id>/
        input/original.<ext>       the upload, stored under a fixed name
        work/                      normalised WAV, chunk plan, checkpoints
        output/transcript.{json,srt,vtt,txt}

The class is the only place that knows paths. Swapping to S3/GCS/Azure Blob
means re-implementing these few methods, nothing else changes.
User-supplied filenames never touch the filesystem (no path traversal).
"""
from __future__ import annotations

import shutil
from pathlib import Path


class LocalStorage:
    def __init__(self, root: Path):
        self.root = Path(root)

    def job_dir(self, job_id: str) -> Path:
        return self.root / "jobs" / job_id

    def input_path(self, job_id: str, input_file: str) -> Path:
        return self.job_dir(job_id) / "input" / input_file

    def work_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "work"

    def output_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "output"

    def output_path(self, job_id: str, fmt: str) -> Path:
        return self.output_dir(job_id) / f"transcript.{fmt}"

    def prepare(self, job_id: str) -> None:
        (self.job_dir(job_id) / "input").mkdir(parents=True, exist_ok=True)
        self.output_dir(job_id).mkdir(parents=True, exist_ok=True)

    def remove_work_dir(self, job_id: str) -> None:
        shutil.rmtree(self.work_dir(job_id), ignore_errors=True)

    def delete_job(self, job_id: str) -> None:
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
