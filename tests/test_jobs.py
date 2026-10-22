import shutil

import pytest

from app.errors import TranscriptionError
from app.jobs import JobManager, JobStatus, JobStore
from app.pipeline import TranscriptionPipeline
from app.storage import LocalStorage
from app.transcriber import MockTranscriber
from tests.helpers import HAS_FFMPEG, make_settings, make_tone, wait_for_status


class FlakyTranscriber(MockTranscriber):
    """Fails the first ``failures`` calls, then behaves."""

    def __init__(self, failures):
        super().__init__()
        self.failures = failures
        self.calls = 0

    def transcribe(self, wav_path, language=None):
        self.calls += 1
        if self.calls <= self.failures:
            raise TranscriptionError(f"transient failure #{self.calls}")
        return super().transcribe(wav_path, language)


def _setup(tmp_path, transcriber, **overrides):
    settings = make_settings(tmp_path, **overrides)
    storage = LocalStorage(settings.data_dir)
    store = JobStore(settings.data_dir / "jobs.db")
    manager = JobManager(settings, storage, store, TranscriptionPipeline(transcriber, settings))
    return settings, storage, store, manager


def _enqueue(storage, store, manager, source, job_id="job1", submit=True):
    storage.prepare(job_id)
    shutil.copy(source, storage.input_path(job_id, "original.wav"))
    store.create(job_id, "demo.wav", "original.wav", None)
    if submit:
        manager.submit(job_id)
    return job_id


def _skip_without_ffmpeg():
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")


def test_job_completes_and_writes_all_outputs(tmp_path):
    _skip_without_ffmpeg()
    _, storage, store, manager = _setup(tmp_path, MockTranscriber())
    job_id = _enqueue(storage, store, manager, make_tone(tmp_path / "a.wav", 12))
    job = wait_for_status(store, job_id, {"completed", "failed"})
    manager.shutdown()
    assert job["status"] == "completed" and job["attempts"] == 1
    for fmt in ("json", "srt", "vtt", "txt"):
        assert storage.output_path(job_id, fmt).exists()
    assert not storage.work_dir(job_id).exists()  # intermediate files cleaned up


def test_transient_failures_are_retried_with_backoff_then_succeed(tmp_path):
    _skip_without_ffmpeg()
    flaky = FlakyTranscriber(failures=2)
    _, storage, store, manager = _setup(tmp_path, flaky, max_attempts=3)
    job_id = _enqueue(storage, store, manager, make_tone(tmp_path / "a.wav", 6))
    job = wait_for_status(store, job_id, {"completed", "failed"})
    manager.shutdown()
    assert job["status"] == "completed"
    assert job["attempts"] == 3 and flaky.calls == 3


def test_job_fails_after_max_attempts(tmp_path):
    _skip_without_ffmpeg()
    flaky = FlakyTranscriber(failures=99)
    _, storage, store, manager = _setup(tmp_path, flaky, max_attempts=3)
    job_id = _enqueue(storage, store, manager, make_tone(tmp_path / "a.wav", 6))
    job = wait_for_status(store, job_id, {"completed", "failed"})
    manager.shutdown()
    assert job["status"] == "failed" and job["attempts"] == 3
    assert "transient failure" in job["error"]


def test_corrupt_audio_fails_immediately_without_retries(tmp_path):
    _skip_without_ffmpeg()
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"garbage" * 200)
    _, storage, store, manager = _setup(tmp_path, MockTranscriber(), max_attempts=3)
    job_id = _enqueue(storage, store, manager, junk)
    job = wait_for_status(store, job_id, {"completed", "failed"})
    manager.shutdown()
    assert job["status"] == "failed" and job["attempts"] == 1
    assert "InvalidAudioError" in job["error"]


def test_unfinished_jobs_are_recovered_on_startup(tmp_path):
    _skip_without_ffmpeg()
    _, storage, store, manager = _setup(tmp_path, MockTranscriber())
    job_id = _enqueue(storage, store, manager, make_tone(tmp_path / "a.wav", 6), submit=False)
    store.update(job_id, status=JobStatus.PROCESSING, attempts=1)  # as if the process died mid-job
    assert manager.recover() == 1
    job = wait_for_status(store, job_id, {"completed", "failed"})
    manager.shutdown()
    assert job["status"] == "completed" and job["attempts"] == 2


def test_crash_loop_protection_gives_up(tmp_path):
    _skip_without_ffmpeg()
    _, storage, store, manager = _setup(tmp_path, MockTranscriber(), max_attempts=2)
    job_id = _enqueue(storage, store, manager, make_tone(tmp_path / "a.wav", 6), submit=False)
    store.update(job_id, status=JobStatus.PROCESSING, attempts=2)  # already used every attempt
    manager.recover()
    job = wait_for_status(store, job_id, {"completed", "failed"})
    manager.shutdown()
    assert job["status"] == "failed" and "maximum attempts" in job["error"]
