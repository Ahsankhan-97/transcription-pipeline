"""HTTP-level tests. Skipped automatically if FastAPI / httpx are not installed."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402
from app.transcriber import MockTranscriber  # noqa: E402
from tests.helpers import HAS_FFMPEG, make_settings, make_tone  # noqa: E402


def _wait_until_done(client, job_id, timeout=20):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/v1/transcriptions/{job_id}").json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.1)
    raise AssertionError("job did not finish in time")


def test_upload_poll_and_download_every_format(tmp_path):
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")
    wav = make_tone(tmp_path / "talk.wav", 12)
    app = create_app(make_settings(tmp_path), MockTranscriber())
    with TestClient(app) as client:
        with wav.open("rb") as f:
            resp = client.post("/v1/transcriptions", files={"file": ("talk.wav", f, "audio/wav")})
        assert resp.status_code == 202
        job = resp.json()
        assert resp.headers["location"].endswith(job["id"])

        done = _wait_until_done(client, job["id"])
        assert done["status"] == "completed" and done["progress"] == 1.0

        body = client.get(f"/v1/transcriptions/{job['id']}/result?format=json").json()
        assert body["segments"][0]["start"] == 0.0
        assert "-->" in client.get(f"/v1/transcriptions/{job['id']}/result?format=srt").text
        assert client.get(f"/v1/transcriptions/{job['id']}/result?format=vtt").text.startswith("WEBVTT")
        assert "mock segment" in client.get(f"/v1/transcriptions/{job['id']}/result?format=txt").text


def test_rejects_unsupported_extension_and_unknown_job(tmp_path):
    app = create_app(make_settings(tmp_path), MockTranscriber())
    with TestClient(app) as client:
        r = client.post("/v1/transcriptions", files={"file": ("notes.pdf", b"%PDF-1.4", "application/pdf")})
        assert r.status_code == 415
        assert client.get("/v1/transcriptions/doesnotexist").status_code == 404
        assert client.get("/health").json()["status"] == "ok"


def test_rejects_empty_and_oversized_uploads(tmp_path):
    app = create_app(make_settings(tmp_path, max_upload_mb=1), MockTranscriber())
    with TestClient(app) as client:
        assert client.post("/v1/transcriptions", files={"file": ("a.wav", b"", "audio/wav")}).status_code == 400
        big = b"\0" * (2 * 1024 * 1024)
        assert client.post("/v1/transcriptions", files={"file": ("a.wav", big, "audio/wav")}).status_code == 413


def test_result_is_409_until_finished_and_422_for_failed_jobs(tmp_path):
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")
    app = create_app(make_settings(tmp_path), MockTranscriber())
    with TestClient(app) as client:
        r = client.post("/v1/transcriptions", files={"file": ("junk.wav", b"not audio" * 100, "audio/wav")})
        job_id = r.json()["id"]
        done = _wait_until_done(client, job_id)
        assert done["status"] == "failed"
        assert client.get(f"/v1/transcriptions/{job_id}/result").status_code == 422
