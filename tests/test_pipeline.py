import pytest

from app.errors import InvalidAudioError, TranscriptionError
from app.pipeline import TranscriptionPipeline
from app.transcriber import MockTranscriber
from tests.helpers import HAS_FFMPEG, convert, make_settings, make_tone


class CountingTranscriber(MockTranscriber):
    def __init__(self, fail_on_call=None):
        super().__init__()
        self.calls = 0
        self.fail_on_call = fail_on_call

    def transcribe(self, wav_path, language=None):
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise TranscriptionError("simulated crash")
        return super().transcribe(wav_path, language)


def _skip_without_ffmpeg():
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")


def test_short_file_single_chunk_with_timestamps(tmp_path):
    _skip_without_ffmpeg()
    src = make_tone(tmp_path / "short.wav", 12)
    pipe = TranscriptionPipeline(MockTranscriber(), make_settings(tmp_path))
    result = pipe.run(src, tmp_path / "work")
    assert abs(result.duration_sec - 12.0) < 0.05
    assert [round(s.start) for s in result.segments] == [0, 5, 10]
    assert result.segments[-1].end == pytest.approx(result.duration_sec, abs=0.01)


def test_long_file_is_chunked_on_silence_and_timestamps_are_global(tmp_path):
    _skip_without_ffmpeg()
    src = make_tone(tmp_path / "long.wav", 50)  # gaps at 8-9, 17-18, 26-27, 35-36, 44-45
    seen = []
    pipe = TranscriptionPipeline(MockTranscriber(), make_settings(tmp_path))
    result = pipe.run(src, tmp_path / "work", on_progress=lambda d, t: seen.append((d, t)))

    total = seen[0][1]
    assert total >= 4
    assert seen[-1] == (total, total)

    # Timeline is continuous, ordered and never overlaps.
    prev_end = 0.0
    for seg in result.segments:
        assert seg.start >= prev_end - 1e-6
        prev_end = seg.end
    assert prev_end == pytest.approx(result.duration_sec, abs=0.05)
    assert [s.id for s in result.segments] == list(range(len(result.segments)))

    # Every internal cut landed inside a silence, not in the middle of the tone.
    import json
    plan = json.loads((tmp_path / "work" / "plan.json").read_text())
    for _, boundary in plan[:-1]:
        in_gap = any(a - 0.05 <= boundary <= a + 1.05 for a in (8, 17, 26, 35, 44))
        assert in_gap, f"cut at {boundary}s is not inside a silence"


def test_resume_reuses_checkpoints_after_a_crash(tmp_path):
    _skip_without_ffmpeg()
    src = make_tone(tmp_path / "long.wav", 50)
    work = tmp_path / "work"
    settings = make_settings(tmp_path)

    crashing = CountingTranscriber(fail_on_call=3)
    with pytest.raises(TranscriptionError):
        TranscriptionPipeline(crashing, settings).run(src, work)
    assert crashing.calls == 3  # chunks 1 and 2 finished, chunk 3 crashed

    healthy = CountingTranscriber()
    result = TranscriptionPipeline(healthy, settings).run(src, work)
    total_chunks = len(__import__("json").loads((work / "plan.json").read_text()))
    assert healthy.calls == total_chunks - 2  # only the missing chunks were redone
    assert result.segments[-1].end == pytest.approx(result.duration_sec, abs=0.05)


def test_common_formats_all_normalise_and_transcribe(tmp_path):
    _skip_without_ffmpeg()
    base = make_tone(tmp_path / "base.wav", 6)
    pipe = TranscriptionPipeline(MockTranscriber(), make_settings(tmp_path))
    tried = 0
    for ext, args in [
        (".mp3", ["-c:a", "libmp3lame"]),
        (".flac", ["-c:a", "flac"]),
        (".ogg", ["-c:a", "libvorbis"]),
        (".m4a", ["-c:a", "aac"]),
        (".opus", ["-c:a", "libopus"]),
    ]:
        sample = tmp_path / f"sample{ext}"
        if not convert(base, sample, *args):
            continue  # this ffmpeg build does not ship that encoder
        result = pipe.run(sample, tmp_path / f"work{ext}")
        assert abs(result.duration_sec - 6.0) < 0.3, ext
        tried += 1
    assert tried >= 2, "expected at least two encoders to be available"


def test_video_container_audio_track_is_extracted(tmp_path):
    _skip_without_ffmpeg()
    base = make_tone(tmp_path / "base.wav", 4)
    video = tmp_path / "clip.mp4"
    if not convert(base, video, "-f", "lavfi", "-i", "color=c=black:s=64x64:d=4",
                   "-map", "1:v", "-map", "0:a", "-c:v", "mpeg4", "-c:a", "aac", "-shortest"):
        pytest.skip("this ffmpeg build cannot make a test video")
    result = TranscriptionPipeline(MockTranscriber(), make_settings(tmp_path)).run(video, tmp_path / "w")
    assert abs(result.duration_sec - 4.0) < 0.3


def test_invalid_file_is_rejected_as_non_retryable(tmp_path):
    _skip_without_ffmpeg()
    bad = tmp_path / "junk.wav"
    bad.write_bytes(b"not audio at all" * 100)
    with pytest.raises(InvalidAudioError) as info:
        TranscriptionPipeline(MockTranscriber(), make_settings(tmp_path)).run(bad, tmp_path / "w")
    assert info.value.retryable is False
