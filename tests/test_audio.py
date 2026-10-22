import pytest

from app import audio
from app.errors import InvalidAudioError
from tests.helpers import HAS_FFMPEG, make_tone

SILENCE_LOG = """
[silencedetect @ 0x1] silence_start: -0.00133
[silencedetect @ 0x1] silence_end: 1.5 | silence_duration: 1.50133
[silencedetect @ 0x1] silence_start: 10.25
[silencedetect @ 0x1] silence_end: 11.0 | silence_duration: 0.75
[silencedetect @ 0x1] silence_start: 58.0
"""


def test_parse_silencedetect_handles_negative_start_and_trailing_silence():
    assert audio.parse_silencedetect(SILENCE_LOG, duration=60.0) == [
        (0.0, 1.5), (10.25, 11.0), (58.0, 60.0),
    ]


def test_plan_chunks_prefers_silence_near_ideal_cut():
    # ideal cut at 600; nearest silence midpoint (602) beats the farther one (580)
    spans = audio.plan_chunks(
        1000.0, [(579, 581), (601, 603)], target=600, window=60, min_chunk=30
    )
    assert spans == [(0.0, 602.0), (602.0, 1000.0)]


def test_plan_chunks_falls_back_to_ideal_cut_without_silence():
    assert audio.plan_chunks(1300.0, [], target=600, window=60, min_chunk=30) == [
        (0.0, 600.0), (600.0, 1200.0), (1200.0, 1300.0),
    ]


def test_plan_chunks_short_file_is_one_chunk():
    assert audio.plan_chunks(620.0, [], target=600, window=60, min_chunk=30) == [(0.0, 620.0)]


def test_plan_chunks_are_contiguous_and_respect_min_length():
    silences = [(i * 37.0, i * 37.0 + 1.0) for i in range(1, 200)]
    spans = audio.plan_chunks(5000.0, silences, target=300, window=40, min_chunk=30)
    assert spans[0][0] == 0.0 and spans[-1][1] == 5000.0
    for (a, b), (c, d) in zip(spans, spans[1:]):
        assert b == c
    assert all(end - start >= 30 for start, end in spans)


def test_probe_rejects_garbage(tmp_path):
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"this is definitely not audio" * 50)
    with pytest.raises(InvalidAudioError):
        audio.probe(bad)


def test_normalize_produces_16k_mono_wav(tmp_path):
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")
    src = make_tone(tmp_path / "in.wav", 5)
    out = tmp_path / "out.wav"
    audio.normalize(src, out)
    info = audio.probe(out)
    assert (info.sample_rate, info.channels) == (16000, 1)
    assert abs(audio.wav_duration(out) - 5.0) < 0.05


def test_detect_silences_finds_the_gaps(tmp_path):
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")
    wav = make_tone(tmp_path / "t.wav", 30, tone_sec=8, silence_sec=1)  # gaps at 8-9, 17-18, 26-27
    silences = audio.detect_silences(wav)
    starts = [round(s) for s, _ in silences]
    assert starts[:3] == [8, 17, 26]
