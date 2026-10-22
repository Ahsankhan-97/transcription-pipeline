import json

from app.exporters import format_timestamp, render, to_srt, to_txt, to_vtt, write_all
from app.models import Segment, TranscriptionResult


def _result():
    return TranscriptionResult(
        language="en", language_probability=0.99, duration_sec=12.0, model="mock",
        segments=[
            Segment(0, 0.0, 4.2, "Hello world."),
            Segment(1, 4.2, 3661.9996, "  Second\n\nline  "),
            Segment(2, 5.0, 6.0, "   "),  # empty after cleaning -> skipped
            Segment(3, 6.0, 9.5, "a --> b"),
        ],
    )


def test_timestamp_rounding_and_separator():
    assert format_timestamp(0) == "00:00:00.000"
    assert format_timestamp(3661.9996, ",") == "01:01:02,000"
    assert format_timestamp(59.9995) == "00:01:00.000"
    assert format_timestamp(-1) == "00:00:00.000"


def test_srt_numbering_and_cleaning():
    srt = to_srt(_result())
    assert srt.startswith("1\n00:00:00,000 --> 00:00:04,200\nHello world.\n")
    assert "2\n00:00:04,200 --> 01:01:02,000\nSecond line\n" in srt
    assert "\n3\n00:00:06,000 --> 00:00:09,500\n" in srt  # empty segment skipped, numbering stays dense


def test_vtt_header_and_reserved_arrow():
    vtt = to_vtt(_result())
    assert vtt.startswith("WEBVTT\n")
    assert "00:00:00.000 --> 00:00:04.200" in vtt
    assert "a -> b" in vtt


def test_txt_one_line_per_segment():
    assert to_txt(_result()) == "Hello world.\nSecond line\na --> b\n"


def test_json_is_valid_and_has_segments():
    data = json.loads(render(_result(), "json"))
    assert data["language"] == "en"
    assert len(data["segments"]) == 4
    assert data["segments"][1]["end"] == 3662.0


def test_write_all_creates_every_format(tmp_path):
    written = write_all(_result(), tmp_path, stem="demo")
    assert set(written) == {"json", "srt", "vtt", "txt"}
    assert all(p.exists() for p in written.values())
