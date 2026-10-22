"""Turn a TranscriptionResult into JSON, SRT, WebVTT or plain text."""
from __future__ import annotations

import json
from pathlib import Path

from .models import TranscriptionResult

FORMATS = ("json", "srt", "vtt", "txt")
MEDIA_TYPES = {
    "json": "application/json",
    "srt": "application/x-subrip",
    "vtt": "text/vtt",
    "txt": "text/plain",
}


def format_timestamp(seconds: float, decimal_sep: str = ".") -> str:
    """HH:MM:SS.mmm (VTT) or HH:MM:SS,mmm (SRT). Rounds once, in integer ms."""
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_sep}{ms:03d}"


def _clean(text: str) -> str:
    # Collapse whitespace/newlines: a blank line would terminate a cue.
    return " ".join(text.split())


def _cues(result: TranscriptionResult):
    for seg in result.segments:
        text = _clean(seg.text)
        if text:
            yield seg, text


def to_json(result: TranscriptionResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n"


def to_srt(result: TranscriptionResult) -> str:
    blocks = []
    for n, (seg, text) in enumerate(_cues(result), start=1):
        start = format_timestamp(seg.start, ",")
        end = format_timestamp(seg.end, ",")
        blocks.append(f"{n}\n{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


def to_vtt(result: TranscriptionResult) -> str:
    blocks = ["WEBVTT\n"]
    for seg, text in _cues(result):
        text = text.replace("-->", "->")  # '-->' is reserved in VTT cue text
        blocks.append(
            f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n{text}\n"
        )
    return "\n".join(blocks)


def to_txt(result: TranscriptionResult) -> str:
    lines = [text for _, text in _cues(result)]
    return "\n".join(lines) + ("\n" if lines else "")


_RENDERERS = {"json": to_json, "srt": to_srt, "vtt": to_vtt, "txt": to_txt}


def render(result: TranscriptionResult, fmt: str) -> str:
    try:
        return _RENDERERS[fmt](result)
    except KeyError:
        raise ValueError(f"Unknown format '{fmt}'. Choose from {', '.join(FORMATS)}") from None


def write_all(
    result: TranscriptionResult,
    out_dir: Path,
    stem: str = "transcript",
    formats: tuple[str, ...] | list[str] = FORMATS,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        path.write_text(render(result, fmt), encoding="utf-8")
        written[fmt] = path
    return written
