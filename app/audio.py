"""Audio helpers: probing, normalisation, silence detection, chunk planning.

Everything that touches a media file shells out to FFmpeg / ffprobe. That is
deliberate: one tool, one code path, and it already understands WAV, MP3, M4A,
FLAC, OGG, Opus, WebM and even video containers (we just drop the video).

Rule of thumb used throughout: never trust the file extension, trust ffprobe.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from .errors import AudioProcessingError, InvalidAudioError, ToolMissingError

log = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000  # what Whisper models expect
MIN_AUDIO_SECONDS = 0.1

# Cheap first gate at the API edge. The real validation is ffprobe's.
SUPPORTED_EXTENSIONS = frozenset(
    {
        ".wav", ".mp3", ".m4a", ".mp4", ".aac", ".flac", ".ogg",
        ".opus", ".webm", ".wma", ".amr", ".aiff", ".mkv", ".mov",
    }
)


# --------------------------------------------------------------------------
# subprocess plumbing
# --------------------------------------------------------------------------
def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise ToolMissingError(
            f"'{tool}' was not found on PATH. Install FFmpeg "
            "(macOS: brew install ffmpeg | Ubuntu: apt install ffmpeg | "
            "Windows: winget install Gyan.FFmpeg) and restart the terminal."
        )
    return path


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, errors="replace", timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioProcessingError(f"{Path(cmd[0]).name} timed out after {timeout}s") from exc


# --------------------------------------------------------------------------
# probing
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AudioInfo:
    codec: str
    sample_rate: int
    channels: int
    container: str
    duration_sec: float | None  # some containers (e.g. raw WebM) do not report it


def probe(path: Path) -> AudioInfo:
    """Validate that ``path`` holds an audio stream and describe it."""
    ffprobe = _require("ffprobe")
    proc = _run(
        [ffprobe, "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        timeout=60,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip()[:300] or "unknown error"
        raise InvalidAudioError(f"Not a readable audio/video file: {detail}")
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise InvalidAudioError("ffprobe returned unreadable output") from exc

    stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None
    )
    if stream is None:
        raise InvalidAudioError("File contains no audio stream")

    fmt = data.get("format", {})
    raw = fmt.get("duration") or stream.get("duration")
    try:
        duration = float(raw) if raw not in (None, "N/A") else None
    except ValueError:
        duration = None

    return AudioInfo(
        codec=stream.get("codec_name", "unknown"),
        sample_rate=int(stream.get("sample_rate") or 0),
        channels=int(stream.get("channels") or 0),
        container=fmt.get("format_name", "unknown"),
        duration_sec=duration,
    )


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------
def normalize(src: Path, dst: Path, timeout: float = 3600) -> None:
    """Decode anything to 16 kHz, mono, 16-bit PCM WAV (written atomically)."""
    ffmpeg = _require("ffmpeg")
    dst.parent.mkdir(parents=True, exist_ok=True)
    partial = dst.with_name(dst.stem + ".part.wav")
    cmd = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-vn", "-map", "0:a:0",
        "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE), "-c:a", "pcm_s16le",
        str(partial),
    ]
    proc = _run(cmd, timeout)
    if proc.returncode != 0 or not partial.exists():
        partial.unlink(missing_ok=True)
        raise AudioProcessingError(f"FFmpeg could not decode the file: {proc.stderr.strip()[:300]}")
    partial.replace(dst)


def wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except (wave.Error, EOFError) as exc:
        raise AudioProcessingError(f"Cannot read WAV header of {path.name}: {exc}") from exc


# --------------------------------------------------------------------------
# long-audio support
# --------------------------------------------------------------------------
_SIL_START = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SIL_END = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?)")


def parse_silencedetect(stderr: str, duration: float) -> list[tuple[float, float]]:
    """Turn ffmpeg's silencedetect log lines into (start, end) pairs."""
    silences: list[tuple[float, float]] = []
    current: float | None = None
    for line in stderr.splitlines():
        m = _SIL_START.search(line)
        if m:
            current = max(0.0, float(m.group(1)))
            continue
        m = _SIL_END.search(line)
        if m and current is not None:
            silences.append((current, min(duration, float(m.group(1)))))
            current = None
    if current is not None:  # file ends in silence
        silences.append((current, duration))
    return silences


def detect_silences(
    wav: Path, noise_db: float = -35.0, min_silence: float = 0.5, timeout: float = 1800
) -> list[tuple[float, float]]:
    ffmpeg = _require("ffmpeg")
    cmd = [
        ffmpeg, "-nostdin", "-hide_banner", "-nostats", "-i", str(wav),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
        "-f", "null", "-",
    ]
    proc = _run(cmd, timeout)
    if proc.returncode != 0:
        raise AudioProcessingError(f"silence detection failed: {proc.stderr.strip()[-300:]}")
    return parse_silencedetect(proc.stderr, wav_duration(wav))


def plan_chunks(
    duration: float,
    silences: list[tuple[float, float]],
    *,
    target: float = 600.0,
    window: float = 60.0,
    min_chunk: float = 30.0,
) -> list[tuple[float, float]]:
    """Split [0, duration] into contiguous chunks of roughly ``target`` seconds.

    Each cut is moved to the middle of the silence closest to the ideal
    position (within +/- ``window``) so we do not slice through a word. If no
    silence is nearby we fall back to the ideal position. No chunk is shorter
    than ``min_chunk``.
    """
    if target <= min_chunk:
        raise ValueError("target must be larger than min_chunk")
    midpoints = sorted((s + e) / 2 for s, e in silences)
    cuts: list[float] = []
    cursor = 0.0
    while True:
        ideal = cursor + target
        if ideal >= duration - min_chunk:
            break
        lo = max(ideal - window, cursor + min_chunk)
        hi = min(ideal + window, duration - min_chunk)
        candidates = [m for m in midpoints if lo <= m <= hi]
        cut = min(candidates, key=lambda m: abs(m - ideal)) if candidates else ideal
        cut = round(cut, 3)
        cuts.append(cut)
        cursor = cut
    points = [0.0, *cuts, duration]
    return list(zip(points[:-1], points[1:]))


def cut_chunk(wav: Path, start: float, end: float, dst: Path, timeout: float = 600) -> None:
    ffmpeg = _require("ffmpeg")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
        "-i", str(wav), "-c:a", "pcm_s16le", str(dst),
    ]
    proc = _run(cmd, timeout)
    if proc.returncode != 0 or not dst.exists():
        raise AudioProcessingError(f"could not cut chunk {start:.1f}-{end:.1f}s: {proc.stderr.strip()[:300]}")
