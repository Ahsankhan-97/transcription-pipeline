"""Speech-to-text engines behind one small interface.

``FasterWhisperTranscriber`` is the real engine (open source, runs locally on
CPU, no API key). ``MockTranscriber`` returns deterministic fake segments so
the rest of the pipeline can be demoed and tested without downloading a model.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Protocol

from . import audio
from .config import Settings
from .errors import TranscriptionError
from .models import ChunkTranscript, Segment

log = logging.getLogger(__name__)


class Transcriber(Protocol):
    name: str

    def transcribe(self, wav_path: Path, language: str | None = None) -> ChunkTranscript:
        ...


class FasterWhisperTranscriber:
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        self.name = f"faster-whisper:{model_size}"
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        # Lazy + guarded: the model loads once, on first use, even with threads.
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel  # imported late: heavy

                log.info("Loading %s (%s, %s)", self._model_size, self._device, self._compute_type)
                self._model = WhisperModel(
                    self._model_size, device=self._device, compute_type=self._compute_type
                )
        return self._model

    def transcribe(self, wav_path: Path, language: str | None = None) -> ChunkTranscript:
        try:
            model = self._load()
            segments, info = model.transcribe(
                str(wav_path),
                language=language,  # None -> auto-detect
                beam_size=5,
                vad_filter=True,  # skip silence, fewer hallucinations
                vad_parameters={"min_silence_duration_ms": 500},
                condition_on_previous_text=False,  # avoids repetition loops on long audio
            )
            out: list[Segment] = []
            for seg in segments:  # generator: decoding happens while we iterate
                text = seg.text.strip()
                if not text:
                    continue
                out.append(
                    Segment(
                        id=len(out),
                        start=float(seg.start),
                        end=float(seg.end),
                        text=text,
                        avg_logprob=seg.avg_logprob,
                        no_speech_prob=seg.no_speech_prob,
                    )
                )
            return ChunkTranscript(info.language, info.language_probability, out)
        except Exception as exc:  # model download, OOM, bad audio buffer ...
            raise TranscriptionError(f"{type(exc).__name__}: {exc}") from exc


class MockTranscriber:
    """Deterministic stand-in: one fake 5-second segment per 5 seconds of audio."""

    name = "mock"

    def __init__(self, seconds_per_segment: float = 5.0):
        self.seconds_per_segment = seconds_per_segment

    def transcribe(self, wav_path: Path, language: str | None = None) -> ChunkTranscript:
        duration = audio.wav_duration(wav_path)
        segments: list[Segment] = []
        t = 0.0
        while t < duration - 1e-6:
            end = min(t + self.seconds_per_segment, duration)
            segments.append(
                Segment(id=len(segments), start=t, end=end,
                        text=f"This is mock segment {len(segments) + 1}.")
            )
            t = end
        return ChunkTranscript(language or "en", 1.0, segments)


def build_transcriber(settings: Settings) -> Transcriber:
    backend = settings.transcriber_backend.lower()
    if backend == "mock":
        return MockTranscriber()
    if backend in {"faster-whisper", "faster_whisper"}:
        return FasterWhisperTranscriber(
            settings.whisper_model, settings.whisper_device, settings.whisper_compute_type
        )
    raise ValueError(f"Unknown TRANSCRIBER_BACKEND '{settings.transcriber_backend}'")
