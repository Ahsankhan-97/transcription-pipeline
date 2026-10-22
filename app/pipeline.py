"""The transcription pipeline: validate -> normalise -> (chunk) -> transcribe -> merge.

Design notes
- Resumable: the chunk plan and each chunk's result are checkpointed as JSON in
  the work directory. If a run dies on chunk 7 of 12, the next run re-uses
  chunks 1-6 and only redoes the rest.
- Timestamps are always relative to the ORIGINAL file: chunk results are shifted
  by the chunk's start offset before merging.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Callable

from . import audio
from .config import Settings
from .errors import InvalidAudioError
from .models import Segment, TranscriptionResult
from .transcriber import Transcriber

log = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]  # (chunks_done, chunks_total)


def _write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


class TranscriptionPipeline:
    def __init__(self, transcriber: Transcriber, settings: Settings):
        self.transcriber = transcriber
        self.settings = settings

    def run(
        self,
        src: Path,
        work_dir: Path,
        *,
        language: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        started = time.monotonic()
        src, work_dir = Path(src), Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        wav = self._normalized_wav(src, work_dir)
        duration = audio.wav_duration(wav)
        if duration < audio.MIN_AUDIO_SECONDS:
            raise InvalidAudioError("Audio is empty or shorter than 0.1 seconds")

        spans = self._plan(wav, duration, work_dir)
        total = len(spans)
        log.info("Audio %.1fs -> %d chunk(s)", duration, total)
        if on_progress:
            on_progress(0, total)

        detected_lang = language
        language_prob: float | None = None
        segments: list[Segment] = []

        for idx, (start, end) in enumerate(spans):
            checkpoint = work_dir / "checkpoints" / f"chunk_{idx:04d}.json"
            if checkpoint.exists():
                log.info("Chunk %d/%d: resuming from checkpoint", idx + 1, total)
                data = json.loads(checkpoint.read_text(encoding="utf-8"))
            else:
                data = self._transcribe_chunk(wav, work_dir, idx, total, start, end, detected_lang)
                _write_json_atomic(checkpoint, data)

            if detected_lang is None and data.get("language"):
                # Detect once (first chunk) and reuse: keeps the language stable
                # across chunks and avoids re-detecting on every 10 minute slice.
                detected_lang = data["language"]
                language_prob = data.get("language_probability")

            segments.extend(Segment(**s) for s in data["segments"])
            if on_progress:
                on_progress(idx + 1, total)

        for i, seg in enumerate(segments):
            seg.id = i

        return TranscriptionResult(
            language=detected_lang,
            language_probability=language_prob,
            duration_sec=duration,
            model=self.transcriber.name,
            segments=segments,
            processing_time_sec=time.monotonic() - started,
        )

    # ------------------------------------------------------------------
    def _normalized_wav(self, src: Path, work_dir: Path) -> Path:
        wav = work_dir / "normalized.wav"
        if wav.exists():
            return wav
        info = audio.probe(src)  # raises InvalidAudioError for junk input
        log.info("Input: container=%s codec=%s %d Hz %d ch",
                 info.container, info.codec, info.sample_rate, info.channels)
        audio.normalize(src, wav)
        return wav

    def _plan(self, wav: Path, duration: float, work_dir: Path) -> list[tuple[float, float]]:
        plan_file = work_dir / "plan.json"
        if plan_file.exists():  # keep the same split when resuming
            return [(float(a), float(b)) for a, b in json.loads(plan_file.read_text())]

        s = self.settings
        if duration <= s.chunk_threshold_sec:
            spans = [(0.0, duration)]
        else:
            silences = audio.detect_silences(wav, s.silence_noise_db, s.silence_min_sec)
            spans = audio.plan_chunks(
                duration, silences,
                target=s.chunk_target_sec,
                window=s.chunk_search_window_sec,
                min_chunk=s.chunk_min_sec,
            )
        _write_json_atomic(plan_file, spans)
        return spans

    def _transcribe_chunk(self, wav, work_dir, idx, total, start, end, language) -> dict:
        log.info("Chunk %d/%d: %.1fs-%.1fs", idx + 1, total, start, end)
        if total == 1:
            chunk_wav, offset = wav, 0.0
        else:
            chunk_wav = work_dir / "chunks" / f"chunk_{idx:04d}.wav"
            audio.cut_chunk(wav, start, end, chunk_wav)
            offset = start
        try:
            out = self.transcriber.transcribe(chunk_wav, language=language)
        finally:
            if total > 1:
                chunk_wav.unlink(missing_ok=True)  # free disk as we go
        return {
            "language": out.language,
            "language_probability": out.language_probability,
            "segments": [s.shifted(offset).to_dict() for s in out.segments],
        }
