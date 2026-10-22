#!/usr/bin/env python3
"""Transcribe one audio/video file from the command line.

    python cli.py talk.mp3
    python cli.py talk.mp3 --model small --language en --out-dir results
    python cli.py long.wav --backend mock          # no model download needed
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
from dataclasses import replace
from pathlib import Path

from app.config import Settings
from app.errors import PipelineError
from app.exporters import FORMATS, format_timestamp, write_all
from app.pipeline import TranscriptionPipeline
from app.transcriber import build_transcriber


def _work_key(path: Path, *extra: str) -> str:
    """Hash of file content + settings, so stale checkpoints are never reused."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    for item in extra:
        h.update(item.encode())
    return h.hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    defaults = Settings.from_env()
    p = argparse.ArgumentParser(description="Transcribe audio to JSON / SRT / VTT / TXT")
    p.add_argument("audio", type=Path, help="audio or video file (wav, mp3, m4a, flac, ogg, mp4 ...)")
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    p.add_argument("--formats", nargs="+", choices=FORMATS, default=list(FORMATS))
    p.add_argument("--model", default=defaults.whisper_model, help="tiny, base, small, medium, large-v3")
    p.add_argument("--language", default=None, help="ISO code such as en, hi, bn. Default: auto-detect")
    p.add_argument("--backend", choices=["faster-whisper", "mock"], default=defaults.transcriber_backend)
    p.add_argument("--keep-work", action="store_true", help="keep intermediate files")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.audio.is_file():
        print(f"error: file not found: {args.audio}", file=sys.stderr)
        return 2

    settings = replace(defaults, whisper_model=args.model, transcriber_backend=args.backend)
    pipeline = TranscriptionPipeline(build_transcriber(settings), settings)

    work_dir = args.out_dir / ".work" / _work_key(
        args.audio, args.model, args.backend, args.language or ""
    )

    def progress(done: int, total: int) -> None:
        if total > 1:
            print(f"  chunk {done}/{total} done", flush=True)

    print(f"Transcribing {args.audio.name} with {pipeline.transcriber.name} ...")
    try:
        result = pipeline.run(args.audio, work_dir, language=args.language, on_progress=progress)
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if not exc.retryable:  # nothing worth resuming
            shutil.rmtree(work_dir, ignore_errors=True)
        return 1

    written = write_all(result, args.out_dir, stem=args.audio.stem, formats=args.formats)
    if not args.keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"\nLanguage : {result.language} ({(result.language_probability or 0):.0%})")
    print(f"Duration : {result.duration_sec:.1f}s   Segments: {len(result.segments)}   "
          f"Took: {result.processing_time_sec:.1f}s")
    for seg in result.segments[:5]:
        print(f"  [{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}] {seg.text}")
    if len(result.segments) > 5:
        print(f"  ... {len(result.segments) - 5} more segments")
    print("\nWrote:")
    for path in written.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
