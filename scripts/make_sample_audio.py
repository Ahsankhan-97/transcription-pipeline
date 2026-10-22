"""Generate synthetic audio (tone bursts separated by silences).

There is no speech in it, so it is meant for testing the PLUMBING (format
handling, chunking, timestamps, retries) together with the mock backend:

    python -m scripts.make_sample_audio --minutes 25 --out samples/long_demo.wav
    python cli.py samples/long_demo.wav --backend mock
"""
from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np


def generate(path, total_sec: float, tone_sec: float = 25.0, silence_sec: float = 1.5,
             sample_rate: int = 16_000, freq: float = 440.0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    total = int(total_sec * sample_rate)
    period = int((tone_sec + silence_sec) * sample_rate)
    tone_len = int(tone_sec * sample_rate)
    block = sample_rate * 30  # write 30 s at a time to keep memory flat
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        for start in range(0, total, block):
            idx = np.arange(start, min(start + block, total))
            tone = 0.3 * np.sin(2 * np.pi * freq * idx / sample_rate)
            samples = np.where((idx % period) < tone_len, tone, 0.0)
            w.writeframes((samples * 32767).astype("<i2").tobytes())
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=float, default=25.0)
    p.add_argument("--out", type=Path, default=Path("samples/long_demo.wav"))
    args = p.parse_args()
    out = generate(args.out, args.minutes * 60)
    print(f"Wrote {out} ({args.minutes:.1f} min)")


if __name__ == "__main__":
    main()
