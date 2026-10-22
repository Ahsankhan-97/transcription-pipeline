"""Plain data objects shared by the pipeline, exporters and API."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Segment:
    id: int
    start: float  # seconds from the start of the original file
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None

    def shifted(self, offset: float) -> "Segment":
        return Segment(
            self.id,
            self.start + offset,
            self.end + offset,
            self.text,
            self.avg_logprob,
            self.no_speech_prob,
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["start"] = round(self.start, 3)
        data["end"] = round(self.end, 3)
        for key in ("avg_logprob", "no_speech_prob"):
            if data[key] is not None:
                data[key] = round(data[key], 4)
        return data


@dataclass
class ChunkTranscript:
    """What a transcriber returns for one audio file or chunk."""

    language: str | None
    language_probability: float | None
    segments: list[Segment] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    language: str | None
    language_probability: float | None
    duration_sec: float
    model: str
    segments: list[Segment]
    processing_time_sec: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "language_probability": (
                round(self.language_probability, 4)
                if self.language_probability is not None
                else None
            ),
            "duration_sec": round(self.duration_sec, 3),
            "model": self.model,
            "processing_time_sec": round(self.processing_time_sec, 3),
            "text": self.text,
            "segments": [s.to_dict() for s in self.segments],
        }
