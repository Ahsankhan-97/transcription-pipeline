"""Exception hierarchy.

Every pipeline error carries a ``retryable`` flag. The job runner uses it to
decide between "try again with back-off" and "fail immediately": a corrupt
upload will never succeed, a transient decode or model-load error might.
"""


class PipelineError(Exception):
    retryable: bool = True


class InvalidAudioError(PipelineError):
    """The input is not usable audio (corrupt, no audio stream, empty)."""

    retryable = False


class ToolMissingError(PipelineError):
    """FFmpeg / ffprobe is not installed. Retrying will not help."""

    retryable = False


class AudioProcessingError(PipelineError):
    """FFmpeg failed or timed out while processing a valid file."""

    retryable = True


class TranscriptionError(PipelineError):
    """The speech-to-text engine failed (model load, inference)."""

    retryable = True
