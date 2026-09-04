"""Pre-validation for audio inputs before speech transcription."""

from app.core.errors import ValidationError

# (magic, format, magic_offset, check_offset, check_bytes)
ALLOWED_MAGIC_BYTES: list[tuple[bytes, str, int, int | None, bytes | None]] = [
    (b"OggS", "ogg", 0, None, None),
    (b"ID3", "mp3", 0, None, None),
    (b"\xff\xfb", "mp3", 0, None, None),
    (b"\xff\xf3", "mp3", 0, None, None),
    (b"\xff\xf2", "mp3", 0, None, None),
    (b"RIFF", "wav", 0, 8, b"WAVE"),
    (b"ftyp", "m4a", 4, 8, b"M4A "),
]

MAX_AUDIO_BYTES = 25 * 1024 * 1024


def validate_audio(audio_bytes: bytes, max_bytes: int = MAX_AUDIO_BYTES) -> str:
    """Validate audio size and format via magic bytes. Returns detected format string.

    Raises ValidationError for empty, oversized, or unsupported audio.
    """
    if not audio_bytes:
        raise ValidationError("Audio is empty")

    if len(audio_bytes) > max_bytes:
        raise ValidationError(
            f"Audio too large: {len(audio_bytes)} bytes (max {max_bytes})"
        )

    for magic, fmt, magic_offset, check_offset, check_bytes in ALLOWED_MAGIC_BYTES:
        if audio_bytes[magic_offset : magic_offset + len(magic)] == magic:
            if check_offset is not None and check_bytes is not None:
                end = check_offset + len(check_bytes)
                if audio_bytes[check_offset:end] != check_bytes:
                    continue
            return fmt

    raise ValidationError(
        "Unsupported audio format. Accepted: OGG, MP3, WAV, M4A"
    )
