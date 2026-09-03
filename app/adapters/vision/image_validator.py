"""Pre-validation for image inputs before vision analysis."""

from app.core.errors import ValidationError

ALLOWED_MAGIC_BYTES: list[tuple[bytes, str, int | None, bytes | None]] = [
    (b"\xff\xd8\xff", "jpeg", None, None),
    (b"\x89PNG\r\n\x1a\n", "png", None, None),
    (b"RIFF", "webp", 8, b"WEBP"),
]

MAX_IMAGE_BYTES = 10 * 1024 * 1024


def validate_image(image_bytes: bytes, max_bytes: int = MAX_IMAGE_BYTES) -> str:
    """Validate image size and format via magic bytes. Returns detected format string.

    Raises ValidationError for empty, oversized, or unsupported images.
    """
    if not image_bytes:
        raise ValidationError("Image is empty")

    if len(image_bytes) > max_bytes:
        raise ValidationError(
            f"Image too large: {len(image_bytes)} bytes (max {max_bytes})"
        )

    for magic, fmt, check_offset, check_bytes in ALLOWED_MAGIC_BYTES:
        if image_bytes[: len(magic)] == magic:
            if check_offset is not None and check_bytes is not None:
                end = check_offset + len(check_bytes)
                if image_bytes[check_offset:end] != check_bytes:
                    continue
            return fmt

    raise ValidationError("Unsupported image format. Accepted: JPEG, PNG, WebP")
