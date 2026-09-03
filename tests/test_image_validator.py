"""Tests for app.adapters.vision.image_validator."""

import pytest

from app.adapters.vision.image_validator import validate_image
from app.core.errors import ValidationError

JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 100
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
WEBP_BYTES = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 100


def test_valid_jpeg():
    assert validate_image(JPEG_BYTES) == "jpeg"


def test_valid_png():
    assert validate_image(PNG_BYTES) == "png"


def test_valid_webp():
    assert validate_image(WEBP_BYTES) == "webp"


def test_empty_bytes():
    with pytest.raises(ValidationError, match="empty"):
        validate_image(b"")


def test_too_large():
    with pytest.raises(ValidationError, match="too large"):
        validate_image(JPEG_BYTES, max_bytes=50)


def test_custom_max_bytes_allows():
    assert validate_image(JPEG_BYTES, max_bytes=len(JPEG_BYTES)) == "jpeg"


def test_unsupported_format():
    with pytest.raises(ValidationError, match="Unsupported"):
        validate_image(b"GIF89a" + b"\x00" * 100)


def test_garbage_bytes():
    with pytest.raises(ValidationError, match="Unsupported"):
        validate_image(b"\x01\x02\x03\x04" * 25)
