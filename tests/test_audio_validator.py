"""Tests for app.adapters.speech.audio_validator."""

import pytest

from app.adapters.speech.audio_validator import validate_audio

# OggS magic + filler
OGG_BYTES = b"OggS" + b"\x00" * 100
# ID3 tag header
MP3_ID3_BYTES = b"ID3" + b"\x00" * 100
# Raw MPEG frame sync (0xFFFB)
MP3_RAW_BYTES = b"\xff\xfb" + b"\x00" * 100
# RIFF....WAVE
WAV_BYTES = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 100
# ....ftypM4A
M4A_BYTES = b"\x00" * 4 + b"ftypM4A " + b"\x00" * 100


def test_valid_ogg():
    assert validate_audio(OGG_BYTES) == "ogg"


def test_valid_mp3_id3():
    assert validate_audio(MP3_ID3_BYTES) == "mp3"


def test_valid_mp3_raw_frame():
    assert validate_audio(MP3_RAW_BYTES) == "mp3"


def test_valid_wav():
    assert validate_audio(WAV_BYTES) == "wav"


def test_valid_m4a():
    assert validate_audio(M4A_BYTES) == "m4a"


def test_empty_bytes():
    with pytest.raises(Exception, match="empty"):
        validate_audio(b"")


def test_too_large():
    with pytest.raises(Exception, match="too large"):
        validate_audio(OGG_BYTES, max_bytes=50)


def test_custom_max_bytes_allows():
    assert validate_audio(OGG_BYTES, max_bytes=200) == "ogg"


def test_unsupported_format():
    with pytest.raises(Exception, match="Unsupported"):
        validate_audio(b"FLAC" + b"\x00" * 100)


def test_garbage_bytes():
    with pytest.raises(Exception, match="Unsupported"):
        validate_audio(b"\x00\x01\x02\x03" + b"\x00" * 100)


def test_riff_without_wave_rejected():
    # RIFF magic present but not followed by WAVE at offset 8
    with pytest.raises(Exception, match="Unsupported"):
        validate_audio(b"RIFF" + b"\x00" * 4 + b"AVI " + b"\x00" * 100)
