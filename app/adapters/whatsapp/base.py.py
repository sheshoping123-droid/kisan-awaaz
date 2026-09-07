from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class WhatsAppMessage:
    from_number: str
    body: str
    media_type: str | None = None
    media_url: str | None = None


class WhatsAppAdapter(ABC):
    @abstractmethod
    async def send_text(self, to: str, body: str) -> None: ...

    @abstractmethod
    async def send_audio(self, to: str, audio_url: str) -> None: ...

    @abstractmethod
    async def download_media(self, media_url: str) -> bytes: ...

    async def send_audio_bytes(self, to: str, audio_bytes: bytes, mime_type: str) -> None:
        """Send raw audio bytes directly (for TTS replies), without needing a
        pre-hosted public URL. Optional capability -- adapters that support
        direct media upload (e.g. Meta's Cloud API) override this; others
        raise NotImplementedError so callers can fall back to text-only."""
        raise NotImplementedError(f"{type(self).__name__} does not support send_audio_bytes")
