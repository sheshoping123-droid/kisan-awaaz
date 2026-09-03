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
