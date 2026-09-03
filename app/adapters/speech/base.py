from abc import ABC, abstractmethod


class SpeechAdapter(ABC):
    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, language: str = "ur") -> str: ...

    @abstractmethod
    async def synthesize(self, text: str, language: str = "ur") -> bytes: ...
