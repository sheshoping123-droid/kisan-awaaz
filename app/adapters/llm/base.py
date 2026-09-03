from abc import ABC, abstractmethod


class LLMAdapter(ABC):
    @abstractmethod
    async def generate(self, prompt: str, context: str = "") -> str: ...
