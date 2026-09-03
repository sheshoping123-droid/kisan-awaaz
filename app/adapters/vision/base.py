from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class VisionResult:
    crop: str
    condition: str
    symptoms: list[str] = field(default_factory=list)
    confidence: str = "low"


class VisionAdapter(ABC):
    @abstractmethod
    async def analyze_image(self, image_bytes: bytes) -> VisionResult: ...
