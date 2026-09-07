from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class VisionResult:
    crop: str
    condition: str
    symptoms: list[str] = field(default_factory=list)
    confidence: str = "low"
    likely_disease: str = ""
    """Specific disease/pest/deficiency name (e.g. 'Cotton Leaf Curl Virus',
    'Wheat Yellow Rust', 'Aphid infestation'), not just the broad category
    in `condition`. Empty when the model can't identify a specific cause."""
    reasoning: str = ""
    """Brief visual evidence the model used to reach its conclusion, e.g.
    'yellow curling leaves with vein thickening, stunted growth'. Not shown
    verbatim to the farmer, but improves diagnosis quality and lets us
    debug misdiagnoses."""
    image_quality_issue: str = ""
    """One of '' (fine), 'blurry', 'not_a_plant', 'too_far', 'poor_lighting'.
    When set, the pipeline should ask the farmer to resend a clearer photo
    instead of guessing from a bad image."""


class VisionAdapter(ABC):
    @abstractmethod
    async def analyze_image(self, image_bytes: bytes) -> VisionResult: ...
