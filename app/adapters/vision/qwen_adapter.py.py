"""Qwen2-VL vision adapter via DashScope OpenAI-compatible API."""

from __future__ import annotations

import base64
import json

import httpx

from app.adapters.vision.base import VisionAdapter, VisionResult
from app.core.errors import AdapterError
from app.core.logging import get_logger

logger = get_logger(__name__)

DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

VISION_PROMPT = (
    "You are an expert agricultural pathologist examining a crop/plant photo "
    "sent by a smallholder farmer in Pakistan via WhatsApp for a diagnosis.\n\n"
    "Step 1 - Image quality check: If the photo is too blurry, too far away, "
    "poorly lit, or does not clearly show a plant, do NOT guess a diagnosis. "
    "Instead set image_quality_issue to one of: 'blurry', 'not_a_plant', "
    "'too_far', 'poor_lighting', and set confidence to 'low'.\n\n"
    "Step 2 - If the image is usable, identify:\n"
    "- The crop/plant species (common name).\n"
    "- Whether it looks healthy, diseased, pest-damaged, nutrient-deficient, "
    "or stressed (drought/heat/waterlogging).\n"
    "- The SPECIFIC disease, pest, or deficiency by its common agricultural "
    "name where you can tell (e.g. 'Cotton Leaf Curl Virus', 'Wheat Yellow "
    "Rust', 'Wheat Stem/Black Rust', 'Rice Blast', 'Bacterial Leaf Blight', "
    "'Tomato Early Blight', 'Tomato Late Blight', 'Potato Late Blight', "
    "'Citrus Canker', 'Citrus Greening (HLB)', 'Mango Anthracnose', "
    "'Sugarcane Red Rot', 'Aphid infestation', 'Whitefly infestation', "
    "'Cotton Bollworm damage', 'Powdery Mildew', 'Nitrogen deficiency', "
    "'Iron/Zinc deficiency chlorosis') -- these are common in Pakistan's "
    "major crops (wheat, cotton, rice, sugarcane, maize, potato, tomato, "
    "citrus, mango) but are only examples; identify whatever you actually "
    "see even if it's not in this list. If multiple diseases could explain "
    "the symptoms and you cannot narrow it down, name the single most "
    "likely one and lower your confidence accordingly rather than leaving "
    "likely_disease blank.\n"
    "- The specific visible symptoms (spots, lesions, discoloration "
    "patterns, wilting, curling, holes, webbing, powdery coating, etc.) -- "
    "be precise about color, shape, and location on the plant (leaf top/"
    "underside, stem, fruit).\n"
    "- Brief reasoning: 1 short sentence on which visual evidence led to "
    "this specific diagnosis (e.g. 'yellow vein mosaic with leaf curling "
    "and stunting is characteristic of a whitefly-transmitted virus').\n\n"
    "Confidence calibration: use 'high' ONLY when the symptom pattern is "
    "textbook-clear for a specific disease. Use 'medium' when the general "
    "condition is clear but the specific cause is a reasonable inference "
    "rather than certain. Use 'low' when the image is ambiguous, shows "
    "early/mild symptoms, or could match several different causes.\n\n"
    "Respond ONLY in valid JSON with this exact structure, no markdown "
    "fences, no extra text:\n"
    '{"crop": "<crop name>", "condition": "<healthy|diseased|pest_damage|'
    'nutrient_deficiency|stressed>", "likely_disease": "<specific name, or '
    'empty string if truly undeterminable>", "symptoms": ["<symptom 1>", '
    '...], "reasoning": "<1 short sentence>", "confidence": "<high|medium|'
    'low>", "image_quality_issue": "<blurry|not_a_plant|too_far|'
    'poor_lighting|empty string if image is fine>"}\n'
    "Do not recommend any treatments here. Only describe and diagnose what "
    "you observe."
)


class QwenVisionAdapter(VisionAdapter):
    """Calls Qwen2-VL via DashScope's OpenAI-compatible chat/completions endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str = "qwen-vl-max",
        timeout: int = 30,
    ):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    async def analyze_image(self, image_bytes: bytes) -> VisionResult:
        b64_image = base64.b64encode(image_bytes).decode("ascii")

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}",
                            },
                        },
                        {
                            "type": "text",
                            "text": VISION_PROMPT,
                        },
                    ],
                }
            ],
            "max_tokens": 700,
            "temperature": 0.0,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{DASHSCOPE_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
        except httpx.TimeoutException:
            raise AdapterError("QwenVision", "API request timed out")
        except httpx.HTTPStatusError as e:
            raise AdapterError(
                "QwenVision",
                f"API returned {e.response.status_code}: {e.response.text[:200]}",
            )
        except httpx.RequestError as e:
            raise AdapterError("QwenVision", f"Request failed: {e}")

        return self._parse_response(response.json())

    def _parse_response(self, data: dict) -> VisionResult:
        """Parse the OpenAI-compatible response into a VisionResult."""
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise AdapterError("QwenVision", f"Unexpected response structure: {e}")

        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            content = content.rsplit("```", 1)[0]
            content = content.strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning("QwenVision: failed to parse JSON from response: %s", e)
            return VisionResult(
                crop="unknown",
                condition="unknown",
                symptoms=[],
                confidence="low",
            )

        return VisionResult(
            crop=parsed.get("crop", "unknown"),
            condition=parsed.get("condition", "unknown"),
            symptoms=parsed.get("symptoms", []),
            confidence=parsed.get("confidence", "low"),
            likely_disease=parsed.get("likely_disease", "") or "",
            reasoning=parsed.get("reasoning", "") or "",
            image_quality_issue=parsed.get("image_quality_issue", "") or "",
        )
