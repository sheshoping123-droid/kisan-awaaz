"""Qwen2-VL vision adapter via DashScope OpenAI-compatible API."""

from __future__ import annotations

import base64
import json

import httpx

from app.adapters.vision.base import VisionAdapter, VisionResult
from app.core.errors import AdapterError
from app.core.logging import get_logger

logger = get_logger(__name__)

# International (Singapore) endpoint -- required for accounts created via
# alibabacloud.com / modelstudio.console.alibabacloud.com/ap-southeast-1.
# The mainland China endpoint (dashscope.aliyuncs.com, no '-intl') will NOT
# accept keys issued to international accounts, and vice versa.
DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

VISION_PROMPT = (
    "Analyze this crop/plant image. Identify the crop type, its condition, "
    "and any visible disease symptoms, pest damage, nutrient deficiency, or stress. "
    "Respond ONLY in valid JSON with this exact structure:\n"
    '{"crop": "<crop name>", "condition": "<healthy|diseased|pest_damage|'
    'nutrient_deficiency|stressed>", "symptoms": ["<symptom 1>", ...], '
    '"confidence": "<high|medium|low>"}\n'
    "Do not recommend any treatments. Only describe what you observe."
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
            "max_tokens": 512,
            "temperature": 0.1,
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
        )
