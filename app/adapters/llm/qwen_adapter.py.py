"""Qwen2.5 LLM adapter via DashScope OpenAI-compatible API."""

from __future__ import annotations

import httpx

from app.adapters.llm.base import LLMAdapter
from app.core.errors import AdapterError
from app.core.logging import get_logger

logger = get_logger(__name__)

# International (Singapore) endpoint -- required for accounts created via
# alibabacloud.com / modelstudio.console.alibabacloud.com/ap-southeast-1.
# The mainland China endpoint (dashscope.aliyuncs.com, no '-intl') will NOT
# accept keys issued to international accounts, and vice versa.
DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

DEFAULT_SYSTEM_PROMPT = (
    "You are Kisan Awaaz, an agricultural advisor helping Pakistani farmers. "
    "Respond in simple Urdu that a farmer can understand."
)


class QwenLLMAdapter(LLMAdapter):
    """Calls Qwen2.5 via DashScope's OpenAI-compatible chat/completions endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str = "qwen-max",
        timeout: int = 60,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._system_prompt = system_prompt

    async def generate(self, prompt: str, context: str = "") -> str:
        user_content = prompt
        if context:
            user_content = f"Retrieved knowledge passages:\n{context}\n\nFarmer question: {prompt}"

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 1024,
            "temperature": 0.3,
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
            raise AdapterError("QwenLLM", "API request timed out")
        except httpx.HTTPStatusError as e:
            raise AdapterError(
                "QwenLLM",
                f"API returned {e.response.status_code}: {e.response.text[:200]}",
            )
        except httpx.RequestError as e:
            raise AdapterError("QwenLLM", f"Request failed: {e}")

        return self._parse_response(response.json())

    def _parse_response(self, data: dict) -> str:
        """Extract the assistant's reply from the OpenAI-compatible response."""
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise AdapterError("QwenLLM", f"Unexpected response structure: {e}")

        content = content.strip()
        if not content:
            raise AdapterError("QwenLLM", "Empty response from LLM")

        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            content = content.rsplit("```", 1)[0]
            content = content.strip()

        return content
