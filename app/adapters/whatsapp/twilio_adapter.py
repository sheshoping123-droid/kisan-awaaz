"""Twilio WhatsApp adapter via the Twilio REST API (httpx, no SDK)."""

from __future__ import annotations

import httpx

from app.adapters.whatsapp.base import WhatsAppAdapter
from app.core.errors import AdapterError
from app.core.logging import get_logger

logger = get_logger(__name__)

TWILIO_API_BASE_URL = "https://api.twilio.com/2010-04-01/Accounts"


class TwilioAdapter(WhatsAppAdapter):
    """Sends WhatsApp messages and downloads media via Twilio's REST API."""

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        whatsapp_number: str,
        timeout: int = 30,
    ):
        self._account_sid = account_sid
        self._auth_token = auth_token
        self._whatsapp_number = whatsapp_number
        self._timeout = timeout

    @property
    def messages_url(self) -> str:
        return f"{TWILIO_API_BASE_URL}/{self._account_sid}/Messages.json"

    @staticmethod
    def _normalize_from(number: str) -> str:
        if number.startswith("whatsapp:"):
            return number
        return f"whatsapp:{number}"

    async def send_text(self, to: str, body: str) -> None:
        await self._send_message(to, body=body)

    async def send_audio(self, to: str, audio_url: str) -> None:
        await self._send_message(to, media_url=audio_url)

    async def _send_message(
        self,
        to: str,
        body: str | None = None,
        media_url: str | None = None,
    ) -> None:
        data: dict[str, str] = {
            "From": self._normalize_from(self._whatsapp_number),
            "To": to,
        }
        if body:
            data["Body"] = body
        if media_url:
            data["MediaUrl"] = media_url

        logger.info("TwilioAdapter: sending message to=%s (media=%s)", to, bool(media_url))

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self.messages_url,
                    data=data,
                    auth=(self._account_sid, self._auth_token),
                )
                response.raise_for_status()
        except httpx.TimeoutException:
            raise AdapterError("Twilio", "API request timed out")
        except httpx.HTTPStatusError as e:
            raise AdapterError(
                "Twilio",
                f"API returned {e.response.status_code}: {e.response.text[:200]}",
            )
        except httpx.RequestError as e:
            raise AdapterError("Twilio", f"Request failed: {e}")

        sid = self._parse_response(response.json())
        logger.info("TwilioAdapter: message sent sid=%s", sid)

    async def download_media(self, media_url: str) -> bytes:
        logger.info("TwilioAdapter: downloading media")
        try:
            async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
                response = await client.get(
                    media_url,
                    auth=(self._account_sid, self._auth_token),
                )
                response.raise_for_status()
                return response.content
        except httpx.TimeoutException:
            raise AdapterError("Twilio", "Media download timed out")
        except httpx.HTTPStatusError as e:
            raise AdapterError(
                "Twilio",
                f"Media download returned {e.response.status_code}",
            )
        except httpx.RequestError as e:
            raise AdapterError("Twilio", f"Media download failed: {e}")

    @staticmethod
    def _parse_response(data: dict) -> str:
        """Extract the message SID from the Twilio response."""
        sid = data.get("sid")
        if not sid:
            raise AdapterError("Twilio", "Unexpected response structure: missing 'sid'")
        return sid
