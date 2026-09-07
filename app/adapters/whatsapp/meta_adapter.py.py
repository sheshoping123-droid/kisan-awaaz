"""Meta WhatsApp Cloud API adapter via the Graph API (httpx, no SDK).

Free alternative to Twilio -- no trial/region restrictions. Requires a
Meta app with the WhatsApp product added, plus META_ACCESS_TOKEN and
META_PHONE_NUMBER_ID (see app/core/config.py for where to find these).
"""

from __future__ import annotations

import httpx

from app.adapters.whatsapp.base import WhatsAppAdapter
from app.core.errors import AdapterError
from app.core.logging import get_logger

logger = get_logger(__name__)


class MetaWhatsAppAdapter(WhatsAppAdapter):
    """Sends WhatsApp messages and downloads media via Meta's Graph API."""

    def __init__(
        self,
        access_token: str,
        phone_number_id: str,
        api_version: str = "v21.0",
        timeout: int = 30,
    ):
        self._access_token = access_token
        self._phone_number_id = phone_number_id
        self._api_version = api_version
        self._timeout = timeout

    @property
    def _base_url(self) -> str:
        return f"https://graph.facebook.com/{self._api_version}"

    @property
    def _messages_url(self) -> str:
        return f"{self._base_url}/{self._phone_number_id}/messages"

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _normalize_to(number: str) -> str:
        """Meta wants a bare number with country code, no '+' and no 'whatsapp:' prefix."""
        return number.replace("whatsapp:", "").lstrip("+")

    async def send_text(self, to: str, body: str) -> None:
        payload = {
            "messaging_product": "whatsapp",
            "to": self._normalize_to(to),
            "type": "text",
            "text": {"body": body},
        }
        await self._post_message(to, payload)

    async def send_audio(self, to: str, audio_url: str) -> None:
        payload = {
            "messaging_product": "whatsapp",
            "to": self._normalize_to(to),
            "type": "audio",
            "audio": {"link": audio_url},
        }
        await self._post_message(to, payload)

    async def _post_message(self, to: str, payload: dict) -> None:
        logger.info("MetaWhatsAppAdapter: sending message to=%s type=%s", to, payload.get("type"))
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(self._messages_url, json=payload, headers=self._headers)
                response.raise_for_status()
        except httpx.TimeoutException:
            raise AdapterError("Meta", "API request timed out")
        except httpx.HTTPStatusError as e:
            raise AdapterError("Meta", f"API returned {e.response.status_code}: {e.response.text[:200]}")
        except httpx.RequestError as e:
            raise AdapterError("Meta", f"Request failed: {e}")

        message_id = self._parse_response(response.json())
        logger.info("MetaWhatsAppAdapter: message sent id=%s", message_id)

    async def send_audio_bytes(self, to: str, audio_bytes: bytes, mime_type: str) -> None:
        """Upload raw audio bytes to Meta's media store, then send it as a
        WhatsApp voice message referencing the resulting media ID. Used for
        TTS replies, which have no public URL to link to directly."""
        media_id = await self._upload_media(audio_bytes, mime_type)
        payload = {
            "messaging_product": "whatsapp",
            "to": self._normalize_to(to),
            "type": "audio",
            "audio": {"id": media_id},
        }
        await self._post_message(to, payload)

    async def _upload_media(self, file_bytes: bytes, mime_type: str) -> str:
        """Upload media to Meta's Graph API, returning a media ID usable in
        an outgoing message. See https://developers.facebook.com/docs/whatsapp/cloud-api/reference/media"""
        url = f"{self._base_url}/{self._phone_number_id}/media"
        extension = "ogg" if "ogg" in mime_type else "mp3"
        files = {
            "file": (f"reply.{extension}", file_bytes, mime_type),
        }
        data = {
            "messaging_product": "whatsapp",
            "type": mime_type,
        }
        headers = {"Authorization": f"Bearer {self._access_token}"}

        logger.info("MetaWhatsAppAdapter: uploading media, size=%d bytes, type=%s", len(file_bytes), mime_type)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, files=files, data=data, headers=headers)
                response.raise_for_status()
        except httpx.TimeoutException:
            raise AdapterError("Meta", "Media upload timed out")
        except httpx.HTTPStatusError as e:
            raise AdapterError("Meta", f"Media upload returned {e.response.status_code}: {e.response.text[:200]}")
        except httpx.RequestError as e:
            raise AdapterError("Meta", f"Media upload failed: {e}")

        media_id = response.json().get("id")
        if not media_id:
            raise AdapterError("Meta", "Media upload response missing 'id'")
        return media_id

    async def download_media(self, media_url: str) -> bytes:
        """Meta gives a media ID (not a direct URL) in incoming webhooks.

        Two-step download: 1) resolve the media ID to a short-lived URL,
        2) fetch the bytes from that URL, both with the same bearer token.
        The parameter is still named media_url to match the shared
        WhatsAppAdapter interface; treat it as a media ID for this adapter.
        """
        media_id = media_url
        logger.info("MetaWhatsAppAdapter: resolving media id=%s", media_id)
        try:
            async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
                lookup = await client.get(f"{self._base_url}/{media_id}", headers=self._headers)
                lookup.raise_for_status()
                resolved_url = lookup.json().get("url")
                if not resolved_url:
                    raise AdapterError("Meta", "Media lookup response missing 'url'")

                response = await client.get(resolved_url, headers=self._headers)
                response.raise_for_status()
                return response.content
        except httpx.TimeoutException:
            raise AdapterError("Meta", "Media download timed out")
        except httpx.HTTPStatusError as e:
            raise AdapterError("Meta", f"Media download returned {e.response.status_code}")
        except httpx.RequestError as e:
            raise AdapterError("Meta", f"Media download failed: {e}")

    @staticmethod
    def _parse_response(data: dict) -> str:
        """Extract the message ID from Meta's response: {"messages": [{"id": "..."}]}"""
        messages = data.get("messages") or []
        if not messages or "id" not in messages[0]:
            raise AdapterError("Meta", "Unexpected response structure: missing message id")
        return messages[0]["id"]
