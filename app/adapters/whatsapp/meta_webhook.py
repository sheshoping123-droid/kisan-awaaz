"""Pure helpers for validating and parsing Meta WhatsApp Cloud API webhook requests.

Meta's webhook payload format is JSON (unlike Twilio's form-urlencoded), and
signature validation uses HMAC-SHA256 over the raw body with the app secret,
sent as the 'X-Hub-Signature-256' header (format: 'sha256=<hex digest>').
"""

from __future__ import annotations

import hashlib
import hmac

from app.adapters.whatsapp.base import WhatsAppMessage
from app.core.errors import ValidationError

META_MEDIA_TYPES = {"image", "audio"}


def validate_meta_signature(raw_body: bytes, signature_header: str, app_secret: str) -> bool:
    """Validate the 'X-Hub-Signature-256: sha256=<hex>' header Meta sends on every webhook POST."""
    if not signature_header or not signature_header.startswith("sha256="):
        return False

    expected_hex = signature_header.removeprefix("sha256=")
    digest = hmac.new(app_secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()

    return hmac.compare_digest(digest, expected_hex)


def parse_meta_incoming_message(payload: dict) -> WhatsAppMessage | None:
    """Parse a Meta Cloud API webhook JSON payload into a WhatsAppMessage.

    Returns None for payloads that carry no actual message (e.g. delivery/read
    status callbacks, which Meta also POSTs to the same webhook URL).

    Expected shape (trimmed):
    {
      "entry": [{
        "changes": [{
          "value": {
            "messages": [{
              "from": "923001234567",
              "type": "text" | "image" | "audio",
              "text": {"body": "..."},
              "image": {"id": "MEDIA_ID", "mime_type": "image/jpeg"},
              "audio": {"id": "MEDIA_ID", "mime_type": "audio/ogg"}
            }]
          }
        }]
      }]
    }
    """
    try:
        entry = payload.get("entry", [])[0]
        change = entry.get("changes", [])[0]
        value = change.get("value", {})
        messages = value.get("messages")
    except (IndexError, AttributeError):
        return None

    if not messages:
        # Status callback (sent/delivered/read) or another change type; nothing to route.
        return None

    msg = messages[0]
    from_number = msg.get("from", "")
    if not from_number:
        raise ValidationError("'from' is required in Meta webhook message payload")

    msg_type = msg.get("type", "")
    body = ""
    media_type = None
    media_url = None  # actually a media ID for Meta; see MetaWhatsAppAdapter.download_media

    if msg_type == "text":
        body = msg.get("text", {}).get("body", "") or ""
    elif msg_type in META_MEDIA_TYPES:
        media_type = msg_type
        media_url = msg.get(msg_type, {}).get("id")

    return WhatsAppMessage(
        from_number=from_number,
        body=body,
        media_type=media_type,
        media_url=media_url,
    )
