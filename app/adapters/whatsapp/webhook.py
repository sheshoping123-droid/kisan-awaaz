"""Pure helpers for validating and parsing Twilio webhook requests."""

from __future__ import annotations

import base64
import hashlib
import hmac

from app.adapters.whatsapp.base import WhatsAppMessage
from app.core.errors import ValidationError

MEDIA_TYPE_PREFIXES = {
    "image/": "image",
    "audio/": "audio",
}


def validate_twilio_signature(
    url: str,
    params: dict,
    signature: str,
    auth_token: str,
) -> bool:
    """Validate an X-Twilio-Signature header per Twilio's request validation.

    Algorithm: base64(HMAC-SHA1(auth_token, full URL + all params
    sorted by name and concatenated as name+value)).
    """
    if not signature:
        return False

    sorted_params = sorted(params.items())
    concatenated = url + "".join(f"{name}{value}" for name, value in sorted_params)

    digest = hmac.new(
        auth_token.encode("utf-8"),
        concatenated.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    expected = base64.b64encode(digest).decode("ascii")

    return hmac.compare_digest(expected, signature)


def parse_incoming_message(form: dict) -> WhatsAppMessage:
    """Parse Twilio webhook form data (From, Body, NumMedia, MediaUrl0) into a WhatsAppMessage."""
    from_number = form.get("From", "")
    if not from_number:
        raise ValidationError("'From' is required in webhook payload")

    body = form.get("Body", "") or ""

    try:
        num_media = int(form.get("NumMedia", "0") or "0")
    except ValueError:
        num_media = 0

    media_type = None
    media_url = None
    if num_media > 0:
        media_url = form.get("MediaUrl0") or None
        content_type = form.get("MediaContentType0", "")
        for prefix, media in MEDIA_TYPE_PREFIXES.items():
            if content_type.startswith(prefix):
                media_type = media
                break

    return WhatsAppMessage(
        from_number=from_number,
        body=body,
        media_type=media_type,
        media_url=media_url,
    )
