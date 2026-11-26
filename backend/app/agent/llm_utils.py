from __future__ import annotations

import time
from typing import Any

from google import genai
from google.genai.errors import APIError

from app.config import get_settings

_LAST_CALL_TS = 0.0
MIN_INTERVAL_SEC = 4.5  # <= 13.3 RPM (safely below 15 RPM free-tier limit)


def generate_with_retry(
    *,
    contents: Any,
    config: Any,
    max_attempts: int = 5,
    base_delay_sec: float = 8.0,
):
    global _LAST_CALL_TS
    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key)

    for attempt in range(1, max_attempts + 1):
        # Pace requests proactively so we stay under 15 RPM
        since_last = time.monotonic() - _LAST_CALL_TS
        if since_last < MIN_INTERVAL_SEC:
            time.sleep(MIN_INTERVAL_SEC - since_last)

        try:
            _LAST_CALL_TS = time.monotonic()
            return client.models.generate_content(
                model=settings.gemini_model,
                contents=contents,
                config=config,
            )
        except APIError as exc:
            status = getattr(exc, "code", None)
            if status in (429, 500, 503, 504) and attempt < max_attempts:
                wait = base_delay_sec * (2 ** (attempt - 1))
                print(
                    f"  [Gemini {status}] Pacing/overload — waiting {wait:.0f}s "
                    f"(attempt {attempt}/{max_attempts})...",
                    flush=True,
                )
                time.sleep(wait)
                continue
            raise