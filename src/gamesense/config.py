from __future__ import annotations

import os


BALLDONTLIE_API_KEY_ENV = "BALLDONTLIE_API_KEY"


def get_balldontlie_api_key() -> str:
    api_key = os.getenv(BALLDONTLIE_API_KEY_ENV, "").strip()
    if not api_key or api_key == "your_key_here":
        raise RuntimeError(
            f"Missing a real {BALLDONTLIE_API_KEY_ENV}. Create a BALLDONTLIE key and export it before syncing real data."
        )
    return api_key
