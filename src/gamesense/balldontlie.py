from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, Callable, Dict, Iterable, List
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from gamesense.config import get_balldontlie_api_key


NBA_BASE_URL = "https://api.balldontlie.io/v1"
NFL_BASE_URL = "https://api.balldontlie.io/nfl/v1"


@dataclass
class BallDontLieClient:
    api_key: str
    request_delay_seconds: float = 12.5
    max_retries: int = 6

    @classmethod
    def from_env(cls) -> "BallDontLieClient":
        return cls(api_key=get_balldontlie_api_key())

    def _get(self, base_url: str, path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        query = ""
        if params:
            items: List[tuple[str, Any]] = []
            for key, value in params.items():
                if value is None:
                    continue
                if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
                    for item in value:
                        items.append((f"{key}[]", item))
                else:
                    items.append((key, value))
            query = f"?{urlencode(items)}" if items else ""

        request = Request(
            f"{base_url}{path}{query}",
            headers={"Authorization": self.api_key, "Accept": "application/json"},
        )
        for attempt in range(self.max_retries):
            try:
                with urlopen(request) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    time.sleep(self.request_delay_seconds)
                    return payload
            except HTTPError as exc:
                if exc.code == 401:
                    raise RuntimeError(
                        "BALLDONTLIE rejected the API key with 401 Unauthorized. "
                        "Make sure BALLDONTLIE_API_KEY is set to your real key from app.balldontlie.io, not the placeholder."
                    ) from exc
                if exc.code == 429 and attempt < self.max_retries - 1:
                    retry_after = exc.headers.get("Retry-After")
                    wait_seconds = float(retry_after) if retry_after else self.request_delay_seconds * (attempt + 1)
                    time.sleep(wait_seconds)
                    continue
                raise RuntimeError(
                    "BALLDONTLIE rate-limited the request with 429 Too Many Requests. "
                    "On the free tier, try syncing one season at a time or wait a minute before retrying."
                ) from exc
        raise RuntimeError("BALLDONTLIE request failed after repeated retries.")

    def get_nba_games(
        self,
        *,
        seasons: list[int],
        per_page: int = 100,
        on_page: Callable[[int, int], None] | None = None,
    ) -> list[dict]:
        return self._collect_pages(
            NBA_BASE_URL,
            "/games",
            {"seasons": seasons, "per_page": per_page},
            on_page=on_page,
        )

    def get_nfl_games(
        self,
        *,
        seasons: list[int],
        per_page: int = 100,
        on_page: Callable[[int, int], None] | None = None,
    ) -> list[dict]:
        return self._collect_pages(
            NFL_BASE_URL,
            "/games",
            {"seasons": seasons, "per_page": per_page},
            on_page=on_page,
        )

    def _collect_pages(
        self,
        base_url: str,
        path: str,
        params: Dict[str, Any],
        *,
        on_page: Callable[[int, int], None] | None = None,
    ) -> list[dict]:
        cursor = None
        rows: list[dict] = []
        page_count = 0
        while True:
            page = self._get(base_url, path, {**params, "cursor": cursor})
            page_rows = page.get("data", [])
            rows.extend(page_rows)
            page_count += 1
            if on_page is not None:
                on_page(page_count, len(rows))
            meta = page.get("meta", {})
            cursor = meta.get("next_cursor")
            if not cursor:
                break
        return rows
