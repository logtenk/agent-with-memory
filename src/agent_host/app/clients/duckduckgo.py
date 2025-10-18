from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List
import re
import time

import httpx
from bs4 import BeautifulSoup


@dataclass
class SearchResult:
    title: str
    link: str
    snippet: str
    position: int


class RateLimiter:
    """Simple rate limiter to keep us polite when hitting DuckDuckGo."""

    def __init__(self, requests_per_minute: int = 30) -> None:
        self.requests_per_minute = requests_per_minute
        self._recent_requests: Deque[float] = deque()

    def acquire(self) -> None:
        window = 60.0
        now = time.monotonic()
        while self._recent_requests and now - self._recent_requests[0] > window:
            self._recent_requests.popleft()
        if len(self._recent_requests) >= self.requests_per_minute:
            sleep_for = window - (now - self._recent_requests[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._recent_requests.append(time.monotonic())


class DuckDuckGoClient:
    BASE_URL = "https://html.duckduckgo.com/html"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    def __init__(self) -> None:
        self._search_rate_limiter = RateLimiter(requests_per_minute=30)
        self._fetch_rate_limiter = RateLimiter(requests_per_minute=20)

    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Perform a DuckDuckGo search and return structured results."""
        try:
            self._search_rate_limiter.acquire()
            data = {"q": query, "b": "", "kl": ""}
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    self.BASE_URL,
                    data=data,
                    headers=self.HEADERS,
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise RuntimeError("Search request timed out") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"HTTP error during search: {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Unexpected error during search: {exc}") from exc

        soup = BeautifulSoup(response.text, "html.parser")
        if soup is None:
            return []

        results: List[SearchResult] = []
        for result in soup.select(".result"):
            title_elem = result.select_one(".result__title")
            if not title_elem:
                continue

            link_elem = title_elem.find("a")
            if not link_elem:
                continue

            title = link_elem.get_text(strip=True)
            link = link_elem.get("href", "")

            if "y.js" in link:
                continue

            if link.startswith("//duckduckgo.com/l/?uddg="):
                parts = link.split("uddg=")
                if len(parts) > 1:
                    link = httpx.URL(parts[1].split("&")[0]).decode()

            snippet_elem = result.select_one(".result__snippet")
            snippet = snippet_elem.get_text(" ", strip=True) if snippet_elem else ""

            results.append(
                SearchResult(
                    title=title,
                    link=link,
                    snippet=snippet,
                    position=len(results) + 1,
                )
            )

            if len(results) >= max_results:
                break

        return results

    @staticmethod
    def format_results_for_llm(results: List[SearchResult]) -> str:
        if not results:
            return (
                "No results were found for your search query. This could be due to "
                "DuckDuckGo's bot detection or the query returned no matches. "
                "Please try rephrasing your search or try again in a few minutes."
            )

        lines = [f"Found {len(results)} search results:\n"]
        for result in results:
            lines.append(f"{result.position}. {result.title}")
            lines.append(f"   URL: {result.link}")
            lines.append(f"   Summary: {result.snippet}")
            lines.append("")
        return "\n".join(lines)

    def fetch_content(self, url: str) -> str:
        try:
            self._fetch_rate_limiter.acquire()
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                response = client.get(
                    url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36"
                        )
                    },
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise RuntimeError("Request timed out while fetching webpage") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"HTTP error while fetching webpage: {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Unexpected error while fetching webpage: {exc}") from exc

        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "nav", "header", "footer"]):
            element.decompose()

        text = soup.get_text(separator=" ")
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) > 8000:
            text = text[:8000] + "... [content truncated]"

        return text


client = DuckDuckGoClient()
