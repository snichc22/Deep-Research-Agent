from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Union

import ollama
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

# ──────────────────────────────────────────────────────────────

MODEL = "gemma4:27b"
MAX_ITERS = 30
N_RESULTS = 6
PAGE_LIMIT = 7500
THINK = True

# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an autonomous deep research agent. Produce a thorough, \
well-cited research report on any topic.

PROTOCOL — follow in order:
1. Call `declare_research_plan` FIRST with 5-10 specific sub-questions and your strategy.
2. Use `web_search` with multiple distinct queries (different angles, keywords).
3. Use `fetch_webpage` on the most relevant URLs to read full articles.
4. Repeat steps 2-3 until you have 8-15 solid sources.
5. Write the final comprehensive report — markdown, with headers, and [Source N] citations.

RULES:
- ALWAYS call `declare_research_plan` before any search.
- Run at least 5 distinct searches from different angles.
- Read at least 8 full pages — snippets alone are not enough.
- Prioritise primary sources: studies, official docs, reputable outlets.
- List all sources at the end of the report.
- The report must be long, detailed, and genuinely useful — not a vague overview."""


# ──────────────────────────────────────────────────────────────

@dataclass
class PlanEvent:
    sub_questions: list[str]
    approach: str


@dataclass
class SearchEvent:
    n: int
    query: str


@dataclass
class FetchEvent:
    n: int
    url: str


@dataclass
class StatusEvent:
    message: str


@dataclass
class DoneEvent:
    n_search: int
    n_fetch: int
    elapsed: float


AgentEvent = Union[PlanEvent, SearchEvent, FetchEvent, StatusEvent, DoneEvent]
EventCallback = Callable[[AgentEvent], None]


# ──────────────────────────────────────────────────────────────

def _web_search(query: str, max_results: int = N_RESULTS) -> list[dict]:
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        return [
            {
                "title": r["title"],
                "url": r["href"],
                "snippet": r["body"]
            }
            for r in raw]
    except Exception as exc:
        return [{"error": str(exc)}]


def _fetch_webpage(url: str) -> str:
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        body = (
                soup.find("article")
                or soup.find("main")
                or soup.find(id="content")
                or soup.body
        )
        lines = [ln.strip() for ln in (body or soup).get_text("\n").splitlines() if ln.strip()]
        text = "\n".join(lines)
        if len(text) > PAGE_LIMIT:
            text = text[:PAGE_LIMIT] + f"\n\n[… truncated at {PAGE_LIMIT} chars]"
        return text
    except requests.exceptions.Timeout:
        return f"[Timeout fetching {url}]"
    except Exception as exc:
        return f"[Error: {exc}]"

