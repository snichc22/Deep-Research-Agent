from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Union

import ollama
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

# ──────────────────────────────────────────────────────────────

MODEL = "gemma4:26b"
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

STRICT RESEARCH MINIMUMS (DO NOT DEVIATE):
- You MUST perform at least 5 distinct `web_search` calls.
- You MUST perform at least 8 `fetch_webpage` calls.
- Do not attempt to write the final report until these minimums are met.

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


# ──────────────────────────────────────────────────────────────

_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "declare_research_plan",
            "description": (
                "CALL THIS FIRST. Declare the sub-questions that will structure your "
                "research and briefly describe your approach before doing any searches."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sub_questions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "5-10 focused sub-questions to investigate.",
                    },
                    "approach": {
                        "type": "string",
                        "description": "One sentence describing your overall research strategy.",
                    },
                },
                "required": ["sub_questions", "approach"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web via DuckDuckGo. Returns titles, URLs, and text snippets. "
                "Use distinct queries to cover different angles of the topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "max_results": {"type": "integer", "default": N_RESULTS},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": (
                "Fetch and read the full text content of a URL. "
                "Use after web_search to read complete articles beyond the snippet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL (https://…) to fetch."},
                },
                "required": ["url"],
            },
        },
    },
]


# ──────────────────────────────────────────────────────────────

def run(
        topic: str,
        on_event: EventCallback = lambda _: None,
) -> str:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Research topic: **{topic}**\n\n"
                "Begin with `declare_research_plan`, then search the web thoroughly "
                "and produce the final comprehensive report."
            ),
        },
    ]

    report = ""
    n_search = 0
    n_fetch = 0
    t0 = time.time()

    for _iter in range(MAX_ITERS):
        on_event(StatusEvent("Thinking..."))

        opts: dict[str, Any] = {"temperature": 0.7}
        if THINK:
            opts["think"] = True

        resp = ollama.chat(model=MODEL, messages=messages, tools=_TOOLS, options=opts)
        msg = resp.message

        entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            entry["tool_calls"] = msg.tool_calls
        messages.append(entry)

        if not msg.tool_calls:
            if n_search < 5 or n_fetch < 8:
                needed_searches = max(0, 5 - n_search)
                needed_fetches = max(0, 8 - n_fetch)
                deficiency_msg = (
                    f"Your research is currently insufficient. "
                    f"You have performed {n_search} searches (minimum 5 required) "
                    f"and {n_fetch} page reads (minimum 8 required). "
                    f"Please continue searching and reading until these thresholds are met before writing the report."
                )
                messages.append({"role": "user", "content": deficiency_msg})
                on_event(StatusEvent(f"Research insufficient: {n_search} searches, {n_fetch} reads. Continuing..."))
                continue

            on_event(StatusEvent("Writing report..."))
            report = msg.content or ""
            break

        for tc in msg.tool_calls:
            name = tc.function.name
            args = tc.function.arguments or {}

            if name == "declare_research_plan":
                qs = args.get("sub_questions", [])
                approach = args.get("approach", "")
                on_event(PlanEvent(sub_questions=qs, approach=approach))
                on_event(StatusEvent(f"Planning: {approach[:80]}"))
                result = json.dumps({"status": "ok", "registered": len(qs)})

            elif name == "web_search":
                n_search += 1
                query = args.get("query", "")
                on_event(SearchEvent(n=n_search, query=query))
                on_event(StatusEvent(f"Searching: {query[:50]}"))
                result = json.dumps(_web_search(**args), ensure_ascii=False)

            elif name == "fetch_webpage":
                n_fetch += 1
                url = args.get("url", "")
                on_event(FetchEvent(n=n_fetch, url=url))
                on_event(StatusEvent(f"Reading: {url[:80]}"))
                result = _fetch_webpage(url)

            else:
                result = f"[Unknown tool: {name}]"

            messages.append({"role": "tool", "content": result})

    if not report:
        on_event(StatusEvent("<!> Max iterations — generating report now..."))
        messages.append({
            "role": "user",
            "content": "Write the final report now from all research gathered so far.",
        })
        fb = ollama.chat(model=MODEL, messages=messages)
        report = fb.message.content or ""

    on_event(DoneEvent(n_search=n_search, n_fetch=n_fetch, elapsed=time.time() - t0))
    return report
