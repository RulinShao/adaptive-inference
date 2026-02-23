"""
Browser + academic search tools for Elastic Serving (Harmony native format).

Tool namespaces:
  browser.*          — built-in Harmony tools (web search, open, find)
                       backed by Serper API + Jina Reader API
  functions.*        — custom tools rendered in the developer message
    snippet_search   — academic paper search via Semantic Scholar API

Uses ``builtin_tools=["browser"]`` + ``tools=[SNIPPET_SEARCH_TOOL]`` with
``tokenizer.apply_chat_template`` so the model sees the native browser
spec *and* the custom snippet_search tool.  After the initial prompt,
multi-round tool calls are handled by extending the raw prompt string
directly (the template cannot render ``browser.*`` tool responses).

Also keeps legacy ``functions.search`` / ``functions.open_url`` helpers
for backward compatibility with ``generate_trajectories.py``.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

# =============================================================================
# Configuration
# =============================================================================

BUILTIN_TOOLS: List[str] = ["browser"]
"""Passed to ``apply_chat_template(builtin_tools=...)``."""

DEFAULT_MAX_TOOL_CALLS = 15
"""Default cap on tool calls per user turn."""

MODEL_IDENTITY = (
    "You are a deep research assistant that can browse the web and search "
    "academic papers to provide thorough, accurate, and well-sourced answers."
)
"""Overrides the default 'You are ChatGPT...' in the Harmony system message."""

SYSTEM_PROMPT = """\
You answer questions through iterative research. You have access to web \
browsing tools and an academic paper search tool.

## Process

Research iteratively until you have enough evidence for a complete answer:

1. **Think** about what information you need and plan your searches.
2. **Search** — use browser.search for web content, or snippet_search for \
academic papers and scientific data.
3. **Read** — use browser.open to read the most promising results in detail. \
Use browser.find to locate specific information in long pages.
4. **Think again** — do you have enough evidence? If not, search more with \
refined queries. Multiple rounds of search → read → search are expected.
5. **Answer** — only provide your final answer when you have sufficient \
evidence. Support every non-trivial claim with retrieved evidence.

## Tools

- **browser.search**(query) — general web search.
- **browser.open**(id) — open a link from search results by its id, or pass \
a URL string directly.
- **browser.find**(pattern) — find exact text matches in the current page.
- **snippet_search**(query) — search academic papers via Semantic Scholar. \
Returns paper titles, abstracts, authors, and URLs. Use optional parameters: \
limit (max results, default 5), year (e.g. "2023-2025"), \
fields_of_study (e.g. "Computer Science,Medicine").

## Citation

Cite information from browsing using the cursor citation format shown in \
the tools section (e.g. 【3†L15-L20】). Support claims with evidence from \
your searches. If sources disagree, note the conflict and explain which \
source is more reliable.

## Answer Format

- Provide a comprehensive, well-structured answer with clear organization.
- For short factual answers, also include the answer as \\boxed{answer}.
- Acknowledge uncertainty when evidence is thin or conflicting."""

STOP_TOKENS = ["<|call|>", "<|end|>", "<|endoftext|>"]
"""vLLM stop strings for Harmony generation."""

STOP_TOKENS_NO_CALL = ["<|end|>", "<|endoftext|>"]
"""Stop strings that force a final answer (no more tool calls)."""


# =============================================================================
# snippet_search tool definition (custom functions namespace)
# =============================================================================

SNIPPET_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "snippet_search",
        "description": (
            "Search academic papers via Semantic Scholar. Returns paper "
            "titles, abstracts, authors, publication year, and URLs. "
            "Use this for scientific questions, research findings, "
            "benchmarks, and peer-reviewed evidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query for academic papers.",
                },
                "limit": {
                    "type": "number",
                    "description": (
                        "Maximum number of papers to return (default: 5)."
                    ),
                },
                "year": {
                    "type": "string",
                    "description": (
                        "Publication year filter. A single year (e.g. '2024') "
                        "or a range (e.g. '2022-2025')."
                    ),
                },
                "fields_of_study": {
                    "type": "string",
                    "description": (
                        "Comma-separated fields of study to filter by. "
                        "Examples: 'Computer Science', 'Medicine', "
                        "'Computer Science,Physics'."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

CUSTOM_TOOLS = [SNIPPET_SEARCH_TOOL]
"""Passed to ``apply_chat_template(tools=...)``."""


# =============================================================================
# BrowserSession — stateful browser backed by Serper + Jina
# =============================================================================


class BrowserSession:
    """
    Stateful browser session tracking search results and opened pages.

    Output is formatted to match what the Harmony-trained model expects:
    - Search results use ``【{id}†{title}】`` link markers
    - Opened pages are prefixed with ``[{cursor}]`` and line-numbered
    - ``find`` returns matching lines with line numbers
    """

    def __init__(self, http_client: httpx.AsyncClient):
        self.http_client = http_client
        self._cursor_counter = 0
        # cursor → {url, title, lines: list[str]}
        self._pages: Dict[int, Dict[str, Any]] = {}
        # cursor → list of {id, url, title, snippet}
        self._search_results: Dict[int, List[Dict[str, Any]]] = {}
        self._current_cursor: Optional[int] = None

    def _next_cursor(self) -> int:
        self._cursor_counter += 1
        return self._cursor_counter

    # ---- browser.search ----

    async def search(self, query: str, topn: int = 10, **_kw) -> str:
        api_key = os.getenv("SERPER_API_KEY", "")
        if not api_key:
            return "Error: SERPER_API_KEY not set."
        try:
            resp = await self.http_client.post(
                "https://google.serper.dev/search",
                json={"q": query, "num": topn},
                headers={
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return f"Search error: {e}"

        results = data.get("organic", [])
        if not results:
            return f"No results found for: {query}"

        cursor = self._next_cursor()
        self._current_cursor = cursor

        result_list: List[Dict[str, Any]] = []
        lines = [f'Searched for "{query}"', ""]
        for i, r in enumerate(results[:topn], 1):
            url = r.get("link", "")
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            result_list.append(
                {"id": i, "url": url, "title": title, "snippet": snippet}
            )
            lines.append(f"【{i}†{title}】")
            lines.append(f"URL: {url}")
            lines.append(snippet)
            lines.append("")

        self._search_results[cursor] = result_list
        self._pages[cursor] = {
            "url": None,
            "title": f"Search: {query}",
            "lines": lines,
        }

        numbered = "\n".join(
            f"L{i + 1}: {line}" for i, line in enumerate(lines)
        )
        return f"[{cursor}]\n{numbered}"

    # ---- browser.open ----

    async def open(
        self,
        id: Any = None,
        cursor: Any = None,
        loc: Any = None,
        num_lines: Any = None,
        view_source: bool = False,
        source: str = None,
        **_kw,
    ) -> str:
        target_url: Optional[str] = None

        # id as a full URL string
        if isinstance(id, str) and id.startswith("http"):
            target_url = id
        # id as a search-result number
        elif id is not None and id != -1:
            search_cursor = (
                cursor if cursor and cursor != -1 else self._current_cursor
            )
            if search_cursor and search_cursor in self._search_results:
                for r in self._search_results[search_cursor]:
                    if r["id"] == id:
                        target_url = r["url"]
                        break
            if not target_url:
                return (
                    f"Error: link id={id} not found. "
                    f"Use browser.search first, then browser.open with a "
                    f"valid id from the results."
                )
        # Scroll within an already-opened page
        elif cursor and cursor != -1 and cursor in self._pages:
            page = self._pages[cursor]
            start = (loc - 1) if loc and loc > 0 else 0
            n = num_lines if num_lines and num_lines > 0 else 50
            view = page["lines"][start : start + n]
            numbered = "\n".join(
                f"L{start + i + 1}: {line}" for i, line in enumerate(view)
            )
            return f"[{cursor}]\n{numbered}"
        else:
            return (
                "Error: provide an id from search results or a URL string. "
                "Example: browser.open({\"id\": 1}) or "
                'browser.open({"id": "https://example.com"})'
            )

        # Fetch the page via Jina Reader (or direct fallback)
        api_key = os.getenv("JINA_API_KEY", "")
        try:
            if api_key:
                resp = await self.http_client.get(
                    f"https://r.jina.ai/{target_url}",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Accept": "text/plain",
                        "X-Return-Format": "text",
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                content = resp.text
            else:
                resp = await self.http_client.get(
                    target_url, follow_redirects=True, timeout=30
                )
                content = resp.text[:20000]
        except Exception as e:
            return f"Error opening URL: {e}"

        if len(content) > 30000:
            content = content[:30000]

        all_lines = content.split("\n")
        new_cursor = self._next_cursor()
        self._current_cursor = new_cursor
        self._pages[new_cursor] = {
            "url": target_url,
            "title": target_url,
            "lines": all_lines,
        }

        # Apply viewport
        start = (loc - 1) if loc and loc > 0 else 0
        n = num_lines if num_lines and num_lines > 0 else len(all_lines)
        view = all_lines[start : start + n]
        numbered = "\n".join(
            f"L{start + i + 1}: {line}" for i, line in enumerate(view)
        )
        return f"[{new_cursor}]\n{numbered}"

    # ---- browser.find ----

    async def find(self, pattern: str, cursor: Any = None, **_kw) -> str:
        target = cursor if cursor and cursor != -1 else self._current_cursor
        if not target or target not in self._pages:
            return "Error: no page open. Use browser.open first."

        page = self._pages[target]
        matches = []
        for i, line in enumerate(page["lines"]):
            if pattern.lower() in line.lower():
                matches.append((i + 1, line))

        if not matches:
            return f'No matches for "{pattern}" in [{target}].'

        out = [f'Found {len(matches)} match(es) for "{pattern}" in [{target}]:']
        for line_num, line in matches[:20]:
            out.append(f"L{line_num}: {line}")
        if len(matches) > 20:
            out.append(f"... and {len(matches) - 20} more matches")
        return "\n".join(out)

    # ---- dispatcher ----

    async def execute(self, tool_name: str, args: dict) -> str:
        """Dispatch a ``browser.*`` tool call."""
        if tool_name == "search":
            return await self.search(
                query=args.get("query", ""),
                topn=args.get("topn", 10),
            )
        elif tool_name == "open":
            return await self.open(
                id=args.get("id"),
                cursor=args.get("cursor"),
                loc=args.get("loc"),
                num_lines=args.get("num_lines"),
                view_source=args.get("view_source", False),
                source=args.get("source"),
            )
        elif tool_name == "find":
            return await self.find(
                pattern=args.get("pattern", ""),
                cursor=args.get("cursor"),
            )
        return f"Unknown browser tool: {tool_name}"


# =============================================================================
# snippet_search — academic paper search via Semantic Scholar
# =============================================================================

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_FIELDS = (
    "title,abstract,authors,year,url,citationCount,externalIds"
)


async def snippet_search(
    query: str,
    http_client: httpx.AsyncClient,
    limit: int = 5,
    year: Optional[str] = None,
    fields_of_study: Optional[str] = None,
) -> str:
    """Search academic papers via Semantic Scholar API."""
    params: Dict[str, Any] = {
        "query": query,
        "limit": min(limit, 20),
        "fields": SEMANTIC_SCHOLAR_FIELDS,
    }
    if year:
        params["year"] = year
    if fields_of_study:
        params["fieldsOfStudy"] = fields_of_study

    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    try:
        resp = await http_client.get(
            SEMANTIC_SCHOLAR_API,
            params=params,
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"Snippet search error: {e}"

    papers = data.get("data", [])
    if not papers:
        return f"No academic papers found for: {query}"

    lines = [f'Academic paper search: "{query}"', ""]
    for i, p in enumerate(papers, 1):
        title = p.get("title", "Untitled")
        year_val = p.get("year", "")
        citations = p.get("citationCount", 0)
        authors = p.get("authors", [])
        author_str = ", ".join(a.get("name", "") for a in authors[:4])
        if len(authors) > 4:
            author_str += " et al."
        abstract = p.get("abstract", "") or ""
        if len(abstract) > 500:
            abstract = abstract[:500] + "..."
        url = p.get("url", "")
        ext = p.get("externalIds", {})
        doi = ext.get("DOI", "")

        lines.append(f"[{i}] {title}")
        if author_str:
            lines.append(f"    Authors: {author_str}")
        if year_val:
            lines.append(f"    Year: {year_val} | Citations: {citations}")
        if doi:
            lines.append(f"    DOI: {doi}")
        if url:
            lines.append(f"    URL: {url}")
        if abstract:
            lines.append(f"    Abstract: {abstract}")
        lines.append("")

    return "\n".join(lines)


async def execute_custom_tool(
    name: str, args: dict, http_client: httpx.AsyncClient
) -> str:
    """Dispatch a ``functions.*`` tool call (custom tools)."""
    if name == "snippet_search":
        return await snippet_search(
            query=args.get("query", ""),
            http_client=http_client,
            limit=int(args.get("limit", 5)),
            year=args.get("year"),
            fields_of_study=args.get("fields_of_study"),
        )
    return f"Unknown custom tool: {name}"


# =============================================================================
# Prompt helpers — build & extend raw Harmony prompts
# =============================================================================


def build_initial_prompt(
    tokenizer,
    user_message: str,
    system_prompt: str = SYSTEM_PROMPT,
    model_identity: str = MODEL_IDENTITY,
    custom_tools: Optional[List[dict]] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Build the initial prompt using ``apply_chat_template``.

    Uses ``builtin_tools=["browser"]`` so the model sees its native browser
    namespace.  ``system_prompt`` goes into the developer message.  Previous
    assistant turns (without tool details) can be passed via ``history``.
    """
    messages: List[Dict[str, str]] = []

    # Developer / system message with our instructions
    messages.append({"role": "system", "content": system_prompt})

    # Previous turns (high-level only — tool details are in the raw prompt)
    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_message})

    kwargs = {
        "builtin_tools": BUILTIN_TOOLS,
        "model_identity": model_identity,
        "tokenize": False,
        "add_generation_prompt": True,
    }
    tools = custom_tools if custom_tools is not None else CUSTOM_TOOLS
    if tools:
        kwargs["tools"] = tools

    return tokenizer.apply_chat_template(messages, **kwargs)


def append_tool_round(
    prompt: str,
    model_output: str,
    tool_name: str,
    tool_response: str,
    namespace: str = "browser",
) -> str:
    """Extend the raw prompt with a tool call + response.

    ``model_output`` is the raw text from vLLM (does NOT include the
    ``<|call|>`` stop token).  We append it, then the tool response in
    native Harmony format, then the generation prompt for the next round.

    ``namespace`` is ``"browser"`` for built-in tools or ``"functions"``
    for custom tools (e.g. snippet_search).
    """
    return (
        f"{prompt}{model_output}"
        f"<|call|>"
        f"<|start|>{namespace}.{tool_name} to=assistant"
        f"<|channel|>commentary<|message|>"
        f"{json.dumps(tool_response)}"
        f"<|end|>"
        f"<|start|>assistant"
    )


def append_user_turn(prompt: str, final_answer_text: str, user_message: str) -> str:
    """Extend the raw prompt with the model's final answer + a new user turn.

    Call this for multi-turn conversations after the model produces a final
    answer (stopped at ``<|end|>``).
    """
    return (
        f"{prompt}{final_answer_text}"
        f"<|end|>"
        f"<|start|>user<|message|>{user_message}<|end|>"
        f"<|start|>assistant"
    )


# =============================================================================
# Harmony format parsing
# =============================================================================


def parse_tool_call(text: str) -> Optional[Tuple[str, str, dict]]:
    """Parse a tool call from raw model output.

    Returns ``(namespace, tool_name, args_dict)`` or ``None``.
    Handles both ``to=browser.search`` and ``to=functions.snippet_search``.
    """
    m = re.search(r"to=(browser|functions)\.(\w+)", text)
    if not m:
        return None

    namespace = m.group(1)
    tool_name = m.group(2)

    # Args after <|message|>
    msg_match = re.search(
        r"<\|message\|>(.*?)(?:<\|call\|>|<\|end\|>|$)", text, re.DOTALL
    )
    if msg_match:
        args_str = msg_match.group(1).strip()
    else:
        after = text[m.end() :]
        json_match = re.search(r"(?:json|code)\s*(\{.*?\})\s*$", after, re.DOTALL)
        args_str = json_match.group(1) if json_match else after.strip()

    # Unescape if JSON-string-wrapped
    if args_str.startswith('"') and args_str.endswith('"'):
        try:
            args_str = json.loads(args_str)
        except Exception:
            args_str = args_str[1:-1]

    try:
        args = json.loads(args_str)
        return namespace, tool_name, args
    except json.JSONDecodeError:
        # Fallback: extract known fields
        for field in ("query", "pattern"):
            match = re.search(rf'"{field}"\s*:\s*"([^"]*)"', args_str)
            if match:
                return namespace, tool_name, {field: match.group(1)}
        id_match = re.search(r'"id"\s*:\s*(\d+)', args_str)
        if id_match:
            return namespace, tool_name, {"id": int(id_match.group(1))}
        # URL as id
        url_match = re.search(r'"id"\s*:\s*"(https?://[^"]*)"', args_str)
        if url_match:
            return namespace, tool_name, {"id": url_match.group(1)}

    return None


def extract_final_answer(raw_text: str) -> Tuple[str, str]:
    """Extract ``(reasoning, final_answer)`` from Harmony channel format.

    Handles both special-token channels and inline markers like
    ``assistantfinal`` (single token).
    """
    reasoning = ""
    answer = raw_text

    if "<|channel|>final<|message|>" in raw_text:
        parts = raw_text.split("<|channel|>final<|message|>", 1)
        reasoning = parts[0].strip()
        answer = parts[1].strip()
        reasoning = re.sub(r"<\|[^|]+\|>", "", reasoning).strip()
        reasoning = re.sub(r"\bassistant\b", "", reasoning).strip()
        reasoning = re.sub(r"\b(analysis|commentary)\b", "", reasoning).strip()
    elif re.search(r"(?:assistant\s*)?final", raw_text):
        # "assistantfinal", "assistant final", or standalone "final"
        # (these appear as Harmony channel tokens without special-token delimiters)
        final_match = re.search(
            r"(?:assistant\s*)?final\s*(.*?)$", raw_text, re.DOTALL
        )
        if final_match:
            answer = final_match.group(1).strip()
            reasoning = raw_text[: final_match.start()].strip()
            reasoning = re.sub(r"\bassistant\b", "", reasoning).strip()
            reasoning = re.sub(r"\b(analysis|commentary)\b", "", reasoning).strip()
    elif raw_text.startswith("analysis"):
        answer = raw_text[len("analysis") :].strip()

    answer = re.sub(r"<\|[^|]+\|>", "", answer).strip()
    return reasoning, answer


# =============================================================================
# Legacy helpers (functions.* namespace — used by generate_trajectories.py)
# =============================================================================

LEGACY_SYSTEM_PROMPT = """\
You are a helpful research assistant. You can search the web and read web \
pages to find accurate, detailed answers to questions.

When answering a question:
1. Think step-by-step about what information you need.
2. Use the search tool to find relevant sources.
3. Use open_url to read promising results in detail.
4. Synthesize information from multiple sources.
5. Provide a clear, well-sourced answer.

Always verify claims across multiple sources when possible."""

LEGACY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the web using a search engine. Returns the top results "
                "with titles, URLs, and short snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": (
                "Open a URL and read its full text content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL to open.",
                    }
                },
                "required": ["url"],
            },
        },
    },
]

# Backward-compat aliases
TOOLS = LEGACY_TOOLS


async def execute_tool(name: str, args: dict, http_client: httpx.AsyncClient) -> str:
    """Legacy dispatcher for ``functions.*`` tool calls."""
    if name == "search":
        query = args.get("query", "")
        if not query:
            return "Error: search requires a 'query' parameter"
        api_key = os.getenv("SERPER_API_KEY", "")
        if not api_key:
            return "Error: SERPER_API_KEY not set."
        try:
            resp = await http_client.post(
                "https://google.serper.dev/search",
                json={"q": query, "num": 10},
                headers={
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("organic", [])
            if not results:
                return f"No results found for: {query}"
            lines = []
            for i, r in enumerate(results, 1):
                lines.append(
                    f"[{i}] {r.get('title', '')}\n"
                    f"    URL: {r.get('link', '')}\n"
                    f"    {r.get('snippet', '')}"
                )
            return "\n\n".join(lines)
        except Exception as e:
            return f"Search error: {e}"
    elif name == "open_url":
        url = args.get("url", "")
        if not url or not url.startswith(("http://", "https://")):
            hint = url or args.get("query", "")
            if hint:
                return (
                    f"Error: open_url requires a valid URL. Got: '{hint[:100]}'. "
                    f"Use the search tool to find URLs first."
                )
            return "Error: open_url requires a 'url' parameter with a valid URL"
        api_key = os.getenv("JINA_API_KEY", "")
        try:
            if api_key:
                resp = await http_client.get(
                    f"https://r.jina.ai/{url}",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Accept": "text/plain",
                        "X-Return-Format": "text",
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                content = resp.text
                if len(content) > 30000:
                    content = content[:30000] + "\n\n[... truncated ...]"
                return content
            else:
                resp = await http_client.get(
                    url, follow_redirects=True, timeout=30
                )
                return resp.text[:20000]
        except Exception as e:
            return f"Error reading URL: {e}"
    return f"Unknown tool: {name}"


def parse_tool_call_from_raw(text: str) -> Optional[Tuple[str, dict]]:
    """Legacy parser for ``to=functions.*`` tool calls."""
    m = re.search(r"to=functions\.(\w+)", text)
    if not m:
        return None

    tool_name = m.group(1)
    msg_match = re.search(
        r"<\|message\|>(.*?)(?:<\|call\|>|<\|end\|>|$)", text, re.DOTALL
    )
    if msg_match:
        args_str = msg_match.group(1).strip()
    else:
        after = text[m.end() :]
        json_match = re.search(r"(?:json|code)\s*(\{.*?\})\s*$", after, re.DOTALL)
        args_str = json_match.group(1) if json_match else after.strip()

    if args_str.startswith('"') and args_str.endswith('"'):
        try:
            args_str = json.loads(args_str)
        except Exception:
            args_str = args_str[1:-1]

    try:
        args = json.loads(args_str)
        return tool_name, args
    except json.JSONDecodeError:
        query_match = re.search(r'"query"\s*:\s*"([^"]*)"', args_str)
        if query_match:
            return tool_name, {"query": query_match.group(1)}
        url_match = re.search(r'"url"\s*:\s*"([^"]*)"', args_str)
        if url_match:
            return tool_name, {"url": url_match.group(1)}

    return None
