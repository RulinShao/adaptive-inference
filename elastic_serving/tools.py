"""
Shared tool definitions and system prompts for Elastic Serving.

Tools use Harmony's native format — the tokenizer's apply_chat_template renders
them as TypeScript-style namespace declarations in the developer message.
This module provides:
  - TOOLS: tool specs (passed to apply_chat_template's `tools=` kwarg)
  - SYSTEM_PROMPT: improved research-assistant prompt
  - Tool execution helpers (Serper search + Jina URL reader)
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

# =============================================================================
# API keys (loaded from env / .env)
# =============================================================================

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
JINA_API_KEY = os.getenv("JINA_API_KEY", "")


# =============================================================================
# Tool definitions (for tokenizer.apply_chat_template)
# =============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the web using a search engine. Returns the top results "
                "with titles, URLs, and short snippets. Use this to discover "
                "relevant sources before reading them in detail with open_url."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The search query. Be specific and include key terms. "
                            "For recent events, include the year. For technical "
                            "topics, use precise terminology."
                        ),
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
                "Open a URL and read its full text content. Use this to read "
                "articles, papers, documentation, or any web page found via "
                "search. Returns the page text (truncated if very long)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": (
                            "The full URL to open, starting with http:// or "
                            "https://. Must be a valid URL from search results "
                            "or a known web address."
                        ),
                    }
                },
                "required": ["url"],
            },
        },
    },
]


# =============================================================================
# System prompt
# =============================================================================

SYSTEM_PROMPT = """\
You are a deep research assistant with access to web search and URL reading tools. Your goal is to provide thorough, accurate, and well-sourced answers by actively researching the web.

## Your Tools

You have two tools available:

1. **search(query)** — Search the web for information. Returns a list of results with titles, URLs, and snippets. Use this to find relevant sources on any topic.
2. **open_url(url)** — Open and read the full content of a web page. Use this to read articles, papers, or documentation found via search.

## Research Strategy

Follow this approach when answering questions:

1. **Analyze the question**: Break it down into the specific pieces of information you need. Identify what you know and what you need to look up.
2. **Search strategically**: Start with a broad search, then refine with more specific queries. Use different phrasings if initial results are insufficient. For recent events, include the current year in your query.
3. **Read primary sources**: Use open_url to read the most promising search results in full. Prefer authoritative sources (official docs, academic papers, reputable news outlets).
4. **Cross-reference**: Verify key claims across multiple independent sources. If sources disagree, note the discrepancy and explain which is more reliable.
5. **Iterate**: If your first round of research is incomplete, search again with refined queries. Don't hesitate to do multiple rounds of search → read → search.

## Response Guidelines

- **Be thorough**: Research until you have enough information to give a comprehensive answer. Don't stop after one search if more detail is needed.
- **Cite your sources**: Reference the URLs where you found key information so the user can verify.
- **Acknowledge uncertainty**: If information is conflicting or unavailable, say so clearly rather than guessing.
- **Stay current**: For time-sensitive topics, prioritize the most recent sources.
- **Structure your answer**: Use clear organization with headers, bullet points, or numbered lists as appropriate for readability."""


# =============================================================================
# Tool implementations
# =============================================================================


async def tool_search(query: str, http_client: httpx.AsyncClient) -> str:
    """Execute a web search via Serper API."""
    api_key = SERPER_API_KEY or os.getenv("SERPER_API_KEY", "")
    if not api_key:
        return "Error: SERPER_API_KEY not set. Cannot perform web search."
    try:
        resp = await http_client.post(
            "https://google.serper.dev/search",
            json={"q": query, "num": 10},
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("organic", [])
        if not results:
            return f"No search results found for: {query}"
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


async def tool_open_url(url: str, http_client: httpx.AsyncClient) -> str:
    """Read a URL's content via Jina Reader API (or direct fetch as fallback)."""
    api_key = JINA_API_KEY or os.getenv("JINA_API_KEY", "")
    if not api_key:
        # Direct fetch fallback
        try:
            resp = await http_client.get(url, follow_redirects=True, timeout=30)
            return resp.text[:20000]
        except Exception as e:
            return f"Error fetching URL: {e}"
    try:
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
            content = content[:30000] + "\n\n[... content truncated ...]"
        return content
    except Exception as e:
        return f"Error reading URL: {e}"


async def execute_tool(name: str, args: dict, http_client: httpx.AsyncClient) -> str:
    """Dispatch a tool call by name."""
    if name == "search":
        query = args.get("query", "")
        if not query:
            return "Error: search requires a 'query' parameter"
        return await tool_search(query, http_client)
    elif name == "open_url":
        url = args.get("url", "")
        if not url or not url.startswith(("http://", "https://")):
            hint = url or args.get("query", "")
            if hint:
                return (
                    f"Error: open_url requires a valid URL starting with "
                    f"http:// or https://. Got: '{hint[:100]}'. "
                    f"Use the search tool to find URLs first."
                )
            return "Error: open_url requires a 'url' parameter with a valid URL"
        return await tool_open_url(url, http_client)
    return f"Unknown tool: {name}"


# =============================================================================
# Harmony format parsing
# =============================================================================


def parse_tool_call_from_raw(text: str) -> Optional[Tuple[str, dict]]:
    """Parse a Harmony-format tool call from raw generated text.

    The model generates one of:
      ... to=functions.TOOLNAME<|channel|>commentary json<|message|>{...}<|call|>
      ... to=functions.TOOLNAME<|channel|>commentary json<|message|>"..."<|call|>
    """
    m = re.search(r'to=functions\.(\w+)', text)
    if not m:
        return None

    tool_name = m.group(1)

    # Try to find JSON args after <|message|>
    msg_match = re.search(
        r'<\|message\|>(.*?)(?:<\|call\|>|<\|end\|>|$)', text, re.DOTALL
    )
    if msg_match:
        args_str = msg_match.group(1).strip()
    else:
        # Fallback: find JSON after the tool name marker
        after = text[m.end():]
        json_match = re.search(r'(?:json|code)\s*(\{.*?\})\s*$', after, re.DOTALL)
        if json_match:
            args_str = json_match.group(1)
        else:
            args_str = after.strip()

    # Unescape if wrapped in quotes
    if args_str.startswith('"') and args_str.endswith('"'):
        try:
            args_str = json.loads(args_str)
        except Exception:
            args_str = args_str[1:-1]

    try:
        args = json.loads(args_str)
        return tool_name, args
    except json.JSONDecodeError:
        # Last resort: extract individual known fields
        query_match = re.search(r'"query"\s*:\s*"([^"]*)"', args_str)
        if query_match:
            return tool_name, {"query": query_match.group(1)}
        url_match = re.search(r'"url"\s*:\s*"([^"]*)"', args_str)
        if url_match:
            return tool_name, {"url": url_match.group(1)}

    return None


def extract_final_answer(raw_text: str) -> Tuple[str, str]:
    """Extract (reasoning, final_answer) from Harmony channel format.

    The raw text may contain:
      analysis ... <|end|><|start|>assistant<|channel|>final<|message|>ANSWER
    Or inline channel markers like:
      analysis ... assistantfinal ANSWER
    """
    reasoning = ""
    answer = raw_text

    # Pattern 1: Harmony special tokens
    if "<|channel|>final<|message|>" in raw_text:
        parts = raw_text.split("<|channel|>final<|message|>", 1)
        reasoning = parts[0].strip()
        answer = parts[1].strip()
        # Clean up reasoning: remove Harmony tokens
        reasoning = re.sub(r'<\|[^|]+\|>', '', reasoning).strip()
        reasoning = re.sub(r'\bassistant\b', '', reasoning).strip()
        reasoning = re.sub(r'\b(analysis|commentary)\b', '', reasoning).strip()
    # Pattern 2: Inline "final" channel marker
    elif re.search(r'(?:assistant)?final\s', raw_text):
        final_match = re.search(r'(?:assistant)?final\s+(.*?)$', raw_text, re.DOTALL)
        if final_match:
            answer = final_match.group(1).strip()
            reasoning = raw_text[:final_match.start()].strip()
            reasoning = re.sub(r'\bassistant\b', '', reasoning).strip()
            reasoning = re.sub(r'\b(analysis|commentary)\b', '', reasoning).strip()
    # Pattern 3: analysis channel only
    elif raw_text.startswith("analysis"):
        # Everything after "analysis" until end is the answer
        answer = raw_text[len("analysis"):].strip()

    # Clean trailing Harmony tokens from answer
    answer = re.sub(r'<\|[^|]+\|>', '', answer).strip()

    return reasoning, answer

