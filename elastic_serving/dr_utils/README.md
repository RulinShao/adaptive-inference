# Deep Research Utils

Tools and prompts for deep research agents using Harmony-native format.

## Tools

### Browser tools (`browser.*` namespace — built-in Harmony)

| Tool | Backend | Description |
|------|---------|-------------|
| `browser.search(query)` | Serper API | Web search. Returns ranked results with `【{id}†{title}】` markers. |
| `browser.open(id)` | Jina Reader | Open a search result by id or a URL string. Returns line-numbered page text. |
| `browser.find(pattern)` | local | Find text in the currently opened page. No API call. |

```
browser.search({"query": "CRISPR gene therapy 2024"})
browser.open({"id": 3})                          # open 3rd search result
browser.open({"id": "https://arxiv.org/..."})     # open URL directly
browser.find({"pattern": "efficacy"})             # find in opened page
browser.open({"cursor": 2, "loc": 50, "num_lines": 20})  # scroll
```

### Custom tools (`functions.*` namespace)

| Tool | Backend | Description |
|------|---------|-------------|
| `paper_search(query)` | Semantic Scholar | Academic paper search. Two modes: `snippets` (default) returns text passages from paper body; `papers` returns metadata + PDF links. |
| `pubmed_search(query)` | NCBI PubMed | Biomedical literature search. Returns titles, structured abstracts, journal, PubMed URL. |

```
paper_search({"query": "Mamba state space model"})                          # snippets (default)
paper_search({"query": "Mamba SSM", "mode": "papers", "limit": 3})         # metadata + PDFs
paper_search({"query": "LLM scaling", "year": "2024-2025", "venue": "NeurIPS"})
pubmed_search({"query": "CRISPR sickle cell disease", "limit": 5})
```

## API Keys

| Env var | Required for | How to get |
|---------|-------------|------------|
| `SERPER_API_KEY` | `browser.search` | https://serper.dev |
| `JINA_API_KEY` | `browser.open` | https://jina.ai (optional — falls back to direct fetch) |
| `S2_API_KEY` | `paper_search` | https://www.semanticscholar.org/product/api#api-key (optional — works without, rate-limited) |
| — | `pubmed_search` | Free, no key needed |

## Files

```
dr_utils/
├── tools.py      # Tool specs, implementations, and dispatchers
├── prompts.py    # SYSTEM_PROMPT, MODEL_IDENTITY
└── __init__.py   # Re-exports
```

## Adding a new tool

1. Add the tool spec dict to `tools.py` (OpenAI function-calling format).
2. Add an `async def` implementation.
3. Append to `CUSTOM_TOOLS` list and add a branch in `execute_custom_tool()`.

