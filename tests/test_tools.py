#!/usr/bin/env python3
"""
Test tool implementations — shows exactly what each tool returns.
No LLM involved, just direct API calls.

Usage:
    python tests/test_tools.py
"""

import asyncio
import os
import sys

import dotenv
import httpx

dotenv.load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elastic_serving.dr_utils.tools import BrowserSession, paper_search


def banner(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


async def main():
    async with httpx.AsyncClient(timeout=30) as http:
        browser = BrowserSession(http)

        # =====================================================================
        # 1. browser.search
        # =====================================================================
        banner("browser.search({'query': 'Mamba state space model'})")
        result = await browser.search("Mamba state space model")
        print(result)

        # =====================================================================
        # 2. browser.open — by search result ID
        # =====================================================================
        banner("browser.open({'id': 1})  — open first search result")
        result = await browser.open(id=1)
        # Show first 60 lines only
        lines = result.split("\n")
        print("\n".join(lines[:60]))
        if len(lines) > 60:
            print(f"\n... ({len(lines) - 60} more lines)")

        # =====================================================================
        # 3. browser.find — search text in the opened page
        # =====================================================================
        banner("browser.find({'pattern': 'selective'})  — find in opened page")
        result = await browser.find("selective")
        print(result)

        # =====================================================================
        # 4. browser.open — scroll to a specific location
        # =====================================================================
        banner("browser.open({'cursor': 2, 'loc': 10, 'num_lines': 15})  — scroll")
        result = await browser.open(cursor=2, loc=10, num_lines=15)
        print(result)

        # =====================================================================
        # 5. browser.open — by direct URL
        # =====================================================================
        banner("browser.open({'id': 'https://en.wikipedia.org/wiki/Mamba_(deep_learning)'})")
        result = await browser.open(id="https://en.wikipedia.org/wiki/Mamba_(deep_learning)")
        lines = result.split("\n")
        print("\n".join(lines[:60]))
        if len(lines) > 60:
            print(f"\n... ({len(lines) - 60} more lines)")

        # =====================================================================
        # 6. browser.open — bad inputs (what the model sometimes does)
        # =====================================================================
        banner("browser.open({'query': '\\\\'})  — bad input (model bug)")
        result = await browser.open(query="\\")
        print(result)

        banner("browser.open({'pattern': 'attendance'})  — wrong tool (should be find)")
        result = await browser.open(pattern="attendance")
        print(result)

        # =====================================================================
        # 7. paper_search — mode="papers" (metadata)
        # =====================================================================
        banner("paper_search(query='Mamba SSM', mode='papers', limit=3)")
        result = await paper_search("Mamba SSM", http, mode="papers", limit=3)
        print(result)

        # =====================================================================
        # 8. paper_search — mode="snippets" (text passages from content)
        # =====================================================================
        await asyncio.sleep(2)  # rate limit courtesy
        banner("paper_search(query='selective state space model inference speed', mode='snippets', limit=5)")
        result = await paper_search(
            "selective state space model inference speed",
            http, mode="snippets", limit=5,
        )
        print(result)

        # =====================================================================
        # 9. paper_search — snippets with year filter
        # =====================================================================
        await asyncio.sleep(2)
        banner("paper_search(query='deep research agent', mode='snippets', limit=3, year='2024-2025')")
        result = await paper_search(
            "deep research agent", http,
            mode="snippets", limit=3, year="2024-2025",
        )
        print(result)


if __name__ == "__main__":
    asyncio.run(main())

