#!/usr/bin/env python3
"""
Interactive CLI Chat with Tool Use (Harmony Format)
=====================================================

Chat with a model served by Elastic Serving. The model can use web search
(Serper) and URL reading (Jina) tools via native Harmony token format.
Reasoning (analysis channel) is shown dimmed; the final answer is highlighted.

Usage:
    python scripts/chat.py --scheduler-url http://localhost:8780
    python scripts/chat.py --scheduler-url http://localhost:8780 --model /path/to/model

Environment:
    SERPER_API_KEY   — for web search (Serper)
    JINA_API_KEY     — for URL reading (Jina Reader)
    ELASTIC_SERVING_URL — default scheduler URL
"""

import argparse
import asyncio
import json
import os
import sys
import time

import dotenv
import httpx

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dotenv.load_dotenv()

from elastic_serving.tools import (
    SYSTEM_PROMPT,
    TOOLS,
    execute_tool,
    extract_final_answer,
    parse_tool_call_from_raw,
)

# =============================================================================
# ANSI colors for terminal output
# =============================================================================

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    GRAY = "\033[90m"


def print_colored(text: str, color: str = "", end: str = "\n"):
    print(f"{color}{text}{C.RESET}", end=end)


def print_tool_call(name: str, args: dict):
    args_short = json.dumps(args, ensure_ascii=False)
    if len(args_short) > 120:
        args_short = args_short[:117] + "..."
    print_colored(f"  🔧 {name}({args_short})", C.YELLOW)


def print_tool_result(result: str, max_lines: int = 8):
    lines = result.split("\n")
    preview = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        preview += f"\n  ... ({len(lines) - max_lines} more lines)"
    for line in preview.split("\n"):
        print_colored(f"  │ {line}", C.GRAY)


def print_reasoning(text: str):
    if not text.strip():
        return
    print_colored("  💭 Reasoning:", C.DIM + C.ITALIC)
    for line in text.strip().split("\n"):
        print_colored(f"  │ {line}", C.DIM)
    print()


def print_answer(text: str):
    print_colored("  📝 Answer:", C.BOLD + C.GREEN)
    print()
    for line in text.strip().split("\n"):
        print(f"  {line}")
    print()


# =============================================================================
# Chat engine
# =============================================================================


async def chat_turn(
    messages: list,
    base_url: str,
    model: str,
    tokenizer,
    http_client: httpx.AsyncClient,
    openai_http: httpx.AsyncClient,
    max_rounds: int = 15,
    max_gen_tokens: int = 8192,
    temperature: float = 0.7,
    verbose: bool = False,
):
    """
    Run one user turn: generate assistant response with tool-call loop.
    Modifies `messages` in place. Returns the final answer string.
    """
    stop_tokens = ["<|call|>", "<|end|>", "<|endoftext|>"]
    round_num = 0

    while round_num < max_rounds:
        round_num += 1
        is_last_round = (round_num == max_rounds)
        current_stops = ["<|end|>", "<|endoftext|>"] if is_last_round else stop_tokens

        # Build prompt via tokenizer
        prompt = tokenizer.apply_chat_template(
            messages, tools=TOOLS, tokenize=False, add_generation_prompt=True,
        )
        prompt_tokens = len(tokenizer.encode(prompt))

        if verbose:
            suffix = " (final)" if is_last_round else ""
            print_colored(
                f"  ⏳ Round {round_num}/{max_rounds} "
                f"({prompt_tokens} prompt tokens){suffix}",
                C.GRAY,
            )

        # Generate via /v1/completions (with retry on 503 / no workers)
        t0 = time.time()
        max_retries = 60  # up to ~5 min of waiting
        for attempt in range(max_retries):
            try:
                resp = await openai_http.post(
                    f"{base_url}/v1/completions",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "max_tokens": max_gen_tokens,
                        "temperature": temperature,
                        "stop": current_stops,
                    },
                    headers={"Authorization": "Bearer EMPTY"},
                    timeout=300,
                )
                if resp.status_code == 503:
                    # No ready workers — wait and retry
                    if attempt == 0:
                        print_colored(
                            "  ⏳ No ready workers yet, waiting for "
                            "SLURM jobs to start...",
                            C.YELLOW,
                        )
                    # Show a dot every 5 retries for progress
                    if attempt > 0 and attempt % 5 == 0:
                        elapsed_wait = attempt * 5
                        print_colored(
                            f"  ⏳ Still waiting... ({elapsed_wait}s)",
                            C.GRAY,
                        )
                    await asyncio.sleep(5)
                    continue
                resp.raise_for_status()
                data = resp.json()
                raw_text = data["choices"][0]["text"]
                finish_reason = data["choices"][0].get("finish_reason", "")
                break
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 503:
                    if attempt == 0:
                        print_colored(
                            "  ⏳ No ready workers yet, waiting for "
                            "SLURM jobs to start...",
                            C.YELLOW,
                        )
                    await asyncio.sleep(5)
                    continue
                print_colored(f"  ❌ Generation error: {e}", C.RED)
                messages.append({"role": "assistant", "content": f"[Error: {e}]"})
                return f"[Error: {e}]"
            except Exception as e:
                print_colored(f"  ❌ Generation error: {e}", C.RED)
                messages.append({"role": "assistant", "content": f"[Error: {e}]"})
                return f"[Error: {e}]"
        else:
            msg = "Timed out waiting for workers to become ready."
            print_colored(f"  ❌ {msg}", C.RED)
            messages.append({"role": "assistant", "content": f"[{msg}]"})
            return f"[{msg}]"

        elapsed = time.time() - t0
        gen_tokens = data.get("usage", {}).get("completion_tokens", len(raw_text) // 4)
        if verbose:
            print_colored(
                f"  ⏱  Generated {gen_tokens} tokens in {elapsed:.1f}s "
                f"({gen_tokens / max(elapsed, 0.01):.0f} tok/s)",
                C.GRAY,
            )

        # Check for tool call
        tool_call = parse_tool_call_from_raw(raw_text) if not is_last_round else None

        if tool_call:
            tool_name, tool_args = tool_call
            print_tool_call(tool_name, tool_args)

            # Record tool call in messages
            call_id = f"call_{round_num}"
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args),
                    },
                }],
            })

            # Execute tool
            result = await execute_tool(tool_name, tool_args, http_client)
            print_tool_result(result)

            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": result,
            })
            continue
        else:
            # Final answer
            reasoning, answer = extract_final_answer(raw_text)
            if reasoning:
                print_reasoning(reasoning)
            print_answer(answer)

            # Store in messages for multi-turn context
            msg = {"role": "assistant", "content": answer}
            if reasoning:
                msg["reasoning_content"] = reasoning
            messages.append(msg)

            return answer

    return "[Max rounds reached without final answer]"


# =============================================================================
# Interactive REPL
# =============================================================================


async def interactive_chat(
    scheduler_url: str,
    model: str,
    tokenizer,
    max_rounds: int = 15,
    max_gen_tokens: int = 8192,
    temperature: float = 0.7,
    verbose: bool = False,
    system_prompt: str = SYSTEM_PROMPT,
):
    http_client = httpx.AsyncClient(timeout=60)
    openai_http = httpx.AsyncClient(timeout=300)

    messages = [
        {"role": "system", "content": system_prompt},
    ]

    base_url = scheduler_url.rstrip("/")

    # Header
    print()
    print_colored("╔══════════════════════════════════════════════════╗", C.CYAN)
    print_colored("║     Elastic Serving — Interactive Research Chat  ║", C.CYAN)
    print_colored("╚══════════════════════════════════════════════════╝", C.CYAN)
    print_colored(f"  Model:    {model}", C.GRAY)
    print_colored(f"  Server:   {base_url}", C.GRAY)
    print_colored(f"  Tools:    search, open_url", C.GRAY)
    print_colored(f"  Rounds:   up to {max_rounds} per turn", C.GRAY)
    print()
    print_colored("  Commands: /clear  — reset conversation", C.GRAY)
    print_colored("            /system — show/change system prompt", C.GRAY)
    print_colored("            /verbose — toggle verbose mode", C.GRAY)
    print_colored("            /quit or Ctrl+C — exit", C.GRAY)
    print()

    while True:
        try:
            user_input = input(f"{C.BOLD}{C.BLUE}You ❯ {C.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print_colored("Goodbye!", C.CYAN)
            break

        if not user_input:
            continue

        # Commands
        if user_input.lower() in ("/quit", "/exit", "/q"):
            print_colored("Goodbye!", C.CYAN)
            break

        if user_input.lower() == "/clear":
            messages = [{"role": "system", "content": system_prompt}]
            print_colored("  ✅ Conversation cleared.", C.GREEN)
            continue

        if user_input.lower() == "/system":
            print_colored("  Current system prompt:", C.GRAY)
            for line in system_prompt.split("\n"):
                print_colored(f"  │ {line}", C.GRAY)
            continue

        if user_input.lower() == "/verbose":
            verbose = not verbose
            print_colored(f"  Verbose mode: {'ON' if verbose else 'OFF'}", C.GREEN)
            continue

        if user_input.lower() == "/history":
            print_colored(f"  Conversation: {len(messages)} messages", C.GRAY)
            for i, m in enumerate(messages):
                role = m["role"]
                content = m.get("content", "")
                preview = content[:80] + "..." if len(content) > 80 else content
                tc = m.get("tool_calls")
                if tc:
                    preview = f"[tool_call: {tc[0]['function']['name']}]"
                print_colored(f"  [{i}] {role}: {preview}", C.GRAY)
            continue

        # Add user message
        messages.append({"role": "user", "content": user_input})

        print()
        print_colored(f"{'─' * 60}", C.GRAY)

        t0 = time.time()
        await chat_turn(
            messages=messages,
            base_url=base_url,
            model=model,
            tokenizer=tokenizer,
            http_client=http_client,
            openai_http=openai_http,
            max_rounds=max_rounds,
            max_gen_tokens=max_gen_tokens,
            temperature=temperature,
            verbose=verbose,
        )
        elapsed = time.time() - t0
        print_colored(f"{'─' * 60}", C.GRAY)
        print_colored(f"  ⏱  Total turn: {elapsed:.1f}s", C.GRAY)
        print()

    await http_client.aclose()
    await openai_http.aclose()


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Interactive CLI chat with web search tools (Harmony format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Basic usage (auto-detects model from scheduler)
  python scripts/chat.py --scheduler-url http://localhost:8780

  # Specify model explicitly
  python scripts/chat.py \\
      --scheduler-url http://localhost:8780 \\
      --model /checkpoint/maestro/models/gpt-oss-120b

  # Verbose mode with higher temperature
  python scripts/chat.py --verbose --temperature 0.9

Environment Variables:
  ELASTIC_SERVING_URL  Default scheduler URL
  SERPER_API_KEY       Serper API key for web search
  JINA_API_KEY         Jina API key for URL reading
""",
    )
    parser.add_argument(
        "--scheduler-url",
        type=str,
        default=os.environ.get("ELASTIC_SERVING_URL", "http://localhost:8780"),
        help="Elastic Serving scheduler URL",
    )
    parser.add_argument("--model", type=str, default=None, help="Model name/path")
    parser.add_argument(
        "--max-rounds", type=int, default=15,
        help="Max tool-call rounds per user turn (default: 15)",
    )
    parser.add_argument(
        "--max-gen-tokens", type=int, default=8192,
        help="Max tokens per generation (default: 8192)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show token counts and timing per round",
    )
    parser.add_argument(
        "--system-prompt", type=str, default=None,
        help="Override system prompt (or path to a .txt file)",
    )
    args = parser.parse_args()

    # Auto-detect model from scheduler
    if not args.model:
        try:
            resp = httpx.get(
                f"{args.scheduler_url.rstrip('/')}/cluster_status", timeout=5
            )
            args.model = resp.json().get("model", "")
            if not args.model:
                print("Error: Could not detect model from scheduler. Use --model.")
                sys.exit(1)
            print(f"Auto-detected model: {args.model}")
        except Exception as e:
            print(f"Error connecting to scheduler at {args.scheduler_url}: {e}")
            print("Make sure the scheduler is running, or specify --model explicitly.")
            sys.exit(1)

    # Check scheduler health
    try:
        resp = httpx.get(f"{args.scheduler_url.rstrip('/')}/health", timeout=5)
        health = resp.json()
        ready = health.get("ready_workers", 0)
        if ready == 0:
            print(f"Warning: No ready workers yet. Chat may fail until workers are ready.")
        else:
            print(f"Scheduler healthy: {ready} worker(s) ready.")
    except Exception:
        print(f"Warning: Cannot reach scheduler at {args.scheduler_url}")

    # Load tokenizer
    print(f"Loading tokenizer for {args.model}...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("Tokenizer loaded.")

    # System prompt
    system_prompt = SYSTEM_PROMPT
    if args.system_prompt:
        if os.path.isfile(args.system_prompt):
            with open(args.system_prompt) as f:
                system_prompt = f.read()
        else:
            system_prompt = args.system_prompt

    asyncio.run(
        interactive_chat(
            scheduler_url=args.scheduler_url,
            model=args.model,
            tokenizer=tokenizer,
            max_rounds=args.max_rounds,
            max_gen_tokens=args.max_gen_tokens,
            temperature=args.temperature,
            verbose=args.verbose,
            system_prompt=system_prompt,
        )
    )


if __name__ == "__main__":
    main()

