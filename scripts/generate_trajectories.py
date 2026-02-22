#!/usr/bin/env python3
"""
SFT Trajectory Generation with Deep Research (gpt-oss-120b)
=============================================================

Uses the gpt-oss-120b model's native Harmony channel format for tool calling.
Generates via /v1/completions with tokenizer.apply_chat_template.

Tools: Serper (search) and Jina (web reader / URL fetcher).
Output: JSONL with full multi-turn trajectories.

Usage:
    python scripts/generate_trajectories.py \
        --scheduler-url http://localhost:8780 \
        --dataset sample --num-samples 3 \
        --output-dir results/trajectories
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
import traceback
from typing import Any, Dict

import dotenv
import httpx

dotenv.load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elastic_serving.tools import (
    SYSTEM_PROMPT,
    TOOLS,
    execute_tool,
    parse_tool_call_from_raw,
)


# =============================================================================
# Trajectory Generation
# =============================================================================

async def generate_one_trajectory(
    question: str,
    qid: Any,
    base_url: str,
    model: str,
    tokenizer,
    http_client: httpx.AsyncClient,
    openai_http: httpx.AsyncClient,
    max_rounds: int = 15,
    max_gen_tokens: int = 8192,
) -> Dict[str, Any]:
    """Generate a single research trajectory."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    round_num = 0
    total_tool_calls = 0

    # <|call|> marks end of tool call; <|end|> marks end of assistant turn
    stop_tokens = ["<|call|>", "<|end|>", "<|endoftext|>"]

    while round_num < max_rounds:
        round_num += 1

        # On the last round, don't stop at <|call|> — force a final answer
        is_last_round = (round_num == max_rounds)
        current_stops = ["<|end|>", "<|endoftext|>"] if is_last_round else stop_tokens

        # Build prompt
        prompt = tokenizer.apply_chat_template(
            messages, tools=TOOLS, tokenize=False, add_generation_prompt=True,
        )
        prompt_tokens = len(tokenizer.encode(prompt))
        suffix = " (final)" if is_last_round else ""
        print(f"  [qid={qid}] Round {round_num}/{max_rounds} ({prompt_tokens} tokens){suffix}")

        # Generate via /v1/completions
        try:
            resp = await openai_http.post(
                f"{base_url}/v1/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_gen_tokens,
                    "temperature": 0.7,
                    "stop": current_stops,
                },
                headers={"Authorization": "Bearer EMPTY"},
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["choices"][0]["text"]
            finish_reason = data["choices"][0].get("finish_reason", "")
        except Exception as e:
            print(f"  [qid={qid}] Generation error: {e}")
            messages.append({"role": "assistant", "content": f"[Error: {e}]"})
            break

        # Check if this is a tool call (stopped at <|call|>) or final answer
        tool_call = parse_tool_call_from_raw(raw_text) if not is_last_round else None

        if tool_call:
            tool_name, tool_args = tool_call
            print(f"  [qid={qid}]   Tool: {tool_name}({json.dumps(tool_args)[:120]})")

            # Record the tool call in messages
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
            total_tool_calls += 1

            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": result,
            })
            continue
        else:
            # Final answer — extract content from harmony channel format
            # Raw text looks like: "analysis...assistantfinal THE ANSWER"
            # or: "analysis...assistantcommentary...assistantfinal THE ANSWER"
            content = raw_text
            reasoning = ""

            # Extract the final channel content (everything after "assistantfinal" or just "final")
            final_match = re.search(r'(?:assistant)?final\s*(.*?)$', content, re.DOTALL)
            if final_match:
                final_content = final_match.group(1).strip()
                # Everything before "final" is reasoning/analysis
                reasoning = content[:final_match.start()].strip()
            else:
                # No "final" channel — treat entire text as content
                # Remove "analysis" prefix if present
                if content.startswith("analysis"):
                    reasoning_match = re.match(r'analysis(.*?)(?:assistant|$)', content, re.DOTALL)
                    if reasoning_match:
                        reasoning = reasoning_match.group(1).strip()
                        final_content = content[reasoning_match.end():].strip()
                    else:
                        final_content = content[len("analysis"):].strip()
                else:
                    final_content = content.strip()

            # Clean up reasoning: remove "assistant" prefixes
            reasoning = re.sub(r'\bassistant\b', '', reasoning).strip()
            # Remove repeated channel markers
            reasoning = re.sub(r'\b(analysis|commentary)\b', '', reasoning).strip()

            msg = {"role": "assistant", "content": final_content}
            if reasoning:
                msg["reasoning_content"] = reasoning

            messages.append(msg)
            print(f"  [qid={qid}] Final answer ({round_num} rounds, {total_tool_calls} tools, {len(final_content)} chars)")
            break

    return {
        "qid": qid,
        "question": question,
        "messages": messages,
        "num_rounds": round_num,
        "num_tool_calls": total_tool_calls,
    }


async def run_generation(
    scheduler_url: str,
    model: str,
    dataset_name: str,
    num_samples: int,
    concurrency: int,
    output_dir: str,
    max_rounds: int,
    max_gen_tokens: int,
):
    from transformers import AutoTokenizer

    print(f"Loading tokenizer for {model}...")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    http_client = httpx.AsyncClient(timeout=60)
    openai_http = httpx.AsyncClient(timeout=300)

    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    if dataset_name == "sample":
        data = [
            {"qid": 1, "question": "What were the key findings of the most recent IPCC report on climate change, and how do they compare to the predictions made in the 2018 special report on 1.5°C warming?"},
            {"qid": 2, "question": "Who is the current CEO of Anthropic, when was the company founded, and what is their stated mission regarding AI safety?"},
            {"qid": 3, "question": "What is the latest breakthrough in room-temperature superconductivity research as of 2024, and what is the scientific consensus on the LK-99 claims?"},
            {"qid": 4, "question": "Describe the architecture and key innovations of the Mamba state space model. How does it compare to Transformers in terms of computational complexity for long sequences?"},
            {"qid": 5, "question": "What is the current state of nuclear fusion energy research? Describe the NIF's ignition achievement and the ITER project timeline."},
        ]
    else:
        try:
            sys.path.insert(0, "/tmp/OpenResearcher")
            from data_utils import load_dataset
            data = load_dataset(dataset_name)
        except Exception as e:
            print(f"Could not load dataset '{dataset_name}': {e}")
            return

    if num_samples > 0:
        data = data[:num_samples]

    # Wait for workers
    async with httpx.AsyncClient() as tmp:
        resp = await tmp.get(f"{scheduler_url.rstrip('/')}/cluster_status")
        status = resp.json()
    print(f"Cluster: {status['ready_workers']} ready workers")

    if status["ready_workers"] == 0:
        print("⚠  No ready workers! Waiting...")
        for _ in range(120):
            await asyncio.sleep(5)
            async with httpx.AsyncClient() as tmp:
                resp = await tmp.get(f"{scheduler_url.rstrip('/')}/cluster_status")
                status = resp.json()
            if status["ready_workers"] > 0:
                print(f"✅ {status['ready_workers']} workers ready")
                break
            print(f"  Waiting... (ready={status['ready_workers']}, loading={status['loading_workers']})")
        else:
            print("Timed out.")
            return

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"trajectories_{dataset_name}.jsonl")

    completed_qids = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                try:
                    completed_qids.add(json.loads(line)["qid"])
                except Exception:
                    pass
        print(f"Resuming: {len(completed_qids)} done")

    pending = [d for d in data if d["qid"] not in completed_qids]
    if not pending:
        print("All done!")
        return

    print(f"Processing {len(pending)} samples (concurrency={concurrency})...\n")

    sem = asyncio.Semaphore(concurrency)
    completed = 0

    async def process_one(item):
        nonlocal completed
        async with sem:
            t0 = time.time()
            try:
                result = await generate_one_trajectory(
                    question=item["question"], qid=item["qid"],
                    base_url=scheduler_url.rstrip("/"), model=model,
                    tokenizer=tokenizer, http_client=http_client,
                    openai_http=openai_http, max_rounds=max_rounds,
                    max_gen_tokens=max_gen_tokens,
                )
                result["answer_ref"] = item.get("answer", "")
                result["latency_s"] = time.time() - t0
                result["status"] = "success"
            except Exception as e:
                result = {
                    "qid": item["qid"], "question": item["question"],
                    "messages": [], "error": traceback.format_exc(),
                    "latency_s": time.time() - t0, "status": "fail",
                }
            completed += 1
            print(f"[{completed}/{len(pending)}] qid={item['qid']} "
                  f"{result['status']} rounds={result.get('num_rounds',0)} "
                  f"tools={result.get('num_tool_calls',0)} "
                  f"time={result['latency_s']:.1f}s")
            return result

    tasks = [asyncio.create_task(process_one(item)) for item in pending]
    with open(output_file, "a") as writer:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            writer.write(json.dumps(result, ensure_ascii=False) + "\n")
            writer.flush()

    await http_client.aclose()
    await openai_http.aclose()
    print(f"\n✅ Done! {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate SFT trajectories")
    parser.add_argument("--scheduler-url", type=str,
                        default=os.environ.get("ELASTIC_SERVING_URL", "http://localhost:8780"))
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="sample")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=15)
    parser.add_argument("--max-gen-tokens", type=int, default=8192)
    parser.add_argument("--output-dir", type=str, default="results/trajectories")
    args = parser.parse_args()

    if not args.model:
        try:
            resp = httpx.get(f"{args.scheduler_url.rstrip('/')}/cluster_status", timeout=5)
            args.model = resp.json().get("model", "default")
        except Exception:
            args.model = "default"

    asyncio.run(run_generation(
        scheduler_url=args.scheduler_url, model=args.model,
        dataset_name=args.dataset, num_samples=args.num_samples,
        concurrency=args.concurrency, output_dir=args.output_dir,
        max_rounds=args.max_rounds, max_gen_tokens=args.max_gen_tokens,
    ))


if __name__ == "__main__":
    main()
