#!/usr/bin/env python3
"""Run GPT-4o judge on existing trajectories without regenerating."""
import argparse
import asyncio
import json
import os
import sys

import dotenv
import httpx

dotenv.load_dotenv()

JUDGE_PROMPT = """\
You are an impartial judge evaluating whether a model's answer is correct.

**Question:** {question}
**Reference answer:** {reference}
**Model's answer:** {prediction}

Evaluate whether the model's answer is correct. Be lenient about formatting.
Respond with a JSON object:
{{"correct": true/false, "explanation": "brief reason"}}"""


async def judge(q, ref, pred, http):
    if not pred.strip():
        return {"correct": False, "explanation": "Empty answer"}
    try:
        resp = await http.post(
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content":
                JUDGE_PROMPT.format(question=q, reference=ref, prediction=pred)}],
                "temperature": 0, "max_tokens": 256},
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
            timeout=60,
        )
        content = resp.json()["choices"][0]["message"]["content"].strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content)
        return {"correct": bool(result.get("correct")), "explanation": result.get("explanation", "")}
    except Exception as e:
        return {"correct": False, "explanation": f"Judge error: {e}"}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    traj_file = os.path.join(args.output_dir, "trajectories.jsonl")
    results_file = os.path.join(args.output_dir, "results.json")

    with open(traj_file) as f:
        trajs = [json.loads(line) for line in f]
    print(f"Judging {len(trajs)} trajectories...")

    sem = asyncio.Semaphore(10)
    done = 0

    async with httpx.AsyncClient(timeout=60) as http:
        async def judge_one(t):
            nonlocal done
            async with sem:
                pred = t.get("boxed_answer") or t.get("answer", "")
                r = await judge(t["question"], t.get("reference_answer", ""), pred, http)
                done += 1
                if done % 200 == 0:
                    print(f"  Judged {done}/{len(trajs)}")
                return r

        results = await asyncio.gather(*[judge_one(t) for t in trajs])

    for t, j in zip(trajs, results):
        t["judge"] = j

    by_qid = {}
    for t in trajs:
        by_qid.setdefault(t["qid"], []).append(t)

    per_question = []
    pass_count = 0
    for qid, ts in by_qid.items():
        nc = sum(1 for t in ts if t.get("judge", {}).get("correct"))
        any_c = nc > 0
        if any_c:
            pass_count += 1
        per_question.append({
            "qid": qid, "num_correct": nc,
            "accuracy": nc / max(len(ts), 1), "pass": any_c,
            "answers": [{"traj_idx": t.get("traj_idx"),
                         "boxed_answer": t.get("boxed_answer", ""),
                         "correct": t.get("judge", {}).get("correct", False),
                         "explanation": t.get("judge", {}).get("explanation", "")}
                        for t in sorted(ts, key=lambda x: x.get("traj_idx", 0))]
        })

    n_q = len(by_qid)
    pass_at_k = pass_count / max(n_q, 1)
    avg_at_k = sum(q["accuracy"] for q in per_question) / max(n_q, 1)
    correct = sum(1 for t in trajs if t.get("judge", {}).get("correct"))

    summary = {
        "pass@4": pass_at_k, "avg@4": avg_at_k,
        "trajectory_accuracy": correct / len(trajs),
        "correct_trajectories": correct,
        "total_trajectories": len(trajs),
        "num_questions": n_q,
        "per_question": per_question,
    }
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\npass@4={pass_at_k:.1%}  avg@4={avg_at_k:.1%}  traj_acc={correct}/{len(trajs)} ({correct/len(trajs):.1%})")


if __name__ == "__main__":
    asyncio.run(main())

