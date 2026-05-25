import argparse
import dataclasses
import json
import os
import re
from collections.abc import Sequence
from typing import Any

import openai
from dotenv import load_dotenv

from .base import Evaluator, strip_think_blocks

load_dotenv()


DATASET_PATH = "rl-rag-2/finsearchcomp"
DEFAULT_ERROR_SCORE = -100000.0


@dataclasses.dataclass(frozen=True)
class FinSearchCompInstance:
    id: str
    question: str
    response_reference: str
    judge_prompt_template: str
    judge_system_prompt: str
    is_completed: bool = True


@dataclasses.dataclass(frozen=True)
class FinSearchCompJudgeResult:
    score: float
    correct: bool
    parse_error: bool
    raw_score: Any = None
    score_key: str | None = None
    error: str | None = None


@dataclasses.dataclass(frozen=True)
class FinSearchCompEvalResult:
    id: str
    task_type: str
    question: str
    response: str
    response_reference: str
    judge_system_prompt: str
    judge_user_input: str | None
    judge_response: str | None
    judge_metadata: dict[str, Any]
    judge_result: FinSearchCompJudgeResult


@dataclasses.dataclass(frozen=True)
class FinSearchCompEvalSummary:
    accuracy_percent: float
    total: int
    correct: int
    parse_errors: int


def create_finsearchcomp_judge_input(
    instance: FinSearchCompInstance,
    response: str,
) -> str:
    return instance.judge_prompt_template.format(
        prompt=instance.question,
        response_reference=instance.response_reference,
        response=response,
    )


def call_openai_judge(
    client: openai.OpenAI,
    judge_user_input: str,
    judge_system_prompt: str,
    model: str,
    max_output_tokens: int,
    reasoning_effort: str | None = None,
) -> Any:
    body: dict[str, Any] = {
        "model": model,
        "max_output_tokens": max_output_tokens,
        "input": judge_user_input,
        "instructions": judge_system_prompt,
    }

    if reasoning_effort is not None:
        body["reasoning"] = {"effort": reasoning_effort, "summary": "detailed"}

    return client.responses.create(**body)


def _iter_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []

    for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
        candidates.append(match.group(1))

    if not candidates:
        pattern = r"(\{[^{}]*answer_score[^{}]*\})"
        for match in re.finditer(pattern, text, re.DOTALL):
            candidates.append(match.group(1))

    return candidates


def parse_finsearchcomp_judge_output(judge_output: str) -> FinSearchCompJudgeResult:
    if not isinstance(judge_output, str) or not judge_output.strip():
        return FinSearchCompJudgeResult(
            score=DEFAULT_ERROR_SCORE,
            correct=False,
            parse_error=True,
        )

    for candidate in _iter_json_candidates(judge_output):
        try:
            judge_json = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if "answer_score" not in judge_json:
            continue

        raw_score = judge_json["answer_score"]
        if isinstance(raw_score, list):
            return FinSearchCompJudgeResult(
                score=DEFAULT_ERROR_SCORE,
                correct=False,
                parse_error=True,
                raw_score=raw_score,
                score_key="answer_score",
            )

        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            return FinSearchCompJudgeResult(
                score=DEFAULT_ERROR_SCORE,
                correct=False,
                parse_error=True,
                raw_score=raw_score,
                score_key="answer_score",
            )

        if score not in (0.0, 1.0):
            return FinSearchCompJudgeResult(
                score=DEFAULT_ERROR_SCORE,
                correct=False,
                parse_error=True,
                raw_score=raw_score,
                score_key="answer_score",
            )

        return FinSearchCompJudgeResult(
            score=score,
            correct=score == 1.0,
            parse_error=False,
            raw_score=raw_score,
            score_key="answer_score",
        )

    return FinSearchCompJudgeResult(
        score=DEFAULT_ERROR_SCORE,
        correct=False,
        parse_error=True,
    )


class FinSearchCompEvaluator(
    Evaluator[
        FinSearchCompInstance,
        FinSearchCompEvalResult,
        FinSearchCompEvalSummary,
    ]
):
    def __init__(
        self,
        model: str = "gpt-4.1",
        max_output_tokens: int = 2048,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.judge_metadata = {
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "reasoning_effort": self.reasoning_effort,
            "strip_think_blocks": True,
        }

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model",
            default="gpt-4.1",
            help="OpenAI model used as the FinSearchComp rubric judge.",
        )
        parser.add_argument(
            "--max_output_tokens",
            type=int,
            default=2048,
            help="Maximum number of tokens the judge may generate.",
        )
        parser.add_argument(
            "--reasoning-effort",
            choices=["low", "medium", "high"],
            default=None,
            help="Reasoning effort sent to reasoning-capable OpenAI judge models; omit for non-reasoning models.",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "FinSearchCompEvaluator":
        return cls(
            model=args.model,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
        )

    def prepare_instances(self) -> dict[str, FinSearchCompInstance]:
        import datasets

        rows = datasets.load_dataset(
            DATASET_PATH,
            split="train",
        )
        instances: dict[str, FinSearchCompInstance] = {}
        for row in rows:
            instance = FinSearchCompInstance(
                id=str(row["id"]),
                question=str(row["question"]),
                response_reference=str(row["response_reference"]),
                judge_prompt_template=str(row["judge_prompt_template"]),
                judge_system_prompt=str(row["judge_system_prompt"]),
            )
            instances[instance.id] = instance
        return instances

    def evaluate(
        self,
        instance: FinSearchCompInstance,
        response: str,
    ) -> FinSearchCompEvalResult:
        response = strip_think_blocks(response)
        task_type = "T2" if instance.id.startswith("(T2)") else "T3"
        if not instance.is_completed or not response:
            error = (
                "Response incomplete or cannot be parsed"
                if not instance.is_completed
                else "Response empty"
            )
            return FinSearchCompEvalResult(
                id=instance.id,
                task_type=task_type,
                question=instance.question,
                response=response,
                response_reference=instance.response_reference,
                judge_system_prompt=instance.judge_system_prompt,
                judge_user_input=None,
                judge_response=None,
                judge_metadata=dict(self.judge_metadata),
                judge_result=FinSearchCompJudgeResult(
                    score=0.0,
                    correct=False,
                    parse_error=True,
                    error=error,
                ),
            )

        judge_user_input = create_finsearchcomp_judge_input(instance, response)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in environment")
        client = openai.OpenAI(api_key=api_key)
        judge_response_obj = call_openai_judge(
            client=client,
            judge_user_input=judge_user_input,
            judge_system_prompt=instance.judge_system_prompt,
            model=self.model,
            max_output_tokens=self.max_output_tokens,
            reasoning_effort=self.reasoning_effort,
        )
        judge_response = judge_response_obj.output_text or ""
        judge_result = parse_finsearchcomp_judge_output(judge_response)
        return FinSearchCompEvalResult(
            id=instance.id,
            task_type=task_type,
            question=instance.question,
            response=response,
            response_reference=instance.response_reference,
            judge_system_prompt=instance.judge_system_prompt,
            judge_user_input=judge_user_input,
            judge_response=judge_response,
            judge_metadata=dict(self.judge_metadata),
            judge_result=judge_result,
        )

    def aggregate(
        self,
        results: Sequence[FinSearchCompEvalResult],
    ) -> FinSearchCompEvalSummary:
        total = len(results)
        correct_count = sum(1 for result in results if result.judge_result.correct)
        parse_errors = sum(1 for result in results if result.judge_result.parse_error)
        accuracy_percent = round((correct_count / total) * 100.0, 2) if total else 0.0

        return FinSearchCompEvalSummary(
            accuracy_percent=accuracy_percent,
            total=total,
            correct=correct_count,
            parse_errors=parse_errors,
        )
