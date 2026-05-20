import argparse
import dataclasses
import os
import re
from collections.abc import Sequence
from typing import Any, Optional

import openai
from dotenv import load_dotenv

from .base import Evaluator, strip_think_blocks

load_dotenv()


DATASET_PATH = "rl-rag-2/frontierscience"


OLYMPIAD_JUDGE_PROMPT = """
You are grading an attempted answer to a science olympiad problem. You will be given the
problem, attempted answer, and reference answer. Evaluate the solution against the provided reference solution, ensuring it is complete and matches the reference solution. Pay close attention to detail and grade it strictly, but fairly.

The reference answer is either a single number or expression in latex formatting, a chemical formula, a compound name, or a phrase referring to a specific name, entity, or method.

Mark the attempted answer as correct if it fully matches the reference answer or is otherwise equivalent (e.g., an equivalent algebraic expression, a numerical number within 1 decimal place rounding of the reference answer, an equivalent name for a compound/formula, equivalent when accounting for units, etc.). Mark it as incorrect if it is not equivalent to the reference answer.

***
The problem: {problem}
***
The reference answer: {reference_answer}
***
The attempted answer: {answer}
***

First, think step-by-step about whether the attempted answer matches the reference answer. If the attempted answer is correct, write "VERDICT: CORRECT" in the last line of your response, with no other text or formatting. If it is incorrect, write "VERDICT: INCORRECT".
""".strip()


RESEARCH_JUDGE_PROMPT = """
You are grading a science exam.

You will be given the problem, attempted answer, and a rubric to grade the answer. The rubric will total up to 10 points.

Evaluate the attempted answer against the provided rubric. Pay close attention to detail and grade it strictly, but fairly. Only evaluate against the rubric, as you yourself should not make any judgements (e.g., even if you think the answer is correct but rubric is wrong, you should treat the rubric as the gold standard). Return the absolute total number of points earned (it can be a decimal based on the rubric).

***
The problem: {problem}
***
The rubric: {rubric}
***
The attempted answer: {answer}
***

First, think step-by-step about each rubric item. Explain your reasoning for each rubric item. Then, tally the points up and write VERDICT: <total_points> in the last line of your response, no other text. For example, VERDICT: 2.5 or VERDICT: 8.
""".strip()


@dataclasses.dataclass(frozen=True)
class FrontierScienceInstance:
    id: str
    question: str
    answer: str
    is_completed: bool = True


@dataclasses.dataclass(frozen=True)
class FrontierScienceJudgeResult:
    points: float | None
    correct: bool
    verdict: str | float | None
    parse_error: bool


@dataclasses.dataclass(frozen=True)
class FrontierScienceEvalResult:
    id: str
    split: str
    question: str
    response: str
    extracted_answer: str
    reference_answer: str
    judge_prompt: str | None
    judge_response: str | None
    judge_metadata: dict[str, Any]
    judge_result: FrontierScienceJudgeResult


@dataclasses.dataclass(frozen=True)
class FrontierScienceEvalSummary:
    split: str
    accuracy_percent: float
    total: int
    correct: int
    parse_errors: int


@dataclasses.dataclass(frozen=True)
class FrontierScienceResearchEvalSummary(FrontierScienceEvalSummary):
    mean_points: float | None


def _extract_verdict_line(text: str) -> str:
    for line in reversed(str(text).splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _clean_extracted_answer(text: str) -> str:
    answer = str(text).strip()
    while answer.startswith("`") and answer.endswith("`") and len(answer) >= 2:
        answer = answer[1:-1].strip()
    if answer.startswith("$") and answer.endswith("$") and len(answer) >= 2:
        answer = answer[1:-1].strip()
    return answer


def _extract_marker_answer(text: str, marker_pattern: str) -> Optional[str]:
    pattern = re.compile(rf"(?im)^\s*{marker_pattern}\s*:?\s*(.*?)\s*$")
    matches = list(pattern.finditer(text))
    for match in reversed(matches):
        same_line = match.group(1).strip()
        if same_line:
            return _clean_extracted_answer(same_line)

        following = text[match.end() :].splitlines()
        for line in following:
            stripped = line.strip()
            if stripped:
                return _clean_extracted_answer(stripped)
    return None


def _extract_balanced_braced_content(text: str, marker: str) -> Optional[str]:
    search_start = 0
    last_content = None
    while True:
        start = text.find(marker, search_start)
        if start == -1:
            return last_content

        cursor = start + len(marker)
        depth = 1
        chars: list[str] = []
        while cursor < len(text):
            char = text[cursor]
            if char == "{":
                depth += 1
                chars.append(char)
            elif char == "}":
                depth -= 1
                if depth == 0:
                    last_content = "".join(chars).strip()
                    break
                chars.append(char)
            else:
                chars.append(char)
            cursor += 1

        search_start = max(cursor + 1, start + len(marker))


def extract_olympiad_attempted_answer(response: str) -> str:
    text = str(response).strip()
    if not text:
        return ""

    final_answer = _extract_marker_answer(text, r"FINAL\s+ANSWER")
    if final_answer is not None:
        return final_answer

    exact_answer = _extract_marker_answer(text, r"Exact\s+Answer")
    if exact_answer is not None:
        return exact_answer

    boxed_answer = _extract_balanced_braced_content(text, r"\boxed{")
    if boxed_answer is not None:
        return _clean_extracted_answer(boxed_answer)

    return text


def parse_olympiad_judge_output(judge_output: str) -> FrontierScienceJudgeResult:
    verdict_line = _extract_verdict_line(judge_output)
    match = re.search(r"VERDICT:\s*(CORRECT|INCORRECT)\s*$", verdict_line, re.I)
    if not match:
        return FrontierScienceJudgeResult(
            points=None,
            correct=False,
            verdict=None,
            parse_error=True,
        )

    verdict = match.group(1).upper()
    return FrontierScienceJudgeResult(
        points=None,
        correct=verdict == "CORRECT",
        verdict=verdict,
        parse_error=False,
    )


def parse_research_judge_output(judge_output: str) -> FrontierScienceJudgeResult:
    verdict_line = _extract_verdict_line(judge_output)
    match = re.search(r"VERDICT:\s*(-?\d+(?:\.\d+)?)\s*$", verdict_line, re.I)
    if not match:
        return FrontierScienceJudgeResult(
            points=0.0,
            correct=False,
            verdict=None,
            parse_error=True,
        )

    raw_points = float(match.group(1))
    points = min(10.0, max(0.0, raw_points))
    return FrontierScienceJudgeResult(
        points=points,
        correct=points >= 7.0,
        verdict=points,
        parse_error=False,
    )


def call_openai_judge(
    client: openai.OpenAI,
    prompt: str,
    model: str,
    max_output_tokens: int,
    reasoning_effort: str | None = None,
) -> Any:
    body: dict[str, Any] = {
        "model": model,
        "max_output_tokens": max_output_tokens,
        "input": prompt,
    }

    if reasoning_effort is not None:
        body["reasoning"] = {"effort": reasoning_effort, "summary": "detailed"}

    return client.responses.create(**body)


class FrontierScienceEvaluator(
    Evaluator[
        FrontierScienceInstance,
        FrontierScienceEvalResult,
        FrontierScienceEvalSummary | FrontierScienceResearchEvalSummary,
    ]
):
    def __init__(
        self,
        split: str,
        model: str = "gpt-5",
        reasoning_effort: str = "high",
        max_output_tokens: int = 40000,
    ) -> None:
        if split not in {"olympiad", "research"}:
            raise ValueError("split must be 'olympiad' or 'research'")
        self.split = split
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.judge_metadata = {
            "split": self.split,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "max_output_tokens": self.max_output_tokens,
            "strip_think_blocks": True,
        }

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--split",
            choices=["olympiad", "research"],
            required=True,
            help="FrontierScience split to evaluate; required.",
        )
        parser.add_argument(
            "--model",
            default="gpt-5",
            help="OpenAI model used as the FrontierScience judge.",
        )
        parser.add_argument(
            "--reasoning-effort",
            choices=["low", "medium", "high"],
            default="high",
            help="Reasoning effort sent to the OpenAI judge model.",
        )
        parser.add_argument(
            "--max_output_tokens",
            type=int,
            default=40000,
            help="Maximum number of tokens the judge may generate.",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "FrontierScienceEvaluator":
        return cls(
            split=args.split,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
        )

    def prepare_instances(self) -> dict[str, FrontierScienceInstance]:
        import datasets

        rows = datasets.load_dataset(
            DATASET_PATH,
            split=self.split,
        )
        instances: dict[str, FrontierScienceInstance] = {}
        for row in rows:
            instance = FrontierScienceInstance(
                id=str(row["id"]),
                question=str(row["question"]),
                answer=str(row["answer"]),
            )
            instances[instance.id] = instance
        return instances

    def evaluate(
        self,
        instance: FrontierScienceInstance,
        response: str,
    ) -> FrontierScienceEvalResult:
        response = strip_think_blocks(response)
        extracted_answer = (
            extract_olympiad_attempted_answer(response)
            if self.split == "olympiad"
            else response
        )

        if not instance.is_completed or not response:
            return FrontierScienceEvalResult(
                id=instance.id,
                split=self.split,
                question=instance.question,
                response=response,
                extracted_answer=extracted_answer,
                reference_answer=instance.answer,
                judge_prompt=None,
                judge_response=None,
                judge_metadata=dict(self.judge_metadata),
                judge_result=FrontierScienceJudgeResult(
                    points=0.0 if self.split == "research" else None,
                    correct=False,
                    verdict=None,
                    parse_error=True,
                ),
            )

        if self.split == "olympiad":
            judge_prompt = OLYMPIAD_JUDGE_PROMPT.format(
                problem=instance.question,
                reference_answer=instance.answer,
                answer=extracted_answer,
            )
            parser = parse_olympiad_judge_output
        else:
            judge_prompt = RESEARCH_JUDGE_PROMPT.format(
                problem=instance.question,
                rubric=instance.answer,
                answer=response,
            )
            parser = parse_research_judge_output

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in environment")
        client = openai.OpenAI(api_key=api_key)
        judge_response_obj = call_openai_judge(
            client=client,
            prompt=judge_prompt,
            model=self.model,
            max_output_tokens=self.max_output_tokens,
            reasoning_effort=self.reasoning_effort,
        )
        judge_response = judge_response_obj.output_text or ""
        judge_result = parser(judge_response)
        return FrontierScienceEvalResult(
            id=instance.id,
            split=self.split,
            question=instance.question,
            response=response,
            extracted_answer=extracted_answer,
            reference_answer=instance.answer,
            judge_prompt=judge_prompt,
            judge_response=judge_response,
            judge_metadata=dict(self.judge_metadata),
            judge_result=judge_result,
        )

    def aggregate(
        self,
        results: Sequence[FrontierScienceEvalResult],
    ) -> FrontierScienceEvalSummary | FrontierScienceResearchEvalSummary:
        total = len(results)
        correct_count = sum(1 for result in results if result.judge_result.correct)
        parse_errors = sum(1 for result in results if result.judge_result.parse_error)
        accuracy_percent = (
            round((correct_count / total) * 100.0, 2) if total else 0.0
        )
        if self.split == "olympiad":
            return FrontierScienceEvalSummary(
                split=self.split,
                accuracy_percent=accuracy_percent,
                total=total,
                correct=correct_count,
                parse_errors=parse_errors,
            )

        valid_points = [
            result.judge_result.points
            for result in results
            if isinstance(result.judge_result.points, (int, float))
        ]

        mean_points = (
            round(sum(float(points) for points in valid_points) / len(valid_points), 6)
            if valid_points
            else None
        )
        return FrontierScienceResearchEvalSummary(
            split=self.split,
            accuracy_percent=accuracy_percent,
            total=total,
            correct=correct_count,
            parse_errors=parse_errors,
            mean_points=mean_points,
        )
