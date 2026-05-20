import argparse
import dataclasses
import os
import re
from collections.abc import Sequence
from typing import Any

import openai
from dotenv import load_dotenv

from .base import Evaluator, strip_think_blocks

load_dotenv()


DATASET_PATH = "rl-rag-2/browsecomp"


GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.
""".strip()


@dataclasses.dataclass(frozen=True)
class BrowseCompInstance:
    id: str
    question: str
    answer: str
    is_completed: bool = True


@dataclasses.dataclass(frozen=True)
class BrowseCompJudgeResult:
    extracted_final_answer: str | None
    reasoning: str | None
    correct: bool | None
    parse_error: bool
    error: str | None = None


@dataclasses.dataclass(frozen=True)
class BrowseCompEvalResult:
    id: str
    question: str
    response: str
    correct_answer: str
    is_completed: bool
    judge_prompt: str | None
    judge_response: str | None
    judge_metadata: dict[str, Any]
    judge_result: BrowseCompJudgeResult


@dataclasses.dataclass(frozen=True)
class BrowseCompEvalSummary:
    accuracy_percent: float
    num_results: int
    num_correct: int


def parse_browsecomp_judge_response(judge_response: str) -> BrowseCompJudgeResult:
    extracted_final_answer = None
    reasoning = None
    correct = None
    parse_error = False

    if not judge_response:
        return BrowseCompJudgeResult(
            extracted_final_answer=extracted_final_answer,
            reasoning=reasoning,
            correct=correct,
            parse_error=True,
        )

    answer_match = re.search(
        r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)",
        judge_response,
        re.IGNORECASE | re.DOTALL,
    )
    if not answer_match:
        answer_match = re.search(
            r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)",
            judge_response,
            re.IGNORECASE | re.DOTALL,
        )
    if not answer_match:
        answer_match = re.search(
            r"extracted_final_answer:\s*(.*?)(?=\n|$)",
            judge_response,
            re.IGNORECASE | re.DOTALL,
        )
    if answer_match:
        extracted_final_answer = answer_match.group(1).strip()

    reasoning_match = re.search(
        r"\*\*reasoning:\*\*\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
        judge_response,
        re.IGNORECASE | re.DOTALL,
    )
    if not reasoning_match:
        reasoning_match = re.search(
            r"\*\*reasoning\*\*:\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
            judge_response,
            re.IGNORECASE | re.DOTALL,
        )
    if not reasoning_match:
        reasoning_match = re.search(
            r"reasoning:\s*(.*?)(?=\ncorrect:|$)",
            judge_response,
            re.IGNORECASE | re.DOTALL,
        )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    correct_match = re.search(
        r"\*\*correct:\*\*\s*(yes|no)",
        judge_response,
        re.IGNORECASE,
    )
    if not correct_match:
        correct_match = re.search(
            r"\*\*correct\*\*:\s*(yes|no)",
            judge_response,
            re.IGNORECASE,
        )
    if not correct_match:
        correct_match = re.search(
            r"correct:\s*(yes|no)",
            judge_response,
            re.IGNORECASE,
        )
    if correct_match:
        correct = correct_match.group(1).lower() == "yes"

    if correct is None:
        parse_error = True

    return BrowseCompJudgeResult(
        extracted_final_answer=extracted_final_answer,
        reasoning=reasoning,
        correct=correct,
        parse_error=parse_error,
    )


def call_openai_judge(
    client: openai.OpenAI,
    judge_prompt: str,
    model: str,
    max_output_tokens: int,
) -> Any:
    return client.responses.create(
        model=model,
        max_output_tokens=max_output_tokens,
        input=judge_prompt,
    )


class BrowseCompEvaluator(
    Evaluator[
        BrowseCompInstance,
        BrowseCompEvalResult,
        BrowseCompEvalSummary,
    ]
):
    def __init__(
        self,
        model: str = "gpt-4.1",
        max_output_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.judge_metadata = {
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "strip_think_blocks": True,
        }

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model",
            default="gpt-4.1",
            help="OpenAI model used as the BrowseComp answer-equivalence judge.",
        )
        parser.add_argument(
            "--max_output_tokens",
            type=int,
            default=1024,
            help="Maximum number of tokens the judge may generate.",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "BrowseCompEvaluator":
        return cls(
            model=args.model,
            max_output_tokens=args.max_output_tokens,
        )

    def prepare_instances(self) -> dict[str, BrowseCompInstance]:
        import datasets

        rows = datasets.load_dataset(
            DATASET_PATH,
            split="train",
        )
        instances: dict[str, BrowseCompInstance] = {}
        for row in rows:
            instance = BrowseCompInstance(
                id=str(row["id"]),
                question=str(row["question"]),
                answer=str(row["answer"]),
            )
            instances[instance.id] = instance
        return instances

    def evaluate(
        self,
        instance: BrowseCompInstance,
        response: str,
    ) -> BrowseCompEvalResult:
        response = strip_think_blocks(response)
        if not response or not instance.is_completed:
            return BrowseCompEvalResult(
                id=instance.id,
                question=instance.question,
                response=response,
                correct_answer=instance.answer,
                is_completed=instance.is_completed,
                judge_prompt=None,
                judge_response=None,
                judge_metadata=dict(self.judge_metadata),
                judge_result=BrowseCompJudgeResult(
                    extracted_final_answer=None,
                    reasoning=None,
                    correct=None,
                    parse_error=True,
                    error="Response incomplete or cannot be parsed",
                ),
            )

        judge_prompt = GRADER_TEMPLATE.format(
            question=instance.question,
            response=response,
            correct_answer=instance.answer,
        )
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in environment")
        client = openai.OpenAI(api_key=api_key)
        judge_response = call_openai_judge(
            client=client,
            judge_prompt=judge_prompt,
            model=self.model,
            max_output_tokens=self.max_output_tokens,
        )
        judge_text = judge_response.output_text or ""

        return BrowseCompEvalResult(
            id=instance.id,
            question=instance.question,
            response=response,
            correct_answer=instance.answer,
            is_completed=instance.is_completed,
            judge_prompt=judge_prompt,
            judge_response=judge_text,
            judge_metadata=dict(self.judge_metadata),
            judge_result=parse_browsecomp_judge_response(judge_text),
        )

    def aggregate(
        self,
        results: Sequence[BrowseCompEvalResult],
    ) -> BrowseCompEvalSummary:
        total = len(results)
        correct_count = sum(1 for result in results if result.judge_result.correct)
        accuracy_percent = round((correct_count / total) * 100.0, 2) if total else 0.0

        return BrowseCompEvalSummary(
            accuracy_percent=accuracy_percent,
            num_results=total,
            num_correct=correct_count,
        )
