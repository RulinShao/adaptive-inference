import argparse
import dataclasses
import json
import math
import os
import random
import re
import time
import urllib.request
from collections import Counter
from collections.abc import Iterable, Sequence
from statistics import mean
from typing import Any
from urllib.parse import urlparse

import numpy as np
import openai
from dotenv import load_dotenv

from .base import Evaluator, strip_think_blocks

load_dotenv()


DATASET_PATH = "rl-rag-2/hle_text_only_curated_600_samples"
DEFAULT_SPLIT = "test"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_REASONING_EFFORT = "low"
REASONING_EFFORT_SUPPORTED_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:
- extracted_final_answer: The final exact answer extracted from the [response]. Put "None" if there is no exact, final answer to extract from the response.
- reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on meaningful differences between [correct_answer] and the extracted_final_answer. Do not solve the problem or argue for another answer.
- correct: "yes" if extracted_final_answer matches [correct_answer], or is within a small margin of error for numerical problems. "no" otherwise, including inconsistency, ambiguity, non-equivalency, or missing answers.
- confidence: The extracted confidence score between 0 and 100 from [response]. Put 100 if no confidence score is available.

Return STRICT JSON only:
{{
  "extracted_final_answer": "string",
  "reasoning": "string",
  "correct": "yes" | "no",
  "confidence": integer
}}
"""


@dataclasses.dataclass(frozen=True)
class HLEInstance:
    id: str
    question: str
    answer: str
    row_index: int | None = None
    image: str = ""
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    is_completed: bool = True


@dataclasses.dataclass(frozen=True)
class HLEJudgeResult:
    status: str
    correct: str | None
    is_correct: bool | None
    confidence: int | None
    extracted_final_answer: str | None
    reasoning: str | None
    parse_error: bool
    error: str | None = None
    raw_response: str | None = None
    usage: dict[str, Any] | None = None


@dataclasses.dataclass(frozen=True)
class HLEEvalResult:
    id: str
    row_index: int | None
    question: str
    response: str
    graded_response: str
    correct_answer: str
    image: str
    metadata: dict[str, Any]
    judge_prompt: str | None
    judge_response: str | None
    judge_metadata: dict[str, Any]
    judge_result: HLEJudgeResult


@dataclasses.dataclass(frozen=True)
class HLEEvalSummary:
    num_records: int
    judged_records: int
    status_counts: dict[str, int]
    correct_count: int
    accuracy: float | None
    judged_accuracy: float | None
    accuracy_percent: float | None
    accuracy_95ci_half_width_percent: float | None
    calibration_error: float | None
    calibration_error_percent: float | None
    mean_confidence: float | None


def is_url(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https"}


def iter_jsonl_lines(source: str) -> Iterable[str]:
    if is_url(source):
        with urllib.request.urlopen(source, timeout=120) as response:
            for raw_line in response:
                yield raw_line.decode("utf-8")
        return

    with open(os.path.expanduser(source), encoding="utf-8") as f:
        yield from f


def load_local_jsonl(source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in iter_jsonl_lines(source):
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def should_load_jsonl(source: str) -> bool:
    expanded = os.path.expanduser(source)
    return is_url(source) or os.path.exists(expanded) or source.endswith((".jsonl", ".json"))


def has_image(row: dict[str, Any]) -> bool:
    image = row.get("image")
    return isinstance(image, str) and bool(image.strip())


def normalize_reasoning_effort(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "none", "null", "off"}:
        return None
    return normalized


def supports_reasoning_effort(model: str) -> bool:
    normalized = model.strip().lower()
    return any(normalized.startswith(prefix) for prefix in REASONING_EFFORT_SUPPORTED_MODEL_PREFIXES)


def resolve_reasoning_effort(model: str, reasoning_effort: str | None) -> str | None:
    if reasoning_effort is None:
        return None
    return reasoning_effort if supports_reasoning_effort(model) else None


def parse_json_object(text: str) -> dict[str, Any]:
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())
    brace = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if brace:
        candidates.append(brace.group(1).strip())
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError(f"could not parse JSON object from: {text[:500]}")


def clamp_confidence(value: Any) -> int:
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        value = float(match.group(0)) if match else 100
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        value = 100
    return max(0, min(100, int(round(float(value)))))


def usage_to_dict(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    if hasattr(usage, "dict"):
        return usage.dict()
    if isinstance(usage, dict):
        return usage
    return {"value": str(usage)}


def response_output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    raw = response.model_dump(mode="json") if hasattr(response, "model_dump") else {}
    parts: list[str] = []
    for item in raw.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if content.get("type") == "output_text" and content.get("text"):
                parts.append(content["text"])
    return "\n".join(parts).strip()


def call_openai_judge(
    client: openai.OpenAI,
    prompt: str,
    *,
    model: str,
    reasoning_effort: str | None,
    max_output_tokens: int | None,
) -> Any:
    body: dict[str, Any] = {
        "model": model,
        "input": prompt,
    }
    if reasoning_effort is not None:
        body["reasoning"] = {"effort": reasoning_effort}
    if max_output_tokens is not None:
        body["max_output_tokens"] = max_output_tokens
    return client.responses.create(**body)


def call_openai_judge_with_retries(
    client: openai.OpenAI,
    prompt: str,
    *,
    model: str,
    reasoning_effort: str | None,
    max_output_tokens: int | None,
    max_retries: int,
    retry_base_seconds: float,
) -> Any:
    last_error: Exception | None = None
    retryable_errors = (
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.RateLimitError,
    )
    for attempt in range(1, max_retries + 1):
        try:
            return call_openai_judge(
                client,
                prompt,
                model=model,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max_output_tokens,
            )
        except retryable_errors as exc:
            last_error = exc
            if attempt == max_retries:
                break
            sleep_seconds = retry_base_seconds * (2 ** (attempt - 1)) + random.uniform(0, 0.2)
            time.sleep(sleep_seconds)
    raise RuntimeError(f"OpenAI judge failed after {max_retries} attempts: {last_error}")


def calibration_error(confidence: np.ndarray, correct: np.ndarray, beta: int = 100) -> float | None:
    if len(confidence) == 0:
        return None
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [
        [i * beta, min((i + 1) * beta, len(confidence))]
        for i in range(math.ceil(len(confidence) / beta))
    ]
    cerr = 0.0
    total = len(confidence)
    for start, end in bins:
        bin_confidence = confidence[start:end]
        bin_correct = correct[start:end]
        if len(bin_confidence) == 0:
            continue
        diff = abs(float(np.nanmean(bin_confidence)) - float(np.nanmean(bin_correct)))
        cerr += len(bin_confidence) / total * diff * diff
    return math.sqrt(cerr)


class HLEEvaluator(Evaluator[HLEInstance, HLEEvalResult, HLEEvalSummary]):
    def __init__(
        self,
        dataset: str = DATASET_PATH,
        split: str = DEFAULT_SPLIT,
        text_only: bool = False,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
        max_output_tokens: int | None = None,
        llm_max_retries: int = 3,
        retry_base_seconds: float = 1.0,
    ) -> None:
        self.dataset = dataset
        self.split = split
        self.text_only = text_only
        self.model = model
        self.requested_reasoning_effort = normalize_reasoning_effort(reasoning_effort)
        self.reasoning_effort = resolve_reasoning_effort(model, self.requested_reasoning_effort)
        self.max_output_tokens = max_output_tokens
        self.llm_max_retries = llm_max_retries
        self.retry_base_seconds = retry_base_seconds
        self.judge_metadata = {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "requested_reasoning_effort": self.requested_reasoning_effort,
            "max_output_tokens": self.max_output_tokens,
            "llm_max_retries": self.llm_max_retries,
            "retry_base_seconds": self.retry_base_seconds,
            "dataset": self.dataset,
            "split": self.split,
            "text_only": self.text_only,
            "strip_think_blocks": True,
        }

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--dataset",
            default=DATASET_PATH,
            help="HLE Hugging Face dataset name or local/URL JSONL with id/question/answer fields.",
        )
        parser.add_argument("--split", default=DEFAULT_SPLIT)
        parser.add_argument(
            "--text-only",
            action="store_true",
            help="Skip instances whose image field is non-empty.",
        )
        parser.add_argument(
            "--model",
            default=DEFAULT_MODEL,
            help="OpenAI model used as the HLE answer-equivalence judge.",
        )
        parser.add_argument(
            "--reasoning-effort",
            choices=["none", "low", "medium", "high"],
            default=DEFAULT_REASONING_EFFORT,
            help="Reasoning effort sent to reasoning-capable judge models.",
        )
        parser.add_argument(
            "--max_output_tokens",
            type=int,
            default=None,
            help="Maximum number of tokens the judge may generate; omit to match gpt-baselines.",
        )
        parser.add_argument(
            "--llm-max-retries",
            type=int,
            default=3,
            help="Maximum number of retries for transient judge-model failures.",
        )
        parser.add_argument(
            "--retry-base-seconds",
            type=float,
            default=1.0,
            help="Base sleep duration for exponential retry backoff.",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "HLEEvaluator":
        return cls(
            dataset=args.dataset,
            split=args.split,
            text_only=args.text_only,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
            llm_max_retries=args.llm_max_retries,
            retry_base_seconds=args.retry_base_seconds,
        )

    def prepare_instances(self) -> dict[str, HLEInstance]:
        if should_load_jsonl(self.dataset):
            rows = load_local_jsonl(self.dataset)
        else:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise RuntimeError("Install `datasets` to load HLE references from Hugging Face.") from exc
            rows = [dict(row) for row in load_dataset(self.dataset, split=self.split)]

        instances: dict[str, HLEInstance] = {}
        for row_index, row in enumerate(rows):
            if self.text_only and has_image(row):
                continue
            example_id = row.get("id")
            if not isinstance(example_id, str) or not example_id.strip():
                example_id = str(row.get("row_index") if isinstance(row.get("row_index"), int) else row_index)
            question = row.get("question")
            answer = row.get("answer")
            if not isinstance(question, str) or not question.strip():
                continue
            if not isinstance(answer, str) or not answer.strip():
                continue
            image = row.get("image")
            metadata = {
                key: value
                for key, value in row.items()
                if key not in {"id", "row_index", "question", "answer", "image"}
            }
            instance = HLEInstance(
                id=example_id,
                row_index=row.get("row_index") if isinstance(row.get("row_index"), int) else row_index,
                question=question,
                answer=answer,
                image=image if isinstance(image, str) else "",
                metadata=metadata,
            )
            instances[example_id] = instance
            row_index_key = str(instance.row_index)
            if row_index_key not in instances:
                instances[row_index_key] = instance
        return instances

    def evaluate(self, instance: HLEInstance, response: str) -> HLEEvalResult:
        graded_response = strip_think_blocks(response)
        judge_prompt = JUDGE_PROMPT.format(
            question=instance.question,
            response=graded_response,
            correct_answer=instance.answer,
        )

        if not instance.is_completed or not graded_response:
            error = (
                "Response incomplete or cannot be parsed"
                if not instance.is_completed
                else "Response empty"
            )
            judge_result = HLEJudgeResult(
                status="completed",
                correct="no",
                is_correct=False,
                confidence=100,
                extracted_final_answer="None",
                reasoning=error,
                parse_error=False,
                error=error,
                raw_response=None,
                usage=None,
            )
            return HLEEvalResult(
                id=instance.id,
                row_index=instance.row_index,
                question=instance.question,
                response=response,
                graded_response=graded_response,
                correct_answer=instance.answer,
                image=instance.image,
                metadata=instance.metadata,
                judge_prompt=None,
                judge_response=None,
                judge_metadata=dict(self.judge_metadata),
                judge_result=judge_result,
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in environment")
        client = openai.OpenAI(api_key=api_key)

        try:
            response_obj = call_openai_judge_with_retries(
                client,
                judge_prompt,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                max_output_tokens=self.max_output_tokens,
                max_retries=self.llm_max_retries,
                retry_base_seconds=self.retry_base_seconds,
            )
            judge_response = response_output_text(response_obj)
            if not judge_response:
                raise ValueError("empty judge response")
            parsed = parse_json_object(judge_response)
            correct = str(parsed.get("correct", "")).strip().lower()
            if correct not in {"yes", "no"}:
                raise ValueError(f"invalid correct field: {parsed.get('correct')!r}")
            judge_result = HLEJudgeResult(
                status="completed",
                correct=correct,
                is_correct=correct == "yes",
                confidence=clamp_confidence(parsed.get("confidence")),
                extracted_final_answer=str(parsed.get("extracted_final_answer") or ""),
                reasoning=str(parsed.get("reasoning") or ""),
                parse_error=False,
                raw_response=judge_response,
                usage=usage_to_dict(response_obj),
            )
            judge_response_value = judge_response
        except Exception as exc:
            judge_result = HLEJudgeResult(
                status="parse_error",
                correct=None,
                is_correct=None,
                confidence=None,
                extracted_final_answer=None,
                reasoning=None,
                parse_error=True,
                error=str(exc),
                raw_response=None,
                usage=None,
            )
            judge_response_value = None

        return HLEEvalResult(
            id=instance.id,
            row_index=instance.row_index,
            question=instance.question,
            response=response,
            graded_response=graded_response,
            correct_answer=instance.answer,
            image=instance.image,
            metadata=instance.metadata,
            judge_prompt=judge_prompt,
            judge_response=judge_response_value,
            judge_metadata=dict(self.judge_metadata),
            judge_result=judge_result,
        )

    def aggregate(self, results: Sequence[HLEEvalResult]) -> HLEEvalSummary:
        status_counts = Counter(result.judge_result.status for result in results)
        correct_values: list[float] = []
        confidences: list[float] = []
        for result in results:
            judge_result = result.judge_result
            if judge_result.status != "completed":
                continue
            correct = 1.0 if judge_result.correct == "yes" else 0.0
            correct_values.append(correct)
            if isinstance(judge_result.confidence, (int, float)) and not isinstance(judge_result.confidence, bool):
                confidences.append(max(0.0, min(1.0, float(judge_result.confidence) / 100.0)))
            else:
                confidences.append(1.0)

        n = len(results)
        judged_n = len(correct_values)
        accuracy = (sum(correct_values) / n) if n and judged_n else None
        judged_accuracy = (sum(correct_values) / judged_n) if judged_n else None
        ci_half_width = None
        if n and accuracy is not None:
            ci_half_width = 1.96 * math.sqrt(accuracy * (1 - accuracy) / n)
        cal_error = None
        if judged_n:
            cal_error = calibration_error(np.array(confidences), np.array(correct_values))

        return HLEEvalSummary(
            num_records=n,
            judged_records=judged_n,
            status_counts=dict(status_counts),
            correct_count=int(sum(correct_values)),
            accuracy=accuracy,
            judged_accuracy=judged_accuracy,
            accuracy_percent=None if accuracy is None else round(100 * accuracy, 4),
            accuracy_95ci_half_width_percent=None
            if ci_half_width is None
            else round(100 * ci_half_width, 4),
            calibration_error=cal_error,
            calibration_error_percent=None if cal_error is None else round(100 * cal_error, 4),
            mean_confidence=mean(confidences) if confidences else None,
        )
