import argparse
import dataclasses
import hashlib
import json
import os
import random
import re
import time
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from statistics import mean, median
from typing import Any
from urllib.parse import urlparse

import numpy as np
import openai
from dotenv import load_dotenv

from .base import Evaluator, strip_think_blocks

load_dotenv()


DATASET_PATH = "openai/healthbench-professional"
HEALTHBENCH_PROFESSIONAL_RAW_URL = (
    "https://huggingface.co/datasets/openai/healthbench-professional/resolve/main/"
    "healthbench_professional_eval.jsonl"
)
HEALTHBENCH_PROFESSIONAL_ALIASES = {
    DATASET_PATH,
    "https://huggingface.co/datasets/openai/healthbench-professional",
    "https://huggingface.co/datasets/openai/healthbench-professional/",
    HEALTHBENCH_PROFESSIONAL_RAW_URL,
}
OFFICIAL_PROFESSIONAL_GRADER_MODEL = "gpt-5.4-2026-03-05"
OFFICIAL_PROFESSIONAL_REASONING_EFFORT = "low"
OFFICIAL_PROFESSIONAL_LENGTH_ADJUSTMENT_CENTER = 2000.0
OFFICIAL_PROFESSIONAL_LENGTH_ADJUSTMENT_PENALTY_PER_500_CHARS = 0.0147


GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


@dataclasses.dataclass(frozen=True)
class HealthBenchProfessionalRubricItem:
    criterion: str
    points: float
    tags: list[str] = dataclasses.field(default_factory=list)

    def __str__(self) -> str:
        return f"[{self.points}] {self.criterion}"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HealthBenchProfessionalRubricItem":
        criterion = payload.get("criterion")
        if not isinstance(criterion, str) or not criterion.strip():
            criterion = payload.get("criterion_text")
        if not isinstance(criterion, str) or not criterion.strip():
            raise ValueError(f"rubric item missing criterion text: {payload!r}")

        points_raw = payload.get("points")
        if isinstance(points_raw, bool):
            raise ValueError(f"rubric points cannot be bool: {payload!r}")
        if isinstance(points_raw, (int, float)):
            points = float(points_raw)
        elif isinstance(points_raw, str) and points_raw.strip():
            points = float(points_raw.strip())
        else:
            raise ValueError(f"rubric item missing numeric points: {payload!r}")

        raw_tags = payload.get("tags") or []
        tags = [str(tag) for tag in raw_tags if tag is not None] if isinstance(raw_tags, list) else []
        return cls(criterion=criterion.strip(), points=points, tags=tags)


@dataclasses.dataclass(frozen=True)
class HealthBenchProfessionalInstance:
    id: str
    prompt_id: str
    question: str
    conversation_messages: list[dict[str, str]]
    gold: str
    rubric_items: list[HealthBenchProfessionalRubricItem]
    example_tags: list[str] = dataclasses.field(default_factory=list)
    row_index: int | None = None
    use_case: str | None = None
    type: str | None = None
    difficulty: str | None = None
    specialty: str | None = None
    is_completed: bool = True


@dataclasses.dataclass(frozen=True)
class HealthBenchProfessionalRubricGrade:
    index: int
    criterion: str
    points: float
    tags: list[str]
    criteria_met: bool | None
    explanation: str
    raw_response_text: str | None
    status: str
    error: str | None = None
    usage: dict[str, int | None] | None = None


@dataclasses.dataclass(frozen=True)
class HealthBenchProfessionalJudgeResult:
    status: str
    overall_score: float | None
    overall_score_clipped: float | None
    overall_score_length_adjusted: float | None
    overall_score_length_adjusted_clipped: float | None
    achieved_points: float | None
    total_positive_points: float | None
    prediction_chars: int
    rubric_grades: list[HealthBenchProfessionalRubricGrade]
    parse_error: bool
    error: str | None = None
    usage: dict[str, int | None] | None = None
    metrics: dict[str, float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class HealthBenchProfessionalEvalResult:
    id: str
    prompt_id: str
    completion_id: str
    row_index: int | None
    question: str
    response: str
    graded_response: str
    gold: str
    conversation_messages: list[dict[str, str]]
    rubric_items: list[HealthBenchProfessionalRubricItem]
    example_tags: list[str]
    use_case: str | None
    type: str | None
    difficulty: str | None
    specialty: str | None
    metrics: dict[str, float]
    judge_metadata: dict[str, Any]
    judge_result: HealthBenchProfessionalJudgeResult


@dataclasses.dataclass(frozen=True)
class HealthBenchProfessionalEvalSummary:
    score: float | None
    metrics: dict[str, float | int]
    num_results: int
    status_counts: dict[str, int]
    scored_examples: int
    mean_overall_score: float | None
    median_overall_score: float | None
    min_overall_score: float | None
    max_overall_score: float | None
    mean_overall_score_clipped: float | None
    mean_overall_score_length_adjusted: float | None
    mean_overall_score_length_adjusted_clipped: float | None
    mean_achieved_points: float | None
    mean_total_positive_points: float | None
    mean_prediction_chars: float | None
    by_use_case: dict[str, dict[str, Any]]
    by_type: dict[str, dict[str, Any]]
    by_difficulty: dict[str, dict[str, Any]]
    by_specialty: dict[str, dict[str, Any]]


def is_url(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https"}


def resolve_reference_source(source: str) -> str:
    normalized = source.strip()
    if normalized in HEALTHBENCH_PROFESSIONAL_ALIASES:
        return HEALTHBENCH_PROFESSIONAL_RAW_URL
    return normalized


def iter_jsonl_lines(source: str) -> Iterable[str]:
    if is_url(source):
        with urllib.request.urlopen(source, timeout=120) as response:
            for raw_line in response:
                yield raw_line.decode("utf-8")
        return

    with open(os.path.expanduser(source), encoding="utf-8") as f:
        for line in f:
            yield line


def render_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def normalize_messages(value: Any) -> list[dict[str, str]] | None:
    if isinstance(value, dict):
        value = value.get("messages")
    if not isinstance(value, list) or not value:
        return None

    normalized: list[dict[str, str]] = []
    for message in value:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str):
            return None
        normalized.append({"role": role, "content": render_text(content)})
    return normalized


def last_user_message_text(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def format_conversation(messages: list[dict[str, str]], response_text: str) -> str:
    convo_with_response = messages + [{"role": "assistant", "content": response_text}]
    return "\n\n".join(
        f"{message['role']}: {message['content']}" for message in convo_with_response
    )


def create_grader_prompt(
    conversation_text: str,
    rubric_item: HealthBenchProfessionalRubricItem,
) -> str:
    return GRADER_TEMPLATE.replace("<<conversation>>", conversation_text).replace(
        "<<rubric_item>>",
        str(rubric_item),
    )


def parse_judge_json(text: str) -> dict[str, Any]:
    json_cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip())
    try:
        payload = json.loads(json_cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON decoding failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"judge JSON was not an object: {payload!r}")
    label = payload.get("criteria_met")
    if label is not True and label is not False:
        raise ValueError(f"judge JSON missing boolean criteria_met: {payload!r}")
    return payload


def get_usage_dict(response_usage: Any) -> dict[str, int | None]:
    if response_usage is None:
        return {
            "input_tokens": None,
            "input_cached_tokens": None,
            "output_tokens": None,
            "output_reasoning_tokens": None,
            "total_tokens": None,
        }

    try:
        return {
            "input_tokens": response_usage.input_tokens,
            "input_cached_tokens": response_usage.input_tokens_details.cached_tokens
            if hasattr(response_usage.input_tokens_details, "cached_tokens")
            else response_usage.input_tokens_details["cached_tokens"],
            "output_tokens": response_usage.output_tokens,
            "output_reasoning_tokens": response_usage.output_tokens_details.reasoning_tokens
            if hasattr(response_usage.output_tokens_details, "reasoning_tokens")
            else response_usage.output_tokens_details["reasoning_tokens"],
            "total_tokens": response_usage.total_tokens,
        }
    except AttributeError:
        return {
            "input_tokens": response_usage.prompt_tokens,
            "input_cached_tokens": response_usage.prompt_tokens_details.cached_tokens
            if hasattr(response_usage.prompt_tokens_details, "cached_tokens")
            else response_usage.prompt_tokens_details["cached_tokens"],
            "output_tokens": response_usage.completion_tokens,
            "output_reasoning_tokens": response_usage.completion_tokens_details.reasoning_tokens
            if hasattr(response_usage.completion_tokens_details, "reasoning_tokens")
            else response_usage.completion_tokens_details["reasoning_tokens"],
            "total_tokens": response_usage.total_tokens,
        }


def merge_usage_dicts(
    left: dict[str, int | None] | None,
    right: dict[str, int | None] | None,
) -> dict[str, int | None] | None:
    if left is None:
        return right
    if right is None:
        return left

    merged: dict[str, int | None] = {}
    for key in (
        "input_tokens",
        "input_cached_tokens",
        "output_tokens",
        "output_reasoning_tokens",
        "total_tokens",
    ):
        left_value = left.get(key)
        right_value = right.get(key)
        if isinstance(left_value, int) and isinstance(right_value, int):
            merged[key] = left_value + right_value
        elif isinstance(left_value, int):
            merged[key] = left_value
        elif isinstance(right_value, int):
            merged[key] = right_value
        else:
            merged[key] = None
    return merged


def calculate_score(
    rubric_items: list[HealthBenchProfessionalRubricItem],
    rubric_grades: list[HealthBenchProfessionalRubricGrade],
) -> float | None:
    total_positive_points = sum(item.points for item in rubric_items if item.points > 0)
    if total_positive_points == 0:
        return None

    achieved_points = sum(
        rubric_item.points
        for rubric_item, rubric_grade in zip(
            rubric_items,
            rubric_grades,
            strict=True,
        )
        if rubric_grade.criteria_met
    )
    return achieved_points / total_positive_points


def calculate_achieved_points(
    rubric_items: list[HealthBenchProfessionalRubricItem],
    rubric_grades: list[HealthBenchProfessionalRubricGrade],
) -> float:
    return sum(
        rubric_item.points
        for rubric_item, grade in zip(rubric_items, rubric_grades, strict=True)
        if grade.criteria_met
    )


def calculate_length_adjusted_score(
    score: float,
    response_text: str,
    *,
    center: float,
    penalty_per_500_chars: float,
) -> float:
    return score - penalty_per_500_chars * ((len(response_text) - center) / 500.0)


def clip_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_clipped_stats(values: list[float], stat: str) -> float | int:
    if stat == "mean":
        return float(np.clip(np.mean(values), 0, 1))
    if stat == "n_samples":
        return len(values)
    if stat == "bootstrap_std":
        bootstrap_samples = [np.random.choice(values, len(values)) for _ in range(1000)]
        bootstrap_means = [
            compute_clipped_stats(list(sample), "mean") for sample in bootstrap_samples
        ]
        return float(np.std(bootstrap_means))
    raise ValueError(f"Unknown stat = {stat!r}")


def normalize_reasoning_effort(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "none", "null", "off"}:
        return None
    return normalized


def call_openai_judge(
    client: openai.OpenAI,
    prompt: str,
    model: str,
    max_output_tokens: int | None,
    reasoning_effort: str | None,
) -> Any:
    body: dict[str, Any] = {
        "model": model,
        "input": [{"role": "user", "content": prompt}],
    }
    if max_output_tokens is not None and reasoning_effort is None:
        body["max_output_tokens"] = max_output_tokens
    if reasoning_effort is not None:
        body["reasoning"] = {"effort": reasoning_effort}
    return client.responses.create(**body)


def grade_rubric_item_with_retries(
    client: openai.OpenAI,
    prompt: str,
    model: str,
    max_output_tokens: int | None,
    reasoning_effort: str | None,
    max_retries: int,
    retry_base_seconds: float,
) -> tuple[dict[str, Any], str, dict[str, int | None] | None]:
    last_error: Exception | None = None
    retryable_errors = (
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.RateLimitError,
        ValueError,
    )
    for attempt in range(1, max_retries + 1):
        try:
            response_obj = call_openai_judge(
                client=client,
                prompt=prompt,
                model=model,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
            )
            judge_text = response_output_text(response_obj)
            parsed = parse_judge_json(judge_text)
            usage = get_usage_dict(getattr(response_obj, "usage", None))
            return parsed, judge_text, usage
        except retryable_errors as exc:
            last_error = exc
            if attempt == max_retries:
                break
            sleep_seconds = retry_base_seconds * (2 ** (attempt - 1 + random.random()))
            time.sleep(sleep_seconds)
    raise RuntimeError(f"OpenAI judge failed after {max_retries} attempts: {last_error}")


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


def group_score_summary(
    results: Sequence[HealthBenchProfessionalEvalResult],
    field: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[HealthBenchProfessionalEvalResult]] = defaultdict(list)
    for result in results:
        value = getattr(result, field)
        if value is not None:
            grouped[str(value)].append(result)

    summary: dict[str, dict[str, Any]] = {}
    for key, group in sorted(grouped.items()):
        completed = [
            result for result in group if result.judge_result.status == "completed"
        ]
        scores = [
            float(result.judge_result.overall_score)
            for result in group
            if isinstance(result.judge_result.overall_score, (int, float))
        ]
        clipped_scores = [
            float(result.judge_result.overall_score_clipped)
            for result in group
            if isinstance(result.judge_result.overall_score_clipped, (int, float))
        ]
        summary[key] = {
            "count": len(group),
            "completed": len(completed),
            "mean_overall_score": compute_clipped_stats(scores, "mean")
            if scores
            else None,
            "mean_overall_score_clipped": mean(clipped_scores)
            if clipped_scores
            else None,
        }
    return summary


def make_healthbench_professional_example_tags(row: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for field in ("use_case", "type", "difficulty", "specialty"):
        value = row.get(field)
        if value is not None:
            tags.append(f"{field}:{value}")
    return tags


def metrics_from_grades(
    *,
    rubric_items: list[HealthBenchProfessionalRubricItem],
    rubric_grades: list[HealthBenchProfessionalRubricGrade],
    example_tags: list[str],
    response_text: str,
    length_adjustment_center: float | None,
    length_adjustment_penalty_per_500_chars: float | None,
) -> dict[str, float]:
    overall_score = calculate_score(rubric_items, rubric_grades)
    if overall_score is None:
        raise ValueError("overall score must not be None")

    metrics = {"overall_score": overall_score}
    if length_adjustment_center is not None:
        if length_adjustment_penalty_per_500_chars is None:
            raise ValueError("length adjustment penalty must be set")
        metrics["overall_score_length_adjusted"] = calculate_length_adjusted_score(
            overall_score,
            response_text,
            center=length_adjustment_center,
            penalty_per_500_chars=length_adjustment_penalty_per_500_chars,
        )

    example_tag_scores = {tag: overall_score for tag in example_tags}
    if len(example_tag_scores) != len(example_tags):
        raise ValueError("duplicate example tags are not supported")
    metrics.update(example_tag_scores)

    rubric_tag_items_grades: dict[
        str,
        list[tuple[HealthBenchProfessionalRubricItem, HealthBenchProfessionalRubricGrade]],
    ] = defaultdict(list)
    for rubric_item, rubric_grade in zip(rubric_items, rubric_grades, strict=True):
        curr_item_tags: set[str] = set()
        for tag in rubric_item.tags:
            rubric_tag_items_grades[tag].append((rubric_item, rubric_grade))
            if tag in curr_item_tags:
                raise ValueError(f"duplicate rubric tag on item: {tag}")
            curr_item_tags.add(tag)

    for tag, items_grades in rubric_tag_items_grades.items():
        items, grades = zip(*items_grades, strict=True)
        tag_score = calculate_score(list(items), list(grades))
        if tag_score is not None:
            metrics[tag] = tag_score
    return metrics


class HealthBenchProfessionalEvaluator(
    Evaluator[
        HealthBenchProfessionalInstance,
        HealthBenchProfessionalEvalResult,
        HealthBenchProfessionalEvalSummary,
    ]
):
    def __init__(
        self,
        reference_source: str = DATASET_PATH,
        model: str = OFFICIAL_PROFESSIONAL_GRADER_MODEL,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = OFFICIAL_PROFESSIONAL_REASONING_EFFORT,
        llm_max_retries: int = 5,
        retry_base_seconds: float = 2.0,
        length_adjustment_center: float | None = (
            OFFICIAL_PROFESSIONAL_LENGTH_ADJUSTMENT_CENTER
        ),
        length_adjustment_penalty_per_500_chars: float | None = (
            OFFICIAL_PROFESSIONAL_LENGTH_ADJUSTMENT_PENALTY_PER_500_CHARS
        ),
    ) -> None:
        if (length_adjustment_center is None) != (
            length_adjustment_penalty_per_500_chars is None
        ):
            raise ValueError(
                "length adjustment requires both center and penalty per 500 chars"
            )
        if length_adjustment_center is not None and length_adjustment_center < 0:
            raise ValueError("length adjustment center must be non-negative")
        if (
            length_adjustment_penalty_per_500_chars is not None
            and length_adjustment_penalty_per_500_chars < 0
        ):
            raise ValueError(
                "length adjustment penalty per 500 chars must be non-negative"
            )

        self.reference_source = resolve_reference_source(reference_source)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = normalize_reasoning_effort(reasoning_effort)
        self.llm_max_retries = llm_max_retries
        self.retry_base_seconds = retry_base_seconds
        self.length_adjustment_center = length_adjustment_center
        self.length_adjustment_penalty_per_500_chars = (
            length_adjustment_penalty_per_500_chars
        )
        self.judge_metadata = {
            "reference_source": self.reference_source,
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "reasoning_effort": self.reasoning_effort,
            "llm_max_retries": self.llm_max_retries,
            "retry_base_seconds": self.retry_base_seconds,
            "strip_think_blocks": True,
            "length_adjustment_center": self.length_adjustment_center,
            "length_adjustment_penalty_per_500_chars": (
                self.length_adjustment_penalty_per_500_chars
            ),
        }

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--reference-source",
            default=DATASET_PATH,
            help=(
                "HealthBench Professional source. Supports the HF dataset alias, "
                "official raw JSONL URL, or a local JSONL path."
            ),
        )
        parser.add_argument(
            "--model",
            default=OFFICIAL_PROFESSIONAL_GRADER_MODEL,
            help="OpenAI model used as the HealthBench Professional rubric judge.",
        )
        parser.add_argument(
            "--max_output_tokens",
            type=int,
            default=None,
            help=(
                "Maximum number of tokens the judge may generate per rubric item. "
                "Official GPT-5.4 low-reasoning mode leaves this unset."
            ),
        )
        parser.add_argument(
            "--reasoning-effort",
            choices=["none", "low", "medium", "high"],
            default=OFFICIAL_PROFESSIONAL_REASONING_EFFORT,
            help="Reasoning effort sent to reasoning-capable OpenAI judge models.",
        )
        parser.add_argument(
            "--llm-max-retries",
            type=int,
            default=5,
            help="Maximum number of retries for transient judge-model failures.",
        )
        parser.add_argument(
            "--retry-base-seconds",
            type=float,
            default=2.0,
            help="Base sleep duration for exponential retry backoff.",
        )
        parser.add_argument(
            "--length-adjustment-center",
            type=float,
            default=OFFICIAL_PROFESSIONAL_LENGTH_ADJUSTMENT_CENTER,
            help="Optional response length center for HealthBench length adjustment.",
        )
        parser.add_argument(
            "--length-adjustment-penalty-per-500-chars",
            type=float,
            default=OFFICIAL_PROFESSIONAL_LENGTH_ADJUSTMENT_PENALTY_PER_500_CHARS,
            help="Optional penalty multiplier for HealthBench length adjustment.",
        )

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
    ) -> "HealthBenchProfessionalEvaluator":
        return cls(
            reference_source=args.reference_source,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
            llm_max_retries=args.llm_max_retries,
            retry_base_seconds=args.retry_base_seconds,
            length_adjustment_center=args.length_adjustment_center,
            length_adjustment_penalty_per_500_chars=(
                args.length_adjustment_penalty_per_500_chars
            ),
        )

    def prepare_instances(self) -> dict[str, HealthBenchProfessionalInstance]:
        instances: dict[str, HealthBenchProfessionalInstance] = {}
        for row_index, line in enumerate(iter_jsonl_lines(self.reference_source)):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue

            row_id = row.get("id") or row.get("prompt_id")
            instance_id = str(row_id) if row_id else str(row_index)
            prompt_id = str(row.get("prompt_id") or instance_id)

            messages = normalize_messages(row.get("prompt"))
            conversation = row.get("conversation")
            if messages is None:
                messages = normalize_messages(conversation)
            if messages is None and isinstance(conversation, dict):
                messages = normalize_messages(conversation.get("messages"))
            if messages is None:
                question = render_text(row.get("question")).strip()
                messages = [{"role": "user", "content": question}]

            question = last_user_message_text(messages).strip()
            if not question:
                question = render_text(row.get("question")).strip()

            rubric_payloads = row.get("rubrics") or row.get("rubric_items") or []
            rubric_items = [
                HealthBenchProfessionalRubricItem.from_dict(item)
                for item in rubric_payloads
            ]
            if not rubric_items:
                raise ValueError(
                    f"Missing rubric_items for HealthBench row_index={row_index}"
                )

            instance = HealthBenchProfessionalInstance(
                id=instance_id,
                prompt_id=prompt_id,
                row_index=row_index,
                question=question,
                conversation_messages=messages,
                gold=render_text(
                    row.get("physician_response")
                    or row.get("gold")
                    or row.get("completion_to_trial")
                ).strip(),
                rubric_items=rubric_items,
                example_tags=list(row.get("example_tags") or make_healthbench_professional_example_tags(row)),
                use_case=row.get("use_case"),
                type=row.get("type"),
                difficulty=row.get("difficulty"),
                specialty=row.get("specialty"),
            )

            instances[instance_id] = instance
            row_index_key = str(row_index)
            if row_index_key not in instances:
                instances[row_index_key] = instance
        return instances

    def evaluate(
        self,
        instance: HealthBenchProfessionalInstance,
        response: str,
    ) -> HealthBenchProfessionalEvalResult:
        graded_response = strip_think_blocks(response)
        completion_id = hashlib.sha256(
            (instance.prompt_id + graded_response).encode("utf-8")
        ).hexdigest()
        total_positive_points = sum(
            item.points for item in instance.rubric_items if item.points > 0
        )

        if not instance.is_completed:
            rubric_grades = [
                HealthBenchProfessionalRubricGrade(
                    index=rubric_index,
                    criterion=rubric_item.criterion,
                    points=rubric_item.points,
                    tags=list(rubric_item.tags),
                    criteria_met=False,
                    explanation="Response was incomplete and was not sent to the judge.",
                    raw_response_text=None,
                    status="skipped",
                    error="Response incomplete",
                )
                for rubric_index, rubric_item in enumerate(instance.rubric_items)
            ]
            overall_score = 0.0 if total_positive_points > 0 else None
            metrics: dict[str, float] = {}
            if overall_score is not None:
                metrics = metrics_from_grades(
                    rubric_items=instance.rubric_items,
                    rubric_grades=rubric_grades,
                    example_tags=instance.example_tags,
                    response_text=graded_response,
                    length_adjustment_center=self.length_adjustment_center,
                    length_adjustment_penalty_per_500_chars=(
                        self.length_adjustment_penalty_per_500_chars
                    ),
                )
                if "overall_score_length_adjusted" in metrics:
                    metrics["overall_score_length_adjusted"] = 0.0

            length_adjusted_score = (
                0.0
                if overall_score is not None
                and self.length_adjustment_center is not None
                else None
            )
            return HealthBenchProfessionalEvalResult(
                id=instance.id,
                prompt_id=instance.prompt_id,
                completion_id=completion_id,
                row_index=instance.row_index,
                question=instance.question,
                response=response,
                graded_response=graded_response,
                gold=instance.gold,
                conversation_messages=instance.conversation_messages,
                rubric_items=instance.rubric_items,
                example_tags=instance.example_tags,
                use_case=instance.use_case,
                type=instance.type,
                difficulty=instance.difficulty,
                specialty=instance.specialty,
                metrics=metrics,
                judge_metadata=dict(self.judge_metadata),
                judge_result=HealthBenchProfessionalJudgeResult(
                    status="incomplete",
                    overall_score=overall_score,
                    overall_score_clipped=overall_score,
                    overall_score_length_adjusted=length_adjusted_score,
                    overall_score_length_adjusted_clipped=length_adjusted_score,
                    achieved_points=0.0 if overall_score is not None else None,
                    total_positive_points=total_positive_points,
                    prediction_chars=len(graded_response),
                    rubric_grades=rubric_grades,
                    parse_error=True,
                    error="Response incomplete",
                    usage=None,
                    metrics=metrics,
                ),
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in environment")
        client = openai.OpenAI(api_key=api_key)
        conversation_text = format_conversation(
            instance.conversation_messages,
            graded_response,
        )

        rubric_grades: list[HealthBenchProfessionalRubricGrade] = []
        usage_total: dict[str, int | None] | None = None
        parse_error = False
        status = "completed"
        error: str | None = None

        for rubric_index, rubric_item in enumerate(instance.rubric_items):
            judge_prompt = create_grader_prompt(conversation_text, rubric_item)
            try:
                parsed, judge_text, usage = grade_rubric_item_with_retries(
                    client=client,
                    prompt=judge_prompt,
                    model=self.model,
                    max_output_tokens=self.max_output_tokens,
                    reasoning_effort=self.reasoning_effort,
                    max_retries=self.llm_max_retries,
                    retry_base_seconds=self.retry_base_seconds,
                )
                usage_total = merge_usage_dicts(usage_total, usage)
                rubric_grades.append(
                    HealthBenchProfessionalRubricGrade(
                        index=rubric_index,
                        criterion=rubric_item.criterion,
                        points=rubric_item.points,
                        tags=list(rubric_item.tags),
                        criteria_met=bool(parsed["criteria_met"]),
                        explanation=parsed.get("explanation", ""),
                        raw_response_text=judge_text,
                        status="completed",
                        usage=usage,
                    )
                )
            except Exception as exc:
                parse_error = True
                status = "error"
                error = str(exc)
                rubric_grades.append(
                    HealthBenchProfessionalRubricGrade(
                        index=rubric_index,
                        criterion=rubric_item.criterion,
                        points=rubric_item.points,
                        tags=list(rubric_item.tags),
                        criteria_met=None,
                        explanation="",
                        raw_response_text=None,
                        status="error",
                        error=str(exc),
                    )
                )

        achieved_points = calculate_achieved_points(
            instance.rubric_items,
            rubric_grades,
        )
        overall_score = calculate_score(instance.rubric_items, rubric_grades)
        metrics: dict[str, float] = {}
        if status == "completed":
            metrics = metrics_from_grades(
                rubric_items=instance.rubric_items,
                rubric_grades=rubric_grades,
                example_tags=instance.example_tags,
                response_text=graded_response,
                length_adjustment_center=self.length_adjustment_center,
                length_adjustment_penalty_per_500_chars=(
                    self.length_adjustment_penalty_per_500_chars
                ),
            )
        overall_score_clipped = (
            clip_unit_interval(overall_score) if overall_score is not None else None
        )
        length_adjusted_score = None
        length_adjusted_score_clipped = None
        if (
            overall_score is not None
            and self.length_adjustment_center is not None
            and self.length_adjustment_penalty_per_500_chars is not None
        ):
            length_adjusted_score = calculate_length_adjusted_score(
                overall_score,
                graded_response,
                center=self.length_adjustment_center,
                penalty_per_500_chars=self.length_adjustment_penalty_per_500_chars,
            )
            length_adjusted_score_clipped = clip_unit_interval(length_adjusted_score)

        return HealthBenchProfessionalEvalResult(
            id=instance.id,
            prompt_id=instance.prompt_id,
            completion_id=completion_id,
            row_index=instance.row_index,
            question=instance.question,
            response=response,
            graded_response=graded_response,
            gold=instance.gold,
            conversation_messages=instance.conversation_messages,
            rubric_items=instance.rubric_items,
            example_tags=instance.example_tags,
            use_case=instance.use_case,
            type=instance.type,
            difficulty=instance.difficulty,
            specialty=instance.specialty,
            metrics=metrics,
            judge_metadata=dict(self.judge_metadata),
            judge_result=HealthBenchProfessionalJudgeResult(
                status=status,
                overall_score=overall_score,
                overall_score_clipped=overall_score_clipped,
                overall_score_length_adjusted=length_adjusted_score,
                overall_score_length_adjusted_clipped=length_adjusted_score_clipped,
                achieved_points=achieved_points,
                total_positive_points=total_positive_points,
                prediction_chars=len(graded_response),
                rubric_grades=rubric_grades,
                parse_error=parse_error,
                error=error,
                usage=usage_total,
                metrics=metrics,
            ),
        )

    def aggregate(
        self,
        results: Sequence[HealthBenchProfessionalEvalResult],
    ) -> HealthBenchProfessionalEvalSummary:
        status_counts = Counter(result.judge_result.status for result in results)
        name2values: dict[str, list[float]] = defaultdict(list)
        for result in results:
            for name, value in result.metrics.items():
                name2values[name].append(value)
            if result.judge_result.overall_score is not None:
                name2values["score"].append(result.judge_result.overall_score)

        final_metrics: dict[str, float | int] = {}
        for name, values in name2values.items():
            for stat in ("mean", "n_samples", "bootstrap_std"):
                key = name if stat == "mean" else f"{name}:{stat}"
                final_metrics[key] = compute_clipped_stats(values, stat)
        score = final_metrics.pop("score", None)

        scores = [
            float(result.judge_result.overall_score)
            for result in results
            if isinstance(result.judge_result.overall_score, (int, float))
        ]
        clipped_scores = [
            float(result.judge_result.overall_score_clipped)
            for result in results
            if isinstance(result.judge_result.overall_score_clipped, (int, float))
        ]
        length_adjusted_scores = [
            float(result.judge_result.overall_score_length_adjusted)
            for result in results
            if isinstance(result.judge_result.overall_score_length_adjusted, (int, float))
        ]
        length_adjusted_clipped_scores = [
            float(result.judge_result.overall_score_length_adjusted_clipped)
            for result in results
            if isinstance(
                result.judge_result.overall_score_length_adjusted_clipped,
                (int, float),
            )
        ]
        achieved_points = [
            float(result.judge_result.achieved_points)
            for result in results
            if isinstance(result.judge_result.achieved_points, (int, float))
        ]
        total_positive_points = [
            float(result.judge_result.total_positive_points)
            for result in results
            if isinstance(result.judge_result.total_positive_points, (int, float))
        ]
        prediction_chars = [
            result.judge_result.prediction_chars for result in results
        ]

        return HealthBenchProfessionalEvalSummary(
            score=float(score) if isinstance(score, (int, float)) else None,
            metrics=final_metrics,
            num_results=len(results),
            status_counts=dict(status_counts),
            scored_examples=len(scores),
            mean_overall_score=compute_clipped_stats(scores, "mean")
            if scores
            else None,
            median_overall_score=median(scores) if scores else None,
            min_overall_score=min(scores) if scores else None,
            max_overall_score=max(scores) if scores else None,
            mean_overall_score_clipped=compute_clipped_stats(clipped_scores, "mean")
            if clipped_scores
            else None,
            mean_overall_score_length_adjusted=compute_clipped_stats(
                length_adjusted_scores,
                "mean",
            )
            if length_adjusted_scores
            else None,
            mean_overall_score_length_adjusted_clipped=compute_clipped_stats(
                length_adjusted_clipped_scores,
                "mean",
            )
            if length_adjusted_clipped_scores
            else None,
            mean_achieved_points=mean(achieved_points) if achieved_points else None,
            mean_total_positive_points=mean(total_positive_points)
            if total_positive_points
            else None,
            mean_prediction_chars=mean(prediction_chars) if prediction_chars else None,
            by_use_case=group_score_summary(results, "use_case"),
            by_type=group_score_summary(results, "type"),
            by_difficulty=group_score_summary(results, "difficulty"),
            by_specialty=group_score_summary(results, "specialty"),
        )
