import argparse
import dataclasses
import json
import logging
import math
import os
import random
import time
from collections.abc import Sequence
from typing import Any, Optional, Protocol

from dotenv import load_dotenv

from .base import Evaluator

load_dotenv()

DATASET_PATH = "rl-rag-2/deepsearchqa"
OFFICIAL_JUDGE_MODEL = "gemini-2.5-flash"
OPENROUTER_DEFAULT_JUDGE_MODEL = "google/gemini-2.5-flash"
OPENAI_DEFAULT_JUDGE_MODEL = "gpt-4.1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
VALID_ANSWER_TYPES = {"Single Answer", "Set Answer"}


DEEPSEARCH_QA_PROMPT = """\
Your task is to evaluate whether a given "AI Response" for a specific "User Prompt" arrived at the correct answer.

**Answer Correctness Task**

*   **Purpose:** Assess whether the AI response provides the correct answer(s) based on the provided "Correct Answer" and "Prompt Type".
*   **Process:**
    *   Identify the "Prompt Type": "<prompt_type>".
    *   Refer to the "Correct Answer": "<answer>".
    *   Based on the "Prompt Type", determine if the "AI Response" contains the expected answer(s).
        *   **'Single Answer'**: Check if the response provides the answer that addresses the user's question. It does not have to match the exact wording of the provided answer.
        *   **'Set Answer'**: Check if the response includes *each* item from the provided ground truth answers. The order might not matter unless specified otherwise. The response might include more answers than the list. Determine the correctness *only* based on the list first and then check if the response includes answers not in the list.
    *   **Explanation:** Provide a brief explanation justifying your assessment of answer correctness, referencing specific parts of the AI response and the correct answer.
    *   **Correctness Details:** Provide a dictionary, one key for each expected answer part, and value is a boolean indicating whether each expected answer part was found.
        *   For 'Set Answer', this will be a list of attributes, one for each item/part in the "Correct Answer". Each key will be a string indicating the expected answer part, and the value will be a boolean indicating whether that part was found in the response.
    *   **Excessive Answers:** Provide a list of strings, each indicating an excessive answer part. If the response provides answers that are **not** in the "Correct Answer" list, add these answers as excessive answers. Return an empty list when there's no excessive answers in the response.


**Output Format:**

Your evaluation *must* be structured as a nested JSON dictionary with the following top-level keys: `"Answer Correctness"`. Please return NULL if any of "Prompt", "AI Response" or "Correct Answer" is empty.
The value for `"Answer Correctness"` should be a dictionary containing `"Explanation"` (a string), `"Correctness Details"` (a dictionary where each key is the expected correct answer, and the value is a boolean indicating whether the response contains the correct answer), and `"Excessive Answers"` (a list of strings indicating the excessive answers).

Make sure you return a valid JSON string. Pay special attention to quotes, commas and special characters in the JSON string. Make sure to escape all special characters and quotes in the JSON string.


"""


GRADER_RATING_OUTPUT_EXAMPLE = r"""**Example (Partial):**

"```json
{{
  "Answer Correctness": {{
    "Explanation": "The response correctly identified Belgium and France but also includes an excessive answer, Italy.",
    "Correctness Details": {{
      "Belgium": true,
      "France": true,
    }},
    "Excessive Answers": [ "Italy" ]
  }}
}}
```"

**Now, proceed with the evaluation using the provided User Prompt, AI Response, and Correct Answer.**

User Prompt (Wrapped in <prompt> and </prompt>):
<prompt>
{prompt}
</prompt>
--------------------
**  Correct Answer (Wrapped in <answer> and </answer>):
Prompt Type: {prompt_type}
<answer>
{answer}
</answer>
--------------------
AI assistant response (Wrapped in <response> and </response>):
<response>
{response}
</response>

--------------------
Rating:"""


@dataclasses.dataclass(frozen=True)
class DeepSearchQAInstance:
    id: str
    question: str
    answer: str
    answer_type: str
    is_completed: bool = True

    def __post_init__(self) -> None:
        if self.answer_type not in VALID_ANSWER_TYPES:
            raise ValueError(
                f"answer_type must be one of {sorted(VALID_ANSWER_TYPES)}, "
                f"got {self.answer_type!r}"
            )


@dataclasses.dataclass
class ItemRating:
    example_id: str
    query: str
    response: str
    category_type: str | None = None
    expected_correct_answer: str | None = None
    answer_correctness_explanation: str | None = None
    expected_correct_answer_list: list[str] | None = None
    response_wrong_answers_list: list[str] | None = None
    grader_ratings_list: list[bool] | None = None
    empty_model_response: bool = False
    empty_auto_rater_response: bool = False
    invalid_auto_rater_response: bool = False
    rating_response: str = ""
    rating_prompt: str = ""
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DeepSearchQAEvalResult:
    id: str
    question: str
    answer_type: str
    response: str
    expected_answer: str
    judge_prompt: str | None
    judge_response: str | None
    judge_metadata: dict[str, Any]
    item_rating: ItemRating
    metrics: dict[str, Any] | None


@dataclasses.dataclass(frozen=True)
class DeepSearchQAEvalSummary:
    num_total_ratings: int = 0
    num_empty_model_response: int = 0
    num_invalid_auto_rater_response: int = 0
    num_empty_auto_rater_response: int = 0
    num_valid_ratings: int = 0
    num_answer_correctness_evaluated: int = 0
    pct_w_ci_all_answers_correct: str = ""
    pct_w_ci_fully_incorrect_items: str = ""
    pct_w_ci_correct_with_excessive_answers: str = ""
    pct_empty_model_response: float = 0.0
    pct_invalid_auto_rater_response: float = 0.0
    pct_empty_auto_rater_response: float = 0.0
    precision: str = ""
    recall: str = ""
    f1_score: str = ""


class JudgeProvider(Protocol):
    name: str
    model: str
    max_output_tokens: Optional[int]

    def judge(self, prompt: str) -> str:
        ...


class GeminiJudgeProvider:
    name = "gemini"

    def __init__(self, model: str, max_output_tokens: Optional[int]) -> None:
        self.model = model
        self.max_output_tokens = max_output_tokens
        api_key = (
            os.getenv("GOOGLE_AIS_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "GOOGLE_AIS_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY must be set"
            )

        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=api_key)

    def judge(self, prompt: str) -> str:
        if self.max_output_tokens is None:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            return getattr(response, "text", "") or ""

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                max_output_tokens=self.max_output_tokens,
            ),
        )
        return getattr(response, "text", "") or ""


class OpenRouterJudgeProvider:
    name = "openrouter"

    def __init__(
        self,
        model: str,
        max_output_tokens: Optional[int],
    ) -> None:
        self.model = model
        self.max_output_tokens = max_output_tokens
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY must be set")

        import openai

        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
        )
    def judge(self, prompt: str) -> str:
        kwargs: dict[str, Any] = {}
        if self.max_output_tokens is not None:
            kwargs["max_tokens"] = self.max_output_tokens

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content or ""


class OpenAIJudgeProvider:
    name = "openai"

    def __init__(self, model: str, max_output_tokens: Optional[int]) -> None:
        self.model = model
        self.max_output_tokens = max_output_tokens
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY must be set")

        import openai

        self._client = openai.OpenAI(api_key=api_key)

    def judge(self, prompt: str) -> str:
        body: dict[str, Any] = {"model": self.model, "input": prompt}
        if self.max_output_tokens is not None:
            body["max_output_tokens"] = self.max_output_tokens
        response = self._client.responses.create(**body)
        return getattr(response, "output_text", "") or ""


def default_model_for_provider(provider_name: str) -> str:
    if provider_name == "openai":
        return OPENAI_DEFAULT_JUDGE_MODEL
    if provider_name == "openrouter":
        return OPENROUTER_DEFAULT_JUDGE_MODEL
    return OFFICIAL_JUDGE_MODEL


def create_judge_provider(
    provider_name: str,
    model: str,
    max_output_tokens: Optional[int],
) -> JudgeProvider:
    if provider_name == "gemini":
        return GeminiJudgeProvider(model=model, max_output_tokens=max_output_tokens)
    if provider_name == "openrouter":
        return OpenRouterJudgeProvider(
            model=model,
            max_output_tokens=max_output_tokens,
        )
    if provider_name == "openai":
        return OpenAIJudgeProvider(model=model, max_output_tokens=max_output_tokens)
    raise ValueError(f"Unsupported judge provider: {provider_name}")


def create_judge_prompt(instance: DeepSearchQAInstance, model_response: str) -> str:
    return DEEPSEARCH_QA_PROMPT + GRADER_RATING_OUTPUT_EXAMPLE.format(
        prompt=instance.question.strip(),
        prompt_type=instance.answer_type.strip(),
        answer=instance.answer.strip(),
        response=str(model_response).strip(),
    )


def _parse_json_response(ori_json_response: str) -> Any:
    try:
        json_str = ori_json_response.strip()
        start_marker = "```json"
        start_idx = json_str.find(start_marker)

        if start_idx != -1:
            json_str = json_str[start_idx + len(start_marker) :].strip()
            end_marker = "```"
            end_idx = json_str.rfind(end_marker)
            if end_idx != -1:
                json_str = json_str[:end_idx].strip()

        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        logging.info("json.JSONDecodeError: %s for response: %s", exc, ori_json_response)
        return None


def _get_answer_correctness_details(json_response: Any) -> dict[str, bool] | None:
    try:
        details = json_response["Answer Correctness"]["Correctness Details"]
        if isinstance(details, dict):
            all_keys_are_strings = all(isinstance(key, str) for key in details.keys())
            all_values_are_booleans = all(
                isinstance(value, bool) for value in details.values()
            )
            if all_keys_are_strings and all_values_are_booleans:
                return details
        logging.warning("Invalid format for Answer Correctness Details: %s", details)
        return None
    except KeyError as exc:
        logging.info(
            'KeyError: %s for path "Answer Correctness.Correctness Details" in'
            " json_response: %s",
            exc,
            json_response,
        )
        return None
    except TypeError:
        logging.warning(
            "TypeError while accessing Correctness Details. JSON response: %s",
            json_response,
        )
        return None


def _get_excessive_answers(json_response: Any) -> list[str] | None:
    try:
        excessive_answers = json_response["Answer Correctness"]["Excessive Answers"]
        if isinstance(excessive_answers, list):
            all_items_are_strings = all(
                isinstance(item, str) for item in excessive_answers
            )
            if all_items_are_strings:
                return excessive_answers
        logging.warning("Invalid format for Excessive Answers: %s", excessive_answers)
        return None
    except KeyError as exc:
        logging.info(
            'KeyError: %s for path "Answer Correctness.Excessive Answers" in'
            " json_response: %s",
            exc,
            json_response,
        )
        return []


def reduce_judge_output_to_item_rating(
    item_rating: ItemRating,
    grader_llm_response_text: str,
    grader_llm_prompt_text: str,
) -> ItemRating:
    item_rating.rating_prompt = grader_llm_prompt_text
    item_rating.rating_response = grader_llm_response_text

    if not item_rating.response:
        item_rating.empty_model_response = True
        item_rating.error_message = "AI response was empty."
        return item_rating

    if not grader_llm_response_text:
        item_rating.empty_auto_rater_response = True
        item_rating.error_message = "Auto-rater response was empty."
        return item_rating

    parsed_json_response = _parse_json_response(grader_llm_response_text)
    if not parsed_json_response:
        item_rating.invalid_auto_rater_response = True
        item_rating.error_message = "Invalid JSON response from auto-rater."
        return item_rating

    try:
        answer_correctness_node = parsed_json_response.get("Answer Correctness")
        if not isinstance(answer_correctness_node, dict):
            item_rating.invalid_auto_rater_response = True
            item_rating.error_message = "Missing or malformed 'Answer Correctness' node."
            return item_rating

        grader_explanation = answer_correctness_node.get("Explanation")
        if not isinstance(grader_explanation, str):
            item_rating.invalid_auto_rater_response = True
            item_rating.error_message = (
                "Missing or malformed 'Explanation' in Answer Correctness."
            )
            return item_rating
        item_rating.answer_correctness_explanation = grader_explanation

        details = _get_answer_correctness_details(parsed_json_response)
        if details is None:
            item_rating.invalid_auto_rater_response = True
            item_rating.error_message = "Invalid 'Correctness Details' in Answer Correctness."
            return item_rating
        item_rating.expected_correct_answer_list = list(details.keys())
        item_rating.grader_ratings_list = list(details.values())

        excessive_answers = _get_excessive_answers(parsed_json_response)
        if excessive_answers is None:
            item_rating.invalid_auto_rater_response = True
            item_rating.error_message = "Invalid 'Excessive Answers' in Answer Correctness."
            return item_rating
        if excessive_answers:
            item_rating.response_wrong_answers_list = excessive_answers
    except (KeyError, TypeError, ValueError) as exc:
        logging.exception("Error processing parsed JSON: %s", exc)
        item_rating.invalid_auto_rater_response = True
        item_rating.error_message = f"Error during JSON processing: {exc}"
        return item_rating

    return item_rating


def call_judge_with_retries(
    judge_provider: JudgeProvider,
    prompt: str,
    max_retries: int,
) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return judge_provider.judge(prompt)
        except Exception as exc:
            last_error = exc
            logging.error(
                "LLM call failed (attempt %s/%s): %s",
                attempt + 1,
                max_retries,
                exc,
            )
            if attempt == max_retries - 1:
                break
            time.sleep(1 + (2 ** (attempt + random.random())))
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_error}")


def _calculate_ci_str(count: int, total: int, z: float = 1.96) -> str:
    if total == 0:
        return f"N/A ({count}/{total})"
    if count < 0:
        logging.warning("CI calculation: count %d is less than 0.", count)
        count = 0
    if count > total:
        logging.warning("CI calculation: count %d is greater than total %d", count, total)
        count = total

    p = count / total
    p_percent = p * 100.0
    try:
        variance = p * (1.0 - p)
        margin_of_error = z * math.sqrt(variance / total)
        moe_percent = margin_of_error * 100.0
        result_str = (
            f"{round(p_percent, 2):.2f} ±"
            f" {round(moe_percent, 2):.2f} ({count}/{total})"
        )
        if total <= 5:
            result_str += " (CI not robust for n<=5)"
        return result_str
    except (ValueError, ZeroDivisionError):
        return "N/A"


def _calculate_metric(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> dict[str, float]:
    precision_val = 0.0
    if (true_positives + false_positives) > 0:
        precision_val = true_positives / (true_positives + false_positives)

    recall_val = 0.0
    if (true_positives + false_negatives) > 0:
        recall_val = true_positives / (true_positives + false_negatives)

    f1_score_val = 0.0
    if (precision_val + recall_val) > 0:
        f1_score_val = (
            2 * (precision_val * recall_val) / (precision_val + recall_val)
        )

    return {
        "precision": precision_val,
        "recall": recall_val,
        "f1_score": f1_score_val,
    }


def _format_mean_percent(values: list[float]) -> str:
    return f"{(sum(values) / len(values)):.2%}" if values else ""


def item_metrics_from_rating(item_rating: ItemRating) -> dict[str, Any] | None:
    if item_rating.grader_ratings_list is None:
        return None

    ratings = item_rating.grader_ratings_list
    num_correct = sum(1 for rating in ratings if rating)
    true_positives = num_correct
    false_negatives = len(ratings) - num_correct
    excessive_answers = item_rating.response_wrong_answers_list
    false_positives = len(excessive_answers) if excessive_answers else 0
    metric = _calculate_metric(true_positives, false_positives, false_negatives)
    metric.update(
        {
            "expected_count": len(ratings),
            "correct_count": num_correct,
            "excessive_count": false_positives,
            "all_correct": bool(ratings)
            and num_correct == len(ratings)
            and false_positives == 0,
            "fully_incorrect": bool(ratings) and num_correct == 0,
        }
    )
    return metric


class DeepSearchQAEvaluator(
    Evaluator[
        DeepSearchQAInstance,
        DeepSearchQAEvalResult,
        DeepSearchQAEvalSummary,
    ]
):
    def __init__(
        self,
        judge_provider: str = "gemini",
        model: str | None = None,
        max_output_tokens: int | None = None,
        llm_max_retries: int = 5,
        injected_judge_provider: JudgeProvider | None = None,
    ) -> None:
        judge_model = model or default_model_for_provider(judge_provider)
        self.llm_max_retries = llm_max_retries
        self.judge_metadata = {
            "model": judge_model,
            "judge_provider": judge_provider,
            "max_output_tokens": max_output_tokens,
            "llm_max_retries": llm_max_retries,
        }
        self.judge_provider = injected_judge_provider or create_judge_provider(
            provider_name=judge_provider,
            model=judge_model,
            max_output_tokens=max_output_tokens,
        )

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model",
            default="gpt-4.1",
            help="Model used as the DeepSearchQA answer-set judge.",
        )
        parser.add_argument(
            "--judge-provider",
            choices=["gemini", "openrouter", "openai"],
            default="openai",
            help="API provider used to call the DeepSearchQA judge model.",
        )
        parser.add_argument(
            "--max_output_tokens",
            type=int,
            default=None,
            help="Maximum number of tokens the judge may generate; omit to use provider default.",
        )
        parser.add_argument(
            "--llm-max-retries",
            type=int,
            default=5,
            help="Maximum number of retries for transient judge-model failures.",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "DeepSearchQAEvaluator":
        return cls(
            judge_provider=args.judge_provider,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
            llm_max_retries=args.llm_max_retries,
        )

    def prepare_instances(self) -> dict[str, DeepSearchQAInstance]:
        import datasets

        rows = datasets.load_dataset(
            DATASET_PATH,
            split="train",
        )
        instances: dict[str, DeepSearchQAInstance] = {}
        for row in rows:
            raw_answer = row["answer"]
            answer = "None" if raw_answer is None else str(raw_answer)
            instance = DeepSearchQAInstance(
                id=str(row["id"]),
                question=str(row["question"]),
                answer=answer,
                answer_type=str(row["answer_type"]).strip(),
            )
            instances[instance.id] = instance
        return instances

    def evaluate(
        self,
        instance: DeepSearchQAInstance,
        response: str,
    ) -> DeepSearchQAEvalResult:
        item_rating = ItemRating(
            example_id=instance.id,
            query=instance.question,
            response=response,
            expected_correct_answer=instance.answer,
        )

        if not instance.is_completed:
            item_rating.answer_correctness_explanation = (
                "Response incomplete or cannot be parsed."
            )
            item_rating.expected_correct_answer_list = [instance.answer]
            item_rating.grader_ratings_list = [False]
            item_rating.error_message = "Response incomplete or cannot be parsed."
            return DeepSearchQAEvalResult(
                id=instance.id,
                question=instance.question,
                answer_type=instance.answer_type,
                response=response,
                expected_answer=instance.answer,
                judge_prompt=None,
                judge_response=None,
                judge_metadata=dict(self.judge_metadata),
                item_rating=item_rating,
                metrics=item_metrics_from_rating(item_rating),
            )

        if not response:
            item_rating = reduce_judge_output_to_item_rating(
                item_rating=item_rating,
                grader_llm_response_text="",
                grader_llm_prompt_text="",
            )
            return DeepSearchQAEvalResult(
                id=instance.id,
                question=instance.question,
                answer_type=instance.answer_type,
                response=response,
                expected_answer=instance.answer,
                judge_prompt=None,
                judge_response=None,
                judge_metadata=dict(self.judge_metadata),
                item_rating=item_rating,
                metrics=item_metrics_from_rating(item_rating),
            )

        judge_prompt = create_judge_prompt(instance, response)
        judge_response = call_judge_with_retries(
            judge_provider=self.judge_provider,
            prompt=judge_prompt,
            max_retries=self.llm_max_retries,
        )
        item_rating = reduce_judge_output_to_item_rating(
            item_rating=item_rating,
            grader_llm_response_text=judge_response,
            grader_llm_prompt_text=judge_prompt,
        )
        return DeepSearchQAEvalResult(
            id=instance.id,
            question=instance.question,
            answer_type=instance.answer_type,
            response=response,
            expected_answer=instance.answer,
            judge_prompt=judge_prompt,
            judge_response=judge_response,
            judge_metadata=dict(self.judge_metadata),
            item_rating=item_rating,
            metrics=item_metrics_from_rating(item_rating),
        )

    def aggregate(
        self,
        results: Sequence[DeepSearchQAEvalResult],
    ) -> DeepSearchQAEvalSummary:
        total_items = len(results)
        num_empty_model_response = 0
        num_invalid_auto_rater_response = 0
        num_empty_auto_rater_response = 0
        num_valid_ratings = 0
        num_answer_correctness_evaluated = 0
        num_answer_correctness_all_correct = 0
        num_fully_incorrect_items = 0
        num_items_correct_with_excessive_answers = 0
        per_item_metrics = {"precision": [], "recall": [], "f1_score": []}

        for result in results:
            item_rating = result.item_rating
            if item_rating.invalid_auto_rater_response:
                num_invalid_auto_rater_response += 1
                continue
            if item_rating.empty_auto_rater_response:
                num_empty_auto_rater_response += 1
                continue
            if item_rating.empty_model_response:
                num_empty_model_response += 1
                continue

            num_valid_ratings += 1
            if item_rating.grader_ratings_list is None:
                continue

            num_answer_correctness_evaluated += 1
            ratings = item_rating.grader_ratings_list
            num_correct = sum(1 for rating in ratings if rating)
            true_positives = num_correct
            false_negatives = len(ratings) - num_correct
            has_expected_answers = bool(ratings)

            all_expected_answers_correct = False
            if has_expected_answers:
                all_expected_answers_correct = num_correct == len(ratings)
                if num_correct == 0:
                    num_fully_incorrect_items += 1

            excessive_answers = item_rating.response_wrong_answers_list
            has_excessive_answers = bool(excessive_answers)
            false_positives = len(excessive_answers) if excessive_answers else 0
            if has_excessive_answers and (
                all_expected_answers_correct or not has_expected_answers
            ):
                num_items_correct_with_excessive_answers += 1

            is_all_correct = (
                all_expected_answers_correct or not has_expected_answers
            ) and not has_excessive_answers
            if is_all_correct:
                num_answer_correctness_all_correct += 1

            per_item_metric = _calculate_metric(
                true_positives, false_positives, false_negatives
            )
            for key, value in per_item_metric.items():
                per_item_metrics[key].append(value)

        pct_empty_model_response = (
            round(num_empty_model_response * 100.0 / total_items, 2)
            if total_items
            else 0.0
        )
        pct_invalid_auto_rater_response = (
            round(num_invalid_auto_rater_response * 100.0 / total_items, 2)
            if total_items
            else 0.0
        )
        pct_empty_auto_rater_response = (
            round(num_empty_auto_rater_response * 100.0 / total_items, 2)
            if total_items
            else 0.0
        )

        if num_answer_correctness_evaluated > 0:
            return DeepSearchQAEvalSummary(
                num_total_ratings=total_items,
                num_empty_model_response=num_empty_model_response,
                num_invalid_auto_rater_response=num_invalid_auto_rater_response,
                num_empty_auto_rater_response=num_empty_auto_rater_response,
                num_valid_ratings=num_valid_ratings,
                num_answer_correctness_evaluated=num_answer_correctness_evaluated,
                pct_w_ci_all_answers_correct=_calculate_ci_str(
                    num_answer_correctness_all_correct,
                    num_answer_correctness_evaluated,
                ),
                pct_w_ci_fully_incorrect_items=_calculate_ci_str(
                    num_fully_incorrect_items,
                    num_answer_correctness_evaluated,
                ),
                pct_w_ci_correct_with_excessive_answers=_calculate_ci_str(
                    num_items_correct_with_excessive_answers,
                    num_answer_correctness_evaluated,
                ),
                pct_empty_model_response=pct_empty_model_response,
                pct_invalid_auto_rater_response=pct_invalid_auto_rater_response,
                pct_empty_auto_rater_response=pct_empty_auto_rater_response,
                precision=_format_mean_percent(per_item_metrics["precision"]),
                recall=_format_mean_percent(per_item_metrics["recall"]),
                f1_score=_format_mean_percent(per_item_metrics["f1_score"]),
            )

        return DeepSearchQAEvalSummary(
            num_total_ratings=total_items,
            num_empty_model_response=num_empty_model_response,
            num_invalid_auto_rater_response=num_invalid_auto_rater_response,
            num_empty_auto_rater_response=num_empty_auto_rater_response,
            num_valid_ratings=num_valid_ratings,
            pct_empty_model_response=pct_empty_model_response,
            pct_invalid_auto_rater_response=pct_invalid_auto_rater_response,
            pct_empty_auto_rater_response=pct_empty_auto_rater_response,
        )
