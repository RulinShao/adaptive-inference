import argparse
import dataclasses
import json
import sys
import types
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator import (  # noqa: E402
    DefaultsHelpFormatter,
    add_evaluator_subcommands,
    evaluator_from_args,
)


def mirror_directory_structure(input_dir: Path, output_dir: Path) -> Path:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    input_parts = input_dir.parts
    runs_index = None
    for i, part in enumerate(input_parts):
        if part == "runs":
            runs_index = i
            break

    if runs_index is not None:
        relative_parts = input_parts[runs_index + 1 :]
    else:
        relative_parts = input_parts[-4:] if len(input_parts) > 4 else input_parts

    mirrored_path = output_dir
    for part in relative_parts:
        mirrored_path = mirrored_path / part

    mirrored_path.mkdir(parents=True, exist_ok=True)
    return mirrored_path


def extract_final_response(run_data: dict[str, Any]) -> str:
    result_items = run_data["result"]
    if not isinstance(result_items, list):
        return ""

    output_texts: list[str] = []
    for item in result_items:
        if not isinstance(item, dict):
            raise TypeError(f"Expected result item to be a dict, got {type(item).__name__}")
        if item["type"] == "output_text" and item["output"]:
            output_texts.append(str(item["output"]))
    return output_texts[-1].strip() if output_texts else ""


def extract_tool_call_counts(run_data: dict[str, Any]) -> dict[str, int]:
    if "tool_call_counts" in run_data:
        tool_call_counts = run_data["tool_call_counts"]
        if not isinstance(tool_call_counts, dict):
            raise TypeError(
                "Expected run_data['tool_call_counts'] to be a dict, "
                f"got {type(tool_call_counts).__name__}"
            )
        return {str(tool_name): int(count) for tool_name, count in tool_call_counts.items()}

    result_items = run_data["result"]
    if not isinstance(result_items, list):
        return {}

    counts: Counter[str] = Counter()
    for item in result_items:
        if not isinstance(item, dict):
            raise TypeError(f"Expected result item to be a dict, got {type(item).__name__}")
        if item["type"] == "tool_call":
            counts[str(item["tool_name"])] += 1
    return dict(counts)


def dataclass_from_dict(dataclass_type: type[Any], data: dict[str, Any]) -> Any:
    if not dataclasses.is_dataclass(dataclass_type):
        raise TypeError(f"Expected dataclass type, got {dataclass_type}")

    kwargs: dict[str, Any] = {}
    for field in dataclasses.fields(dataclass_type):
        kwargs[field.name] = coerce_loaded_value(field.type, data[field.name])
    return dataclass_type(**kwargs)


def coerce_loaded_value(field_type: Any, value: Any) -> Any:
    if value is None:
        return None

    origin = get_origin(field_type)
    args = get_args(field_type)
    if origin in (types.UnionType, Union):
        non_none_args = [arg for arg in args if arg is not type(None)]
        for arg in non_none_args:
            if dataclasses.is_dataclass(arg):
                return dataclass_from_dict(arg, value)
        return value

    if dataclasses.is_dataclass(field_type):
        return dataclass_from_dict(field_type, value)
    if origin is list:
        item_type = args[0]
        return [coerce_loaded_value(item_type, item) for item in value]
    if origin is dict:
        return value
    return value


def evaluator_result_type(evaluator: Any) -> type[Any]:
    return_type = get_type_hints(evaluator.evaluate)["return"]
    if not isinstance(return_type, type) or not dataclasses.is_dataclass(return_type):
        raise TypeError(f"Evaluator returned unsupported result type: {return_type}")
    return return_type


def add_non_colliding_fields(
    result_dict: dict[str, Any],
    extra_fields: dict[str, Any],
) -> dict[str, Any]:
    for key, value in extra_fields.items():
        if key not in result_dict:
            result_dict[key] = value
    return result_dict


def average_tool_counts(results: list[dict[str, Any]]) -> dict[str, float]:
    totals: Counter[str] = Counter()
    for result in results:
        for tool_name, count in result["tool_call_counts"].items():
            totals[tool_name] += count

    total_results = len(results)
    if total_results == 0:
        return {}
    return {
        tool_name: count / total_results
        for tool_name, count in sorted(totals.items())
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=DefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing run JSON files.",
    )
    parser.add_argument(
        "--eval_dir",
        required=True,
        help="Root directory to write mirrored evaluation results.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of parallel judge worker threads.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation of existing judge result files.",
    )
    add_evaluator_subcommands(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if args.num_threads < 1:
        raise ValueError("--num-threads must be >= 1")

    evaluator = evaluator_from_args(args)
    eval_result_type = evaluator_result_type(evaluator)
    instances = evaluator.prepare_instances()

    output_dir = mirror_directory_structure(input_dir, eval_dir)
    judge_results_dir = output_dir / "judge_results"
    judge_results_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Loaded {len(instances)} instances")
    print(f"Evaluations will be saved to {output_dir}")
    print(f"Evaluating {len(json_files)} run JSON files with {args.num_threads} threads")

    eval_results: list[Any] = []
    saved_results: list[dict[str, Any]] = []
    skipped = 0

    def evaluate_file(json_path: Path) -> tuple[Any, dict[str, Any]] | None:
        with json_path.open("r", encoding="utf-8") as f:
            run_data = json.load(f)

        instance_id = str(run_data["query_id"])
        if instance_id not in instances:
            print(f"No instance found for id {instance_id} in {json_path}")
            return None

        is_completed = run_data["status"] == "completed"
        response = extract_final_response(run_data)
        tool_call_counts = extract_tool_call_counts(run_data)
        instance = dataclasses.replace(instances[instance_id], is_completed=is_completed)
        eval_result = evaluator.evaluate(instance, response)
        result_dict = dataclasses.asdict(eval_result)
        result_dict = add_non_colliding_fields(
            result_dict,
            {
                "json_path": str(json_path),
                "is_completed": is_completed,
                "tool_call_counts": tool_call_counts,
            },
        )

        eval_path = judge_results_dir / f"{json_path.stem}_eval.json"
        with eval_path.open("w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        return eval_result, result_dict

    json_files_to_eval: list[Path] = []
    for json_path in json_files:
        eval_path = judge_results_dir / f"{json_path.stem}_eval.json"
        if eval_path.exists() and not args.force:
            try:
                with eval_path.open("r", encoding="utf-8") as f:
                    existing_result = json.load(f)
                eval_results.append(dataclass_from_dict(eval_result_type, existing_result))
                saved_results.append(existing_result)
                skipped += 1
                continue
            except Exception as exc:
                print(f"Error loading existing evaluation {eval_path}: {exc}")
        json_files_to_eval.append(json_path)

    if json_files_to_eval:
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor, tqdm(
            total=len(json_files_to_eval), desc="Evaluating", unit="file"
        ) as pbar:
            futures = {
                executor.submit(evaluate_file, json_path): json_path
                for json_path in json_files_to_eval
            }
            for future in as_completed(futures):
                json_path = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"Error evaluating {json_path}: {exc}")
                    result = None
                finally:
                    pbar.update(1)
                if result is None:
                    continue
                eval_result, saved_result = result
                eval_results.append(eval_result)
                saved_results.append(saved_result)

    print(f"\nProcessed {len(saved_results)} evaluations ({skipped} skipped)")

    if not eval_results:
        print("No run JSON files were evaluated.")
        return

    summary = dataclasses.asdict(evaluator.aggregate(eval_results))
    complete_count = sum(1 for result in saved_results if result["is_completed"])
    summary["avg_tool_call_counts"] = average_tool_counts(saved_results)
    summary["avg_complete_rate"] = complete_count / len(saved_results)

    summary_path = output_dir / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary saved to {summary_path}")
    print(f"Individual judge results saved to {judge_results_dir}")


if __name__ == "__main__":
    main()
