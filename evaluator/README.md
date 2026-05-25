# Evaluators

Evaluators unify benchmark data formats and provide a lenient interface for downstream evaluation scripts.

## Interface

The evaluators for different datasets have been minimally abstracted in `base.py` so that each evaluator implements:
- `InstanceT`, `EvalResultT`, `EvalSummaryT`: type variables specific to this dataset, defining the fields for each instance before evaluation, each instance's evaluation result, and the aggregate summary result, respectively.
- `prepare_instances(self) -> dict[str, InstanceT]`
  Loads benchmark instances and returns them keyed by instance id.
- `evaluate(self, instance: InstanceT, response: str) -> EvalResultT`
  Evaluates one model response for one typed eval instance.
- `aggregate(self, results: Sequence[EvalResultT]) -> EvalSummaryT`
  Aggregates per-instance results for the dataset into a summary.
- `add_args(cls, parser: argparse.ArgumentParser) -> None`
  Adds evaluator-specific CLI flags necessary for construction.
- `from_args(cls, args: argparse.Namespace) -> Evaluator`
  Constructs the evaluator from the parsed driver namespace.

## Runner Example

The interface and evaluators in this folder can be called from your custom python scripts, fitting to your input / output shape. To eval a single instance, pass in the model's `response` to the `evaluate()` method; you can match by `id` to get the desired instance within the dataset.

After you have evaluated all instances, you can aggregate the results using the `aggregate()` method.

For an example eval script, see `scripts/run_judge_jsons.py`, which assumes that your model outputs follow the format as:
```
{
    "query_id": str, # The instance id
    "tool_call_counts": dict[str, int], # The number of tool calls for each tool
    "status": str, # The status of the response, use "completed" for success, otherwise treated as failure (e.g. reached max tokens)
    "result": [
        {
            "type": "output_text",
            "output": str, # the final output of the agent
        }
    ]
}
```

From which you may run:
```bash
python scripts/run_judge_jsons.py \
  --input_dir {your_run_dir_of_jsons} \
  --num-threads {num_threads} \
  deepsearchqa
```
or other datasets to evaluate.

Ultimately, the goal of the evaluators is to provide abstraction so that the many model output formats we currently have (and will have) can be easily adapted to different benchmarks.

## Supported Datasets

| Subcommand | Evaluator | Hugging Face dataset | Instance fields |
| --- | --- | --- | --- |
| `browsecomp` | `BrowseCompEvaluator` | `rl-rag-2/browsecomp` | `id`, `question`, `answer`, `is_completed` |
| `deepsearchqa` | `DeepSearchQAEvaluator` | `rl-rag-2/deepsearchqa` | `id`, `question`, `answer`, `answer_type`, `is_completed` |
| `finsearchcomp` | `FinSearchCompEvaluator` | `rl-rag-2/finsearchcomp` | `id`, `question`, `response_reference`, `judge_prompt_template`, `judge_system_prompt`, `is_completed` |
| `frontierscience` | `FrontierScienceEvaluator` | `rl-rag-2/frontierscience` | `id`, `question`, `answer`, `is_completed` |
| `healthbench-professional` | `HealthBenchProfessionalEvaluator` | `openai/healthbench-professional` | `id`, `row_index`, `question`, `conversation_messages`, `gold`, `rubric_items`, `is_completed` |
| `hle` | `HLEEvaluator` | `rl-rag-2/hle_text_only_curated_600_samples` | `id`, `row_index`, `question`, `answer`, `image`, `is_completed` |

`frontierscience` requires `--split {olympiad,research}` and loads that split from the same Hugging Face dataset.

## Dataset Caveats

For all datasets, we assume that model outputs that were incomplete due to context length (marked as `is_completed=False`) are incorrect, and do not call the judge for them.

All evaluators strip exposed `<think>...</think>` blocks from model responses before judging, because local reasoning-model outputs may include hidden reasoning traces that should not be graded as answer content.

### BrowseComp

We do not compute calibration error for now, as it may not be necessary at this initial stage.

### DeepSearchQA

- Official judge uses Gemini 2.5 Flash. We default to use GPT-4.1 instead, and the two judges are close.
- Incomplete runs are now counted as incorrect without judging.
- The official script counts accuracy by only considering completed runs, but we count all runs (including incomplete ones) as incorrect without judging. This makes the accuracy lower than the original judge.

### FinSearchComp

- The evaluator loads only the T2/T3 subset.
- The official parser requires a fenced JSON code block and reads `answer_score` as a nested value like `[[score]]`, which seems stale and inconsistent with their prompts. We changed this to just parse scalar.
- The official judge called some custom unknown Azure endpoint for judge model. We use GPT-4.1 instead.

### FrontierScience

- The research subset has 1 duplicate question. This has been removed, so we eval on 59/60 questions.
- Be mindful whether the results reported used search or not.

### HealthBench Professional

- The evaluator loads the official raw JSONL from `openai/healthbench-professional` by default. Use `--reference-source` to pass either a local HealthBench Professional JSONL or a simple-evals HealthBench-format JSONL.
- It follows the `openai/simple-evals` HealthBench reference path after response preprocessing: the full HealthBench grader prompt, independent rubric-item grading, negative-point rubric handling, `achieved_points / total_positive_points`, example-tag metrics, rubric-tag metrics, clipped aggregate means, `n_samples`, and bootstrap standard deviations.
- The default grader settings match the simple-evals HealthBench Professional option bundle: `gpt-5.4-2026-03-05`, low reasoning effort, no max output token cap for the reasoning grader, length-adjustment center `2000`, and penalty `0.0147` per 500 response characters.
- Instances are keyed by both official `id` and zero-based `row_index` when possible, so run outputs may use either value as `query_id`.

### HLE

- The evaluator follows `gpt-baselines/hle/run_judge_results.py`: it asks a judge to extract the final answer, compare it against the official answer, extract confidence, and returns accuracy plus calibration error.
- The default source is the curated 600-row text-only dataset at `rl-rag-2/hle_text_only_curated_600_samples`. If Hugging Face access requires authentication in the runtime, pass a local JSONL path to `--dataset`, for example `../gpt-baselines/hle/outputs/hle_text_only_curated_600_samples.jsonl`.
- Use `--text-only` to skip rows whose `image` field is non-empty.
