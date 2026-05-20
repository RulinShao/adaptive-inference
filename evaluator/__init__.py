import argparse
from collections.abc import Mapping
from typing import Any

from .base import Evaluator, EvalResultT, EvalSummaryT, InstanceT
from .browsecomp_evaluator import (
    BrowseCompEvalResult,
    BrowseCompEvalSummary,
    BrowseCompEvaluator,
    BrowseCompInstance,
    BrowseCompJudgeResult,
)
from .deepsearchqa_evaluator import (
    DeepSearchQAEvalResult,
    DeepSearchQAEvalSummary,
    DeepSearchQAEvaluator,
    DeepSearchQAInstance,
)
from .finsearchcomp_evaluator import (
    FinSearchCompEvalResult,
    FinSearchCompEvalSummary,
    FinSearchCompEvaluator,
    FinSearchCompInstance,
)
from .frontierscience_evaluator import (
    FrontierScienceEvalResult,
    FrontierScienceResearchEvalSummary,
    FrontierScienceEvalSummary,
    FrontierScienceEvaluator,
    FrontierScienceInstance,
)
from .healthbench_professional_evaluator import (
    HealthBenchProfessionalEvalResult,
    HealthBenchProfessionalEvalSummary,
    HealthBenchProfessionalEvaluator,
    HealthBenchProfessionalInstance,
    HealthBenchProfessionalJudgeResult,
    HealthBenchProfessionalRubricGrade,
    HealthBenchProfessionalRubricItem,
)
from .hle_evaluator import (
    HLEEvalResult,
    HLEEvalSummary,
    HLEEvaluator,
    HLEInstance,
    HLEJudgeResult,
)

EvaluatorClass = type[Evaluator[Any, Any, Any]]


class DefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action: argparse.Action) -> str:
        help_text = action.help or ""
        if "%(default)" in help_text:
            return help_text
        if action.required:
            return help_text
        if action.default is argparse.SUPPRESS or action.default is None:
            return help_text
        return f"{help_text} (default: %(default)s)"


EVALUATORS: dict[str, EvaluatorClass] = {
    "browsecomp": BrowseCompEvaluator,
    "deepsearchqa": DeepSearchQAEvaluator,
    "finsearchcomp": FinSearchCompEvaluator,
    "frontierscience": FrontierScienceEvaluator,
    "healthbench-professional": HealthBenchProfessionalEvaluator,
    "hle": HLEEvaluator,
}


def add_evaluator_subcommands(
    parser: argparse.ArgumentParser,
    evaluators: Mapping[str, EvaluatorClass] | None = None,
    dest: str = "dataset",
) -> argparse._SubParsersAction:
    evaluator_map = evaluators or EVALUATORS
    subparsers = parser.add_subparsers(dest=dest, required=True)
    for name, evaluator_cls in evaluator_map.items():
        subparser = subparsers.add_parser(
            name,
            formatter_class=DefaultsHelpFormatter,
        )
        evaluator_cls.add_args(subparser)
        subparser.set_defaults(evaluator_cls=evaluator_cls)
    return subparsers


def evaluator_from_args(args: argparse.Namespace) -> Evaluator[Any, Any, Any]:
    return args.evaluator_cls.from_args(args)


__all__ = [
    "EVALUATORS",
    "Evaluator",
    "EvaluatorClass",
    "EvalResultT",
    "EvalSummaryT",
    "DefaultsHelpFormatter",
    "InstanceT",
    "add_evaluator_subcommands",
    "BrowseCompEvalResult",
    "BrowseCompEvalSummary",
    "BrowseCompEvaluator",
    "BrowseCompInstance",
    "BrowseCompJudgeResult",
    "DeepSearchQAEvalResult",
    "DeepSearchQAEvalSummary",
    "DeepSearchQAEvaluator",
    "DeepSearchQAInstance",
    "evaluator_from_args",
    "FinSearchCompEvalResult",
    "FinSearchCompEvalSummary",
    "FinSearchCompEvaluator",
    "FinSearchCompInstance",
    "FrontierScienceEvalResult",
    "FrontierScienceResearchEvalSummary",
    "FrontierScienceEvalSummary",
    "FrontierScienceEvaluator",
    "FrontierScienceInstance",
    "HealthBenchProfessionalEvalResult",
    "HealthBenchProfessionalEvalSummary",
    "HealthBenchProfessionalEvaluator",
    "HealthBenchProfessionalInstance",
    "HealthBenchProfessionalJudgeResult",
    "HealthBenchProfessionalRubricGrade",
    "HealthBenchProfessionalRubricItem",
    "HLEEvalResult",
    "HLEEvalSummary",
    "HLEEvaluator",
    "HLEInstance",
    "HLEJudgeResult",
]
