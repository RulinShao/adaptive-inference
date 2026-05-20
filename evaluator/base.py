import argparse
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Generic, TypeVar


InstanceT = TypeVar("InstanceT")
EvalResultT = TypeVar("EvalResultT")
EvalSummaryT = TypeVar("EvalSummaryT")


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.IGNORECASE | re.DOTALL)
THINK_TAG_RE = re.compile(r"</?think>", flags=re.IGNORECASE)


def strip_think_blocks(value: Any) -> str:
    """Remove exposed reasoning blocks from model responses before judging."""
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    text = text.strip()
    if not text:
        return ""
    return THINK_TAG_RE.sub("", THINK_BLOCK_RE.sub("", text)).strip()


class Evaluator(ABC, Generic[InstanceT, EvalResultT, EvalSummaryT]):
    @abstractmethod
    def prepare_instances(self) -> dict[str, InstanceT]:
        """Load benchmark instances keyed by instance id."""

    @abstractmethod
    def evaluate(self, instance: InstanceT, response: str) -> EvalResultT:
        """Evaluate one model response for one dataset instance."""

    @abstractmethod
    def aggregate(self, results: Sequence[EvalResultT]) -> EvalSummaryT:
        """Aggregate per-instance evaluation results."""

    @classmethod
    @abstractmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """Register evaluator-specific CLI arguments on a dataset subparser."""

    @classmethod
    @abstractmethod
    def from_args(cls, args: argparse.Namespace) -> Any:
        """Construct this evaluator from a parsed driver namespace."""
