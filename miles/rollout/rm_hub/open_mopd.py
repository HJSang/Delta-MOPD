"""Adapter for the Open-MOPD/verl evaluation score functions."""

from __future__ import annotations

import importlib.util
import os
import re
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

_REWARD_SCORE_SUBDIR = Path("training/verl/verl/utils/reward_score")
_TERMINAL_CONTROL_TOKENS = re.compile(r"(?:(?:<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>)\s*)+$")


def clean_verifier_response(response: str) -> str:
    """Remove terminal transport markers only; preserve content/think tags and raw traces."""
    return _TERMINAL_CONTROL_TOKENS.sub("", response)


def _candidate_roots() -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get("OPEN_MOPD_REWARD_SCORE_ROOT")
    if configured:
        candidates.append(Path(configured))

    current = Path(__file__).resolve()
    for parent in (current, *current.parents):
        candidates.append(parent / "Open-MOPD" / _REWARD_SCORE_SUBDIR)
        candidates.append(parent / _REWARD_SCORE_SUBDIR)
    candidates.extend(
        [
            Path("/persistent/Open-MOPD") / _REWARD_SCORE_SUBDIR,
            Path("/root/Open-MOPD") / _REWARD_SCORE_SUBDIR,
        ]
    )
    return candidates


def _reward_score_root() -> Path:
    for candidate in _candidate_roots():
        if (candidate / "math_dapo.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Open-MOPD reward scorers were not found. Set "
        "OPEN_MOPD_REWARD_SCORE_ROOT to Open-MOPD/training/verl/verl/utils/reward_score."
    )


@lru_cache(maxsize=8)
def _load_scorer(root: str, filename: str) -> ModuleType:
    path = Path(root) / filename
    if not path.is_file():
        raise FileNotFoundError(f"Open-MOPD scorer {path} does not exist.")
    module_name = f"miles_open_mopd_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Open-MOPD scorer from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _family_for_data_source(data_source: str) -> str:
    source = str(data_source).lower()
    if source in {"math", "math_dapo", "math_dapo_reasoning", "math_dapo_boxed"} or source.startswith("aime"):
        return "math"
    if source in {
        "codecontests",
        "apps",
        "codeforces",
        "taco",
        "primeintellect",
        "humaneval",
        "humanevalplus",
    } or source.startswith("livecodebench"):
        return "code"
    if "ifeval" in source or source in {"nemotron_if", "nemotron_if_rl", "ifbench", "instruction_following"}:
        return "if"
    raise NotImplementedError(f"No Open-MOPD scorer is registered for data_source={data_source!r}.")


def compute_open_mopd_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    scorer: str | None = None,
    **reward_kwargs: Any,
) -> dict[str, Any]:
    """Call an Open-MOPD scorer and preserve its structured result."""

    root = _reward_score_root()
    solution_str = clean_verifier_response(solution_str)
    family = scorer or _family_for_data_source(data_source)
    extra_info = extra_info or {}

    if family == "math":
        module = _load_scorer(str(root), "math_dapo.py")
        # The boxed math datasets ask the model to finish with ``\boxed{...}``.
        # Open-MOPD's historical default parser instead looks for a literal
        # ``Answer:`` line, which turns otherwise valid boxed responses into
        # ``[INVALID]`` and makes the math eval score -1 for every sample.
        # Keep an explicit caller setting authoritative, but select the
        # dataset's native boxed verifier when no override was supplied.
        strict_box_verify = reward_kwargs.get("strict_box_verify")
        if strict_box_verify is None:
            strict_box_verify = str(data_source).lower() == "math_dapo_boxed"
        return module.compute_score(
            solution_str,
            str(ground_truth or ""),
            strict_box_verify=bool(strict_box_verify),
        )
    if family == "code":
        module = _load_scorer(str(root), "rllm_code_reward.py")
        return module.compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **reward_kwargs,
        )
    if family == "if":
        module = _load_scorer(root=str(root), filename="instruction_following.py")
        return module.compute_score(
            solution_str=solution_str,
            ground_truth=str(ground_truth or ""),
            extra_info=extra_info,
            data_source=data_source,
            **reward_kwargs,
        )
    raise ValueError(f"Unknown Open-MOPD scorer family: {family!r}.")


async def open_mopd_rm(
    args: Any,
    sample: Any,
    scorer: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Miles RM hook using the sample's dataset metadata and label."""

    del args
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    data_source = metadata.get("data_source") or metadata.get("dataset")
    if not data_source:
        raise ValueError("Open-MOPD evaluation samples require metadata.data_source.")
    ground_truth = sample.label if sample.label is not None else metadata.get("ground_truth")
    # ``async_rm`` supplies the scorer selected from ``rm_type``.  Keep the
    # metadata value as a fallback, and remove any duplicated scorer forwarded
    # through generic RM kwargs before calling the concrete scorer.
    scorer = scorer or metadata.get("open_mopd_scorer") or kwargs.pop("scorer", None)
    if "strict_box_verify" not in kwargs and "strict_box_verify" in metadata:
        kwargs["strict_box_verify"] = metadata["strict_box_verify"]
    return compute_open_mopd_score(
        data_source=str(data_source),
        solution_str=sample.response,
        ground_truth=ground_truth,
        extra_info=metadata,
        scorer=scorer,
        **kwargs,
    )
