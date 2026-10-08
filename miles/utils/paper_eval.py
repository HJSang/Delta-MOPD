"""Appendix A evaluation contract; Avg@K is NOT best-of-K/pass@K."""

from dataclasses import dataclass

import numpy as np

from miles.utils.eval_config import EvalDatasetConfig


@dataclass(frozen=True)
class PaperSuite:
    name: str
    items: int
    samples: int


PAPER_SUITES = (
    PaperSuite("amc2023", 40, 16),
    PaperSuite("math500", 500, 1),
    PaperSuite("aime2025", 30, 16),
    PaperSuite("gpqa_diamond", 198, 4),
    PaperSuite("ifeval", 541, 4),
)


def paper_eval_configs(paths: dict[str, str], rm_types: dict[str, str], *, protocol: str) -> list[EvalDatasetConfig]:
    """Build actual Miles eval configs; caller supplies frozen data paths/scorers.

    avg: acquisition/mechanism Avg@K; greedy: matched 16K greedy;
    scaling: the scaling/teacher-reference 10K greedy protocol.
    IFEval retains the evaluation driver's 2048-token cap in each protocol.
    """
    names = {suite.name for suite in PAPER_SUITES}
    if protocol not in ("avg", "greedy", "scaling") or set(paths) != names or set(rm_types) != names:
        raise ValueError("Specify the protocol and paths/verifiers for all five paper suites.")
    configs = []
    for suite in PAPER_SUITES:
        sampled = protocol == "avg" and suite.samples > 1
        cap = 10000 if protocol == "scaling" else 16384
        configs.append(
            EvalDatasetConfig(
                name=suite.name,
                path=paths[suite.name],
                rm_type=rm_types[suite.name],
                n_samples_per_eval_prompt=suite.samples if sampled else 1,
                temperature=1.0 if sampled else 0.0,
                top_p=0.95 if sampled else 1.0,
                top_k=-1,
                min_p=0.0,
                max_response_len=2048 if suite.name == "ifeval" else cap,
            )
        )
    return configs


def summarize_paper_eval(scores: dict[str, np.ndarray], *, protocol: str) -> dict[str, float]:
    """Mean binary verifier accuracy per item/sample, then unweighted five-suite mean."""
    if protocol not in ("avg", "greedy", "scaling") or set(scores) != {suite.name for suite in PAPER_SUITES}:
        raise ValueError("Incomplete or unknown five-suite evaluation protocol.")
    metrics = {}
    for suite in PAPER_SUITES:
        values = np.asarray(scores[suite.name], dtype=np.float64)
        k = suite.samples if protocol == "avg" else 1
        if values.shape != (suite.items, k) or not np.isin(values, [0, 1]).all():
            raise ValueError(
                f"{suite.name}: expected complete [{suite.items},{k}] binary outcomes; no silently dropped items."
            )
        metrics[suite.name] = float(values.mean())
    metrics["five_suite_macro"] = sum(metrics.values()) / len(PAPER_SUITES)
    return metrics
