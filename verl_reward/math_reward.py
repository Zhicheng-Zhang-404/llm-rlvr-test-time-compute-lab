"""Rule-based MATH reward loaded by VERL's custom reward hook."""

from __future__ import annotations

from typing import Any


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """Score mathematical equivalence; unparsable generations receive zero."""
    del data_source, extra_info
    try:
        from math_verify import LatexExtractionConfig, parse, verify

        gold = parse(ground_truth, extraction_mode="first_match")
        answer = parse(
            solution_str,
            extraction_config=[LatexExtractionConfig(boxed_match_priority=0)],
            extraction_mode="first_match",
        )
        if not gold or not answer:
            return 0.0
        return 1.0 if verify(gold, answer) else 0.0
    except Exception:
        return 0.0

