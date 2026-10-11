"""Math-Verify adapter with explicit parse failure handling."""

from __future__ import annotations

from typing import Any


def verify_answer(gold: str, response: str) -> bool:
    """Return whether ``response`` is mathematically equivalent to ``gold``.

    Parsing errors and answers that cannot be extracted count as incorrect.
    """
    try:
        from math_verify import LatexExtractionConfig, parse, verify
    except ImportError as exc:
        raise RuntimeError("math-verify is required for answer verification") from exc

    try:
        gold_parsed = parse(gold, extraction_mode="first_match")
        answer_parsed = parse(
            response,
            extraction_config=[LatexExtractionConfig(boxed_match_priority=0)],
            extraction_mode="first_match",
        )
        if not gold_parsed or not answer_parsed:
            return False
        return bool(verify(gold_parsed, answer_parsed))
    except Exception:
        # A malformed model completion is an evaluation miss, not a crashed run.
        return False


def score_record(record: dict[str, Any]) -> dict[str, Any]:
    outputs = record.get("outputs")
    if not isinstance(outputs, list) or not all(isinstance(output, str) for output in outputs):
        raise ValueError("record.outputs must be a list of strings")
    gold = record.get("answer")
    if not isinstance(gold, str):
        raise ValueError("record.answer must be a string")
    correctness = [verify_answer(gold, output) for output in outputs]
    return {**record, "correct": correctness, "num_correct": sum(correctness)}

