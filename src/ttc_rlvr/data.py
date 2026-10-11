"""MATH-500 loading and prompt construction."""

from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = """You are a careful mathematical problem solver. Work through the problem step by step. End with exactly one final answer in the form \\boxed{...}."""


def load_math500(limit: int = 100, split: str = "test") -> list[dict[str, Any]]:
    """Load a deterministic prefix of HuggingFaceH4/MATH-500."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies before loading MATH-500") from exc

    dataset = load_dataset("HuggingFaceH4/MATH-500", split=split)
    limit = min(limit, len(dataset))
    rows: list[dict[str, Any]] = []
    for index in range(limit):
        item = dict(dataset[index])
        problem = item.get("problem")
        answer = item.get("answer") or item.get("solution")
        if not isinstance(problem, str) or not isinstance(answer, str):
            raise ValueError(f"row {index} is missing string problem/answer fields")
        rows.append(
            {
                "id": index,
                "problem": problem,
                "answer": answer,
                "subject": item.get("subject", "unknown"),
                "level": item.get("level", "unknown"),
            }
        )
    return rows


def chat_messages(problem: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]

