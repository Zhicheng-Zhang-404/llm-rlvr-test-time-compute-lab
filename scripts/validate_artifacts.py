#!/usr/bin/env python3
"""Validate generated artifacts before accepting an experiment checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttc_rlvr.io_utils import read_jsonl


def validate_base(directory: Path) -> None:
    generated = read_jsonl(directory / "generations.jsonl")
    scored = read_jsonl(directory / "scored.jsonl")
    if len(generated) != 100 or len(scored) != 100:
        raise ValueError(f"expected 100 generated/scored rows, got {len(generated)}/{len(scored)}")
    if {row["id"] for row in generated} != set(range(100)):
        raise ValueError("generation IDs must be exactly 0..99")
    for row in generated:
        if len(row.get("outputs", [])) != 16:
            raise ValueError(f"problem {row.get('id')} does not contain 16 outputs")
    for row in scored:
        correct = row.get("correct", [])
        if len(correct) != 16 or not all(isinstance(value, bool) for value in correct):
            raise ValueError(f"problem {row.get('id')} has invalid correctness flags")

    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    expected_ks = [1, 2, 4, 8, 16]
    values = [float(metrics["pass_at_k"][str(k)]) for k in expected_ks]
    if values != sorted(values) or not all(0 <= value <= 1 for value in values):
        raise ValueError(f"pass@k must be monotone and within [0,1], got {values}")
    if not (directory / "pass_at_k.png").stat().st_size:
        raise ValueError("plot is empty")
    print(f"CHECKPOINT PASSED: base artifacts valid; pass@k={dict(zip(expected_ks, values))}")


def validate_grpo_log(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    markers = ("step:0", "step: 0", "global_step")
    if not any(marker in text for marker in markers):
        raise ValueError("log contains no confirmed training step")
    failure_markers = ("Traceback (most recent call last)", "CUDA out of memory", "RayTaskError")
    failures = [marker for marker in failure_markers if marker in text]
    if failures:
        raise ValueError(f"log contains failure markers: {failures}")
    print("CHECKPOINT PASSED: GRPO log contains a training step and no known fatal marker")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="kind", required=True)
    base = subparsers.add_parser("base")
    base.add_argument("--dir", default="artifacts/base", type=Path)
    grpo = subparsers.add_parser("grpo-log")
    grpo.add_argument("--path", default="artifacts/grpo_smoke.log", type=Path)
    args = parser.parse_args()
    if args.kind == "base":
        validate_base(args.dir)
    else:
        validate_grpo_log(args.path)


if __name__ == "__main__":
    main()

