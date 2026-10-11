#!/usr/bin/env python3
"""Convert MATH-500 to VERL parquet while keeping the first 100 held out."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, load_dataset

from ttc_rlvr.data import SYSTEM_PROMPT


def convert(row: dict, index: int, split: str) -> dict:
    return {
        "data_source": "math500",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["problem"]},
        ],
        "ability": row.get("subject", "math"),
        "reward_model": {"style": "rule", "ground_truth": row["answer"]},
        "extra_info": {
            "index": index,
            "split": split,
            "level": row.get("level", "unknown"),
            "subject": row.get("subject", "unknown"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/verl_data")
    parser.add_argument("--eval-size", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.eval_size < 500:
        raise ValueError("eval-size must be between 1 and 499")

    raw = load_dataset("HuggingFaceH4/MATH-500", split="test")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    validation = [convert(dict(raw[i]), i, "validation") for i in range(args.eval_size)]
    training = [convert(dict(raw[i]), i, "train") for i in range(args.eval_size, len(raw))]
    Dataset.from_list(training).to_parquet(output / "train.parquet")
    Dataset.from_list(validation).to_parquet(output / "validation.parquet")
    print(f"Wrote {len(training)} train rows and {len(validation)} held-out validation rows to {output}")
    print("CHECKPOINT PASSED: VERL parquet schema generated")


if __name__ == "__main__":
    main()

