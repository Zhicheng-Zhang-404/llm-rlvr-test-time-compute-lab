"""Aggregate and plot pass@k results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .pass_at_k import mean_pass_at_k


def aggregate(records: Sequence[dict[str, Any]], ks: Sequence[int]) -> dict[str, Any]:
    if not records:
        raise ValueError("records cannot be empty")
    correctness = [record["correct"] for record in records]
    overall = mean_pass_at_k(correctness, ks)
    by_level: dict[str, dict[str, Any]] = {}
    levels = sorted({str(record.get("level", "unknown")) for record in records})
    for level in levels:
        group = [record["correct"] for record in records if str(record.get("level", "unknown")) == level]
        by_level[level] = {"count": len(group), "pass_at_k": mean_pass_at_k(group, ks)}
    return {
        "num_problems": len(records),
        "samples_per_problem": len(correctness[0]),
        "pass_at_k": overall,
        "by_level": by_level,
    }


def write_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def plot_metrics(metrics: dict[str, Any], path: str | Path, include_levels: bool = False) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    overall = metrics["pass_at_k"]
    ks = sorted(int(key) for key in overall)
    values = [float(overall.get(str(k), overall.get(k))) for k in ks]
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.plot(ks, values, marker="o", linewidth=2.2, label="Overall")
    if include_levels:
        for level, group in metrics["by_level"].items():
            level_metrics = group["pass_at_k"]
            axis.plot(
                ks,
                [float(level_metrics.get(str(k), level_metrics.get(k))) for k in ks],
                marker=".",
                alpha=0.72,
                label=f"{level} (n={group['count']})",
            )
    axis.set_xscale("log", base=2)
    axis.set_xticks(ks, labels=[str(k) for k in ks])
    axis.set_ylim(0, 1.0)
    axis.set_xlabel("Number of samples k")
    axis.set_ylabel("pass@k")
    axis.set_title("MATH-500: test-time compute scaling")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)

