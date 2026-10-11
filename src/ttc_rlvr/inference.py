"""Batched, resumable vLLM generation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .data import chat_messages
from .io_utils import append_jsonl, read_jsonl


def _completed_ids(path: Path, expected_samples: int) -> set[int]:
    if not path.exists():
        return set()
    done: set[int] = set()
    for row in read_jsonl(path):
        outputs = row.get("outputs")
        if isinstance(row.get("id"), int) and isinstance(outputs, list) and len(outputs) == expected_samples:
            done.add(row["id"])
    return done


def generate_with_vllm(
    problems: Sequence[dict[str, Any]],
    output_path: str | Path,
    *,
    model: str = "Qwen/Qwen3-0.6B",
    samples: int = 16,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 2048,
    tensor_parallel_size: int = 1,
    batch_size: int = 25,
    seed: int = 42,
) -> None:
    """Generate samples, writing one problem per JSONL line for safe resume."""
    if samples <= 0 or batch_size <= 0 or tensor_parallel_size <= 0:
        raise ValueError("samples, batch_size and tensor_parallel_size must be positive")
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("vLLM and transformers are required for generation") from exc

    destination = Path(output_path)
    completed = _completed_ids(destination, samples)
    pending = [problem for problem in problems if problem["id"] not in completed]
    if not pending:
        print(f"All {len(problems)} problems already exist in {destination}")
        return

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        seed=seed,
    )
    sampling = SamplingParams(
        n=samples,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        seed=seed,
    )

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                chat_messages(item["problem"]),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for item in batch
        ]
        generated = llm.generate(prompts, sampling)
        records = []
        for item, request_output in zip(batch, generated, strict=True):
            outputs = [candidate.text for candidate in request_output.outputs]
            if len(outputs) != samples:
                raise RuntimeError(f"problem {item['id']}: expected {samples} outputs, got {len(outputs)}")
            records.append({**item, "model": model, "outputs": outputs})
        append_jsonl(destination, records)
        print(f"Saved {min(start + batch_size, len(pending))}/{len(pending)} pending problems")

