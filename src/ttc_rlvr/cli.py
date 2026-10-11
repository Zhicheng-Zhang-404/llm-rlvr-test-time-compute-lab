"""Command-line entry points for the base experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_math500
from .evaluate import aggregate, plot_metrics, write_metrics
from .inference import generate_with_vllm
from .io_utils import append_jsonl, read_jsonl
from .verification import score_record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ttc-eval")
    subcommands = parser.add_subparsers(dest="command", required=True)

    generate = subcommands.add_parser("generate", help="sample MATH-500 responses with vLLM")
    generate.add_argument("--output", default="artifacts/base/generations.jsonl")
    generate.add_argument("--model", default="Qwen/Qwen3-0.6B")
    generate.add_argument("--limit", type=int, default=100)
    generate.add_argument("--samples", type=int, default=16)
    generate.add_argument("--temperature", type=float, default=0.6)
    generate.add_argument("--top-p", type=float, default=0.95)
    generate.add_argument("--top-k", type=int, default=20)
    generate.add_argument("--max-tokens", type=int, default=2048)
    generate.add_argument("--tensor-parallel-size", type=int, default=1)
    generate.add_argument("--batch-size", type=int, default=25)
    generate.add_argument("--seed", type=int, default=42)

    score = subcommands.add_parser("score", help="verify generated answers")
    score.add_argument("--input", default="artifacts/base/generations.jsonl")
    score.add_argument("--output", default="artifacts/base/scored.jsonl")

    report = subcommands.add_parser("report", help="aggregate pass@k and make a plot")
    report.add_argument("--input", default="artifacts/base/scored.jsonl")
    report.add_argument("--metrics", default="artifacts/base/metrics.json")
    report.add_argument("--plot", default="artifacts/base/pass_at_k.png")
    report.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    report.add_argument("--by-level", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "generate":
        generate_with_vllm(
            load_math500(args.limit),
            args.output,
            model=args.model,
            samples=args.samples,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    elif args.command == "score":
        destination = Path(args.output)
        if destination.exists():
            destination.unlink()
        append_jsonl(destination, (score_record(record) for record in read_jsonl(args.input)))
    elif args.command == "report":
        metrics = aggregate(read_jsonl(args.input), args.ks)
        write_metrics(metrics, args.metrics)
        plot_metrics(metrics, args.plot, include_levels=args.by_level)
        print(f"pass@k: {metrics['pass_at_k']}")


if __name__ == "__main__":
    main()

