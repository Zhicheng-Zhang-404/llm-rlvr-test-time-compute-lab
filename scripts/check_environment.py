#!/usr/bin/env python3
"""Fail-fast environment gate used before expensive GPU work."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import subprocess
import sys
from typing import Any


def module_version(name: str) -> str:
    module = importlib.import_module(name)
    return str(getattr(module, "__version__", "installed"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-verl", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    required = ["torch", "vllm", "transformers", "datasets", "math_verify", "matplotlib", "numpy"]
    if args.require_verl:
        required.append("verl")
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "modules": {},
    }
    failures: list[str] = []
    for name in required:
        try:
            report["modules"][name] = module_version(name)
        except Exception as exc:
            report["modules"][name] = f"ERROR: {type(exc).__name__}: {exc}"
            failures.append(f"cannot import {name}")

    try:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_device_count"] = torch.cuda.device_count()
        report["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if args.require_gpu and not torch.cuda.is_available():
            failures.append("CUDA is required but unavailable")
    except Exception:
        report["cuda_available"] = False

    if shutil.which("nvidia-smi"):
        command = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
        report["nvidia_smi"] = subprocess.run(command, text=True, capture_output=True, check=False).stdout.strip()
    else:
        report["nvidia_smi"] = "not found"

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        print("FAILED: " + "; ".join(failures), file=sys.stderr)
        raise SystemExit(1)
    print("CHECKPOINT PASSED: environment is ready")


if __name__ == "__main__":
    main()

