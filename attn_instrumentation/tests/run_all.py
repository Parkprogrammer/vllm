#!/usr/bin/env python3
"""Run all attn instrumentation stress tests."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

SUITES = [
    "test_01_sequential_matrix.py",
    "test_02_layer_selection.py",
    "test_03_concurrent_mixed_stress.py",
    "test_04_interleaved_modalities.py",
]


def main() -> int:
    root = Path(__file__).resolve().parent
    py = sys.executable
    started = time.time()
    results: list[dict] = []
    failed = 0

    for suite in SUITES:
        path = root / suite
        ts = time.time()
        proc = subprocess.run(
            [py, str(path)],
            cwd=str(root),
            check=False,
            text=True,
            capture_output=True,
        )
        elapsed = round(time.time() - ts, 3)
        ok = proc.returncode == 0
        if not ok:
            failed += 1
        results.append(
            {
                "suite": suite,
                "ok": ok,
                "returncode": proc.returncode,
                "elapsed_sec": elapsed,
            }
        )
        # Pass through suite output for live inspection.
        if proc.stdout:
            print(proc.stdout.rstrip())
        if proc.stderr:
            print(proc.stderr.rstrip(), file=sys.stderr)
        print(
            json.dumps(
                {
                    "type": "suite_summary",
                    "suite": suite,
                    "ok": ok,
                    "returncode": proc.returncode,
                    "elapsed_sec": elapsed,
                },
                ensure_ascii=False,
            )
        )

    out = {
        "status": "ok" if failed == 0 else "failed",
        "total_suites": len(SUITES),
        "failed_suites": failed,
        "elapsed_sec": round(time.time() - started, 3),
        "results": results,
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

