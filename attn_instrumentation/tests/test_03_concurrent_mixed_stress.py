#!/usr/bin/env python3
"""Concurrent mixed stress test for KV hook capture and layer routing."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import RequestCase, ensure_server_alive, print_json_line, resolve_model_name, run_case


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Concurrent mixed stress test.")
    p.add_argument("--num-requests", type=int, default=24)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_case(i: int, rng: random.Random) -> RequestCase:
    mode = rng.choice(["text", "text", "text_image", "image_only"])
    capture = 1 if rng.random() < 0.85 else 0
    layers = None
    if capture:
        layers = rng.choice(["12", "23", "33", "all", "12,23"])

    vllm_xargs = None
    if capture and rng.random() < 0.5:
        vllm_xargs = {"attn_capture_prefix": rng.choice(["0:48", "8:72", "16:96"])}

    allow_http_error = mode == "image_only"
    allow_no_kv = mode == "image_only"
    return RequestCase(
        name=f"concurrent_{i:03d}",
        mode=mode,
        capture=capture,
        layers=layers,
        vllm_xargs=vllm_xargs,
        max_tokens=rng.choice([24, 32, 40, 48]),
        allow_http_error=allow_http_error,
        allow_no_kv=allow_no_kv,
        meta={"idx": i},
    )


def main() -> int:
    args = parse_args()
    health = ensure_server_alive()
    if not health.get("ok"):
        print_json_line({"status": "error", "stage": "server_check", **health})
        return 2

    rng = random.Random(args.seed)
    model = resolve_model_name()
    image_path = Path(__file__).resolve().parents[1] / "sample_image.webp"

    cases = [build_case(i, rng) for i in range(args.num_requests)]
    started = time.time()
    results: list[dict] = []
    hard_failures: list[dict] = []

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        future_map = {
            ex.submit(run_case, case, model, image_path): case
            for case in cases
        }
        for fut in as_completed(future_map):
            case = future_map[fut]
            try:
                res = fut.result()
            except Exception as e:  # defensive; run_case normally catches
                res = {
                    "name": case.name,
                    "mode": case.mode,
                    "capture": case.capture,
                    "layers": case.layers,
                    "ok": False,
                    "error": str(e),
                }
            results.append(res)
            print_json_line({"type": "case_result", **res})
            if not res.get("ok", False):
                hard_failures.append(res)

    elapsed = round(time.time() - started, 3)
    by_mode: dict[str, int] = {}
    by_capture: dict[str, int] = {}
    for res in results:
        by_mode[res.get("mode", "unknown")] = by_mode.get(res.get("mode", "unknown"), 0) + 1
        cap_key = str(res.get("capture"))
        by_capture[cap_key] = by_capture.get(cap_key, 0) + 1

    summary = {
        "status": "ok" if not hard_failures else "failed",
        "suite": "concurrent_mixed_stress",
        "model": model,
        "workers": args.workers,
        "num_requests": args.num_requests,
        "elapsed_sec": elapsed,
        "total": len(results),
        "passed": len(results) - len(hard_failures),
        "failed": len(hard_failures),
        "by_mode": by_mode,
        "by_capture": by_capture,
        "failed_cases": [x.get("name") for x in hard_failures],
    }
    print_json_line({"type": "summary", **summary})

    if hard_failures:
        print(json.dumps({"status": "error", "failures": hard_failures}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
