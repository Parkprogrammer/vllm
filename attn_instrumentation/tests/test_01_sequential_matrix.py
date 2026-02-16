#!/usr/bin/env python3
"""Sequential matrix test for mixed request modalities and layer routing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from common import RequestCase, ensure_server_alive, print_json_line, resolve_model_name, run_case


def build_cases() -> list[RequestCase]:
    return [
        RequestCase(name="text_l12", mode="text", layers="12"),
        RequestCase(name="text_l23", mode="text", layers="23"),
        RequestCase(name="text_l33", mode="text", layers="33"),
        RequestCase(name="text_all_layers", mode="text", layers="all", min_layers_for_all=2),
        RequestCase(
            name="text_l12_with_prefix",
            mode="text",
            layers="12",
            vllm_xargs={"attn_capture_prefix": "0:64"},
        ),
        RequestCase(name="text_image_l12", mode="text_image", layers="12"),
        RequestCase(name="text_image_l23", mode="text_image", layers="23"),
        RequestCase(name="text_image_all", mode="text_image", layers="all", min_layers_for_all=2),
        RequestCase(name="text_capture_off", mode="text", capture=0, layers=None),
        # Some model/chat-template combinations may reject image-only prompts.
        RequestCase(
            name="image_only_l12_soft",
            mode="image_only",
            layers="12",
            allow_http_error=True,
            allow_no_kv=True,
        ),
    ]


def main() -> int:
    health = ensure_server_alive()
    if not health.get("ok"):
        print_json_line({"status": "error", "stage": "server_check", **health})
        return 2

    model = resolve_model_name()
    image_path = Path(__file__).resolve().parents[1] / "sample_image.webp"
    results: list[dict] = []
    hard_failures: list[dict] = []

    for case in build_cases():
        res = run_case(case, model=model, image_path=image_path)
        results.append(res)
        print_json_line({"type": "case_result", **res})
        if not res.get("ok", False):
            hard_failures.append(res)

    summary = {
        "status": "ok" if not hard_failures else "failed",
        "suite": "sequential_matrix",
        "model": model,
        "total": len(results),
        "passed": len(results) - len(hard_failures),
        "failed": len(hard_failures),
        "failed_cases": [x.get("name") for x in hard_failures],
    }
    print_json_line({"type": "summary", **summary})

    if hard_failures:
        print(json.dumps({"status": "error", "failures": hard_failures}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

