#!/usr/bin/env python3
"""Interleaved modality and capture toggling test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from common import RequestCase, ensure_server_alive, print_json_line, resolve_model_name, run_case


def main() -> int:
    health = ensure_server_alive()
    if not health.get("ok"):
        print_json_line({"status": "error", "stage": "server_check", **health})
        return 2

    model = resolve_model_name()
    image_path = Path(__file__).resolve().parents[1] / "sample_image.webp"

    seq = [
        RequestCase(name="step_01_text_l12", mode="text", layers="12"),
        RequestCase(name="step_02_mm_l12", mode="text_image", layers="12"),
        RequestCase(name="step_03_text_l23", mode="text", layers="23"),
        RequestCase(name="step_04_mm_l23", mode="text_image", layers="23"),
        RequestCase(name="step_05_text_capture_off", mode="text", capture=0, layers=None),
        RequestCase(name="step_06_mm_all", mode="text_image", layers="all", min_layers_for_all=2),
        RequestCase(name="step_07_text_all", mode="text", layers="all", min_layers_for_all=2),
        RequestCase(name="step_08_mm_l12_23", mode="text_image", layers="12,23", expected_layers={12, 23}),
        RequestCase(
            name="step_09_text_l12_prefix",
            mode="text",
            layers="12",
            vllm_xargs={"attn_capture_prefix": "8:88"},
        ),
        RequestCase(
            name="step_10_mm_l23_prefix",
            mode="text_image",
            layers="23",
            vllm_xargs={"attn_capture_prefix": "4:72"},
        ),
        RequestCase(
            name="step_11_image_only_soft",
            mode="image_only",
            layers="12",
            allow_http_error=True,
            allow_no_kv=True,
        ),
        RequestCase(name="step_12_text_l33", mode="text", layers="33"),
    ]

    seen_request_ids: set[str] = set()
    results: list[dict] = []
    hard_failures: list[dict] = []

    for case in seq:
        res = run_case(case, model=model, image_path=image_path)
        results.append(res)
        print_json_line({"type": "case_result", **res})

        if not res.get("ok", False):
            hard_failures.append(res)
            continue

        req_id = res.get("request_id")
        if isinstance(req_id, str):
            if req_id in seen_request_ids:
                hard_failures.append(
                    {
                        "name": case.name,
                        "ok": False,
                        "error": f"duplicate request_id detected: {req_id}",
                    }
                )
            seen_request_ids.add(req_id)

        # Modality-specific sanity checks using first token_meta diagnostics.
        vision_ranges_len = res.get("vision_ranges_len")
        if case.mode == "text" and case.capture == 1 and isinstance(vision_ranges_len, int):
            if vision_ranges_len != 0:
                hard_failures.append(
                    {
                        "name": case.name,
                        "ok": False,
                        "error": f"text-only request unexpectedly has vision ranges: {vision_ranges_len}",
                    }
                )
        if case.mode == "text_image" and case.capture == 1 and vision_ranges_len is None:
            hard_failures.append(
                {
                    "name": case.name,
                    "ok": False,
                    "error": "text+image request missing vision range diagnostics",
                }
            )

    summary = {
        "status": "ok" if not hard_failures else "failed",
        "suite": "interleaved_modalities",
        "model": model,
        "total": len(results),
        "passed": len(results) - len(hard_failures),
        "failed": len(hard_failures),
        "failed_cases": [x.get("name") for x in hard_failures],
        "unique_request_ids": len(seen_request_ids),
    }
    print_json_line({"type": "summary", **summary})

    if hard_failures:
        print(json.dumps({"status": "error", "failures": hard_failures}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

