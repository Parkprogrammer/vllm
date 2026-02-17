#!/usr/bin/env python3
"""Layer selection focused tests: single, multi, all, and default behavior."""

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

    cases = [
        RequestCase(name="single_layer_12", mode="text", layers="12", expected_layers={12}),
        RequestCase(name="single_layer_23", mode="text", layers="23", expected_layers={23}),
        RequestCase(
            name="multi_layer_12_23",
            mode="text",
            layers="12,23",
            expected_layers={12, 23},
            require_all_expected_layers=True,
        ),
        RequestCase(
            name="multi_layer_with_spaces",
            mode="text",
            layers=" 12 , 23 ",
            expected_layers={12, 23},
            require_all_expected_layers=True,
        ),
        RequestCase(name="all_layers", mode="text", layers="all", min_layers_for_all=2),
        # When layers is omitted and server is configured with "all", it should still capture.
        RequestCase(name="default_layers_from_server", mode="text", layers=None),
        RequestCase(name="capture_off_with_layers_specified", mode="text", capture=0, layers="12,23"),
    ]

    results: list[dict] = []
    hard_failures: list[dict] = []
    for case in cases:
        res = run_case(case, model=model, image_path=image_path)
        results.append(res)
        print_json_line({"type": "case_result", **res})
        if not res.get("ok", False):
            hard_failures.append(res)

    summary = {
        "status": "ok" if not hard_failures else "failed",
        "suite": "layer_selection",
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

