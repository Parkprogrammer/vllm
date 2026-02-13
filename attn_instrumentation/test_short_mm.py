#!/usr/bin/env python3
"""Short multimodal test with normal response."""
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

MODEL = "gemma-3-4b-it"  # Use short name as returned by /v1/models
API_BASE = "http://127.0.0.1:8000/v1"
API_KEY = "EMPTY"
IMAGE = str(Path(__file__).with_name("sample_image.webp"))
KV_HOOK_CAPTURE = 1
KV_HOOK_LAYERS = "33"
MAX_TOKENS = 50  # Short response
TIMEOUT = 60.0


def img_data_url(path: Path) -> str:
    return f"data:image/webp;base64,{base64.b64encode(path.read_bytes()).decode()}"


def call_vllm(messages: list[dict]) -> dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,  # Non-zero for natural output
        "max_tokens": MAX_TOKENS,
        "kv_hook_capture": KV_HOOK_CAPTURE,
        "kv_hook_layers": KV_HOOK_LAYERS,
    }

    req = urllib.request.Request(
        url=API_BASE.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code} Error:\n{detail}", file=sys.stderr)
        raise


def main() -> int:
    image = Path(IMAGE)
    assert image.exists(), f"image not found: {image}"

    # Simple, natural prompt
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in 2-3 sentences."},
            {"type": "image_url", "image_url": {"url": img_data_url(image)}},
        ],
    }]

    print("Sending request...", file=sys.stderr)
    body = call_vllm(messages)

    # Extract response
    response_text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    kv_data = body.get("kv_hook_data") or []

    # Print results
    result = {
        "status": "ok",
        "request_id": body.get("id"),
        "model": MODEL,
        "prompt_tokens": body.get("usage", {}).get("prompt_tokens"),
        "completion_tokens": body.get("usage", {}).get("completion_tokens"),
        "response_text": response_text,
        "kv_hook_captured": len(kv_data) > 0,
        "kv_hook_layers": [item.get("layer_idx") for item in kv_data],
    }

    if kv_data:
        token_meta = kv_data[0].get("token_meta", {})
        result.update({
            "token_idx_basis": token_meta.get("token_idx_basis"),
            "window_offset": token_meta.get("window_offset_candidate"),
            "prompt_len": token_meta.get("prompt_len"),
            "total_len": token_meta.get("total_len"),
            "vision_ranges": token_meta.get("vision_ranges"),
        })

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
