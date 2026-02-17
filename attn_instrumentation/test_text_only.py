#!/usr/bin/env python3
"""Simple text-only test to verify KV hook works correctly."""
import json
import sys
import urllib.request

MODEL = "gemma-3-4b-it"
API_BASE = "http://127.0.0.1:8000/v1"
API_KEY = "EMPTY"
KV_HOOK_CAPTURE = 1
KV_HOOK_LAYERS = "33"
MAX_TOKENS = 20
TIMEOUT = 60.0


def call_vllm(messages: list[dict]) -> dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
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
    # Simple text-only prompt
    messages = [{
        "role": "user",
        "content": "What is 2+2? Answer briefly.",
    }]

    print("Sending text-only request...", file=sys.stderr)
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
        "response_length": len(response_text),
        "kv_hook_captured": len(kv_data) > 0,
        "kv_hook_layers": [item.get("layer_idx") for item in kv_data],
    }

    if kv_data:
        token_meta = kv_data[0].get("token_meta", {})
        result.update({
            "token_idx_basis": token_meta.get("token_idx_basis"),
            "window_offset": token_meta.get("win_offset"),
            "prompt_len": token_meta.get("prompt_len"),
            "total_len": token_meta.get("total_len"),
            "vision_ranges": token_meta.get("vision_ranges"),
            "lang_ranges": token_meta.get("lang_ranges"),
        })

    print(json.dumps(result, ensure_ascii=False, indent=2))

    # Verdict
    if len(kv_data) > 0:
        print("\n✅ KV hook is working correctly!", file=sys.stderr)
    else:
        print("\n❌ KV hook data missing!", file=sys.stderr)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
