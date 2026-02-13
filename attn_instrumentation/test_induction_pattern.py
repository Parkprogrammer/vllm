#!/usr/bin/env python3
"""Test induction head behavior with repeated patterns (Transformer Circuits style).

This test uses a short text-only prompt with repeated sequences to check
if the model exhibits induction head behavior: when it sees "A B ... A",
does it attend strongly to the B that followed the previous A?
"""
import json
import sys
import urllib.request
from pathlib import Path

MODEL = "gemma-3-4b-it"
API_BASE = "http://127.0.0.1:8000/v1"
API_KEY = "EMPTY"
KV_HOOK_CAPTURE = 1
KV_HOOK_LAYERS = "33"  # Check one layer
MAX_TOKENS = 100  # Short generation
TIMEOUT = 60.0

# Classic induction prompt: repeated sequences
# Pattern: "The cat sat. The dog ran. The bird flew. The cat"
# When model generates after "The cat" the second time, it should attend to "sat"
INDUCTION_PROMPT = """The cat sat on the mat. The dog ran in the yard. The bird flew in the sky. The fish swam in the pond. The cat"""


def call_vllm(prompt: str) -> dict:
    """Call vLLM API with text-only prompt."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,  # Deterministic for clear patterns
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
    print("=" * 80, file=sys.stderr)
    print("Induction Head Test - Transformer Circuits Style", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"\nPrompt:\n{INDUCTION_PROMPT}\n", file=sys.stderr)
    print("Expected: Model should attend to 'sat on the mat' when continuing 'The cat'", file=sys.stderr)
    print("-" * 80, file=sys.stderr)

    # Make request
    print("\nSending request...", file=sys.stderr)
    body = call_vllm(INDUCTION_PROMPT)

    # Extract response
    response_text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    kv_data = body.get("kv_hook_data") or []
    request_id = body.get("id")

    print(f"\nResponse: {response_text}\n", file=sys.stderr)

    # Save artifacts
    artifact_dir = Path(__file__).parent / "artifacts" / "induction" / request_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Save raw response
    (artifact_dir / "raw_response.json").write_text(
        json.dumps(body, ensure_ascii=False, indent=2)
    )

    # Save context info
    context = {
        "mode": "induction_test",
        "model": MODEL,
        "layer": int(KV_HOOK_LAYERS),
        "prompt_text": INDUCTION_PROMPT,
        "response_text": response_text,
        "request_id": request_id,
    }
    (artifact_dir / "request_context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2)
    )

    # Print summary
    result = {
        "status": "ok",
        "request_id": request_id,
        "model": MODEL,
        "layer": int(KV_HOOK_LAYERS),
        "prompt_length": len(INDUCTION_PROMPT),
        "response_length": len(response_text),
        "prompt_tokens": body.get("usage", {}).get("prompt_tokens"),
        "completion_tokens": body.get("usage", {}).get("completion_tokens"),
        "kv_hook_captured": len(kv_data) > 0,
        "artifact_dir": str(artifact_dir),
    }

    if kv_data:
        token_meta = kv_data[0].get("token_meta", {})
        result.update({
            "token_idx_basis": token_meta.get("token_idx_basis"),
            "prompt_len": token_meta.get("prompt_len"),
            "total_len": token_meta.get("total_len"),
        })

    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n" + "=" * 80, file=sys.stderr)
    print(f"Artifacts saved to: {artifact_dir}", file=sys.stderr)
    print("Next step: Run test_visualization_by_sentence.py on this artifact", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
