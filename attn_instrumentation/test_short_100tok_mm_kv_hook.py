#!/usr/bin/env python3
import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

MODEL = "google/gemma-3-4b-it"
API_BASE = "http://127.0.0.1:8000/v1"
API_KEY = "EMPTY"
IMAGE = str(Path(__file__).with_name("sample_image.webp"))
KV_HOOK_CAPTURE = 1
KV_HOOK_LAYERS = "33"
TARGET_INPUT_TOKENS = 100
MAX_TOKENS = 256
TIMEOUT = 120.0


def list_served_models() -> list[str]:
    url = API_BASE.rstrip("/") + "/models"
    req = urllib.request.Request(
        url=url,
        headers={"Authorization": f"Bearer {API_KEY}"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        body = json.loads(r.read().decode("utf-8"))
    return [
        item.get("id")
        for item in body.get("data", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]


def resolve_model_name(preferred: str) -> str:
    try:
        served = list_served_models()
    except Exception:
        return preferred
    if preferred in served:
        return preferred
    if served:
        return served[0]
    return preferred


def img_data_url(path: Path) -> str:
    b = path.read_bytes()
    return f"data:image/webp;base64,{base64.b64encode(b).decode()}"


def build_short_prompt(target: int = 100) -> str:
    chunks = [
        "Analyze this market photo using only visible evidence.",
        "Mention tents colors tables people path trees and shadows.",
        "Estimate crowd density and list three uncertain observations.",
        "Describe left center right layout with concise grounded cues.",
        "Do not invent unreadable text or hidden objects.",
        "Focus on object relations depth and occlusion patterns.",
        "Return brief factual findings with confidence notes.",
    ]
    words = []
    i = 0
    while len(words) < target:
        words.extend(chunks[i % len(chunks)].split())
        i += 1
    return " ".join(words[:target])


def post_with_urllib(payload: dict) -> dict:
    req = urllib.request.Request(
        url=API_BASE.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    image = Path(IMAGE)
    assert image.exists(), f"image not found: {image}"

    requested_model = os.getenv("VLLM_TEST_MODEL", MODEL)
    model = resolve_model_name(requested_model)

    prompt = build_short_prompt(TARGET_INPUT_TOKENS)
    image_url = img_data_url(image)

    extra_body = {"kv_hook_capture": KV_HOOK_CAPTURE}
    if KV_HOOK_LAYERS:
        extra_body["kv_hook_layers"] = KV_HOOK_LAYERS

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]

    body: dict
    content = ""
    transport = "openai"
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=API_KEY, base_url=API_BASE)
        resp = client.chat.completions.with_raw_response.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            extra_body=extra_body,
            timeout=TIMEOUT,
        )
        parsed = resp.parse()
        body = json.loads(resp.text)
        try:
            content = parsed.choices[0].message.content or ""
        except Exception:
            pass
    except Exception:
        transport = "urllib"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": MAX_TOKENS,
            "kv_hook_capture": KV_HOOK_CAPTURE,
        }
        if KV_HOOK_LAYERS:
            payload["kv_hook_layers"] = KV_HOOK_LAYERS
        try:
            body = post_with_urllib(payload)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            if e.code == 404:
                served = list_served_models()
                if served and model != served[0]:
                    payload["model"] = served[0]
                    model = served[0]
                    body = post_with_urllib(payload)
                else:
                    raise RuntimeError(f"HTTP {e.code}: {detail}") from e
            else:
                raise RuntimeError(f"HTTP {e.code}: {detail}") from e

    usage = body.get("usage", {})
    kv = body.get("kv_hook_data")
    diagnostics: dict[str, Any] = {}
    if kv and isinstance(kv, list):
        token_meta = (kv[0] or {}).get("token_meta") if isinstance(kv[0], dict) else None
        if isinstance(token_meta, dict):
            token_idx = token_meta.get("token_idx") or []
            if isinstance(token_idx, list) and token_idx and all(
                isinstance(v, int) for v in token_idx
            ):
            diagnostics["prompt_len"] = token_meta.get("prompt_len")
            diagnostics["offset_candidate"] = token_meta.get("window_offset_candidate")
            diagnostics["boundary_local"] = token_meta.get("prompt_boundary_local")
            diagnostics["boundary_with_offset_candidate"] = token_meta.get(
                "prompt_boundary_with_offset_candidate"
            )

    out = {
        "transport": transport,
        "model": model,
        "request_id": body.get("id"),
        "input_word_tokens": len(prompt.split()),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "kv_hook_data": bool(kv),
        "kv_items": 0 if not kv else len(kv),
        "diagnostics": diagnostics,
        "preview": content[:180].replace("\n", " "),
    }
    if kv:
        k0 = kv[0]
        out["first"] = {
            "layer_idx": k0.get("layer_idx"),
            "shape": k0.get("shape"),
            "dtype": k0.get("dtype"),
        }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
