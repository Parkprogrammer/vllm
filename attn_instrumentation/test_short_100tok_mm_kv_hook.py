#!/usr/bin/env python3
import base64
import json
from pathlib import Path

from openai import OpenAI


MODEL = "google/gemma-3-4b-it"
API_BASE = "http://127.0.0.1:8000/v1"
API_KEY = "EMPTY"
IMAGE = "/home/parkprogrammer/jehyun/vllm/attn_instrumentation/sample_image.webp"
KV_HOOK_CAPTURE = 1
KV_HOOK_LAYERS = "33"
TARGET_INPUT_TOKENS = 100
MAX_TOKENS = 256
TIMEOUT = 120.0


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


def main():
    image = Path(IMAGE)
    assert image.exists(), f"image not found: {image}"

    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    prompt = build_short_prompt(TARGET_INPUT_TOKENS)
    image_url = img_data_url(image)

    extra_body = {"kv_hook_capture": KV_HOOK_CAPTURE}
    if KV_HOOK_LAYERS:
        extra_body["kv_hook_layers"] = KV_HOOK_LAYERS

    resp = client.chat.completions.with_raw_response.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        extra_body=extra_body,
        timeout=TIMEOUT,
    )

    parsed = resp.parse()
    body = json.loads(resp.text)
    usage = body.get("usage", {})
    kv = body.get("kv_hook_data")

    content = ""
    try:
        content = parsed.choices[0].message.content or ""
    except Exception:
        pass

    out = {
        "request_id": body.get("id"),
        "input_word_tokens": len(prompt.split()),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "kv_hook_data": bool(kv),
        "kv_items": 0 if not kv else len(kv),
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

