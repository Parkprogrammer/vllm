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
TARGET_PROMPT_TOKENS = 4000
MAX_TOKENS = 1024
TIMEOUT = 240.0


def img_data_url(path: Path) -> str:
    b = path.read_bytes()
    return f"data:image/webp;base64,{base64.b64encode(b).decode()}"


def tokenish_len(text: str) -> int:
    return len(text.split())


def build_long_text(target_tokens: int = 4000) -> tuple[str, int]:
    seed = (
        "You are given one image of an outdoor street market with colorful tents, tables, goods, trees, and pedestrians. "
        "Use only visible evidence from the image. If uncertain, label uncertainty explicitly. "
        "Do not invent text that is not legible.\n"
    )
    tasks = [
        "Count likely stalls by color cluster and note confidence.",
        "Describe left/middle/right regions with object density estimates.",
        "List nearest foreground objects and probable material types.",
        "Track pedestrian positions and likely walking directions.",
        "Compare shade vs sunlit zones and expected contrast effects.",
        "Identify repeated patterns across tables, fabrics, and bags.",
        "Estimate depth layering from foreground to background.",
        "Flag occluded or partially visible objects and why.",
        "Infer marketplace activity level from crowd and layout cues.",
        "Provide two alternative interpretations for ambiguous items.",
        "Map attention-worthy anchors: tents, pathways, people, goods.",
        "Find geometry cues from table edges, canopy lines, and path.",
    ]
    text = [seed]
    i = 0
    tok = tokenish_len(seed)
    while tok < target_tokens:
        i += 1
        t = tasks[(i - 1) % len(tasks)]
        zone = ["left", "center", "right"][i % 3]
        focus = ["color", "shape", "texture", "spatial relation", "occlusion"][i % 5]
        text.append(
            f"[SCENE-{i:04d}] Task: {t} "
            f"Primary zone: {zone}. Focus: {focus}. "
            f"Cross-check with nearby stalls and pedestrian context. "
            f"If evidence is weak, output low-confidence alternatives. "
            f"Tag anchors A{i%11} B{i%13} C{i%17}.\n"
        )
        text.append(
            f"[DETAIL-{i:04d}] Evaluate canopy colors, merchandise arrangement, path continuity, "
            f"and person-to-stall proximity. Mention at least one uncertainty and one confident cue.\n"
        )
        if i % 16 == 0:
            tok = tokenish_len("".join(text))
    text.append(
        "\nFinal output format:\n"
        "1) 8 bullet findings grounded in image evidence.\n"
        "2) 1 compact table: region | key objects | uncertainty.\n"
        "3) 1 short action plan for next visual verification pass.\n"
    )
    prompt = "".join(text)
    tok = tokenish_len(prompt)
    return prompt, tok


def run_once(client: OpenAI, model: str, image_url: str, text: str):
    extra_body = {"kv_hook_capture": KV_HOOK_CAPTURE}
    if KV_HOOK_LAYERS:
        extra_body["kv_hook_layers"] = KV_HOOK_LAYERS
    resp = client.chat.completions.with_raw_response.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ], 
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        extra_body=extra_body,
        timeout=TIMEOUT,
    )
    data = resp.parse()
    raw = resp.text
    body = json.loads(raw)
    kv = body.get("kv_hook_data")
    content = ""
    try:
        content = data.choices[0].message.content or ""
    except Exception:
        pass
    usage = body.get("usage", {})
    out = {
        "request_id": body.get("id"),
        "ok": bool(content),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "kv_hook_data": bool(kv),
        "kv_items": 0 if not kv else len(kv),
        "preview": content[:220].replace("\n", " "),
    }
    if kv:
        k0 = kv[0]
        out["first"] = {
            "layer_idx": k0.get("layer_idx"),
            "shape": k0.get("shape"),
            "dtype": k0.get("dtype"),
        }
    print(json.dumps(out, ensure_ascii=False))


def main():
    image = Path(IMAGE)
    assert image.exists(), f"image not found: {image}"
    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    image_url = img_data_url(image)
    long_text, token_est = build_long_text(TARGET_PROMPT_TOKENS)
    print(json.dumps({"token_estimate": token_est, "model": MODEL}, ensure_ascii=False))
    run_once(client, MODEL, image_url, long_text)


if __name__ == "__main__":
    main()
