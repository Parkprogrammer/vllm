#!/usr/bin/env python3
import base64
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

MODEL = "google/gemma-3-4b-it"
API_BASE = "http://127.0.0.1:8000/v1"
API_KEY = "EMPTY"
IMAGE = str(Path(__file__).with_name("sample_image.webp"))
KV_HOOK_CAPTURE = 1
KV_HOOK_LAYERS = "33"
MAX_TOKENS = 1024
TIMEOUT = 240.0
OUT_DIR = Path(__file__).resolve().parent / "artifacts" / "mm_kv_viz"
TARGET_PROMPT_TOKENS = 4000
HEATMAP_SIZE = 1024


def list_served_models() -> list[str]:
    url = API_BASE.rstrip("/") + "/models"
    req = urllib.request.Request(
        url=url,
        headers={
            "Authorization": f"Bearer {API_KEY}",
        },
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


def build_prompt(n: int = TARGET_PROMPT_TOKENS) -> str:
    seed = "Analyze one market image only with visible evidence. Mark uncertainty explicitly.\n"
    lines, i, wc = [seed], 0, len(seed.split())
    while wc < n:
        i += 1
        s = (
            f"[SCENE-{i:04d}] tents paths people goods shade sunlight occlusion depth. "
            f"anchors A{i%11} B{i%13} C{i%17}. cite concrete visual cues only.\n"
        )
        lines.append(s)
        wc += len(s.split())
    lines.append("Return 8 grounded bullets, 1 region table, and a short verification plan.")
    return "".join(lines)


def img_data_url(path: Path) -> str:
    return f"data:image/webp;base64,{base64.b64encode(path.read_bytes()).decode()}"


def _post_with_urllib(payload: dict) -> tuple[dict, str, str]:
    url = API_BASE.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        txt = r.read().decode("utf-8")
    return json.loads(txt), txt, "urllib"


def build_http_payload(model: str, messages: list[dict]) -> dict:
    # Raw HTTP path expects custom fields at top-level, not nested in "extra_body".
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "kv_hook_capture": KV_HOOK_CAPTURE,
    }
    if KV_HOOK_LAYERS is not None:
        payload["kv_hook_layers"] = KV_HOOK_LAYERS
    return payload


def call_vllm(messages: list[dict]) -> tuple[dict, str, str, str]:
    requested_model = os.getenv("VLLM_TEST_MODEL", MODEL)
    model = resolve_model_name(requested_model)
    http_payload = build_http_payload(model, messages)
    extra_body = {"kv_hook_capture": KV_HOOK_CAPTURE}
    if KV_HOOK_LAYERS is not None:
        extra_body["kv_hook_layers"] = KV_HOOK_LAYERS
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=API_KEY, base_url=API_BASE)
        if hasattr(client.chat.completions, "with_raw_response"):
            raw = client.chat.completions.with_raw_response.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=MAX_TOKENS,
                extra_body=extra_body,
                timeout=TIMEOUT,
            )
            txt = raw.text
            return json.loads(txt), txt, "openai", model
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            extra_body=extra_body,
            timeout=TIMEOUT,
        )
        body = resp.model_dump() if hasattr(resp, "model_dump") else json.loads(resp.json())
        txt = json.dumps(body, ensure_ascii=False)
        return body, txt, "openai", model
    except Exception:
        try:
            body, txt, transport = _post_with_urllib(http_payload)
            return body, txt, transport, model
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            # Retry once with first served model if the requested model mismatches.
            if e.code == 404:
                served = list_served_models()
                if served and model != served[0]:
                    http_payload["model"] = served[0]
                    body, txt, transport = _post_with_urllib(http_payload)
                    return body, txt, transport, served[0]
            raise RuntimeError(f"HTTP {e.code}: {detail}") from e


def decode_attn(item: dict) -> np.ndarray:
    raw = gzip.decompress(base64.b64decode(item["data"]))
    return np.frombuffer(raw, dtype=np.float16).reshape(item["shape"]).astype(np.float32)


def to_image(attn_2d: np.ndarray, title: str, boundary: int | None) -> Image.Image:
    p = float(np.percentile(attn_2d, 99.5)) if attn_2d.size else 1.0
    x = np.clip(attn_2d / max(p, 1e-8), 0.0, 1.0)
    rgb = np.stack(
        [
            (255 * x).astype(np.uint8),
            (255 * np.sqrt(x)).astype(np.uint8),
            (255 * (1.0 - x)).astype(np.uint8),
        ],
        axis=-1,
    )
    img = Image.fromarray(rgb, mode="RGB").resize(
        (HEATMAP_SIZE, HEATMAP_SIZE), Image.Resampling.NEAREST
    )
    dr = ImageDraw.Draw(img)
    dr.rectangle((0, 0, HEATMAP_SIZE - 1, 22), fill=(0, 0, 0))
    dr.text((6, 5), title, fill=(255, 255, 255))
    if boundary is not None and 0 < boundary < attn_2d.shape[0]:
        pos = int(boundary / attn_2d.shape[0] * HEATMAP_SIZE)
        dr.line((pos, 0, pos, HEATMAP_SIZE), fill=(255, 255, 255), width=2)
        dr.line((0, pos, HEATMAP_SIZE, pos), fill=(255, 255, 255), width=2)
    return img


def classify_indexing_status(
    boundary_local: int | None,
    boundary_with_offset_candidate: int | None,
    t_dim: int,
) -> str:
    local_in = boundary_local is not None and 0 <= boundary_local < t_dim
    offset_in = (
        boundary_with_offset_candidate is not None
        and 0 <= boundary_with_offset_candidate < t_dim
    )
    if local_in:
        return "aligned"
    if offset_in:
        return "candidate_fixable"
    return "mismatch_local"


def main() -> int:
    image = Path(IMAGE)
    assert image.exists(), f"image not found: {image}"
    run_dir = OUT_DIR / str(int(time.time()))
    run_dir.mkdir(parents=True, exist_ok=True)

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": build_prompt()},
            {"type": "image_url", "image_url": {"url": img_data_url(image)}},
        ],
    }]
    body, raw_json, transport, used_model = call_vllm(messages)
    (run_dir / "raw_response.json").write_text(raw_json)

    kv = body.get("kv_hook_data") or []
    assert kv, (
        "kv_hook_data missing "
        f"(capture_on={KV_HOOK_CAPTURE}, model={used_model}, request_id={body.get('id')})"
    )
    pngs, layer_summaries = [], []
    first_diag: dict[str, Any] | None = None

    for item in kv:
        layer = int(item.get("layer_idx", -1))
        attn = decode_attn(item)
        assert attn.ndim == 3 and attn.shape[0] == attn.shape[2], f"bad shape: {attn.shape}"
        hm = attn.mean(axis=1)
        t = hm.shape[0]

        token_meta = item.get("token_meta")
        assert isinstance(token_meta, dict), "token_meta missing"
        token_idx = token_meta.get("token_idx")
        assert isinstance(token_idx, list), "token_idx missing"
        assert len(token_idx) == t, f"len(token_idx)={len(token_idx)} != T={t}"
        assert all(isinstance(v, int) for v in token_idx), "token_idx has non-int"
        assert all(token_idx[i] <= token_idx[i + 1] for i in range(t - 1)), "token_idx not monotonic"

        order = np.argsort(np.asarray(token_idx, dtype=np.int64))
        indexed = hm[np.ix_(order, order)]
        prompt_len = token_meta.get("prompt_len")
        raw_boundary = int(np.searchsorted(np.asarray(token_idx), prompt_len, side="left")) if isinstance(prompt_len, int) else None
        sorted_idx = np.asarray(token_idx, dtype=np.int64)[order]
        idx_boundary = int(np.searchsorted(sorted_idx, prompt_len, side="left")) if isinstance(prompt_len, int) else None
        boundary_local = token_meta.get("prompt_boundary_local")
        boundary_with_offset_candidate = token_meta.get(
            "prompt_boundary_with_offset_candidate"
        )
        offset_candidate = token_meta.get("window_offset_candidate")
        token_index_basis = token_meta.get("token_idx_basis")
        if not isinstance(boundary_local, int) and isinstance(prompt_len, int):
            boundary_local = raw_boundary
        if (
            not isinstance(boundary_with_offset_candidate, int)
            and isinstance(offset_candidate, int)
            and isinstance(prompt_len, int)
        ):
            shifted = np.asarray(token_idx, dtype=np.int64) + int(offset_candidate)
            boundary_with_offset_candidate = int(
                np.searchsorted(shifted, prompt_len, side="left")
            )
        indexing_status = classify_indexing_status(
            boundary_local if isinstance(boundary_local, int) else None,
            boundary_with_offset_candidate
            if isinstance(boundary_with_offset_candidate, int)
            else None,
            t,
        )

        raw_img = to_image(hm, f"layer={layer} raw T={t}", raw_boundary)
        idx_img = to_image(indexed, f"layer={layer} indexed T={t}", idx_boundary)
        raw_path = run_dir / f"layer_{layer}_raw.png"
        idx_path = run_dir / f"layer_{layer}_indexed.png"
        raw_img.save(raw_path)
        idx_img.save(idx_path)
        pngs.extend([raw_path, idx_path])

        meta = {
            "layer_idx": layer,
            "shape": list(attn.shape),
            "prompt_len": prompt_len,
            "total_len": token_meta.get("total_len"),
            "token_idx_head": token_idx[:16],
            "token_idx_tail": token_idx[-16:],
            "token_idx_basis": token_index_basis,
            "offset_candidate": offset_candidate,
            "boundary_local": boundary_local,
            "boundary_with_offset_candidate": boundary_with_offset_candidate,
            "indexing_status": indexing_status,
            "raw_png": raw_path.name,
            "indexed_png": idx_path.name,
        }
        (run_dir / f"layer_{layer}_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2)
        )
        layer_summaries.append(meta)
        if first_diag is None:
            first_diag = {
                "token_index_basis": token_index_basis,
                "boundary_local": boundary_local,
                "boundary_with_offset_candidate": boundary_with_offset_candidate,
                "offset_candidate": offset_candidate,
                "indexing_status": indexing_status,
            }

    pages = [Image.open(p).convert("RGB") for p in pngs]
    assert pages, "no visualization pages"
    pages[0].save(run_dir / "report.pdf", save_all=True, append_images=pages[1:])
    for p in pages:
        p.close()

    summary = {
        "status": "ok",
        "transport": transport,
        "model": used_model,
        "request_id": body.get("id"),
        "prompt_tokens": (body.get("usage") or {}).get("prompt_tokens"),
        "completion_tokens": (body.get("usage") or {}).get("completion_tokens"),
        "layers": [m["layer_idx"] for m in layer_summaries],
        "diagnostics": first_diag or {},
        "output_dir": str(run_dir),
        "files": sorted(x.name for x in run_dir.iterdir()),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
