#!/usr/bin/env python3
import argparse
import base64
import gzip
import json
import os
import random
import shutil
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_MODEL = "google/gemma-3-4b-it"
DEFAULT_API_BASE = "http://127.0.0.1:8000/v1"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_IMAGE = str(Path(__file__).with_name("sample_image.webp"))
DEFAULT_MODEL_PATH = "/workspace/models/gemma-3-4b-it"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 240.0
DEFAULT_CAPTURE = 1
DEFAULT_LAYER = 33
DEFAULT_NUM_RANDOM_QUERIES = 5
DEFAULT_SEED = 42
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "artifacts" / "visualization"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text/image attention heatmaps from KV hook output."
    )
    parser.add_argument("--artifact-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--api-base", type=str, default=DEFAULT_API_BASE)
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image-path", type=str, default=DEFAULT_IMAGE)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument(
        "--num-random-queries", type=int, default=DEFAULT_NUM_RANDOM_QUERIES
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--compile-latex", action="store_true")
    return parser.parse_args()


def list_served_models(api_base: str, api_key: str, timeout: float) -> list[str]:
    req = urllib.request.Request(
        url=api_base.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return [
        item.get("id")
        for item in body.get("data", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]


def resolve_model_name(
    preferred: str, api_base: str, api_key: str, timeout: float
) -> str:
    try:
        served = list_served_models(api_base, api_key, timeout)
    except Exception:
        return preferred
    if preferred in served:
        return preferred
    if served:
        return served[0]
    return preferred


def build_online_prompt(target_words: int = 600) -> str:
    seed = "Analyze one market image only with visible evidence.\n"
    chunks = [
        "Describe tents paths people goods shadows and depth cues.",
        "Separate left center right observations with confidence tags.",
        "Call out occlusion and uncertainty explicitly.",
        "Avoid guessing text or hidden objects.",
        "Compare foreground middle and background details.",
    ]
    words = seed.split()
    i = 0
    while len(words) < target_words:
        words.extend(chunks[i % len(chunks)].split())
        i += 1
    return " ".join(words[:target_words])


def img_data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    suffix = path.suffix.lower().lstrip(".") or "png"
    return f"data:image/{suffix};base64,{data}"


def _post_with_urllib(
    api_base: str, api_key: str, timeout: float, payload: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    req = urllib.request.Request(
        url=api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        txt = resp.read().decode("utf-8")
    return json.loads(txt), txt


def extract_response_text(body: dict[str, Any]) -> str:
    try:
        return (((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    except Exception:
        return ""


def call_vllm_and_save_context(
    *,
    api_base: str,
    api_key: str,
    model: str,
    image_path: Path,
    layer: int,
    max_tokens: int,
    timeout: float,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt_text = build_online_prompt()
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": img_data_url(image_path)}},
        ],
    }]
    used_model = resolve_model_name(model, api_base, api_key, timeout)
    extra_body = {
        "kv_hook_capture": DEFAULT_CAPTURE,
        "kv_hook_layers": str(layer),
    }
    payload = {
        "model": used_model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "kv_hook_capture": DEFAULT_CAPTURE,
        "kv_hook_layers": str(layer),
    }
    body: dict[str, Any]
    raw_text: str

    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key, base_url=api_base)
        if hasattr(client.chat.completions, "with_raw_response"):
            raw = client.chat.completions.with_raw_response.create(
                model=used_model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                extra_body=extra_body,
                timeout=timeout,
            )
            raw_text = raw.text
            body = json.loads(raw_text)
        else:
            resp = client.chat.completions.create(
                model=used_model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                extra_body=extra_body,
                timeout=timeout,
            )
            body = (
                resp.model_dump() if hasattr(resp, "model_dump") else json.loads(resp.json())
            )
            raw_text = json.dumps(body, ensure_ascii=False)
    except Exception:
        try:
            body, raw_text = _post_with_urllib(api_base, api_key, timeout, payload)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            if e.code == 404:
                served = list_served_models(api_base, api_key, timeout)
                if served and used_model != served[0]:
                    payload["model"] = served[0]
                    used_model = served[0]
                    body, raw_text = _post_with_urllib(api_base, api_key, timeout, payload)
                else:
                    raise RuntimeError(f"HTTP {e.code}: {detail}") from e
            else:
                raise RuntimeError(f"HTTP {e.code}: {detail}") from e

    (output_dir / "raw_response.json").write_text(raw_text)
    context = {
        "mode": "online",
        "model": used_model,
        "layer": layer,
        "api_base": api_base,
        "image_path": str(image_path),
        "prompt_text": prompt_text,
        "response_text": extract_response_text(body),
        "request_id": body.get("id"),
        "timestamp": int(time.time()),
    }
    (output_dir / "request_context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2)
    )
    return body, context


def load_artifact_payload(artifact_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_path = artifact_dir / "raw_response.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"raw_response.json not found: {raw_path}")
    raw_text = raw_path.read_text()
    body = json.loads(raw_text)
    context_path = artifact_dir / "request_context.json"
    context: dict[str, Any] = {"mode": "artifact"}
    if context_path.exists():
        context = json.loads(context_path.read_text())
        context["mode"] = "artifact"
    return body, context


def decode_attn(item: dict[str, Any]) -> np.ndarray:
    raw = gzip.decompress(base64.b64decode(item["data"]))
    return np.frombuffer(raw, dtype=np.float16).reshape(item["shape"]).astype(np.float32)


def resolve_token_indices(
    token_meta: dict[str, Any],
) -> tuple[list[int], list[int], int | None]:
    token_idx_local = token_meta.get("token_idx")
    if not isinstance(token_idx_local, list) or not all(
        isinstance(v, int) for v in token_idx_local
    ):
        raise ValueError("token_meta.token_idx must be a list[int]")

    basis = token_meta.get("token_idx_basis")
    offset = token_meta.get("window_offset_candidate")
    offset_used = offset if isinstance(offset, int) and basis == "window_local" else None
    if offset_used is not None:
        token_idx_abs = [int(i) + int(offset_used) for i in token_idx_local]
    else:
        token_idx_abs = [int(i) for i in token_idx_local]
    return token_idx_local, token_idx_abs, offset_used


def load_mm_config(model_path: Path) -> dict[str, int]:
    cfg = {"image_token_index": 262144, "mm_tokens_per_image": 256}
    path = model_path / "config.json"
    if not path.exists():
        return cfg
    try:
        raw = json.loads(path.read_text())
        if isinstance(raw.get("image_token_index"), int):
            cfg["image_token_index"] = int(raw["image_token_index"])
        if isinstance(raw.get("mm_tokens_per_image"), int):
            cfg["mm_tokens_per_image"] = int(raw["mm_tokens_per_image"])
    except Exception:
        pass
    return cfg


def build_abs_idx_to_text_map(
    *,
    model_path: Path,
    prompt_text: str | None,
    response_text: str | None,
    image_path: Path | None,
    prompt_len: int,
    total_len: int,
    token_idx_abs: list[int],
) -> tuple[dict[int, str], dict[str, Any], list[int] | None]:
    abs_to_text: dict[int, str] = {}
    decode_info: dict[str, Any] = {
        "processor_loaded": False,
        "prompt_decoded": 0,
        "generated_decoded": 0,
    }
    prompt_input_ids: list[int] | None = None

    try:
        from transformers import AutoProcessor  # type: ignore

        processor = AutoProcessor.from_pretrained(
            str(model_path), trust_remote_code=True
        )
        tokenizer = getattr(processor, "tokenizer", None)
        decode_info["processor_loaded"] = True

        if (
            tokenizer is not None
            and prompt_text
            and image_path is not None
            and image_path.exists()
            and hasattr(processor, "apply_chat_template")
        ):
            try:
                image = Image.open(image_path).convert("RGB")
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image", "image": image},
                    ],
                }]
                chat_text = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                enc = processor(text=[chat_text], images=[image], return_tensors="np")
                prompt_input_ids = [int(x) for x in np.asarray(enc["input_ids"][0]).tolist()]
                upto = min(prompt_len, len(prompt_input_ids))
                for idx in range(upto):
                    tok_id = int(prompt_input_ids[idx])
                    tok = tokenizer.convert_ids_to_tokens(tok_id)
                    abs_to_text[idx] = str(tok)
                decode_info["prompt_decoded"] = upto
                decode_info["prompt_input_ids_len"] = len(prompt_input_ids)
            except Exception as e:
                decode_info["prompt_decode_error"] = str(e)

        if tokenizer is not None and response_text:
            try:
                gen_ids = tokenizer.encode(response_text, add_special_tokens=False)
                gen_decoded = 0
                for j, tok_id in enumerate(gen_ids):
                    abs_pos = prompt_len + j
                    if abs_pos >= total_len:
                        break
                    abs_to_text[int(abs_pos)] = str(tokenizer.convert_ids_to_tokens(int(tok_id)))
                    gen_decoded += 1
                decode_info["generated_decoded"] = gen_decoded
                decode_info["generated_ids_len"] = len(gen_ids)
            except Exception as e:
                decode_info["generated_decode_error"] = str(e)
    except Exception as e:
        decode_info["processor_error"] = str(e)

    covered = sum(1 for idx in token_idx_abs if int(idx) in abs_to_text)
    decode_info["window_covered_tokens"] = covered
    decode_info["window_total_tokens"] = len(token_idx_abs)
    decode_info["window_coverage_ratio"] = (
        float(covered) / float(len(token_idx_abs)) if token_idx_abs else 0.0
    )
    return abs_to_text, decode_info, prompt_input_ids


def pick_random_queries(
    decoded_query_positions: list[int], n: int, seed: int
) -> list[int]:
    if not decoded_query_positions:
        return []
    uniq = sorted(set(int(v) for v in decoded_query_positions))
    if len(uniq) <= n:
        return uniq
    rng = random.Random(seed)
    picked = rng.sample(uniq, n)
    return sorted(picked)


def pick_diverse_queries_with_image_attention(
    *,
    attn_3d: np.ndarray,
    candidate_positions: list[int],
    image_positions: list[int],
    n: int,
    seed: int,
) -> list[int]:
    if not candidate_positions:
        return []
    if not image_positions:
        return pick_random_queries(candidate_positions, n, seed)
    if attn_3d.ndim != 3:
        return pick_random_queries(candidate_positions, n, seed)

    candidates = sorted(set(int(v) for v in candidate_positions))
    if len(candidates) <= n:
        return candidates

    # Mean over heads -> [T, T], then restrict keys to image positions.
    hm = attn_3d.mean(axis=1)
    img_pos = np.asarray(sorted(set(int(v) for v in image_positions)), dtype=np.int64)
    img_pos = img_pos[(img_pos >= 0) & (img_pos < hm.shape[1])]
    if img_pos.size == 0:
        return pick_random_queries(candidates, n, seed)
    vecs = hm[np.asarray(candidates, dtype=np.int64)][:, img_pos].astype(np.float32)
    if vecs.ndim != 2 or vecs.shape[1] == 0:
        return pick_random_queries(candidates, n, seed)

    rng = random.Random(seed)
    norms = np.linalg.norm(vecs, axis=1)
    start = int(np.argmax(norms))
    # A small seed jitter to keep deterministic but not always same anchor.
    if len(candidates) > 4:
        start = int((start + rng.randint(0, 3)) % len(candidates))
    selected_idx = [start]
    min_d = np.linalg.norm(vecs - vecs[start], axis=1)
    for _ in range(1, n):
        min_d[selected_idx] = -1.0
        nxt = int(np.argmax(min_d))
        if nxt in selected_idx:
            break
        selected_idx.append(nxt)
        d = np.linalg.norm(vecs - vecs[nxt], axis=1)
        min_d = np.minimum(min_d, d)
    out = sorted(candidates[i] for i in selected_idx[:n])
    if len(out) < n:
        remain = [c for c in candidates if c not in out]
        out.extend(pick_random_queries(remain, n - len(out), seed + 1))
        out = sorted(set(out))[:n]
    return out


def aggregate_key_attention(attn_3d: np.ndarray, query_idx: int) -> np.ndarray:
    if attn_3d.ndim != 3:
        raise ValueError(f"expected attn_3d ndim=3, got {attn_3d.ndim}")
    if query_idx < 0 or query_idx >= attn_3d.shape[0]:
        raise IndexError(f"query_idx out of range: {query_idx}")
    return attn_3d[int(query_idx)].mean(axis=0).astype(np.float32)


def rescale(input_list: list[float]) -> list[float]:
    arr = np.asarray(input_list, dtype=np.float32)
    if arr.size == 0:
        return []
    hi = float(np.max(arr))
    lo = float(np.min(arr))
    if hi <= lo:
        return [50.0 for _ in input_list]
    return (((arr - lo) / (hi - lo)) * 100.0).tolist()


def clean_word(word_list: list[str]) -> list[str]:
    out: list[str] = []
    for word in word_list:
        text = str(word)
        for latex_sensitive in ["\\", "%", "&", "^", "#", "_", "{", "}"]:
            if latex_sensitive in text:
                text = text.replace(latex_sensitive, "\\" + latex_sensitive)
        out.append(text)
    return out


def merge_subword_tokens_for_display(
    text_list: list[str], attention_list: list[float]
) -> tuple[list[str], list[float]]:
    if len(text_list) != len(attention_list):
        raise ValueError("text_list and attention_list length mismatch")
    if not text_list:
        return [], []

    markers = 0
    for tok in text_list:
        t = str(tok)
        if t.startswith("▁") or t.startswith("Ġ"):
            markers += 1
    marker_ratio = float(markers) / float(len(text_list))
    # If marker ratio is very low, keep token-level display (e.g., CJK-heavy pieces).
    if marker_ratio < 0.05:
        return [str(t) for t in text_list], [float(v) for v in attention_list]

    merged_tokens: list[str] = []
    merged_scores: list[float] = []
    cur_parts: list[str] = []
    cur_scores: list[float] = []

    def flush() -> None:
        if not cur_parts:
            return
        token = "".join(cur_parts).strip()
        if not token:
            token = "▢"
        merged_tokens.append(token)
        merged_scores.append(float(max(cur_scores)))
        cur_parts.clear()
        cur_scores.clear()

    for tok, score in zip(text_list, attention_list):
        t = str(tok)
        s = float(score)
        special = (t.startswith("<") and t.endswith(">")) or t.startswith("tok@")
        if special:
            flush()
            merged_tokens.append(t)
            merged_scores.append(s)
            continue

        starts_new = t.startswith("▁") or t.startswith("Ġ")
        piece = t.replace("▁", "").replace("Ġ", "").replace("</w>", "")
        if starts_new:
            flush()
        if not piece.strip():
            continue
        cur_parts.append(piece)
        cur_scores.append(s)

    flush()
    if not merged_tokens:
        return [str(t) for t in text_list], [float(v) for v in attention_list]
    return merged_tokens, merged_scores


def sanitize_display_token(token: str) -> str:
    text = str(token)
    if text.startswith("tok@"):
        return ""
    if text.startswith("<") and text.endswith(">"):
        return ""
    text = text.replace("\n", " ")
    cleaned: list[str] = []
    for ch in text:
        code = ord(ch)
        if ch == " ":
            cleaned.append(ch)
            continue
        if 32 <= code <= 126:
            cleaned.append(ch)
            continue
        # Drop non-ASCII by default to avoid tofu boxes in fallback rendering.
        cat = unicodedata.category(ch)
        if cat.startswith("P") and code < 256:
            cleaned.append(ch)
    out = "".join(cleaned)
    out = " ".join(out.split())
    return out


def select_tokens_for_text_view(
    text_list: list[str], attention_list: list[float], max_tokens: int = 260
) -> tuple[list[str], list[float]]:
    if len(text_list) != len(attention_list):
        raise ValueError("text_list and attention_list length mismatch")

    items: list[tuple[int, str, float]] = []
    for idx, (tok, score) in enumerate(zip(text_list, attention_list)):
        clean = sanitize_display_token(str(tok))
        if not clean:
            continue
        items.append((idx, clean, float(score)))

    if not items:
        return ["[no-token]"], [0.0]

    if len(items) > max_tokens:
        scores = np.asarray([s for _, _, s in items], dtype=np.float32)
        anchor_count = max(6, min(16, max_tokens // 20))
        top_local = np.argsort(scores)[-anchor_count:]
        selected_idx: set[int] = set()
        half = max(4, max_tokens // max(anchor_count * 2, 1))
        for a in top_local:
            start = max(0, int(a) - half)
            end = min(len(items), int(a) + half + 1)
            for k in range(start, end):
                selected_idx.add(k)
        selected = [items[i] for i in sorted(selected_idx)]
        if len(selected) < max_tokens:
            need = max_tokens - len(selected)
            remain = [items[i] for i in range(len(items)) if i not in selected_idx]
            if remain:
                step = max(1, len(remain) // max(1, need))
                selected.extend(remain[::step][:need])
                selected = sorted(selected, key=lambda x: x[0])
        if len(selected) > max_tokens:
            step = max(1, len(selected) // max_tokens)
            selected = selected[::step][:max_tokens]
    else:
        selected = list(items)

    out_tokens = [tok for _, tok, _ in selected]
    out_scores = [score for _, _, score in selected]
    # Collapse consecutive duplicates for readability.
    collapsed_tokens: list[str] = []
    collapsed_scores: list[float] = []
    token_cap: Counter[str] = Counter()
    max_repeat_per_token = 2
    for tok, score in zip(out_tokens, out_scores):
        if token_cap[tok] >= max_repeat_per_token:
            continue
        token_cap[tok] += 1
        if collapsed_tokens and collapsed_tokens[-1] == tok:
            collapsed_scores[-1] = max(collapsed_scores[-1], float(score))
            continue
        collapsed_tokens.append(tok)
        collapsed_scores.append(float(score))
    return collapsed_tokens, collapsed_scores


def generated_tokens_are_noisy(tokens: list[str]) -> bool:
    cleaned = [sanitize_display_token(t).lower() for t in tokens]
    cleaned = [t for t in cleaned if t and not t.startswith("<")]
    if len(cleaned) < 60:
        return False
    uniq_ratio = float(len(set(cleaned))) / float(len(cleaned))
    counts = Counter(cleaned)
    top_ratio = float(max(counts.values())) / float(len(cleaned)) if counts else 0.0
    short_ratio = float(sum(1 for t in cleaned if len(t) <= 3)) / float(len(cleaned))
    return (uniq_ratio < 0.25) or (top_ratio > 0.10) or (short_ratio > 0.55)


def recover_prompt_text_for_artifact(body: dict[str, Any]) -> str | None:
    usage = body.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    try:
        p = int(prompt_tokens) if prompt_tokens is not None else 0
    except Exception:
        p = 0
    try:
        if p >= 2000:
            try:
                from test_long_context_mm_kv_hook import build_prompt  # type: ignore
            except Exception:
                from attn_instrumentation.test_long_context_mm_kv_hook import build_prompt

            return str(build_prompt())
        if p > 0:
            try:
                from test_short_100tok_mm_kv_hook import build_short_prompt  # type: ignore
            except Exception:
                from attn_instrumentation.test_short_100tok_mm_kv_hook import (
                    build_short_prompt,
                )

            return str(build_short_prompt())
    except Exception:
        return None
    return None


def generate_latex_heatmap(
    text_list: list[str],
    attention_list: list[float],
    latex_file: Path,
    color: str = "red",
    rescale_value: bool = True,
) -> None:
    if len(text_list) != len(attention_list):
        raise ValueError("text_list and attention_list length mismatch")
    text_list, attention_list = select_tokens_for_text_view(
        text_list, attention_list
    )
    text_list, attention_list = merge_subword_tokens_for_display(
        text_list, attention_list
    )
    if rescale_value:
        attention_list = rescale(attention_list)
    text_list = clean_word(text_list)
    att_vals = [max(0, min(100, int(round(float(v))))) for v in attention_list]
    with latex_file.open("w") as f:
        f.write(
            r"""\documentclass[varwidth]{standalone}
\special{papersize=210mm,297mm}
\usepackage{color}
\usepackage{tcolorbox}
\usepackage{CJK}
\usepackage{adjustbox}
\tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
\begin{document}
\begin{CJK*}{UTF8}{gbsn}
"""
        )
        buf = r"""{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{""" + "\n"
        for idx, word in enumerate(text_list):
            buf += "\\colorbox{%s!%s}{" % (color, att_vals[idx]) + "\\strut " + word + "} "
        buf += "\n}}}"
        f.write(buf + "\n")
        f.write(r"""\end{CJK*}
\end{document}""")


def save_text_heatmap_png(
    text_list: list[str],
    attention_list: list[float],
    out_path: Path,
    title: str,
) -> None:
    if len(text_list) != len(attention_list):
        raise ValueError("text_list and attention_list length mismatch")
    text_list, attention_list = select_tokens_for_text_view(
        text_list, attention_list
    )
    text_list, attention_list = merge_subword_tokens_for_display(
        text_list, attention_list
    )

    if not text_list:
        Image.new("RGB", (1200, 300), color=(230, 230, 230)).save(out_path)
        return

    def normalize_token_for_display(token: str) -> str:
        tok = str(token).replace("\n", " ")
        if tok.startswith("<") and tok.endswith(">"):
            return tok
        tok = tok.replace("▁", "")
        tok = tok.replace("Ġ", "")
        tok = tok.replace("</w>", "")
        tok = tok.strip()
        return tok if tok else "▢"

    def choose_font_size(token_count: int) -> int:
        if token_count > 1200:
            return 20
        if token_count > 900:
            return 22
        if token_count > 700:
            return 24
        if token_count > 500:
            return 26
        if token_count > 350:
            return 30
        return 34

    def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",
        ]
        for path in candidates:
            if Path(path).exists():
                try:
                    return ImageFont.truetype(path, size=size)
                except Exception:
                    pass
        return ImageFont.load_default()

    def score_to_red(score_0_100: float) -> tuple[int, int, int]:
        s = max(0.0, min(100.0, float(score_0_100))) / 100.0
        # white -> red gradient close to the LaTeX style image
        g = int(round(255.0 - 210.0 * s))
        b = int(round(255.0 - 210.0 * s))
        return (255, g, b)

    tokens = [normalize_token_for_display(t) for t in text_list]
    scores_0_100 = rescale([float(v) for v in attention_list])

    font_size = choose_font_size(len(tokens))
    font = load_font(font_size)

    max_width = 2400
    margin_x = 20
    margin_y = 20
    gap_x = 10
    gap_y = 10
    pad_x = 8
    pad_y = 5

    dummy = Image.new("RGB", (max_width, 200), color=(240, 240, 240))
    dummy_draw = ImageDraw.Draw(dummy)

    # Two-pass layout: first pass computes required height.
    x = margin_x
    y = margin_y
    line_h = 0
    placed: list[tuple[int, int, int, int, str, float]] = []
    max_token_box_w = max_width - margin_x * 2

    for tok, score in zip(tokens, scores_0_100):
        bbox = dummy_draw.textbbox((0, 0), tok, font=font)
        tw = max(1, bbox[2] - bbox[0])
        th = max(1, bbox[3] - bbox[1])
        bw = min(max_token_box_w, tw + 2 * pad_x)
        bh = th + 2 * pad_y
        if x + bw > (max_width - margin_x) and x > margin_x:
            x = margin_x
            y += line_h + gap_y
            line_h = 0
        line_h = max(line_h, bh)
        placed.append((x, y, bw, bh, tok, float(score)))
        x += bw + gap_x

    img_h = y + line_h + margin_y
    image = Image.new("RGB", (max_width, img_h), color=(230, 230, 230))
    draw = ImageDraw.Draw(image)

    for x0, y0, bw, bh, tok, score in placed:
        fill = score_to_red(score)
        draw.rectangle((x0, y0, x0 + bw, y0 + bh), fill=fill)
        draw.text((x0 + pad_x, y0 + pad_y - 1), tok, fill=(0, 0, 0), font=font)

    # Save title metadata at the bottom-left in small text instead of top banner.
    meta_font = load_font(max(14, font_size // 2))
    draw.text((margin_x, img_h - margin_y), title[:200], fill=(80, 80, 80), font=meta_font)
    image.save(out_path)


def extract_image_key_positions(
    token_meta: dict[str, Any],
    key_abs_idx: np.ndarray,
    prompt_input_ids: list[int] | None,
    image_token_index: int,
    max_image_tokens: int,
) -> dict[str, Any]:
    found_pairs: list[tuple[int, int]] = []
    source = "none"

    vision_ranges = token_meta.get("vision_ranges")
    if isinstance(vision_ranges, list):
        intervals: list[tuple[int, int]] = []
        for item in vision_ranges:
            if not isinstance(item, dict):
                continue
            start, end = item.get("start"), item.get("end")
            if isinstance(start, int) and isinstance(end, int) and start < end:
                intervals.append((int(start), int(end)))
        if intervals:
            for start, end in intervals:
                key_positions = np.where(
                    (key_abs_idx >= start) & (key_abs_idx < end)
                )[0].tolist()
                span = max(1, (end - start - 1))
                for key_pos in key_positions:
                    abs_idx = int(key_abs_idx[int(key_pos)])
                    rel = abs_idx - int(start)
                    patch_idx = int(round((float(rel) / float(span)) * (max_image_tokens - 1)))
                    if 0 <= patch_idx < max_image_tokens:
                        found_pairs.append((int(key_pos), int(patch_idx)))
            if found_pairs:
                source = "vision_ranges"

    if not found_pairs and prompt_input_ids is not None:
        image_positions = [
            i for i, tok_id in enumerate(prompt_input_ids) if int(tok_id) == int(image_token_index)
        ]
        if image_positions:
            abs_to_patch = {int(abs_pos): idx for idx, abs_pos in enumerate(image_positions)}
            for i, abs_idx in enumerate(key_abs_idx.tolist()):
                patch_idx = abs_to_patch.get(int(abs_idx))
                if patch_idx is not None and 0 <= int(patch_idx) < max_image_tokens:
                    found_pairs.append((int(i), int(patch_idx)))
            if found_pairs:
                source = "image_token_index"

    unique_pairs = sorted(set(found_pairs), key=lambda x: x[1])
    if len(unique_pairs) > max_image_tokens:
        unique_pairs = unique_pairs[:max_image_tokens]

    key_positions = [p[0] for p in unique_pairs]
    return {
        "positions": key_positions,
        "pairs": unique_pairs,
        "source": source,
        "image_tokens_found": len(unique_pairs),
        "full_grid_available": len(unique_pairs) >= max_image_tokens,
        "partial_image_tokens": 0 < len(unique_pairs) < max_image_tokens,
    }


def build_image_grid(
    scores: np.ndarray,
    image_pairs: list[tuple[int, int]],
    grid_size: int = 16,
    max_tokens: int = 256,
) -> np.ndarray:
    grid = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
    for key_pos, patch_idx in image_pairs:
        if patch_idx < 0 or patch_idx >= min(max_tokens, grid_size * grid_size):
            continue
        if key_pos < 0 or key_pos >= scores.size:
            continue
        row, col = divmod(int(patch_idx), grid_size)
        grid[row, col] = float(scores[int(key_pos)])
    return grid


def render_image_overlay(
    image: Image.Image, grid_16x16: np.ndarray, out_path: Path, title: str
) -> None:
    grid = np.asarray(grid_16x16, dtype=np.float32)
    valid = ~np.isnan(grid)
    norm = np.zeros_like(grid, dtype=np.float32)
    if bool(np.any(valid)):
        vals = grid[valid]
        hi = float(np.max(vals))
        lo = float(np.min(vals))
        if hi > lo:
            norm[valid] = (vals - lo) / (hi - lo)
        else:
            norm[valid] = 0.5

    color = np.zeros((16, 16, 3), dtype=np.uint8)
    color[:, :, 0] = (255 * norm).astype(np.uint8)
    color[:, :, 1] = (255 * np.sqrt(norm)).astype(np.uint8)
    color[:, :, 2] = (255 * (1.0 - norm)).astype(np.uint8)
    alpha = np.zeros((16, 16), dtype=np.uint8)
    alpha[valid] = (120 + 120 * norm[valid]).astype(np.uint8)

    overlay_small = np.dstack([color, alpha]).astype(np.uint8)
    overlay = Image.fromarray(overlay_small, mode="RGBA").resize(
        image.size, Image.Resampling.NEAREST
    )
    composed = Image.alpha_composite(image.convert("RGBA"), overlay)

    draw = ImageDraw.Draw(composed)
    draw.rectangle((0, 0, composed.width - 1, 28), fill=(0, 0, 0, 180))
    draw.text((8, 8), title, fill=(255, 255, 255, 255))
    composed.save(out_path)


def write_summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def maybe_compile_latex(tex_files: list[Path], output_dir: Path) -> dict[str, Any]:
    report = {
        "requested": True,
        "available": False,
        "compiled": 0,
        "failed": [],
    }
    exe = shutil.which("pdflatex")
    if not exe:
        report["failed"].append("pdflatex not found")
        return report
    report["available"] = True
    for tex_path in tex_files:
        proc = subprocess.run(
            [exe, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=str(output_dir),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if proc.returncode == 0:
            report["compiled"] += 1
        else:
            report["failed"].append(f"{tex_path.name}: rc={proc.returncode}")
    return report


def select_kv_item(kv_data: list[dict[str, Any]], layer: int) -> dict[str, Any]:
    for item in kv_data:
        try:
            if int(item.get("layer_idx")) == int(layer):
                return item
        except Exception:
            continue
    available = sorted(
        {
            int(v.get("layer_idx"))
            for v in kv_data
            if isinstance(v, dict) and isinstance(v.get("layer_idx"), int)
        }
    )
    raise RuntimeError(f"layer {layer} not found in kv_hook_data. available={available}")


def main() -> int:
    args = parse_args()
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT / str(int(time.time()))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_mode = bool(args.artifact_dir)
    context: dict[str, Any]
    if artifact_mode:
        body, context = load_artifact_payload(Path(args.artifact_dir))
        (out_dir / "raw_response.json").write_text(
            json.dumps(body, ensure_ascii=False, indent=2)
        )
        if context:
            (out_dir / "request_context.json").write_text(
                json.dumps(context, ensure_ascii=False, indent=2)
            )
    else:
        image_path = Path(args.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"image not found: {image_path}")
        body, context = call_vllm_and_save_context(
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            image_path=image_path,
            layer=args.layer,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            output_dir=out_dir,
        )

    kv = body.get("kv_hook_data") or []
    if not kv:
        raise RuntimeError(
            "kv_hook_data missing "
            f"(request_id={body.get('id')}, model={context.get('model', args.model)})"
        )

    item = select_kv_item(kv, args.layer)
    attn = decode_attn(item)
    if attn.ndim != 3 or attn.shape[0] != attn.shape[2]:
        raise RuntimeError(f"invalid attention shape: {attn.shape}")
    t_dim, h_dim, _ = attn.shape

    token_meta = item.get("token_meta")
    if not isinstance(token_meta, dict):
        raise RuntimeError("token_meta missing")
    for required in ("token_idx", "prompt_len", "total_len"):
        if required not in token_meta:
            raise RuntimeError(f"token_meta missing required field: {required}")

    token_idx_local, token_idx_abs, offset_used = resolve_token_indices(token_meta)
    if len(token_idx_local) != t_dim:
        raise RuntimeError(
            f"token_idx length mismatch: len(token_idx)={len(token_idx_local)} vs T={t_dim}"
        )
    key_abs_idx = np.asarray(token_idx_abs, dtype=np.int64)

    model_path = Path(args.model_path)
    mm_cfg = load_mm_config(model_path)
    prompt_len = int(token_meta.get("prompt_len", 0))
    total_len = int(token_meta.get("total_len", prompt_len))
    prompt_text = context.get("prompt_text")
    prompt_text_source = "context"
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        recovered_prompt = recover_prompt_text_for_artifact(body)
        if isinstance(recovered_prompt, str) and recovered_prompt.strip():
            prompt_text = recovered_prompt
            prompt_text_source = "recovered_template"
        else:
            prompt_text = None
            prompt_text_source = "missing"
    response_text = context.get("response_text") or extract_response_text(body)

    context_image_path = context.get("image_path")
    active_image_path = (
        Path(context_image_path) if isinstance(context_image_path, str) else Path(args.image_path)
    )
    abs_to_text, decode_info, prompt_input_ids = build_abs_idx_to_text_map(
        model_path=model_path,
        prompt_text=prompt_text if isinstance(prompt_text, str) else None,
        response_text=response_text if isinstance(response_text, str) else None,
        image_path=active_image_path if active_image_path.exists() else None,
        prompt_len=prompt_len,
        total_len=total_len,
        token_idx_abs=token_idx_abs,
    )

    decoded_abs = set(abs_to_text.keys())
    decoded_query_positions = [
        i for i, abs_idx in enumerate(token_idx_abs) if int(abs_idx) in decoded_abs
    ]
    non_special_query_positions = [
        i
        for i in decoded_query_positions
        if not str(abs_to_text.get(int(token_idx_abs[i]), "")).startswith("<")
    ]
    query_fallback = False
    if not decoded_query_positions:
        decoded_query_positions = list(range(t_dim))
        query_fallback = True

    selected_records: list[dict[str, Any]] = []
    key_tokens = [abs_to_text.get(int(a), f"tok@{int(a)}") for a in token_idx_abs]
    tex_files: list[Path] = []
    text_png_files: list[Path] = []
    image_files: list[Path] = []
    grids: list[np.ndarray] = []

    image_positions_info = extract_image_key_positions(
        token_meta=token_meta,
        key_abs_idx=key_abs_idx,
        prompt_input_ids=prompt_input_ids,
        image_token_index=mm_cfg["image_token_index"],
        max_image_tokens=mm_cfg["mm_tokens_per_image"],
    )
    image_positions = image_positions_info["positions"]
    image_pairs = image_positions_info["pairs"]
    candidate_for_pick = (
        non_special_query_positions
        if len(non_special_query_positions) >= max(1, int(args.num_random_queries))
        else decoded_query_positions
    )
    picked_queries = pick_diverse_queries_with_image_attention(
        attn_3d=attn,
        candidate_positions=candidate_for_pick,
        image_positions=image_positions,
        n=max(1, int(args.num_random_queries)),
        seed=int(args.seed),
    )
    if not picked_queries:
        raise RuntimeError("no query positions selected")

    prompt_decoded_positions = [
        i for i, abs_idx in enumerate(token_idx_abs)
        if abs_idx < prompt_len and int(abs_idx) in decoded_abs
    ]
    prompt_language_positions = [
        i
        for i in prompt_decoded_positions
        if not str(abs_to_text.get(int(token_idx_abs[i]), "")).startswith("<")
    ]
    generated_decoded_positions = [
        i for i, abs_idx in enumerate(token_idx_abs)
        if abs_idx >= prompt_len and int(abs_idx) in decoded_abs
    ]
    generated_decoded_tokens = [
        abs_to_text[int(token_idx_abs[i])] for i in generated_decoded_positions
    ]
    generated_noisy = generated_tokens_are_noisy(generated_decoded_tokens)

    base_image: Image.Image | None = None
    if active_image_path.exists():
        base_image = Image.open(active_image_path).convert("RGB")

    for q_idx in picked_queries:
        q_abs = int(token_idx_abs[q_idx])
        q_text = abs_to_text.get(q_abs, f"tok@{q_abs}")
        scores = aggregate_key_attention(attn, q_idx)
        if generated_noisy and len(prompt_language_positions) >= 20:
            render_positions = prompt_language_positions
            text_render_mode = "prompt_language_only_due_noisy_generated"
        elif generated_noisy and len(prompt_decoded_positions) >= 40:
            render_positions = prompt_decoded_positions
            text_render_mode = "prompt_only_due_noisy_generated"
        else:
            render_positions = [
                i for i in range(t_dim) if int(token_idx_abs[i]) in decoded_abs
            ]
            if len(render_positions) < 32:
                render_positions = list(range(t_dim))
            text_render_mode = (
                "decoded_all_noisy_generated"
                if generated_noisy
                else "decoded_all"
            )
        render_tokens = [key_tokens[i] for i in render_positions]
        render_scores = [float(scores[i]) for i in render_positions]
        selected_records.append(
            {
                "query_local_idx": int(q_idx),
                "query_abs_idx": q_abs,
                "token_text": q_text,
                "segment": "prompt" if q_abs < prompt_len else "generated",
                "text_render_mode": text_render_mode,
            }
        )

        tex_path = out_dir / f"text_heatmap_q{int(q_idx):04d}.tex"
        png_path = out_dir / f"text_heatmap_q{int(q_idx):04d}.png"
        generate_latex_heatmap(
            render_tokens,
            render_scores,
            tex_path,
            color="red",
            rescale_value=True,
        )
        save_text_heatmap_png(
            render_tokens,
            render_scores,
            png_path,
            title=f"layer={int(item.get('layer_idx'))} q={int(q_idx)} abs={q_abs}",
        )
        tex_files.append(tex_path)
        text_png_files.append(png_path)

        if base_image is not None and image_pairs:
            grid = build_image_grid(
                scores,
                image_pairs=image_pairs,
                grid_size=16,
                max_tokens=mm_cfg["mm_tokens_per_image"],
            )
            overlay_path = out_dir / f"image_overlay_q{int(q_idx):04d}.png"
            render_image_overlay(
                base_image,
                grid,
                overlay_path,
                title=f"q={int(q_idx)} abs={q_abs} tok={q_text[:24]}",
            )
            grids.append(grid)
            image_files.append(overlay_path)

    avg_overlay_path: Path | None = None
    if base_image is not None and grids:
        stack = np.stack(grids, axis=0)
        valid = ~np.isnan(stack)
        summed = np.nansum(stack, axis=0)
        counts = valid.sum(axis=0)
        avg = np.full(summed.shape, np.nan, dtype=np.float32)
        nonzero = counts > 0
        avg[nonzero] = (summed[nonzero] / counts[nonzero]).astype(np.float32)
        avg_overlay_path = out_dir / "image_overlay_avg.png"
        render_image_overlay(base_image, avg, avg_overlay_path, title="average over selected queries")
        image_files.append(avg_overlay_path)
    if base_image is not None:
        base_image.close()

    selected_path = out_dir / "selected_queries.json"
    selected_path.write_text(json.dumps(selected_records, ensure_ascii=False, indent=2))

    latex_report = {
        "requested": bool(args.compile_latex),
        "available": False,
        "compiled": 0,
        "failed": [],
    }
    if args.compile_latex:
        latex_report = maybe_compile_latex(tex_files, out_dir)

    summary = {
        "status": "ok",
        "mode": "artifact" if artifact_mode else "online",
        "request_id": body.get("id"),
        "model": context.get("model", args.model),
        "layer": int(item.get("layer_idx")),
        "t_dim": int(t_dim),
        "h_dim": int(h_dim),
        "token_index_basis": token_meta.get("token_idx_basis"),
        "offset_used": offset_used,
        "token_decode_coverage": decode_info.get("window_coverage_ratio"),
        "decode_info": decode_info,
        "prompt_text_source": prompt_text_source,
        "generated_noisy": generated_noisy,
        "prompt_decoded_positions_in_window": len(prompt_decoded_positions),
        "selected_queries": selected_records,
        "query_selection_fallback": query_fallback,
        "image_tokens_found": image_positions_info["image_tokens_found"],
        "image_token_source": image_positions_info["source"],
        "full_grid_available": image_positions_info["full_grid_available"],
        "partial_image_tokens": image_positions_info["partial_image_tokens"],
        "latex_compile": latex_report,
        "output_dir": str(out_dir),
        "files": sorted(p.name for p in out_dir.iterdir()),
    }
    write_summary(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
