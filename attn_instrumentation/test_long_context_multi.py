#!/usr/bin/env python3
import argparse
import base64
import gzip
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from openai import OpenAI

DEFAULT_MODEL = "google/gemma-3-4b-it"
DEFAULT_API_BASE = "http://127.0.0.1:8000/v1"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_IMAGE = "/home/parkprogrammer/jehyun/vllm/attn_instrumentation/sample_image.webp"
DEFAULT_KV_HOOK_CAPTURE = 1
DEFAULT_KV_HOOK_LAYERS = "31,33"
DEFAULT_TARGET_PROMPT_TOKENS = 7000
DEFAULT_MAX_TOKENS = 256
DEFAULT_TIMEOUT = 240.0

SAMPLE_Q = 64
SAMPLE_H = 4
ROW_SUM_ATOL = 5e-2


def parse_on_off(v: str) -> bool:
    t = v.strip().lower()
    if t in {"on", "true", "1", "yes"}:
        return True
    if t in {"off", "false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid on/off value: {v}")


def parse_layers(s: str) -> set[int]:
    out = set()
    for p in s.split(","):
        p = p.strip()
        if p:
            out.add(int(p))
    return out


def img_data_url(path: Path) -> str:
    b = path.read_bytes()
    return f"data:image/webp;base64,{base64.b64encode(b).decode()}"


def build_prompt(tag: str, target_words: int) -> str:
    head = (
        "You are analyzing ONE market photo only. "
        "Use visible evidence only. "
        "If uncertain, say uncertain.\n"
    )
    units = [
        "Describe tent colors and map them to left/center/right.",
        "Track people location, walking direction, and occlusion.",
        "Estimate table object density and texture categories.",
        "Compare shaded vs sunlit regions and contrast cues.",
        "List repeated patterns in bags, fabrics, and arrangement.",
        "Detect perspective/depth cues from path and stall lines.",
        "Separate foreground/midground/background observations.",
        "Provide one ambiguity and one high-confidence cue.",
    ]
    out, n, i = [f"[{tag}] {head}"], len(head.split()), 0
    while n < target_words:
        i += 1
        t = units[i % len(units)]
        s = (
            f"[{tag}-{i:04d}] {t} "
            f"Anchor A{i%11} B{i%13} C{i%17}. "
            f"Cross-check with nearby stall and pedestrian context.\n"
        )
        out.append(s)
        n += len(s.split())
    out.append(
        f"[{tag}-FINAL] Return: 8 bullets + 1 short table(region|objects|uncertainty) + 1 action plan.\n"
    )
    return "".join(out)


def decode_and_check(item: dict) -> dict:
    layer = item.get("layer_idx")
    shape = item.get("shape")
    dtype = str(item.get("dtype"))
    raw = gzip.decompress(base64.b64decode(item["data"]))

    if not (isinstance(shape, list) and len(shape) == 3):
        return {
            "ok": False,
            "layer_idx": layer,
            "shape": shape,
            "dtype": dtype,
            "error": "invalid_shape_metadata",
        }

    expected = int(np.prod(shape))
    arr_flat = np.frombuffer(raw, dtype=np.float16)
    if arr_flat.size != expected:
        return {
            "ok": False,
            "layer_idx": layer,
            "shape": shape,
            "dtype": dtype,
            "error": f"size_mismatch:{arr_flat.size}!={expected}",
        }

    arr = arr_flat.reshape(shape)
    t, h, k = arr.shape
    qs = min(t, SAMPLE_Q)
    hs = min(h, SAMPLE_H)
    sample = arr[:qs, :hs, :]
    row_sum = sample.astype(np.float32).sum(axis=-1)
    max_err = float(np.max(np.abs(row_sum - 1.0)))
    finite_ok = bool(np.isfinite(sample).all())
    neg_ratio = float((sample < -1e-3).mean())
    gt1_ratio = float((sample > 1.0 + 1e-3).mean())
    ok = finite_ok and (max_err <= ROW_SUM_ATOL)
    return {
        "ok": ok,
        "layer_idx": layer,
        "shape": [int(t), int(h), int(k)],
        "dtype": dtype,
        "finite_ok": finite_ok,
        "row_sum_max_abs_err": max_err,
        "neg_ratio": neg_ratio,
        "gt1_ratio": gt1_ratio,
    }


def dump_failure(payload: dict, req_id: str | None, tag: str) -> str:
    p = Path(f"/tmp/kv_test_{(req_id or tag).replace('-', '_')}.json")
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return str(p)


def run_case_once(
    client: OpenAI,
    image_url: str,
    model: str,
    kv_hook_capture: int,
    kv_hook_layers: str,
    expected_layers: set[int],
    max_tokens: int,
    timeout: float,
    tag: str,
    words: int,
) -> dict:
    prompt = build_prompt(tag, words)
    extra_body = {
        "kv_hook_capture": kv_hook_capture,
        "kv_hook_layers": kv_hook_layers,
    }

    t0 = time.time()
    resp = client.chat.completions.with_raw_response.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }],
        temperature=0.0,
        max_tokens=max_tokens,
        extra_body=extra_body,
        timeout=timeout,
    )
    body = json.loads(resp.text)
    dt = round(time.time() - t0, 3)

    rid = body.get("id")
    usage = body.get("usage", {})
    kv = body.get("kv_hook_data") or []
    msg = (((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "").replace("\n", " ")[:160]

    checks = []
    decode_err = None
    try:
        for it in kv:
            checks.append(decode_and_check(it))
    except Exception as e:
        decode_err = str(e)

    layers_seen = sorted(
        int(c["layer_idx"])
        for c in checks
        if c.get("layer_idx") is not None and str(c.get("layer_idx")).isdigit()
    )
    layers_seen_set = set(layers_seen)
    layer_match = bool(layers_seen_set & expected_layers)
    all_checks_ok = bool(checks) and all(c.get("ok", False) for c in checks)
    kv_ok = bool(kv) and decode_err is None and all_checks_ok and layer_match
    first_shape = checks[0]["shape"] if checks else None
    t_dim = first_shape[0] if isinstance(first_shape, list) and len(first_shape) == 3 else None

    out = {
        "tag": tag,
        "request_id": rid,
        "elapsed_sec": dt,
        "input_words": len(prompt.split()),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "kv_hook_data": bool(kv),
        "kv_items": len(kv),
        "expected_layers": sorted(expected_layers),
        "layers_seen": layers_seen,
        "layer_match": layer_match,
        "all_checks_ok": all_checks_ok,
        "kv_ok": kv_ok,
        "decode_error": decode_err,
        "first_shape": first_shape,
        "t_dim": t_dim,
        "preview": msg,
        "checks": checks,
    }
    return out


def run_case_with_retries(
    image_url: str,
    model: str,
    api_base: str,
    api_key: str,
    kv_hook_capture: int,
    kv_hook_layers: str,
    expected_layers: set[int],
    max_tokens: int,
    timeout: float,
    tag: str,
    start_words: int,
    target_prompt_tokens: int,
) -> dict:
    words = max(64, int(start_words))
    last = None
    for attempt in range(3):
        client = OpenAI(api_key=api_key, base_url=api_base)
        out = run_case_once(
            client=client,
            image_url=image_url,
            model=model,
            kv_hook_capture=kv_hook_capture,
            kv_hook_layers=kv_hook_layers,
            expected_layers=expected_layers,
            max_tokens=max_tokens,
            timeout=timeout,
            tag=tag,
            words=words,
        )
        out["attempt"] = attempt + 1
        out["target_prompt_tokens"] = target_prompt_tokens
        print(json.dumps(out, ensure_ascii=False))
        last = out

        pt = out.get("prompt_tokens")
        if isinstance(pt, int) and pt >= target_prompt_tokens:
            break
        if attempt < 2:
            words = int(math.ceil(words * 1.25))
    assert last is not None

    if not last.get("kv_ok", False):
        dump = {
            "summary": last,
            "hint": "kv_hook_data empty or decode/numeric/layer check failed",
        }
        last["dump_file"] = dump_failure(dump, last.get("request_id"), tag)
        print(json.dumps({"tag": tag, "dump_file": last["dump_file"]}, ensure_ascii=False))
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--api-base", default=DEFAULT_API_BASE)
    ap.add_argument("--api-key", default=DEFAULT_API_KEY)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--kv-hook-layers", default=DEFAULT_KV_HOOK_LAYERS)
    ap.add_argument("--target-prompt-tokens", type=int, default=DEFAULT_TARGET_PROMPT_TOKENS)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--run-parallel", type=parse_on_off, default=True)
    ap.add_argument("--run-sequential", type=parse_on_off, default=True)
    args = ap.parse_args()

    image = Path(args.image)
    if not image.exists():
        raise FileNotFoundError(f"image not found: {image}")
    image_url = img_data_url(image)
    expected_layers = parse_layers(args.kv_hook_layers)
    base_words = max(1500, int(args.target_prompt_tokens * 0.5))

    all_cases = []
    sequential_ok = True
    parallel_ok = True

    if args.run_sequential:
        a = run_case_with_retries(
            image_url=image_url,
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            kv_hook_capture=DEFAULT_KV_HOOK_CAPTURE,
            kv_hook_layers=args.kv_hook_layers,
            expected_layers=expected_layers,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            tag="SEQ_A",
            start_words=base_words,
            target_prompt_tokens=args.target_prompt_tokens,
        )
        b = run_case_with_retries(
            image_url=image_url,
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            kv_hook_capture=DEFAULT_KV_HOOK_CAPTURE,
            kv_hook_layers=args.kv_hook_layers,
            expected_layers=expected_layers,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            tag="SEQ_B",
            start_words=int(base_words * 1.12),
            target_prompt_tokens=args.target_prompt_tokens,
        )
        all_cases.extend([a, b])
        sequential_ok = (
            a.get("kv_ok") and b.get("kv_ok")
            and a.get("request_id") != b.get("request_id")
            and (
                a.get("prompt_tokens") != b.get("prompt_tokens")
                or a.get("t_dim") != b.get("t_dim")
            )
        )
        print(json.dumps({
            "check": "sequential",
            "ok": bool(sequential_ok),
            "a_request_id": a.get("request_id"),
            "b_request_id": b.get("request_id"),
            "a_prompt_tokens": a.get("prompt_tokens"),
            "b_prompt_tokens": b.get("prompt_tokens"),
            "a_first": {"shape": a.get("first_shape"), "layers": a.get("layers_seen")},
            "b_first": {"shape": b.get("first_shape"), "layers": b.get("layers_seen")},
        }, ensure_ascii=False))

    if args.run_parallel:
        def task(tag: str, words: int) -> dict:
            return run_case_with_retries(
                image_url=image_url,
                model=args.model,
                api_base=args.api_base,
                api_key=args.api_key,
                kv_hook_capture=DEFAULT_KV_HOOK_CAPTURE,
                kv_hook_layers=args.kv_hook_layers,
                expected_layers=expected_layers,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                tag=tag,
                start_words=words,
                target_prompt_tokens=args.target_prompt_tokens,
            )

        with ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(task, "PAR_A", int(base_words * 1.03))
            f2 = ex.submit(task, "PAR_B", int(base_words * 1.20))
            p1, p2 = f1.result(), f2.result()
        all_cases.extend([p1, p2])
        parallel_ok = (
            p1.get("kv_ok") and p2.get("kv_ok")
            and p1.get("request_id") != p2.get("request_id")
            and (
                p1.get("prompt_tokens") != p2.get("prompt_tokens")
                or p1.get("t_dim") != p2.get("t_dim")
            )
        )
        print(json.dumps({
            "check": "parallel",
            "ok": bool(parallel_ok),
            "a_request_id": p1.get("request_id"),
            "b_request_id": p2.get("request_id"),
            "a_prompt_tokens": p1.get("prompt_tokens"),
            "b_prompt_tokens": p2.get("prompt_tokens"),
            "a_first": {"shape": p1.get("first_shape"), "layers": p1.get("layers_seen")},
            "b_first": {"shape": p2.get("first_shape"), "layers": p2.get("layers_seen")},
        }, ensure_ascii=False))

    multilayer_ok = all(c.get("layer_match", False) for c in all_cases) if all_cases else False
    final_ok = sequential_ok and parallel_ok and multilayer_ok

    summary = {
        "final_ok": bool(final_ok),
        "sequential_ok": bool(sequential_ok),
        "parallel_ok": bool(parallel_ok),
        "multilayer_ok": bool(multilayer_ok),
        "num_cases": len(all_cases),
        "expected_layers": sorted(expected_layers),
        "run_sequential": bool(args.run_sequential),
        "run_parallel": bool(args.run_parallel),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if final_ok else 1


if __name__ == "__main__":
    sys.exit(main())
