#!/usr/bin/env python3
"""Common utilities for KV-hook instrumentation stress tests."""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

API_BASE = os.getenv("VLLM_API_BASE", "http://127.0.0.1:8000/v1")
API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
DEFAULT_MODEL = os.getenv("VLLM_TEST_MODEL", "gemma-3-4b-it")
DEFAULT_TIMEOUT = float(os.getenv("VLLM_TEST_TIMEOUT", "60"))
DEFAULT_MAX_TOKENS = int(os.getenv("VLLM_TEST_MAX_TOKENS", "48"))
DEFAULT_TEMPERATURE = float(os.getenv("VLLM_TEST_TEMPERATURE", "0.0"))
SAMPLE_IMAGE = Path(__file__).resolve().parents[1] / "sample_image.webp"


@dataclass
class RequestCase:
    name: str
    mode: str = "text"  # text | text_image | image_only
    capture: int = 1
    layers: str | None = "33"
    prompt: str = ""
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    timeout: float = DEFAULT_TIMEOUT
    expected_layers: set[int] | None = None
    require_all_expected_layers: bool = True
    min_layers_for_all: int = 2
    vllm_xargs: dict[str, Any] | None = None
    allow_http_error: bool = False
    allow_no_kv: bool = False
    strict_capture_off_no_kv: bool = True
    meta: dict[str, Any] = field(default_factory=dict)


def _safe_json_loads(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {"raw": raw}


def _http_request_json(url: str, method: str, headers: dict[str, str], data: bytes | None, timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(url=url, method=method, headers=headers, data=data)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_served_models(timeout: float = DEFAULT_TIMEOUT) -> list[str]:
    body = _http_request_json(
        url=API_BASE.rstrip("/") + "/models",
        method="GET",
        headers={"Authorization": f"Bearer {API_KEY}"},
        data=None,
        timeout=timeout,
    )
    out: list[str] = []
    for item in body.get("data", []):
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            out.append(item["id"])
    return out


def resolve_model_name(preferred: str = DEFAULT_MODEL, timeout: float = DEFAULT_TIMEOUT) -> str:
    try:
        served = list_served_models(timeout=timeout)
    except Exception:
        return preferred
    if preferred in served:
        return preferred
    if served:
        return served[0]
    return preferred


def ensure_server_alive(timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    try:
        served = list_served_models(timeout=timeout)
        return {
            "ok": True,
            "served_models": served,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "api_base": API_BASE,
        }


def _img_data_url(path: Path) -> str:
    raw = path.read_bytes()
    suffix = path.suffix.lower().lstrip(".") or "png"
    payload = base64.b64encode(raw).decode("utf-8")
    return f"data:image/{suffix};base64,{payload}"


def _default_prompt(case_name: str) -> str:
    return (
        f"[{case_name}] Analyze only visible evidence. "
        "Mention objects, relations, and confidence briefly."
    )


def _build_messages(case: RequestCase, image_path: Path = SAMPLE_IMAGE) -> list[dict[str, Any]]:
    prompt = case.prompt or _default_prompt(case.name)
    if case.mode == "text":
        return [{"role": "user", "content": prompt}]
    if case.mode == "text_image":
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _img_data_url(image_path)}},
                ],
            }
        ]
    if case.mode == "image_only":
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _img_data_url(image_path)}},
                ],
            }
        ]
    raise ValueError(f"unsupported mode: {case.mode}")


def _parse_expected_layers(case: RequestCase) -> set[int] | None:
    if case.expected_layers is not None:
        return set(case.expected_layers)
    if case.capture != 1 or not case.layers:
        return None
    s = case.layers.strip().lower()
    if s == "all":
        return None
    try:
        return {int(x.strip()) for x in case.layers.split(",") if x.strip()}
    except Exception:
        return None


def _validate_kv_contract(case: RequestCase, body: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    details: dict[str, Any] = {}
    kv = body.get("attn_capture_data")
    choices = body.get("choices")
    usage = body.get("usage")
    if not isinstance(choices, list) or not choices:
        return False, "choices missing or empty", details
    if not isinstance(usage, dict):
        return False, "usage missing", details

    if case.capture == 0:
        details["kv_items"] = 0 if not kv else len(kv)
        if case.strict_capture_off_no_kv and kv not in (None, []):
            return False, "capture=0 but attn_capture_data present", details
        return True, "ok", details

    # capture=1
    if not isinstance(kv, list) or not kv:
        if case.allow_no_kv:
            return True, "ok (no kv allowed)", details
        return False, "capture=1 but attn_capture_data missing", details

    returned_layers: set[int] = set()
    token_meta_first: dict[str, Any] | None = None
    for idx, item in enumerate(kv):
        if not isinstance(item, dict):
            return False, f"kv item {idx} is not a dict", details
        if not isinstance(item.get("layer_idx"), int):
            return False, f"kv item {idx} missing layer_idx", details
        returned_layers.add(int(item["layer_idx"]))
        if "data" not in item or "shape" not in item or "dtype" not in item:
            return False, f"kv item {idx} missing encoded tensor fields", details
        token_meta = item.get("token_meta")
        if not isinstance(token_meta, dict):
            return False, f"kv item {idx} missing token_meta", details
        for req_field in ("token_idx", "prompt_len", "total_len"):
            if req_field not in token_meta:
                return False, f"token_meta missing {req_field}", details
        if token_meta_first is None:
            token_meta_first = token_meta

    expected_layers = _parse_expected_layers(case)
    if expected_layers:
        if case.require_all_expected_layers:
            missing = expected_layers - returned_layers
            if missing:
                return False, f"missing requested layers: {sorted(missing)}", details
        else:
            if not (expected_layers & returned_layers):
                return False, (
                    f"none of requested layers present: requested={sorted(expected_layers)} "
                    f"returned={sorted(returned_layers)}"
                ), details

    if case.layers and case.layers.strip().lower() == "all":
        if len(returned_layers) < max(1, case.min_layers_for_all):
            return False, (
                f"layers=all but too few layers returned: {len(returned_layers)}"
            ), details

    details["kv_items"] = len(kv)
    details["returned_layers"] = sorted(returned_layers)
    if token_meta_first is not None:
        details["token_idx_basis"] = token_meta_first.get("token_idx_basis")
        vr = token_meta_first.get("vision_ranges")
        details["vision_ranges_len"] = len(vr) if isinstance(vr, list) else None
        details["prompt_len"] = token_meta_first.get("prompt_len")
        details["total_len"] = token_meta_first.get("total_len")
    return True, "ok", details


def run_case(case: RequestCase, model: str | None = None, image_path: Path = SAMPLE_IMAGE) -> dict[str, Any]:
    started = time.time()
    chosen_model = model or resolve_model_name()
    result: dict[str, Any] = {
        "name": case.name,
        "mode": case.mode,
        "capture": case.capture,
        "layers": case.layers,
        "ok": False,
        "soft": False,
        "model": chosen_model,
    }
    if case.meta:
        result["meta"] = case.meta

    if case.mode in {"text_image", "image_only"} and not image_path.exists():
        result["error"] = f"image not found: {image_path}"
        return result

    payload: dict[str, Any] = {
        "model": chosen_model,
        "messages": _build_messages(case, image_path=image_path),
        "temperature": case.temperature,
        "max_tokens": case.max_tokens,
        "attn_capture": int(case.capture),
    }
    if case.layers:
        payload["attn_capture_layers"] = case.layers
    if case.vllm_xargs:
        payload["vllm_xargs"] = case.vllm_xargs

    req = urllib.request.Request(
        url=API_BASE.rstrip("/") + "/chat/completions",
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        data=json.dumps(payload).encode("utf-8"),
    )

    try:
        with urllib.request.urlopen(req, timeout=case.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        result["elapsed_sec"] = round(time.time() - started, 3)
        result["http_status"] = e.code
        result["http_error"] = _safe_json_loads(detail)
        if case.allow_http_error:
            result["ok"] = True
            result["soft"] = True
            result["message"] = f"allowed HTTP error ({e.code})"
            return result
        result["error"] = f"HTTP {e.code}"
        return result
    except Exception as e:
        result["elapsed_sec"] = round(time.time() - started, 3)
        result["error"] = str(e)
        return result

    result["request_id"] = body.get("id")
    usage = body.get("usage") or {}
    result["prompt_tokens"] = usage.get("prompt_tokens")
    result["completion_tokens"] = usage.get("completion_tokens")
    choice0 = (body.get("choices") or [{}])[0]
    msg0 = choice0.get("message") if isinstance(choice0, dict) else {}
    if isinstance(msg0, dict):
        content = msg0.get("content")
        if isinstance(content, str):
            result["response_preview"] = content[:120].replace("\n", " ")

    ok, reason, details = _validate_kv_contract(case, body)
    result["elapsed_sec"] = round(time.time() - started, 3)
    result.update(details)
    if ok:
        result["ok"] = True
        result["message"] = reason
    else:
        result["error"] = reason
    return result


def print_json_line(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False))
