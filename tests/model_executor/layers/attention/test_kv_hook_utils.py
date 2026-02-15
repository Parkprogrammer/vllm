# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import gzip
import os
import uuid
from types import SimpleNamespace

import numpy as np
import torch

from vllm.model_executor.layers.attention.kv_hook_utils import (
    HookConfig,
    KVHook,
    load_kv_snapshot_data,
)


def _unique_req_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def test_load_kv_snapshot_data_round_trip_and_cleanup():
    req_id = _unique_req_id("kvhook-load")
    req_safe = req_id.replace("-", "_")
    path = f"/tmp/vllm_snapshot_{req_safe}_layer33_T2.pt"
    attn = torch.randn(2, 3, 2, dtype=torch.float16)
    payload = {
        "attn_scores": attn,
        "layer_idx": 33,
        "token_meta": {"token_idx": [0, 1]},
        "extra_args": {"kv_hook_capture": "1"},
    }
    torch.save(payload, path)
    assert os.path.exists(path)

    out = load_kv_snapshot_data(req_id, prefix=None)
    assert out is not None
    assert len(out) == 1
    item = out[0]
    assert item["shape"] == [2, 3, 2]
    assert item["layer_idx"] == 33
    assert item["token_meta"] == {"token_idx": [0, 1]}

    raw = gzip.decompress(base64.b64decode(item["data"]))
    arr = np.frombuffer(raw, dtype=np.float16).reshape(item["shape"])
    np.testing.assert_allclose(arr, attn.numpy(), rtol=0, atol=0)
    assert not os.path.exists(path)


def test_load_kv_snapshot_data_returns_none_when_capture_off():
    req_id = _unique_req_id("kvhook-capture-off")
    req_safe = req_id.replace("-", "_")
    path = f"/tmp/vllm_snapshot_{req_safe}_layer33_T1.pt"
    torch.save(
        {
            "attn_scores": torch.zeros((1, 1, 1), dtype=torch.float16),
            "layer_idx": 33,
            "extra_args": {"kv_hook_capture": "0"},
        },
        path,
    )
    try:
        out = load_kv_snapshot_data(req_id, prefix=None)
        assert out is None
        # capture off path exits early and keeps file untouched.
        assert os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_build_token_meta_contains_diagnostics():
    hook = KVHook(HookConfig(enabled=True, layers={33}, topk=10))
    req_state = SimpleNamespace(
        num_prompt_tokens=8,
        num_tokens=20,
        mm_features=[
            SimpleNamespace(mm_position=SimpleNamespace(offset=2, length=3)),
            SimpleNamespace(mm_position=SimpleNamespace(offset=4, length=2)),
        ],
    )
    meta = hook.build_token_meta(
        req_state=req_state,
        token_idx=[0, 1, 2, 3],
        ordered_slots_len=4,
    )

    assert meta["token_idx_basis"] == "window_local"
    assert meta["window_offset_candidate"] == 16
    assert meta["prompt_boundary_local"] == 4
    assert meta["prompt_boundary_with_offset_candidate"] == 0
    assert meta["vision_ranges"] == [{"start": 2, "end": 6}]
    assert meta["language_ranges"] == [{"start": 0, "end": 2}, {"start": 6, "end": 8}]
    assert meta["prompt_len"] == 8
    assert meta["total_len"] == 20


def test_buffer_query_uses_layer_and_slot_mapping():
    hook = KVHook(HookConfig(enabled=True, layers={33}, topk=10))
    query = torch.randn(3, 2, 4, dtype=torch.float32)
    key = torch.randn(3, 2, 4, dtype=torch.float32)
    attn_metadata = SimpleNamespace(slot_mapping=torch.tensor([10, -1, 12], dtype=torch.int64))

    hook.buffer_query(
        query=query,
        key=key,
        attn_metadata=attn_metadata,
        layer_name="model.layers.33.self_attn",
    )

    assert (33, 10) in hook.q_buffer
    assert (33, 12) in hook.q_buffer
    assert (33, -1) not in hook.q_buffer
    assert hook.q_buffer[(33, 10)][0].dtype == torch.float16
    assert hook.k_buffer[(33, 12)][0].dtype == torch.float16
