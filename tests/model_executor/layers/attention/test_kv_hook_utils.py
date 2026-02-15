# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import gzip
import uuid
from types import SimpleNamespace

import numpy as np
import torch

from vllm.model_executor.layers.attention.kv_hook_utils import (
    HookConfig,
    KVHook,
    _shm_read,
    _shm_write,
    load_kv_snapshot_data,
)


def _unique_req_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def test_load_kv_snapshot_data_round_trip_and_cleanup():
    """Write snapshots to shared memory and verify round-trip read."""
    req_id = _unique_req_id("kvhook-shm")
    attn = torch.randn(2, 3, 2, dtype=torch.float16)
    compressed = gzip.compress(attn.numpy().tobytes())
    snapshot = [{
        "data": base64.b64encode(compressed).decode("utf-8"),
        "shape": list(attn.shape),
        "dtype": str(attn.dtype),
        "layer_idx": 33,
        "token_meta": {"token_idx": [0, 1]},
    }]
    _shm_write(req_id, snapshot)

    out = load_kv_snapshot_data(req_id)
    assert out is not None
    assert len(out) == 1
    item = out[0]
    assert item["shape"] == [2, 3, 2]
    assert item["layer_idx"] == 33
    assert item["token_meta"] == {"token_idx": [0, 1]}

    raw = gzip.decompress(base64.b64decode(item["data"]))
    arr = np.frombuffer(raw, dtype=np.float16).reshape(item["shape"])
    np.testing.assert_allclose(arr, attn.numpy(), rtol=0, atol=0)


def test_load_kv_snapshot_data_returns_none_when_no_segment():
    """Returns None when no shared-memory segment exists for the request."""
    req_id = _unique_req_id("kvhook-missing")
    out = _shm_read(req_id, timeout=0.1)  # short timeout for test speed
    assert out is None


def test_build_token_meta_contains_diagnostics():
    hook = KVHook(HookConfig(enabled=True, layers={33}))
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
    hook = KVHook(HookConfig(enabled=True, layers={33}))
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
    assert hook.q_buffer[(33, 12)][0].dtype == torch.float16


def test_buffer_query_respects_capture_slots():
    """Only slots in capture_slots should be buffered."""
    hook = KVHook(HookConfig(enabled=True, layers={33}))
    hook.capture_slots = {10}  # only slot 10 is capture-enabled
    query = torch.randn(3, 2, 4, dtype=torch.float32)
    key = torch.randn(3, 2, 4, dtype=torch.float32)
    attn_metadata = SimpleNamespace(
        slot_mapping=torch.tensor([10, 11, 12], dtype=torch.int64))

    hook.buffer_query(
        query=query, key=key,
        attn_metadata=attn_metadata,
        layer_name="model.layers.33.self_attn",
    )

    assert (33, 10) in hook.q_buffer
    assert (33, 11) not in hook.q_buffer  # filtered out
    assert (33, 12) not in hook.q_buffer  # filtered out


def test_cleanup_request_buffers_removes_stale_entries():
    """cleanup_request_buffers removes Q buffer entries for freed blocks."""
    hook = KVHook(HookConfig(enabled=True, layers={5}))
    # Simulate buffered Q for block 0 (slots 0-3) and block 1 (slots 4-7)
    for slot in range(8):
        hook.q_buffer[(5, slot)] = [torch.zeros(2, 4)]
    # Also buffer a slot from a different request (block 2, slot 8)
    hook.q_buffer[(5, 8)] = [torch.zeros(2, 4)]

    # Clean up request that owned blocks [0, 1] with block_size=4
    hook.cleanup_request_buffers(block_ids=[[0, 1]], block_size=4)

    # Slots 0-7 should be cleaned; slot 8 should remain
    for slot in range(8):
        assert (5, slot) not in hook.q_buffer
    assert (5, 8) in hook.q_buffer
