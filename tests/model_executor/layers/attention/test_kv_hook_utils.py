import base64
import glob
import gzip
import importlib.util
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import torch

_KV_PATH = (
    Path(__file__).resolve().parents[4]
    / "vllm/model_executor/layers/attention/kv_hook_utils.py"
)
_SPEC = importlib.util.spec_from_file_location("kv_hook_utils_test_local", _KV_PATH)
assert _SPEC and _SPEC.loader
_KV = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_KV)
HookConfig = _KV.HookConfig
KVHook = _KV.KVHook
load_kv_snapshot_data = _KV.load_kv_snapshot_data


def _decode_blob(item: dict) -> torch.Tensor:
    raw = gzip.decompress(base64.b64decode(item["data"]))
    return torch.frombuffer(raw, dtype=torch.float16).reshape(item["shape"])


def test_extract_layer_idx_patterns():
    hook = KVHook(HookConfig(enabled=True, layers={0}))
    assert hook._extract_layer_idx("model.layers.12.self_attn") == 12
    assert hook._extract_layer_idx("transformer.h.7.attn") == 7
    assert hook._extract_layer_idx("decoder.layers.3.self_attn") == 3
    assert hook._extract_layer_idx("bad.layer.name") == -1


def test_compute_attention_dense_shape_for_gqa():
    hook = KVHook(HookConfig(enabled=True, layers={0}))
    q = torch.randn(5, 8, 16, dtype=torch.float16)
    k = torch.randn(5, 2, 16, dtype=torch.float16)
    out = hook._compute_attention(q, k, 1.0 / (16**0.5))
    assert out is not None
    assert out.shape == (5, 8, 5)


def test_compute_attention_incompatible_shapes_return_none():
    hook = KVHook(HookConfig(enabled=True, layers={0}))
    q = torch.randn(5, 8, 16, dtype=torch.float16)
    k_bad_t = torch.randn(4, 8, 16, dtype=torch.float16)
    k_bad_d = torch.randn(5, 8, 32, dtype=torch.float16)
    assert hook._compute_attention(q, k_bad_t, 1.0) is None
    assert hook._compute_attention(q, k_bad_d, 1.0) is None


def test_buffer_qk_pair_uses_slot_mapping():
    hook = KVHook(HookConfig(enabled=True, layers={1}))
    q = torch.randn(3, 2, 4)
    k = torch.randn(3, 2, 4)
    meta = SimpleNamespace(slot_mapping=torch.tensor([10, -1, 11]))
    hook.buffer_qk_pair(q, k, meta, "model.layers.1.self_attn")
    assert (1, 10) in hook.q_buffer and (1, 11) in hook.q_buffer
    assert (1, -1) not in hook.q_buffer


def test_buffer_qk_pair_noop_on_shape_mismatch():
    hook = KVHook(HookConfig(enabled=True, layers={1}))
    q = torch.randn(3, 2, 4)
    k = torch.randn(3, 2, 4)
    meta = SimpleNamespace(slot_mapping=torch.tensor([10, 11]))
    hook.buffer_qk_pair(q, k, meta, "model.layers.1.self_attn")
    assert hook.q_buffer == {}
    assert hook.k_buffer == {}


def test_load_kv_snapshot_roundtrip_and_cleanup(monkeypatch):
    monkeypatch.setattr(_KV.time, "sleep", lambda _: None)
    req_id = f"req-{uuid.uuid4()}"
    safe = req_id.replace("-", "_")
    path = f"/tmp/vllm_snapshot_{safe}_layer3_T4.pt"
    attn = torch.arange(4 * 2 * 4, dtype=torch.float16).reshape(4, 2, 4)
    torch.save({"attn_scores": attn, "layer_idx": 3}, path)

    items = load_kv_snapshot_data(req_id)
    assert items is not None and len(items) == 1
    assert items[0]["shape"] == [4, 2, 4]
    assert items[0]["layer_idx"] == 3
    got = _decode_blob(items[0])
    assert torch.equal(got, attn)
    assert not os.path.exists(path)


def test_load_kv_snapshot_prefix_slice(monkeypatch):
    monkeypatch.setattr(_KV.time, "sleep", lambda _: None)
    req_id = f"req-{uuid.uuid4()}"
    safe = req_id.replace("-", "_")
    path = f"/tmp/vllm_snapshot_{safe}_layer0_T6.pt"
    attn = torch.randn(6, 4, 6, dtype=torch.float16)
    torch.save({"attn_scores": attn, "layer_idx": 0}, path)

    items = load_kv_snapshot_data(req_id, prefix="1:4")
    assert items is not None and len(items) == 1
    assert items[0]["shape"] == [3, 4, 6]


def test_load_kv_snapshot_bad_prefix_returns_none(monkeypatch):
    monkeypatch.setattr(_KV.time, "sleep", lambda _: None)
    req_id = f"req-{uuid.uuid4()}"
    safe = req_id.replace("-", "_")
    path = f"/tmp/vllm_snapshot_{safe}_layer0_T4.pt"
    torch.save(
        {"attn_scores": torch.randn(4, 2, 4, dtype=torch.float16), "layer_idx": 0},
        path,
    )
    assert load_kv_snapshot_data(req_id, prefix="bad") is None
    assert not os.path.exists(path)


def test_snapshot_keys_immediate_respects_layer_override_and_prefix():
    req_id = f"req-{uuid.uuid4()}"
    safe = req_id.replace("-", "_")
    hook = KVHook(HookConfig(enabled=True, layers={0}))
    t0 = torch.ones(2, 4, dtype=torch.float16)
    t1 = torch.full((2, 4), 2, dtype=torch.float16)
    hook.q_buffer[(0, 10)] = [t0]
    hook.k_buffer[(0, 10)] = [t0]
    hook.q_buffer[(1, 20)] = [t1, t1]
    hook.k_buffer[(1, 20)] = [t1, t1]
    req_state = SimpleNamespace(
        req_id=req_id,
        sampling_params=SimpleNamespace(extra_args={"kv_hook_layers": "1"}),
    )
    hook.snapshot_keys_immediate(req_state, block_size=16, kv_caches=None, prefix="1:2")
    snap = hook.snapshots[req_id]
    assert snap["layer_idx"] == 1
    assert list(snap["attn_scores"].shape) == [1, 2, 2]
    files = glob.glob(f"/tmp/vllm_snapshot_{safe}_layer1_T*.pt")
    assert files
    for f in files:
        os.remove(f)
