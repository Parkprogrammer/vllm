from types import SimpleNamespace
import sys
import types

from vllm.outputs import CompletionOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.output_processor import RequestState


def _mk_state(parent_req=None):
    return RequestState(
        request_id="req-int",
        external_req_id="req-ext",
        parent_req=parent_req,
        request_index=0,
        lora_request=None,
        output_kind=RequestOutputKind.FINAL_ONLY,
        prompt="p",
        prompt_token_ids=[1],
        prompt_embeds=None,
        logprobs_processor=SimpleNamespace(prompt_logprobs=None),
        detokenizer=None,
        max_tokens_param=8,
        arrival_time=0.0,
        queue=None,
        log_stats=False,
        stream_interval=1,
    )


def test_new_request_output_loads_kv_data_when_finished(monkeypatch):
    captured = {}

    def fake_loader(req_id, prefix=None):
        captured["args"] = (req_id, prefix)
        return [{"layer_idx": 1, "shape": [2, 2, 2], "dtype": "torch.float16", "data": "x"}]

    fake_mod = types.ModuleType("vllm.model_executor.layers.attention.kv_hook_utils")
    fake_mod.load_kv_snapshot_data = fake_loader
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.attention.kv_hook_utils",
        fake_mod,
    )
    parent = SimpleNamespace(sampling_params=SimpleNamespace(extra_args={"kv_hook_prefix": "2:5"}))
    rs = _mk_state(parent_req=parent)
    out = rs._new_request_output(
        external_req_id="req-ext",
        outputs=[CompletionOutput(0, "ok", [1], None, None, finish_reason="stop")],
        finished=True,
        kv_transfer_params=None,
    )
    assert captured["args"] == ("req-int", "2:5")
    assert out.kv_hook_data is not None


def test_new_request_output_skips_loader_when_not_finished(monkeypatch):
    called = {"v": 0}

    def fake_loader(*args, **kwargs):
        called["v"] += 1
        return []

    fake_mod = types.ModuleType("vllm.model_executor.layers.attention.kv_hook_utils")
    fake_mod.load_kv_snapshot_data = fake_loader
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.attention.kv_hook_utils",
        fake_mod,
    )
    rs = _mk_state()
    out = rs._new_request_output(
        external_req_id="req-ext",
        outputs=[CompletionOutput(0, "ok", [1], None, None, finish_reason=None)],
        finished=False,
        kv_transfer_params=None,
    )
    assert called["v"] == 0
    assert out.kv_hook_data is None


def test_new_request_output_finished_no_parent_uses_none_prefix(monkeypatch):
    captured = {}

    def fake_loader(req_id, prefix=None):
        captured["args"] = (req_id, prefix)
        return None

    fake_mod = types.ModuleType("vllm.model_executor.layers.attention.kv_hook_utils")
    fake_mod.load_kv_snapshot_data = fake_loader
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.attention.kv_hook_utils",
        fake_mod,
    )
    rs = _mk_state(parent_req=None)
    out = rs._new_request_output(
        external_req_id="req-ext",
        outputs=[CompletionOutput(0, "ok", [1], None, None, finish_reason="stop")],
        finished=True,
        kv_transfer_params=None,
    )
    assert captured["args"] == ("req-int", None)
    assert out.kv_hook_data is None


def test_new_request_output_finished_parent_no_extra_args(monkeypatch):
    captured = {}

    def fake_loader(req_id, prefix=None):
        captured["args"] = (req_id, prefix)
        return None

    fake_mod = types.ModuleType("vllm.model_executor.layers.attention.kv_hook_utils")
    fake_mod.load_kv_snapshot_data = fake_loader
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.attention.kv_hook_utils",
        fake_mod,
    )
    parent = SimpleNamespace(sampling_params=SimpleNamespace(extra_args=None))
    rs = _mk_state(parent_req=parent)
    rs._new_request_output(
        external_req_id="req-ext",
        outputs=[CompletionOutput(0, "ok", [1], None, None, finish_reason="stop")],
        finished=True,
        kv_transfer_params=None,
    )
    assert captured["args"] == ("req-int", None)
