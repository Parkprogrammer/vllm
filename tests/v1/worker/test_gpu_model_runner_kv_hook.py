# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np

import vllm.v1.worker.gpu_model_runner as gpu_model_runner_mod
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class _DummyInputBatch:
    def __init__(self, req_ids: list[str]):
        self.req_id_to_index = {rid: i for i, rid in enumerate(req_ids)}

    def remove_request(self, req_id: str) -> None:
        self.req_id_to_index.pop(req_id, None)

    def condense(self) -> None:
        return

    def refresh_metadata(self) -> None:
        return


def _empty_cached_reqs():
    return SimpleNamespace(
        resumed_req_ids=set(),
        req_ids=[],
        num_computed_tokens=[],
        new_block_ids=[],
        num_output_tokens=[],
        new_token_ids=[],
        all_token_ids={},
    )


def _minimal_scheduler_output_for_finished(req_id: str):
    return SimpleNamespace(
        finished_req_ids={req_id},
        free_encoder_mm_hashes=[],
        num_scheduled_tokens={},
        scheduled_cached_reqs=_empty_cached_reqs(),
        scheduled_new_reqs=[],
        scheduled_spec_decode_tokens={},
    )


def _build_minimal_runner(req_id: str):
    runner = object.__new__(GPUModelRunner)
    runner.requests = {
        req_id: SimpleNamespace(
            req_id=req_id,
            sampling_params=SimpleNamespace(
                extra_args={"attn_capture": "1", "attn_capture_prefix": "2:6"}
            ),
            block_ids=[],
        )
    }
    runner.num_prompt_logprobs = {}
    runner.input_batch = _DummyInputBatch([req_id])
    runner.encoder_cache = {}
    runner.cache_config = SimpleNamespace(block_size=16)
    runner.kv_caches = "dummy-kv-caches"
    runner._get_valid_sampled_token_count = lambda: np.asarray([], dtype=np.int64)
    runner._may_reorder_batch = lambda scheduler_output: None
    runner.is_pooling_model = False
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner.use_async_scheduling = False
    return runner


def test_init_attn_capture_sets_runner_and_global(monkeypatch):
    runner = object.__new__(GPUModelRunner)
    created = []
    set_calls = []

    class DummyCapture:
        def __init__(self, config):
            created.append(config)

    monkeypatch.setattr(
        "vllm.model_executor.layers.attention.attn_capture.AttentionCapture",
        DummyCapture,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.attention.attn_capture.set_attn_capture",
        lambda inst: set_calls.append(inst),
    )

    cfg = SimpleNamespace(enabled=True, layers={1, 2}, topk=10)
    runner.init_attn_capture(cfg)

    assert created == [cfg]
    assert isinstance(runner.attn_capture, DummyCapture)
    assert set_calls == [runner.attn_capture]


def test_update_states_captures_finished_request_when_enabled(monkeypatch):
    req_id = "req-snapshot"
    runner = _build_minimal_runner(req_id)
    snapshot_calls = []
    runner.attn_capture = SimpleNamespace(
        config=SimpleNamespace(enabled=True),
        capture=lambda **kwargs: snapshot_calls.append(kwargs),
        cleanup_request_buffers=lambda block_ids, block_size: None,
    )

    monkeypatch.setattr(
        gpu_model_runner_mod,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    )

    runner._update_states(_minimal_scheduler_output_for_finished(req_id))

    assert len(snapshot_calls) == 1
    call = snapshot_calls[0]
    assert call["req_state"].req_id == req_id
    assert call["block_size"] == 16
    assert call["kv_caches"] == "dummy-kv-caches"
    assert call["prefix"] == "2:6"
    assert req_id not in runner.requests


def test_update_states_skips_capture_when_disabled(monkeypatch):
    req_id = "req-no-snapshot"
    runner = _build_minimal_runner(req_id)
    runner.requests[req_id].sampling_params.extra_args["attn_capture"] = "0"
    snapshot_calls = []
    runner.attn_capture = SimpleNamespace(
        config=SimpleNamespace(enabled=True),
        capture=lambda **kwargs: snapshot_calls.append(kwargs),
        cleanup_request_buffers=lambda block_ids, block_size: None,
    )

    monkeypatch.setattr(
        gpu_model_runner_mod,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    )

    runner._update_states(_minimal_scheduler_output_for_finished(req_id))

    assert snapshot_calls == []
    assert req_id not in runner.requests
