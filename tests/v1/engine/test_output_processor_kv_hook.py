# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.outputs import CompletionOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.output_processor import RequestState


class _DummyLogprobsProcessor:
    def __init__(self):
        self.prompt_logprobs = [{"dummy": 1}]

    def pop_prompt_logprobs(self):
        return self.prompt_logprobs


def _make_request_state(
    *,
    output_kind: RequestOutputKind = RequestOutputKind.FINAL_ONLY,
    prompt_token_ids: list[int] | None = None,
    prompt_embeds: torch.Tensor | None = None,
) -> RequestState:
    state = object.__new__(RequestState)
    state.logprobs_processor = _DummyLogprobsProcessor()
    state.output_kind = output_kind
    state.prompt_token_ids = prompt_token_ids
    state.prompt_embeds = prompt_embeds
    state.request_id = "req-attn-capture"
    state.lora_request = None
    state.prompt = "hello"
    state.num_cached_tokens = 0
    state.stats = None
    return state


def _one_completion_output() -> CompletionOutput:
    return CompletionOutput(
        index=0,
        text="ok",
        token_ids=[1],
        cumulative_logprob=0.0,
        logprobs=None,
    )


def test_new_request_output_loads_attn_capture_data_when_finished(monkeypatch):
    captured_calls: list[tuple[str, str | None]] = []

    def _fake_load(req_id: str, prefix: str | None = None):
        captured_calls.append((req_id, prefix))
        return [{"layer_idx": 33, "shape": [1, 1, 1]}]

    monkeypatch.setattr(
        "vllm.model_executor.layers.attention.attn_capture.load_attn_snapshot",
        _fake_load,
    )

    state = _make_request_state()
    out = state._new_request_output(
        external_req_id="external-id",
        outputs=[_one_completion_output()],
        finished=True,
    )

    assert captured_calls == [("req-attn-capture", None)]
    assert out.attn_capture_data == [{"layer_idx": 33, "shape": [1, 1, 1]}]


def test_new_request_output_skips_attn_capture_load_when_not_finished(monkeypatch):
    captured_calls: list[tuple[str, str | None]] = []

    def _fake_load(req_id: str, prefix: str | None = None):
        captured_calls.append((req_id, prefix))
        return [{"layer_idx": 99}]

    monkeypatch.setattr(
        "vllm.model_executor.layers.attention.attn_capture.load_attn_snapshot",
        _fake_load,
    )

    state = _make_request_state()
    out = state._new_request_output(
        external_req_id="external-id",
        outputs=[_one_completion_output()],
        finished=False,
    )

    assert captured_calls == []
    assert out.attn_capture_data is None


def test_new_request_output_uses_placeholder_prompt_ids_for_prompt_embeds():
    state = _make_request_state(
        prompt_token_ids=None,
        prompt_embeds=torch.randn(3, 8),
    )
    out = state._new_request_output(
        external_req_id="external-id",
        outputs=[_one_completion_output()],
        finished=False,
    )
    assert out.prompt_token_ids == [0, 0, 0]
