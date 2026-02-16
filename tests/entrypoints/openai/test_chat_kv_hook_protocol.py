# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)


def _basic_messages() -> list[dict[str, str]]:
    return [{"role": "user", "content": "hello"}]


def test_attn_capture_fields_propagate_to_sampling_params_extra_args():
    req = ChatCompletionRequest(
        model="test-model",
        messages=_basic_messages(),
        attn_capture=1,
        attn_capture_layers="3,7",
        kv_transfer_params={"mode": "test"},
        vllm_xargs={"custom_flag": 42},
    )

    sampling_params = req.to_sampling_params(max_tokens=16, default_sampling_params={})
    assert sampling_params.extra_args is not None
    assert sampling_params.extra_args["attn_capture"] == "1"
    assert sampling_params.extra_args["attn_capture_layers"] == "3,7"
    assert sampling_params.extra_args["kv_transfer_params"] == {"mode": "test"}
    assert sampling_params.extra_args["custom_flag"] == 42


def test_attn_capture_fields_absent_when_not_requested():
    req = ChatCompletionRequest(
        model="test-model",
        messages=_basic_messages(),
    )
    sampling_params = req.to_sampling_params(max_tokens=8, default_sampling_params={})
    assert sampling_params.extra_args is None


def test_chat_completion_response_accepts_attn_capture_data():
    resp = ChatCompletionResponse(
        model="test-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="ok"),
            )
        ],
        usage=UsageInfo(prompt_tokens=2, completion_tokens=1, total_tokens=3),
        attn_capture_data=[{"layer_idx": 33, "shape": [4, 8, 4]}],
    )
    dumped = resp.model_dump()
    assert dumped["attn_capture_data"] == [{"layer_idx": 33, "shape": [4, 8, 4]}]
