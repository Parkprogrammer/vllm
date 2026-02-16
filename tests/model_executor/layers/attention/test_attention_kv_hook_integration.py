# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.attention import attention as attention_mod
from vllm.model_executor.layers.attention.attn_capture import set_attn_capture


def test_unified_attention_with_output_calls_capture_when_runtime_enabled(monkeypatch):
    query = torch.randn(2, 3, 4, dtype=torch.float16)
    key = torch.randn(2, 3, 4, dtype=torch.float16)
    value = torch.randn(2, 3, 4, dtype=torch.float16)
    output = torch.empty_like(query)

    forward_calls: list[str] = []
    buffer_calls: list[str] = []

    class DummyImpl:
        def forward(
            self,
            layer_self,
            q,
            k,
            v,
            kv_cache,
            attn_metadata,
            *,
            output,
            output_scale,
            output_block_scale,
        ):
            forward_calls.append("forward")
            output.copy_(q)

    layer_self = SimpleNamespace(impl=DummyImpl())
    attn_metadata = SimpleNamespace(slot_mapping=torch.tensor([0, 1], dtype=torch.int64))
    kv_cache = torch.empty(0)

    monkeypatch.setattr(
        attention_mod,
        "get_attention_context",
        lambda layer_name: (attn_metadata, layer_self, kv_cache),
    )

    dummy_capture = SimpleNamespace(
        config=SimpleNamespace(enabled=True),
        runtime_enabled_this_step=True,
        buffer_query=lambda **kwargs: buffer_calls.append("buffer"),
    )
    set_attn_capture(dummy_capture)
    try:
        attention_mod.unified_attention_with_output(
            query=query,
            key=key,
            value=value,
            output=output,
            layer_name="model.layers.0.self_attn",
        )
    finally:
        set_attn_capture(None)

    assert buffer_calls == ["buffer"]
    assert forward_calls == ["forward"]


def test_unified_attention_with_output_skips_capture_when_runtime_disabled(monkeypatch):
    query = torch.randn(1, 2, 4, dtype=torch.float16)
    key = torch.randn(1, 2, 4, dtype=torch.float16)
    value = torch.randn(1, 2, 4, dtype=torch.float16)
    output = torch.empty_like(query)

    buffer_calls: list[str] = []
    forward_calls: list[str] = []

    class DummyImpl:
        def forward(
            self,
            layer_self,
            q,
            k,
            v,
            kv_cache,
            attn_metadata,
            *,
            output,
            output_scale,
            output_block_scale,
        ):
            forward_calls.append("forward")

    layer_self = SimpleNamespace(impl=DummyImpl())
    attn_metadata = SimpleNamespace(slot_mapping=torch.tensor([0], dtype=torch.int64))
    kv_cache = torch.empty(0)
    monkeypatch.setattr(
        attention_mod,
        "get_attention_context",
        lambda layer_name: (attn_metadata, layer_self, kv_cache),
    )

    dummy_capture = SimpleNamespace(
        config=SimpleNamespace(enabled=True),
        runtime_enabled_this_step=False,
        buffer_query=lambda **kwargs: buffer_calls.append("buffer"),
    )
    set_attn_capture(dummy_capture)
    try:
        attention_mod.unified_attention_with_output(
            query=query,
            key=key,
            value=value,
            output=output,
            layer_name="model.layers.1.self_attn",
        )
    finally:
        set_attn_capture(None)

    assert buffer_calls == []
    assert forward_calls == ["forward"]
