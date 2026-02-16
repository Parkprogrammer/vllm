# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser


def _parse_engine_args(argv: list[str]) -> EngineArgs:
    # Some CI/sandbox environments report an unspecified platform.
    # Force a stable device type so CLI parser construction is deterministic.
    if not current_platform.device_type:
        current_platform.device_type = "cpu"
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    parsed = parser.parse_args(argv)
    return EngineArgs.from_cli_args(parsed)


def test_attn_capture_cli_defaults():
    args = _parse_engine_args([])
    assert args.enable_attention_instrumentation is False
    assert args.attention_instrumentation_layers is None


def test_attn_capture_cli_parsing_enabled_and_layers():
    args = _parse_engine_args(
        [
            "--enable-attention-instrumentation",
            "--attention-instrumentation-layers",
            "0,5,11",
        ]
    )
    assert args.enable_attention_instrumentation is True
    assert args.attention_instrumentation_layers == "0,5,11"


def test_attn_capture_cli_parsing_all_layers():
    args = _parse_engine_args(
        [
            "--enable-attention-instrumentation",
            "--attention-instrumentation-layers",
            "all",
        ]
    )
    assert args.enable_attention_instrumentation is True
    assert args.attention_instrumentation_layers == "all"
