from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest


def _request(**kwargs):
    data = {
        "model": "facebook/opt-125m",
        "messages": [{"role": "user", "content": "hi"}],
    }
    data.update(kwargs)
    return ChatCompletionRequest.model_validate(data)


def test_sampling_params_propagates_kv_hook_fields():
    req = _request(kv_hook_capture=1, kv_hook_layers="0,5")
    sp = req.to_sampling_params(
        max_tokens=8,
        logits_processor_pattern=None,
        default_sampling_params={},
    )
    assert sp.extra_args["kv_hook_capture"] == "1"
    assert sp.extra_args["kv_hook_layers"] == "0,5"


def test_sampling_params_without_kv_hook_fields():
    req = _request()
    sp = req.to_sampling_params(
        max_tokens=8,
        logits_processor_pattern=None,
        default_sampling_params={},
    )
    assert sp.extra_args is None or "kv_hook_capture" not in sp.extra_args
    assert sp.extra_args is None or "kv_hook_layers" not in sp.extra_args


def test_sampling_params_propagates_capture_zero():
    req = _request(kv_hook_capture=0)
    sp = req.to_sampling_params(
        max_tokens=8,
        logits_processor_pattern=None,
        default_sampling_params={},
    )
    assert sp.extra_args["kv_hook_capture"] == "0"


def test_sampling_params_merges_xargs_transfer_and_kv_fields():
    req = _request(
        vllm_xargs={"foo": "bar"},
        kv_transfer_params={"mode": "x"},
        kv_hook_capture=1,
        kv_hook_layers="2,3",
    )
    sp = req.to_sampling_params(
        max_tokens=8,
        logits_processor_pattern=None,
        default_sampling_params={},
    )
    assert sp.extra_args["foo"] == "bar"
    assert sp.extra_args["kv_transfer_params"] == {"mode": "x"}
    assert sp.extra_args["kv_hook_capture"] == "1"
    assert sp.extra_args["kv_hook_layers"] == "2,3"
