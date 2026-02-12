from pathlib import Path


def test_gpu_model_runner_has_kv_hook_initializer():
    src = (
        Path(__file__).resolve().parents[3]
        / "vllm/v1/worker/gpu_model_runner.py"
    ).read_text()
    assert "def init_kv_hook(self, config)" in src
