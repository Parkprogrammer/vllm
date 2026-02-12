#!/bin/bash

#  This script is for setting up vllm for ARM archiectures.
#  The Spark CPU is ARM based.
#  You can adjust the vllm version according to your vllm development version.


VLLM_VERSION=0.15.1
PYTORCH_VERSION=2.9.1
CUDA_VERSION=130
CPU_ARCH=$(uname -m)

# For dependency issues, install torch first. 
pip install torch==${PYTORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}

# Installing ARM-based CUDA libs, whl from the public repo
pip install "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu${CUDA_VERSION}-cp38-abi3-manylinux_2_35_${CPU_ARCH}.whl"

#   From here, the rebuild of vllm for local serving is optional
#   Must build without FLASH_ATTN since Multi-Modal Encoder 
#    enforces the engine to use FLASH_ATTN

pip uninstall -y vllm
pip install setuptools_scm

export VLLM_USE_FLASH_ATTN_V2=0
export MAX_JOBS=4

pip install -e . --no-build-isolation --no-deps