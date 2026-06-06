# Optimized vLLM Dockerfile for Blackwell GPUs (NVFP4)
FROM nvidia/cuda:13.0.2-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    ninja-build \
    python3-pip \
    python3-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"

# Install vLLM, FlashInfer, and DeepGEMM
RUN uv pip install --system "packaging>=24.2" && \
    uv pip install --system "vllm[flashinfer]==0.20.2" && \
    uv pip install --system "huggingface_hub[hf_transfer]" && \
    uv pip install --system git+https://github.com/deepseek-ai/DeepGEMM.git@714dd1a4a980f7937a74343d19a8eba4fe321480 --no-build-isolation

# Set environment variables for Blackwell / NVFP4 optimization
# Using both VLLM_ and standard prefixes for maximum compatibility with RunPod templates
ENV MODEL_PATH=/model
ENV VLLM_MODEL=/model
ENV MODEL_NAME=sakamakismile/Huihui-gemma-4-26B-A4B-it-abliterated-NVFP4
ENV VLLM_SERVED_MODEL_NAME=gemma-4-26b-it-nvfp4
ENV SERVED_MODEL_NAME=gemma-4-26b-it-nvfp4

ENV VLLM_MAX_MODEL_LEN=131072
ENV MAX_MODEL_LEN=131072

ENV VLLM_GPU_MEMORY_UTILIZATION=0.90
ENV GPU_MEMORY_UTILIZATION=0.90

ENV VLLM_DTYPE=bfloat16
ENV DTYPE=bfloat16

ENV VLLM_KV_CACHE_DTYPE=fp8
ENV KV_CACHE_DTYPE=fp8

ENV VLLM_USE_DEEP_GEMM=1
ENV VLLM_ENABLE_AUTO_TOOL_CHOICE=1
ENV VLLM_TOOL_CALL_PARSER=gemma4
ENV VLLM_REASONING_PARSER=gemma4
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Download and prebake the model into the image
COPY download_model.py /download_model.py
RUN --mount=type=secret,id=HF_TOKEN \
    mkdir -p /model && \
    HF_HUB_ENABLE_HF_TRANSFER=0 python3 /download_model.py

# Expose vLLM port
EXPOSE 8000

# Start vLLM OpenAI-compatible API server using direct exec form
# This avoids shell expansion issues with positional parameters
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--model", "/model"]
