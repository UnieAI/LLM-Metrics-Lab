#!/bin/bash

# A script to launch a model server using a specified framework.

# --- Check for Input Arguments ---
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "🚨 Error: Missing arguments."
    echo "Usage: $0 {model_name} {framework}"
    echo "  - model_name: {mixtral|llama3.1-70b|llama3.1-8b|llama3.2-vision|qwen3-32b}"
    echo "  - framework: {vllm|sglang|uifw}"
    exit 1
fi

MODEL_NAME=$1
FW=$2

echo "🚀 Preparing to launch model '$MODEL_NAME' using framework '$FW'..."

# --- Select Framework ---
case $FW in
    vllm)
        ### vLLM FRAMEWORK ###
        echo "Selected framework: vLLM"
        VLLM_COMMON_ARGS="--port 8910 --host 0.0.0.0 --served-model-name=test --trust_remote_code"

        case $MODEL_NAME in
            mixtral)
                echo "Using 4 GPUs for Mixtral-8x7B-Instruct-v0.1..."
                CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve /root/weights/mistralai/Mixtral-8x7B-Instruct-v0.1 \
                    $VLLM_COMMON_ARGS --tensor-parallel-size=4 --enable-prefix-caching
                ;;
            llama3.1-70b)
                echo "Using 4 GPUs for Llama-3.1-70B-Instruct..."
                CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve /root/weights/meta-llama/Llama-3.1-70B-Instruct \
                    $VLLM_COMMON_ARGS --tensor-parallel-size=4 --enable-prefix-caching
                ;;
            llama3.1-8b)
                echo "Using 1 GPU for Llama-3.1-8B-Instruct..."
                CUDA_VISIBLE_DEVICES=4 vllm serve /root/weights/meta-llama/Llama-3.1-8B-Instruct \
                    $VLLM_COMMON_ARGS --tensor-parallel-size=1 --enable-prefix-caching
                ;;
            llama3.2-vision)
                echo "Using 2 GPUs for Llama-3.2-11B-Vision-Instruct..."
                CUDA_VISIBLE_DEVICES=4,5 vllm serve /root/weights/meta-llama/Llama-3.2-11B-Vision-Instruct \
                    $VLLM_COMMON_ARGS --tensor-parallel-size=2 --limit-mm-per-prompt image=1,video=1 \
                    --max-model-len=32768 --max-num-seqs=128
                ;;
            qwen3-32b)
                echo "Using 1 GPU for Qwen3-32B..."
                CUDA_VISIBLE_DEVICES=4 vllm serve /root/weights/Qwen/Qwen3-32B \
                    $VLLM_COMMON_ARGS --tensor-parallel-size=1 --enable-prefix-caching
                ;;
            *)
                echo "🚨 Error: Invalid model name '$MODEL_NAME' for framework '$FW'."
                exit 1
                ;;
        esac
        ;;

    sglang)
        ### SGLang FRAMEWORK ###
        echo "Selected framework: SGLang"
        SGLANG_COMMON_ARGS="--host=0.0.0.0 --port=8910 --enable-torch-compile"

        case $MODEL_NAME in
            mixtral)
                echo "Using 4 GPUs for Mixtral-8x7B-Instruct-v0.1..."
                CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m sglang.launch_server \
                    --model-path /root/weights/mistralai/Mixtral-8x7B-Instruct-v0.1 \
                    $SGLANG_COMMON_ARGS --tp-size=4
                ;;
            llama3.1-70b)
                echo "Using 4 GPUs for Llama-3.1-70B-Instruct..."
                CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m sglang.launch_server \
                    --model-path /root/weights/meta-llama/Llama-3.1-70B-Instruct \
                    $SGLANG_COMMON_ARGS --tp-size=4
                ;;
            llama3.1-8b)
                echo "Using 1 GPU for Llama-3.1-8B-Instruct..."
                CUDA_VISIBLE_DEVICES=4 python3 -m sglang.launch_server \
                    --model-path /root/weights/meta-llama/Llama-3.1-8B-Instruct \
                    $SGLANG_COMMON_ARGS --tp-size=1
                ;;
            llama3.2-vision)
                echo "Using 2 GPUs for Llama-3.2-11B-Vision-Instruct..."
                CUDA_VISIBLE_DEVICES=4,5 python3 -m sglang.launch_server \
                    --model-path /root/weights/meta-llama/Llama-3.2-11B-Vision-Instruct \
                    $SGLANG_COMMON_ARGS --tp-size=2 --context-length=32768
                ;;
            qwen3-32b)
                echo "Using 1 GPU for Qwen3-32B..."
                CUDA_VISIBLE_DEVICES=4 python3 -m sglang.launch_server \
                    --model-path /root/weights/Qwen/Qwen3-32B \
                    $SGLANG_COMMON_ARGS --tp-size=1
                ;;
            *)
                echo "🚨 Error: Invalid model name '$MODEL_NAME' for framework '$FW'."
                exit 1
                ;;
        esac
        ;;

    uifw)
        ### UIFW FRAMEWORK ###
        echo "Selected framework: UIFW"
        UIFW_COMMON_ARGS="--backend=turbomind --server-port=8910 --model-name=test --enable-prefix-caching"

        case $MODEL_NAME in
            mixtral)
                echo "Using 4 GPUs for Mixtral-8x7B-Instruct-v0.1..."
                CUDA_VISIBLE_DEVICES=4,5,6,7 with-u010 uifw serve api_server /root/weights/mistralai/Mixtral-8x7B-Instruct-v0.1 \
                    $UIFW_COMMON_ARGS --tp=4 --chat-template=mixtral
                ;;
            llama3.1-70b)
                echo "Using 4 GPUs for Llama-3.1-70B-Instruct..."
                CUDA_VISIBLE_DEVICES=4,5,6,7 with-u010 uifw serve api_server /root/weights/meta-llama/Llama-3.1-70B-Instruct \
                    $UIFW_COMMON_ARGS --tp=4 --chat-template=llama3_1
                ;;
            llama3.1-8b)
                echo "Using 1 GPU for Llama-3.1-8B-Instruct..."
                CUDA_VISIBLE_DEVICES=4 with-u010 uifw serve api_server /root/weights/meta-llama/Llama-3.1-8B-Instruct \
                    $UIFW_COMMON_ARGS --tp=1 --chat-template=llama3_1
                ;;
            llama3.2-vision)
                echo "Using 2 GPUs for Llama-3.2-11B-Vision-Instruct..."
                CUDA_VISIBLE_DEVICES=4,5 with-u010 uifw serve api_server /root/weights/meta-llama/Llama-3.2-11B-Vision-Instruct \
                    $UIFW_COMMON_ARGS --tp=2 --chat-template=llama3_2 --session-len=32768
                ;;
            qwen3-32b)
                echo "Using 1 GPU for Qwen3-32B..."
                CUDA_VISIBLE_DEVICES=4 with-u010 uifw serve api_server /root/weights/Qwen/Qwen3-32B \
                    $UIFW_COMMON_ARGS --tp=1 --chat-template=qwen2d5
                ;;
            *)
                echo "🚨 Error: Invalid model name '$MODEL_NAME' for framework '$FW'."
                exit 1
                ;;
        esac
        ;;

    *)
        echo "🚨 Error: Invalid framework '$FW'."
        echo "Available frameworks: {vllm|sglang|uifw}"
        exit 1
        ;;
esac
