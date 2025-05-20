#!/bin/bash

#SBATCH --job-name=MAS-CIExMAS-EXEC
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --mail-user=max.lautenbach@yahoo.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:2
#SBATCH --time=06:00:00
#SBATCH --chdir=/home/mlautenb/CIExMAS
#SBATCH --partition=gpu-vram-48gb

# === Environment Setup ===
source activate base
conda activate CIExMAS
source ./.env

# === Trap and Cleanup ===
cleanup() {
    echo "Interrupt received. Cleaning up..."

    echo "Killing ollama processes if running..."
    pkill -f ollama 2>/dev/null

    if [[ "$LLM_MODEL_PROVIDER" == "vLLM" ]]; then
        echo "Killing vllm processes..."
        pkill -f vllm 2>/dev/null
    fi

    exit 1
}
trap cleanup SIGINT SIGTERM

echo "Starting Ollama server..."
OLLAMA_MODELS=/work/$(whoami)/ollama_models OLLAMA_LOAD_TIMEOUT=30m ollama serve &
OLLAMA_PID=$!
sleep 2  # Wait for server to spin up

echo "Pulling embedding model: $EMBEDDING_MODEL_ID"
ollama pull $EMBEDDING_MODEL_ID

# === Start Ollama Server if Needed ===
if [[ "$LLM_MODEL_PROVIDER" == "Ollama" ]]; then
    echo "Pulling model: $LLM_MODEL_ID"
    ollama pull "$LLM_MODEL_ID"
fi

# === Start vLLM if Needed ===
if [[ "$LLM_MODEL_PROVIDER" == "vLLM" ]]; then
    echo "Starting vLLM server with model: $LLM_MODEL_ID"
    if [[ "$LLM_MODEL_ID" == *"gemma"* ]]; then
        chat_template_name="gemma-it.jinja"
    else
        chat_template_name="llama-3-instruct.jinja"
    fi

    chat_template_path="/home/mlautenb/CIExMAS/helper_tools/chat_templates/${chat_template_name}"

    vllm serve "$LLM_MODEL_ID" \
            --port 19123 \
            --chat-template "$chat_template_path" \
            --download-dir /work/mlautenb/CIExMAS/models \
            --gpu-memory-utilization 0.95 \
            --max_model_len 8192 &

    VLLM_PID=$!
    echo "Waiting for vLLM server to be ready on port 19123..."
    for i in {1..60}; do
        if curl -s http://localhost:19123 > /dev/null; then
            echo "vLLM server is up!"
            break
        fi
        echo "Still waiting... ($i/60)"
        sleep 10
    done

    # If still not ready, exit with error
    if ! curl -s http://localhost:19123 > /dev/null; then
        echo "Error: vLLM server did not start within expected time."
        cleanup
    fi

fi

# === Run Agent System ===
echo "Running agent system..."
python3 ./approaches/Network/Gen2/slurm/agent_system.py test_text 50 'SVT'

# === Final Cleanup ===
echo "Job completed. Cleaning up..."

echo "Killing ollama if still running..."
pkill -f ollama 2>/dev/null

if [[ "$LLM_MODEL_PROVIDER" == "vLLM" ]]; then
    echo "Killing vLLM process..."
    pkill -f vllm 2>/dev/null
fi
