#!/bin/bash

#SBATCH --job-name=MAS-CIExMAS-EXEC
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/mlautenb/CIExMAS
#SBATCH --partition=gpu-vram-48gb

# Function to cleanup background processes
cleanup() {
    echo "Interrupt received. Killing ollama process..."
    # Kill any process with "ollama" in its name.
    kill $(pgrep ollama) 2>/dev/null
    exit 1
}

# Trap SIGINT and SIGTERM signals (like when you press Ctrl+C)
trap cleanup SIGINT SIGTERM

conda activate CIExMAS

pip install -r requirements.txt

# Start ollama serve in the background
OLLAMA_MODELS=/work/$(whoami)/ollama_models OLLAMA_LOAD_TIMEOUT=30m ollama serve &
# Optionally, capture its PID if needed:
OLLAMA_PID=$!

# Allow ollama to initialize
sleep 2

ollama pull llama3.3

# Pull the nomic-embed-text model
ollama pull nomic-embed-text

# Run your python agent system
python3 ./approaches/baseline/slurm/agent_system.py

# After the script finishes, cleanup ollama (if still running)
kill $(pgrep ollama) 2>/dev/null
