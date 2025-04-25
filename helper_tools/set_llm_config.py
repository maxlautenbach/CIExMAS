import os
from pathlib import Path
from dotenv import load_dotenv, set_key

# Updated AVAILABLE_MODELS to include RPM values
AVAILABLE_MODELS = {
    "OpenAI": [
        {"model": "gpt-4o-search-preview-2025-03-11", "rpm": 0}
    ],
    "SambaNova": [
        {"model": "Meta-Llama-3.3-70B-Instruct", "rpm": 80},
        {"model": "QwQ-32B", "rpm": 40},
        {"model": "Llama-4-Maverick-17B-128E-Instruct", "rpm": 40},
        {"model": "Llama-4-Scout-17B-16E-Instruct", "rpm": 40}
    ],
    "DeepInfra": [
        {"model": "meta-llama/Llama-3.3-70B-Instruct", "rpm": 0},
        {"model": "google/gemma-3-27b-it", "rpm": 0},
        {"model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "rpm": 0},
        {"model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "rpm": 0},
        {"model": "Qwen/QwQ-32B", "rpm": 0}
    ],
    "vLLM": [
        {"model": "ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g", "rpm": 0},
        {"model": "kosbu/Llama-3.3-70B-Instruct-AWQ", "rpm": 0},
        {"model": "unsloth/c4ai-command-a-03-2025-unsloth-bnb-4bit", "rpm": 0}
    ],
    "Cerebras": [
        {"model": "llama-4-scout-17b-16e-instruct", "rpm": 30},
        {"model": "llama-3.3-70b", "rpm": 30}
    ]
}

def get_env_path():
    """Get the path to the .env file"""
    return Path(__file__).parent.parent / ".env"

def select_provider():
    """Let the user select a provider"""
    print("\nAvailable providers:")
    for i, provider in enumerate(AVAILABLE_MODELS.keys(), 1):
        print(f"{i}. {provider}")
    
    while True:
        try:
            choice = int(input("\nSelect a provider (number): "))
            if 1 <= choice <= len(AVAILABLE_MODELS):
                return list(AVAILABLE_MODELS.keys())[choice - 1]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

# Updated select_model to return both model and RPM
def select_model(provider):
    """Let the user select a model for the chosen provider"""
    models = AVAILABLE_MODELS[provider]
    print(f"\nAvailable models for {provider}:")
    for i, model_info in enumerate(models, 1):
        print(f"{i}. {model_info['model']} (RPM: {model_info['rpm']})")

    while True:
        try:
            choice = int(input("\nSelect a model (number): "))
            if 1 <= choice <= len(models):
                return models[choice - 1]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

# Updated main to set LLM_RPM in .env
def main():
    env_path = get_env_path()

    if not env_path.exists():
        print("Error: .env file not found!")
        return

    # Load existing .env file
    load_dotenv(env_path)

    # Get user selections
    provider = select_provider()
    model_info = select_model(provider)
    model = model_info['model']
    rpm = model_info['rpm']

    # Update .env file
    set_key(env_path, "LLM_MODEL_PROVIDER", provider)
    set_key(env_path, "LLM_MODEL_ID", model)
    set_key(env_path, "LLM_RPM", str(rpm))

    print(f"\nSuccessfully set:")
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"RPM: {rpm}")

if __name__ == "__main__":
    main()