import os
from pathlib import Path
from dotenv import load_dotenv, set_key

# Dictionary of available providers and their models
AVAILABLE_MODELS = {
    "OpenAI": [
        "gpt-4o-search-preview-2025-03-11"
    ],
    "SambaNova": [
        "Meta-Llama-3.3-70B-Instruct",
        "QwQ-32B",
        "Llama-4-Maverick-17B-128E-Instruct",
        "Llama-4-Scout-17B-16E-Instruct"
    ],
    "DeepInfra": [
        "meta-llama/Llama-3.3-70B-Instruct",
        "google/gemma-3-27b-it",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "Qwen/QwQ-32B"
    ],
    "vLLM": [
        "ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g",
        "kosbu/Llama-3.3-70B-Instruct-AWQ",
        "unsloth/c4ai-command-a-03-2025-unsloth-bnb-4bit"
    ],
    "Cerebras": [
        "llama-4-scout-17b-16e-instruct",
        "llama-3.3-70b"
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

def select_model(provider):
    """Let the user select a model for the chosen provider"""
    models = AVAILABLE_MODELS[provider]
    print(f"\nAvailable models for {provider}:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    while True:
        try:
            choice = int(input("\nSelect a model (number): "))
            if 1 <= choice <= len(models):
                return models[choice - 1]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def main():
    env_path = get_env_path()
    
    if not env_path.exists():
        print("Error: .env file not found!")
        return
    
    # Load existing .env file
    load_dotenv(env_path)
    
    # Get user selections
    provider = select_provider()
    model = select_model(provider)
    
    # Update .env file
    set_key(env_path, "LLM_MODEL_PROVIDER", provider)
    set_key(env_path, "LLM_MODEL_ID", model)
    
    print(f"\nSuccessfully set:")
    print(f"Provider: {provider}")
    print(f"Model: {model}")

if __name__ == "__main__":
    main() 