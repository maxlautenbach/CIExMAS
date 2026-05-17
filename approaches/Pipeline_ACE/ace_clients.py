"""
Initialize OpenAI-compatible clients for ACE using CIExMAS .env configuration.

ACE expects raw openai.OpenAI clients. This module maps CIExMAS providers
(DeepInfra, SambaNova, Cerebras, etc.) to their OpenAI-compatible endpoints.
"""

import os
import openai
import git
from dotenv import load_dotenv

repo = git.Repo(search_parent_directories=True)
load_dotenv(os.path.join(repo.working_dir, ".env"), override=True)

PROVIDER_CONFIG = {
    "DeepInfra": {
        "base_url": "https://api.deepinfra.com/v1/openai",
        "api_key_env": "DEEPINFRA_API_TOKEN",
    },
    "SambaNova": {
        "base_url": "https://api.sambanova.ai/v1",
        "api_key_env": "SAMBANOVA_API_KEY",
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "Cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
    },
    "Cerebras-Paid": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_PAID_API_KEY",
    },
    "Cohere": {
        "base_url": "https://api.cohere.ai/compatibility/v1",
        "api_key_env": "COHERE_API_KEY",
    },
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
    },
    "vLLM": {
        "base_url": "http://localhost:19123/v1",
        "api_key_env": None,
    },
    "Ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key_env": None,
    },
}


def get_provider_name() -> str:
    return os.getenv("LLM_MODEL_PROVIDER", "DeepInfra")


def get_model_id() -> str:
    return os.getenv("LLM_MODEL_ID", "")


def initialize_ciexmas_clients():
    """
    Create three OpenAI-compatible clients (generator, reflector, curator)
    using the provider and API key from .env.

    Returns the same client three times since we use one model for all roles.
    """
    provider = get_provider_name()

    if provider not in PROVIDER_CONFIG:
        raise ValueError(
            f"Unknown LLM_MODEL_PROVIDER '{provider}'. "
            f"Supported: {list(PROVIDER_CONFIG.keys())}"
        )

    cfg = PROVIDER_CONFIG[provider]
    api_key = "EMPTY"
    if cfg["api_key_env"]:
        api_key = os.getenv(cfg["api_key_env"], "")
        if not api_key:
            raise ValueError(
                f"{cfg['api_key_env']} not set in .env for provider {provider}"
            )

    client = openai.OpenAI(api_key=api_key, base_url=cfg["base_url"])

    return client, client, client
