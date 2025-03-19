import json
from typing import TypedDict

from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langfuse.callback import CallbackHandler
from qdrant_client import QdrantClient
import git

repo = git.Repo(search_parent_directories=True)

from dotenv import load_dotenv
import os

load_dotenv(repo.working_dir + "/.env")

# DeepInfra
# model = ChatOpenAI(
#     api_key=os.getenv("DEEPINFRA_API_TOKEN"),
#     base_url="https://api.deepinfra.com/v1/openai",
#     model="google/gemma-3-27b-it"
# )

# SambaNova
model = ChatOpenAI(
    api_key=os.getenv("SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
    model="Meta-Llama-3.3-70B-Instruct"
)

embeddings = OllamaEmbeddings(model='nomic-embed-text')

langfuse_handler = CallbackHandler(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

client = QdrantClient("localhost", port=6333)
label_vector_store = QdrantVectorStore(
    client=client,
    collection_name="wikidata_labels",
    embedding=embeddings
)

description_vector_store = QdrantVectorStore(
    client=client,
    collection_name="wikidata_descriptions",
    embedding=embeddings
)


class cIEState(TypedDict):
    text: str
    call_trace: list[tuple[str]]
    results: list[str]
    comments: list[str]
    instruction: str
    debug: bool
