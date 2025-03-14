import json
from typing import TypedDict

from langchain_core.messages import AIMessage
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import git

repo = git.Repo(search_parent_directories=True)

from dotenv import load_dotenv
import os

load_dotenv(repo.working_dir + "/.env")

model = ChatOpenAI(model_name="Meta-Llama-3.3-70B-Instruct", base_url="https://api.sambanova.ai/v1",
                   api_key=os.getenv("SAMBANOVA_API_KEY"))
embeddings = OllamaEmbeddings(model='nomic-embed-text')

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
    comments: list[AIMessage]
    instruction: str
