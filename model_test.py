from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_community.llms import VLLM
from langchain_ollama.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langfuse.callback import CallbackHandler
from qdrant_client import QdrantClient
import git
import faiss

repo = git.Repo(search_parent_directories=True)

from dotenv import load_dotenv
import os

load_dotenv(repo.working_dir + "/.env", override=True)

llm_provider = os.getenv("LLM_MODEL_PROVIDER")
model_id = os.getenv("LLM_MODEL_ID")
print(llm_provider)
print(model_id)

model = ChatOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    model=model_id
)

response = model.invoke("Wer war zur Wende in Deutschland Bundeskanzler?")

print(response)