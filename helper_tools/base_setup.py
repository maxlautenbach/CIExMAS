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

if llm_provider == "DeepInfra":
    model = ChatOpenAI(
        api_key=os.getenv("DEEPINFRA_API_TOKEN"),
        base_url="https://api.deepinfra.com/v1/openai",
        model=model_id
    )

elif llm_provider == "SambaNova":
    model = ChatOpenAI(
        api_key=os.getenv("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
        model=model_id
    )

elif llm_provider == "OpenAI":
    model = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=model_id
    )

elif llm_provider == "vLLM":
    model = ChatOpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        model=model_id
    )

elif llm_provider == "Ollama":
    model = ChatOllama(
        model=model_id,
        num_ctx=25600
    )

embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL_ID"),
)

langfuse_handler = CallbackHandler(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

if os.getenv("VECTOR_STORE") == "qdrant":

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_port = os.getenv("QDRANT_PORT")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    client = QdrantClient(qdrant_url, port=qdrant_port, api_key=qdrant_api_key)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="wikidata_labels",
        embedding=embeddings
    )

    label_vector_store = vector_store

    description_vector_store = QdrantVectorStore(
        client=client,
        collection_name="wikidata_descriptions",
        embedding=embeddings
    )

else:
    vector_store = FAISS(
        embedding_function=embeddings,
        index=faiss.IndexFlatL2(len(embeddings.embed_query("hello world"))),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    label_vector_store = vector_store

    description_vector_store = FAISS(
        embedding_function=embeddings,
        index=faiss.IndexFlatL2(len(embeddings.embed_query("hello world"))),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
