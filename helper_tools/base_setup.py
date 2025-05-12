import uuid

import faiss
import git
from SPARQLWrapper import SPARQLWrapper
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_qdrant import QdrantVectorStore
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from qdrant_client import QdrantClient

repo = git.Repo(search_parent_directories=True)

from dotenv import load_dotenv
import os
from rdflib import Graph

load_dotenv(repo.working_dir + "/.env", override=True)

llm_provider = os.getenv("LLM_MODEL_PROVIDER")
model_id = os.getenv("LLM_MODEL_ID")
req_per_second = int(os.getenv("LLM_RPM")) / 60
print(f"Initializing {model_id} at {llm_provider} - {req_per_second * 60} RPM")

if req_per_second > 0:
    rate_limiter = InMemoryRateLimiter(requests_per_second=req_per_second, check_every_n_seconds=0.1)
else:
    rate_limiter = InMemoryRateLimiter(requests_per_second=100, check_every_n_seconds=0.1)

if llm_provider == "DeepInfra":
    model = ChatOpenAI(
        api_key=os.getenv("DEEPINFRA_API_TOKEN"),
        base_url="https://api.deepinfra.com/v1/openai",
        model=model_id,
        temperature=0,
    )

elif llm_provider == "SambaNova":
    model = ChatOpenAI(
        api_key=os.getenv("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
        model=model_id,
        rate_limiter=rate_limiter,
        temperature=0
    )

elif llm_provider == "OpenAI":
    model = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=model_id,
        temperature=0,
    )

elif llm_provider == "vLLM":
    model = ChatOpenAI(
        api_key="EMPTY",
        base_url="http://localhost:19123/v1",
        model=model_id,
        temperature=0,
    )

elif llm_provider == "Ollama":
    model = ChatOllama(
        model=model_id,
        num_ctx=25600,
        temperature=0,
    )

elif llm_provider == "Cerebras":
    model = ChatOpenAI(
        api_key=os.getenv("CEREBRAS_API_KEY"),
        base_url="https://api.cerebras.ai/v1",
        model=model_id,
        rate_limiter=rate_limiter,
        temperature=0,
    )

elif llm_provider == "Cohere":
    model = ChatOpenAI(
        api_key=os.getenv("COHERE_API_KEY"),
        base_url="https://api.cohere.ai/compatibility/v1",
        model=model_id,
        rate_limiter=rate_limiter,
        temperature=0,
    )

embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL_ID"),
)

session_id = str(uuid.uuid4())

langfuse_handler = CallbackHandler(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
    session_id=session_id
)

langfuse_client = Langfuse(
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

    example_vector_store = QdrantVectorStore(
        client=client,
        collection_name="wikidata_examples",
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

wikidata_predicate_graph = Graph()
wikidata_predicate_graph.parse(repo.working_dir + "/infrastructure/all_properties.ttl", format="ttl")

sparql = SPARQLWrapper("https://query.wikidata.org/sparql", agent="CIExMAS-SPARQL-Loader-Bot/1.0 (mlautenb@students.uni-mannheim.de)")