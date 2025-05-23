import uuid
import logging
import pickle

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
from .fuseki_handler import FusekiClient, check_datasets, init_db

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

repo = git.Repo(search_parent_directories=True)

from dotenv import load_dotenv
import os

load_dotenv(repo.working_dir + "/.env", override=True)

llm_provider = os.getenv("LLM_MODEL_PROVIDER")
model_id = os.getenv("LLM_MODEL_ID")
req_per_second = int(os.getenv("LLM_RPM")) / 60
logger.info(f"Initializing {model_id} at {llm_provider} - {req_per_second * 60} RPM")

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
        seed=1337
    )

elif llm_provider == "SambaNova":
    model = ChatOpenAI(
        api_key=os.getenv("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
        model=model_id,
        rate_limiter=rate_limiter,
        temperature=0,
        seed=1337
    )

elif llm_provider == "OpenAI":
    model = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=model_id,
        temperature=0,
        seed=1337
    )

elif llm_provider == "vLLM":
    model = ChatOpenAI(
        api_key="EMPTY",
        base_url="http://localhost:19123/v1",
        model=model_id,
        temperature=0,
        seed=1337
    )

elif llm_provider == "Ollama":
    model = ChatOllama(
        model=model_id,
        num_ctx=25600,
        temperature=0,
        seed=1337
    )

elif llm_provider == "Cerebras":
    model = ChatOpenAI(
        api_key=os.getenv("CEREBRAS_API_KEY"),
        base_url="https://api.cerebras.ai/v1",
        model=model_id,
        rate_limiter=rate_limiter,
        temperature=0,
        seed=1337
    )

elif llm_provider == "Cohere":
    model = ChatOpenAI(
        api_key=os.getenv("COHERE_API_KEY"),
        base_url="https://api.cohere.ai/compatibility/v1",
        model=model_id,
        rate_limiter=rate_limiter,
        temperature=0,
        seed=1337
    )

embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL_ID"),
)
logger.info(f"Embeddings model {os.getenv('EMBEDDING_MODEL_ID')} initialized")

session_id = str(uuid.uuid4())

langfuse_handler = CallbackHandler(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
    session_id=session_id
)
logger.info("Langfuse handler initialized")

langfuse_client = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)
logger.info("Langfuse client initialized")

if os.getenv("VECTOR_STORE") == "qdrant":
    logger.info("Initializing Qdrant vector store")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_port = os.getenv("QDRANT_PORT")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    client = QdrantClient(qdrant_url, port=qdrant_port, api_key=qdrant_api_key)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="wikidata_labels",
        embedding=embeddings
    )
    logger.info("Qdrant vector store initialized")

    label_vector_store = vector_store

    description_vector_store = QdrantVectorStore(
        client=client,
        collection_name="wikidata_descriptions",
        embedding=embeddings
    )
    logger.info("Qdrant description vector store initialized")

    example_vector_store = QdrantVectorStore(
        client=client,
        collection_name="wikidata_examples",
        embedding=embeddings
    )
    logger.info("Qdrant example vector store initialized")

else:
    logger.info("Initializing FAISS vector store")
    vector_store = FAISS(
        embedding_function=embeddings,
        index=faiss.IndexFlatL2(len(embeddings.embed_query("hello world"))),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    logger.info("FAISS vector store initialized")

    label_vector_store = vector_store

    description_vector_store = FAISS(
        embedding_function=embeddings,
        index=faiss.IndexFlatL2(len(embeddings.embed_query("hello world"))),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    logger.info("FAISS description vector store initialized")

# Initialize Fuseki datasets if they don't exist
datasets_status = check_datasets()
if not all(status['exists'] for status in datasets_status.values()):
    logger.info("Initializing Fuseki datasets...")
    init_db()

# Create Fuseki clients for the datasets
wikidata_predicate_graph = FusekiClient("wikidata_predicates")
wikidata_class_hierarchy = FusekiClient("wikidata_class_hierarchy")
logger.info("Fuseki clients initialized")

sparql = SPARQLWrapper("https://query.wikidata.org/sparql", agent="CIExMAS-SPARQL-Loader-Bot/1.0 (mlautenb@students.uni-mannheim.de)")
logger.info("SPARQL wrapper initialized")