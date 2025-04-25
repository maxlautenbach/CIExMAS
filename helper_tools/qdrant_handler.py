from langchain_core.documents import Document
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from helper_tools.redis_handler import get_element_info, element_info_upload, clear_redis
from dotenv import load_dotenv
import os
import git

from helper_tools.wikidata_loader import get_description

repo = git.Repo(search_parent_directories=True)
load_dotenv(repo.working_dir + "/.env", override=True)

def upload_wikidata_element(uri, label):
    if "^^" in uri:
        return None

    # Check if entity is already tracked in Redis
    tracking_info = get_element_info(uri)
    if tracking_info:
        return None

    # Determine element type from URI
    element_type = "predicate" if "entity/P" in uri else "entity" if "entity/Q" in uri else "unknown"

    client = QdrantClient(os.getenv("QDRANT_URL"), port=os.getenv("QDRANT_PORT"), api_key=os.getenv("QDRANT_API_KEY"))

    embeddings = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL_ID"))
    qdrant_wikidata_labels = QdrantVectorStore(
        client=client,
        collection_name="wikidata_labels",
        embedding=embeddings
    )
    qdrant_wikidata_descriptions = QdrantVectorStore(
        client=client,
        collection_name="wikidata_descriptions",
        embedding=embeddings
    )

    description = get_description(uri)

    label_doc = Document(page_content=label, metadata={"uri": uri, "description": description, "type": element_type})
    description_doc = Document(page_content=description, metadata={"uri": uri, "label": label, "type": element_type})
    qdrant_wikidata_labels.add_documents([label_doc])
    qdrant_wikidata_descriptions.add_documents([description_doc])
    
    # Track the upload in Redis with type information
    element_info_upload(uri, label, description)
    return None

def init_collections():
    client = QdrantClient(os.getenv("QDRANT_URL"), port=os.getenv("QDRANT_PORT"), api_key=os.getenv("QDRANT_API_KEY"))
    client.create_collection(collection_name="wikidata_labels",
                             vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE))
    client.create_collection(collection_name="wikidata_descriptions",
                             vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE))
    clear_redis()

def clear_collections():
    client = QdrantClient(os.getenv("QDRANT_URL"), port=os.getenv("QDRANT_PORT"), api_key=os.getenv("QDRANT_API_KEY"))

    # Delete the collection
    client.delete_collection(collection_name="wikidata_labels")
    client.delete_collection(collection_name="wikidata_descriptions")

    client.create_collection(collection_name="wikidata_labels", vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE))

    client.create_collection(collection_name="wikidata_descriptions", vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE))

    # Clear Redis
    clear_redis()

if __name__ == "__main__":
    clear_collections()