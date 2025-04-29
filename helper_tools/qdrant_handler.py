from langchain_core.documents import Document
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from helper_tools.base_setup import client
from helper_tools.redis_handler import get_element_info, element_info_upload, clear_redis
from dotenv import load_dotenv
import os
import git

from helper_tools.wikidata_loader import get_description

repo = git.Repo(search_parent_directories=True)
load_dotenv(repo.working_dir + "/.env", override=True)


def upload_wikidata_element(uri, label, description=None):
    if "^^" in uri:
        return None

    # Check if entity is already tracked in Redis
    tracking_info = get_element_info(uri)
    if tracking_info:
        return None

    # Determine element type from URI
    element_type = "predicate" if "entity/P" in uri else "entity" if "entity/Q" in uri else "unknown"

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

    if not description:
        description = get_description(uri)

    label_doc = Document(page_content=label, metadata={"uri": uri, "description": description, "type": element_type})
    description_doc = Document(page_content=description, metadata={"uri": uri, "label": label, "type": element_type})
    qdrant_wikidata_labels.add_documents([label_doc])
    qdrant_wikidata_descriptions.add_documents([description_doc])

    # Track the upload in Redis with type information
    element_info_upload(uri, label, description)
    return None


def upload_wikidata_elements(elements_dict):
    """
    Upload multiple Wikidata elements in bulk, processing in batches of 100.
    
    Args:
        elements_dict: A dictionary where each key is a URI and each value is a dict with 'label' and optionally 'description'
                      Example: {'http://wikidata.org/entity/Q123': {'label': 'Example', 'description': 'This is an example'}}
    """
    if not elements_dict:
        return None
    
    import tqdm
    
    # Initialize embedding model once for all uploads
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
    
    # Process elements in batches of 100
    batch_size = 100
    all_uris = list(elements_dict.keys())
    total_batches = (len(all_uris) + batch_size - 1) // batch_size  # Ceiling division
    
    print(f"Processing {len(elements_dict)} elements in {total_batches} batches of {batch_size}")
    
    # Process each batch with a progress bar
    for batch_idx in tqdm.tqdm(range(total_batches), desc="Uploading batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_uris))
        batch_uris = all_uris[start_idx:end_idx]
        
        label_docs = []
        description_docs = []
        
        for uri in batch_uris:
            if "^^" in uri:
                continue
                
            content = elements_dict[uri]
            label = content.get('label')
            description = content.get('description')
            
            # Determine element type from URI
            element_type = "predicate" if "entity/P" in uri else "entity" if "entity/Q" in uri else "unknown"
            
            if not description:
                description = get_description(uri)
            
            # Create document objects
            label_doc = Document(page_content=label, metadata={"uri": uri, "description": description, "type": element_type})
            description_doc = Document(page_content=description, metadata={"uri": uri, "label": label, "type": element_type})
            
            label_docs.append(label_doc)
            description_docs.append(description_doc)
            
            # Track the upload in Redis with type information
            element_info_upload(uri, label, description)
        
        # Batch upload the current batch of documents
        if label_docs:
            qdrant_wikidata_labels.add_documents(label_docs)
        if description_docs:
            qdrant_wikidata_descriptions.add_documents(description_docs)
    
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

    client.create_collection(collection_name="wikidata_labels",
                             vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE))

    client.create_collection(collection_name="wikidata_descriptions",
                             vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE))

    # Clear Redis
    clear_redis()


if __name__ == "__main__":
    clear_collections()
