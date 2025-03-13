from langchain_core.documents import Document
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from helper_tools.wikidata_loader import get_description


def upload_wikidata_entity(uri, label):
    if "^^" in uri:
        return None

    client = QdrantClient("localhost", port=6333)

    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    qdrant_wikidata_labels = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="wikidata_labels",
        url="http://localhost:6333",
    )
    qdrant_wikidata_descriptions = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="wikidata_descriptions",
        url="http://localhost:6333",
    )

    # Erstelle ein Filter-Objekt, das nach einem Dokument mit der entsprechenden URI sucht.
    filter_obj = models.Filter(must=[
        models.FieldCondition(key="metadata.uri", match=models.MatchValue(value=uri))
    ])

    existing_label_docs = client.scroll(
        collection_name="wikidata_labels",
        limit=1,
        scroll_filter=filter_obj
    )
    existing_desc_docs = client.scroll(
        collection_name="wikidata_descriptions",
        limit=1,
        scroll_filter=filter_obj
    )

    if len(existing_label_docs[0]) != 0 or len(existing_desc_docs[0]) != 0:
        return None

    description = get_description(uri)

    label_doc = Document(page_content=label, metadata={"uri": uri})
    description_doc = Document(page_content=description, metadata={"uri": uri, "label": label})
    qdrant_wikidata_labels.add_documents([label_doc])
    qdrant_wikidata_descriptions.add_documents([description_doc])
    return None


def clear_collections():
    client = QdrantClient("localhost", port=6333)

    # Delete the collection
    client.delete_collection(collection_name="wikidata_labels")
    client.delete_collection(collection_name="wikidata_descriptions")

    client.create_collection(collection_name="wikidata_labels", vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE))
    client.create_collection(collection_name="wikidata_descriptions", vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE))


if __name__ == "__main__":
    upload_wikidata_entity("http://www.wikidata.org/entity/Q61053", "Olaf Scholz")