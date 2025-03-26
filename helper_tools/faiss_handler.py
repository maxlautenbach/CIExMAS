from langchain_core.documents import Document
from helper_tools.wikidata_loader import get_description
from helper_tools.base_setup import label_vector_store, description_vector_store, repo


def upload_wikidata_entity(uri, label):
    if "^^" in uri:
        return None

    description = get_description(uri)

    label_doc = Document(page_content=label, metadata={"uri": uri, "description": description})
    description_doc = Document(page_content=description, metadata={"uri": uri, "label": label})
    label_vector_store.add_documents([label_doc])
    description_vector_store.add_documents([description_doc])
    return None
