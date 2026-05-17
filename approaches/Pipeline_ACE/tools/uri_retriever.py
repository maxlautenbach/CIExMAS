from langchain_core.runnables import RunnableConfig
from qdrant_client.http import models

from approaches.Pipeline_ACE.setup import cIEState, label_vector_store


def agent(state: cIEState, config: RunnableConfig = None) -> dict:
    """Vector search returning top-k candidates per entity and predicate."""
    uri_candidates = {}

    for entity in state["entities"]:
        results = label_vector_store.similarity_search(
            entity, k=3,
            filter=models.Filter(must=[
                models.FieldCondition(key="metadata.type", match=models.MatchValue(value="entity"))
            ])
        )
        uri_candidates[entity] = [
            {"uri": doc.metadata["uri"], "label": doc.page_content, "description": doc.metadata.get("description", "")}
            for doc in results
        ]

    # Extract unique predicates from triples
    for triple in state["triples"]:
        parts = [p.strip() for p in triple.split(";")]
        if len(parts) == 3:
            predicate = parts[1]
            if predicate not in uri_candidates:
                results = label_vector_store.similarity_search(
                    predicate, k=3,
                    filter=models.Filter(must=[
                        models.FieldCondition(key="metadata.type", match=models.MatchValue(value="predicate"))
                    ])
                )
                uri_candidates[predicate] = [
                    {"uri": doc.metadata["uri"], "label": doc.page_content, "description": doc.metadata.get("description", "")}
                    for doc in results
                ]

    return {"uri_candidates": uri_candidates}
