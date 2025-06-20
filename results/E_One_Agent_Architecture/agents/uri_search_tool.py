import re
from typing import Literal

from langgraph.types import Command

from results.E_One_Agent_Architecture.setup import cIEState, model, langfuse_handler, label_vector_store, description_vector_store


def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
    search_terms = state["instruction"].split(",")
    label_search_terms = [term.replace("[LABEL]", "") for term in search_terms if "[LABEL]" in term]
    description_search_terms = [term.replace("[DESCR]", "") for term in search_terms if "[DESCR]" in term]
    search_response = ""
    for term in label_search_terms:
        search_response += f'Most Similar rdfs:label Search Results for {term}:{[{"label": doc.page_content, "uri": doc.metadata["uri"], "description": doc.metadata["description"]} for doc in label_vector_store.similarity_search(term, k=3)]}\n\n'
    for term in description_search_terms:
        search_response += f'Most Similar schema:description Search Results for {term}:{[{"label": doc.metadata["label"], "uri": doc.metadata["uri"], "description": doc.page_content} for doc in description_vector_store.similarity_search(term, k=3)]}\n\n'
    search_response = search_response.replace("},", "},\n")

    if state["debug"]:
        state["instruction"] = ""
        state["messages"] += [search_response]
        return state, f"Search Results:\n{search_response}"

    return Command(goto="main_agent", update={"messages": state["messages"] + [search_response], "instruction": ""})
