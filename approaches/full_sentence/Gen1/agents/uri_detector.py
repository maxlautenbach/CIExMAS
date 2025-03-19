import re
from typing import Literal

from langgraph.types import Command

from approaches.full_sentence.Gen1.setup import cIEState, model, langfuse_handler, label_vector_store, description_vector_store
from approaches.full_sentence.Gen1.prompts import uri_detector_prompt as prompt
import importlib
import approaches.full_sentence.Gen1.prompts
importlib.reload(approaches.full_sentence.Gen1.prompts)


def agent(state: cIEState) -> Command[Literal["agent_instructor_agent"]] | tuple[cIEState, str]:
    search_terms = state["instruction"].split(",")
    label_search_terms = [term.replace("[LABEL]", "") for term in search_terms if "[LABEL]" in term]
    description_search_terms = [term.replace("[DESCR]", "") for term in search_terms if "[DESCR]" in term]
    search_response = ""
    for term in label_search_terms:
        search_response += f'Most Similar rdfs:label Search Results for {term}:{[{"label": doc.page_content, "uri": doc.metadata["uri"], "description": doc.metadata["description"]} for doc in label_vector_store.similarity_search(term, k=3)]}\n\n'
    for term in description_search_terms:
        search_response += f'Most Similar rdfs:label Search Results for {term}:{[{"label": doc.metadata["label"], "uri": doc.metadata["uri"], "description": doc.page_content} for doc in description_vector_store.similarity_search(term, k=3)]}\n\n'
    search_response = search_response.replace("},", "},\n")

    response_chain = prompt | model

    config = {}

    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    llm_response = response_chain.invoke({"response": search_response, "text": state["text"]}, config=config)

    result = f"Output of uri_detection_agent: {llm_response.content}"

    if state["debug"]:
        state["instruction"] = ""
        state["results"] += [result]
        return state, f"Search Results:\n{search_response}\n{llm_response.content}"

    return Command(goto="result_checker_agent", update={"instruction": "", "results": state["results"] + [result]})
