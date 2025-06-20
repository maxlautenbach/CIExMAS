import re
from typing import Literal

from langgraph.types import Command

from results.C_Error_Incorporation.Gen1_PredEx.setup import cIEState, model, langfuse_handler, label_vector_store, description_vector_store
from results.C_Error_Incorporation.Gen1_PredEx.prompts import uri_detector_prompt as prompt
import importlib
import results.C_Error_Incorporation.Gen1_PredEx.prompts
importlib.reload(results.C_Error_Incorporation.Gen1_PredEx.prompts)


def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
    search_terms_match = re.search(r'<search_terms>(.*?)</search_terms>', state["instruction"], re.DOTALL)
    if search_terms_match:
        search_terms = search_terms_match.group(1)
    else:
        search_terms = "NONE"
    additional_instruction_match = re.search(r'<additional_instruction>(.*?)</additional_instruction>', state["instruction"], re.DOTALL)
    if additional_instruction_match:
        additional_instruction = additional_instruction_match.group(1)
    else:
        additional_instruction = ""
    search_terms = search_terms.split(",")
    label_search_terms = [term.replace("[LABEL]", "") for term in search_terms if "[LABEL]" in term]
    description_search_terms = [term.replace("[DESCR]", "") for term in search_terms if "[DESCR]" in term]
    search_response = ""
    for term in label_search_terms:
        search_response += f'Most Similar rdfs:label Search Results for {term}:{[{"label": doc.page_content, "uri": doc.metadata["uri"], "description": doc.metadata["description"]} for doc in label_vector_store.similarity_search(term, k=3)]}\n\n'
    for term in description_search_terms:
        search_response += f'Most Similar schema:description Search Results for {term}:{[{"label": doc.metadata["label"], "uri": doc.metadata["uri"], "description": doc.page_content} for doc in description_vector_store.similarity_search(term, k=3)]}\n\n'
    search_response = search_response.replace("},", "},\n")

    response_chain = prompt | model

    config = {}

    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    llm_response = response_chain.invoke({"response": search_response, "text": state["text"], "instruction": additional_instruction}, config=config)

    result = f"Output of uri_detection_agent: {llm_response.content}"

    if state["debug"]:
        state["instruction"] = ""
        state["results"] += [result]
        return state, f"Search Results:\n{search_response}\n{llm_response.content}"

    return Command(goto="result_checker_agent", update={"instruction": "", "results": state["results"] + [result]})
