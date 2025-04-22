import importlib
import re
from typing import Literal, Set, Tuple

from langgraph.types import Command

from approaches.full_sentence.Gen1v2.setup import cIEState, model, langfuse_handler, label_vector_store, description_vector_store
from approaches.full_sentence.Gen1v2.prompts import uri_retriever_prompt as prompt
import approaches.full_sentence.Gen1v2.prompts
importlib.reload(approaches.full_sentence.Gen1v2.prompts)


def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
    search_terms = state["instruction"].split(",")
    print(search_terms)
    label_search_terms = [term.replace("[LABEL]", "").strip() for term in search_terms if "[LABEL]" in term]
    description_search_terms = [term.replace("[DESCR]", "").strip() for term in search_terms if "[DESCR]" in term]

    # Perform similarity searches
    search_response = ""
    for term in label_search_terms:
        search_response += f'Most Similar rdfs:label Search Results for {term}:{[{"label": doc.page_content, "uri": doc.metadata["uri"], "description": doc.metadata["description"]} for doc in label_vector_store.similarity_search(term, k=3)]}\n\n'
    for term in description_search_terms:
        search_response += f'Most Similar schema:description Search Results for {term}:{[{"label": doc.metadata["label"], "uri": doc.metadata["uri"], "description": doc.page_content} for doc in description_vector_store.similarity_search(term, k=3)]}\n\n'
    search_response = search_response.replace("},", "},\n")

    # Process the search results with the LLM
    response_chain = prompt | model
    config = {}
    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    response = response_chain.invoke(
        {
            "text": state["text"],
            "search_response": search_response
        },
        config=config
    )

    # Extract URI mappings from the response
    content = response.content
    mappings_match = re.search(r'<uri_mappings>(.*?)</uri_mappings>', content, re.DOTALL)
    new_mappings: Set[Tuple[str, str]] = set()
    
    if mappings_match:
        mappings_text = mappings_match.group(1)
        for line in mappings_text.split('\n'):
            line = line.strip()
            if line and '->' in line:
                text, uri = line.split('->', 1)
                new_mappings.add((text.strip(), uri.strip()))
    
    # Update state with new URI mappings
    update = {
        "uri_mapping": state["uri_mapping"].union(new_mappings),
        "agent_response": content
    }

    if state["debug"]:
        state.update(update)
        return state, f"Search Results:\n{search_response}\nLLM Response: {content}"

    return Command(
        goto="planner",
        update=update
    ) 