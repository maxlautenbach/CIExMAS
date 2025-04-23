import importlib
import re
import json
from typing import Literal, Dict

from langgraph.types import Command

import approaches.full_sentence.Gen1v2.prompts
importlib.reload(approaches.full_sentence.Gen1v2.prompts)

from approaches.full_sentence.Gen1v2.setup import cIEState, model, langfuse_handler, label_vector_store, description_vector_store
from approaches.full_sentence.Gen1v2.prompts import uri_retriever_prompt as prompt
from qdrant_client.http import models



def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
    search_terms = state["instruction"].split(",")
    
    # Parse search terms and their modes
    label_search_terms = []
    description_search_terms = []
    label_filters = []
    description_filters = []
    
    for term in search_terms:
        if "[LABEL" in term:
            # Extract filter mode if present
            filter_match = re.search(r'\[LABEL-([QP])\]', term)
            filter_mode = filter_match.group(1) if filter_match else None
            clean_term = term.replace(f"[LABEL-{filter_mode}]", "").replace("[LABEL]", "").strip()
            label_search_terms.append(clean_term)
            label_filters.append(filter_mode)
        elif "[DESCR" in term:
            filter_match = re.search(r'\[DESCR-([QP])\]', term)
            filter_mode = filter_match.group(1) if filter_match else None
            clean_term = term.replace(f"[DESCR-{filter_mode}]", "").replace("[DESCR]", "").strip()
            description_search_terms.append(clean_term)
            description_filters.append(filter_mode)

    # Perform similarity searches with filters
    search_response = ""
    for i, term in enumerate(label_search_terms):
        filter_mode = label_filters[i]
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.type",
                    match=models.MatchValue(value="entity" if filter_mode == "Q" else "predicate" if filter_mode == "P" else None)
                )
            ]
        )
        
        results = label_vector_store.similarity_search(
            term, 
            k=3,
            filter=filter_condition
        )
        search_response += f'Most Similar rdfs:label Search Results for {term}:{[{"label": doc.page_content, "uri": doc.metadata["uri"], "description": doc.metadata["description"]} for doc in results]}\n\n'
    
    for i, term in enumerate(description_search_terms):
        filter_mode = description_filters[i]
        filter_criteria = {"type": "entity"} if filter_mode == "Q" else {"type": "predicate"} if filter_mode == "P" else None
        
        results = description_vector_store.similarity_search(
            term, 
            k=3,
            filter=filter_criteria
        )
        search_response += f'Most Similar schema:description Search Results for {term}:{[{"label": doc.metadata["label"], "uri": doc.metadata["uri"], "description": doc.page_content} for doc in results]}\n\n'
    
    search_response = search_response.replace("},", "},\n")

    # Process the search results with the LLM
    response_chain = prompt | model
    config = {}
    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    response = response_chain.invoke(
        {
            "text": state["text"],
            "search_response": search_response,
            "uri_mapping": state["uri_mapping"]
        },
        config=config
    )

    # Extract URI mappings from the response
    content = response.content
    mappings_match = re.search(r'<uri_mappings>(.*?)</uri_mappings>', content, re.DOTALL)
    updated_mappings: Dict[str, Dict[str, str]] = {}
    
    if mappings_match:
        try:
            # Parse JSON-like format
            mappings_text = mappings_match.group(1).strip()
            mappings_list = json.loads(mappings_text)
            
            # Convert list to dictionary format
            for mapping in mappings_list:
                text = mapping.get("text", "").strip()
                if text and "uri" in mapping:
                    updated_mappings[text] = {
                        "uri": mapping.get("uri", "").strip(),
                        "label": mapping.get("label", text).strip(),
                        "description": mapping.get("description", "").strip(),
                        "note": mapping.get("note", "").strip()
                    }
        except json.JSONDecodeError as e:
            # If JSON parsing fails, add an error message
            if state["debug"]:
                print(f"Error parsing LLM response: {str(e)}")
                print(f"Response content: {mappings_text}")
    
    # Update state with the complete URI mappings
    update = {
        "uri_mapping": updated_mappings,
        "agent_response": content
    }

    if state["debug"]:
        state.update(update)
        return state, f"Search Results:\n{search_response}\nLLM Response: {content}"

    return Command(
        goto="planner",
        update=update
    ) 