import re
import traceback
from typing import Literal
import json

from langgraph.types import Command

from approaches.full_sentence.One_Agent.setup import cIEState, model, langfuse_handler, label_vector_store, description_vector_store
from qdrant_client.http import models


def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
    try:
        search_terms = state["instruction"].split("|")
        
        # Parse search terms and their modes
        label_search_terms = []
        description_search_terms = []
        label_filters = []
        description_filters = []
        
        for term in search_terms:
            if "[LABEL" in term:
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
            ) if filter_mode else label_vector_store.similarity_search(
                term, 
                k=3
            )
            
            search_response += f'Most Similar Search Results for "{term}" - Search Mode [LABEL]:\n'
            for idx, doc in enumerate(results):
                search_response += f"  {idx+1}. Label: {doc.page_content}\n"
                search_response += f"     URI: {doc.metadata['uri']}\n"
                search_response += f"     Description: {doc.metadata['description']}\n"
            search_response += "\n"
        
        for i, term in enumerate(description_search_terms):
            filter_mode = description_filters[i]
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.type",
                        match=models.MatchValue(value="entity" if filter_mode == "Q" else "predicate" if filter_mode == "P" else None)
                    )
                ]
            )
            
            results = description_vector_store.similarity_search(
                term, 
                k=3,
                filter=filter_condition
            ) if filter_mode else description_vector_store.similarity_search(
                term, 
                k=3
            )
            
            search_response += f'Most Similar Search Results for "{term}" - Search Mode [DESCR]:\n'
            for idx, doc in enumerate(results):
                search_response += f"  {idx+1}. Label: {doc.metadata['label']}\n"
                search_response += f"     URI: {doc.metadata['uri']}\n"
                search_response += f"     Description: {doc.page_content}\n"
            search_response += "\n"

        if state["debug"]:
            state["instruction"] = ""
            state["messages"] += [search_response]
            return state, f"Search Results:\n{search_response}"

        return Command(goto="main_agent", update={"messages": state["messages"] + [search_response], "instruction": ""})
        
    except Exception as e:
        error_message = f"SYSTEM MESSAGE: Error occurred in URI search tool: {str(e)}\n{traceback.format_exc()}"
        if state["debug"]:
            state["instruction"] = error_message
            return state, f"Error in search tool: {str(e)}"
        return Command(goto="main_agent", update={"instruction": error_message})

