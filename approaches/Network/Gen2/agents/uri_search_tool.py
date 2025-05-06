import re
import traceback
from typing import Literal

from langgraph.types import Command

from approaches.Network.Gen2.setup import cIEState, label_vector_store, description_vector_store, example_vector_store
from qdrant_client.http import models


def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
    try:
        # Update call_trace with current tool call information
        tool_id = "uri_search_tool"
        tool_input = state.get("tool_input", "")
        
        search_terms = state["tool_input"].split("|")
        search_response = ""
        
        for term in search_terms:
            # Parse search term and mode
            mode = None
            clean_term = term
            
            # Extract mode from term using regex
            if "[" in term and "]" in term:
                mode_match = re.search(r'\[([QP]|PX)\]', term)
                if mode_match:
                    mode = mode_match.group(1)
                    clean_term = re.sub(r'\[([QP]|PX)\]', '', term).strip()
            
            # Default behavior if no mode is specified
            if not mode:
                # Assume it's a label search
                results = label_vector_store.similarity_search_with_score(clean_term, k=3)
                search_response += f'Most Similar Search Results for "{clean_term}" - Default Search Mode (LABEL):\n'
                
                for idx, (doc, sim_score) in enumerate(results):
                    search_response += f"  {idx+1}. Label: {doc.page_content}\n"
                    search_response += f"     URI: {doc.metadata['uri']}\n"
                    search_response += f"     Description: {doc.metadata['description']}\n"
                    search_response += f"     Similarity Score: {sim_score}\n"
                search_response += "\n"
                continue
            
            # Create filter condition if mode is Q or P
            filter_condition = None
            if mode in ['Q', 'P']:
                filter_condition = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.type",
                            match=models.MatchValue(value="entity" if mode == "Q" else "predicate")
                        )
                    ]
                )
            
            # Perform search based on the mode
            if mode in ['Q', 'P']:
                # Search in labels with type filter
                results = label_vector_store.similarity_search_with_score(
                    clean_term, 
                    k=3,
                    filter=filter_condition
                )
                search_response += f'Most Similar Search Results for "{clean_term}" - Search Mode [{mode}]:\n'
                
                for idx, (doc, sim_score) in enumerate(results):
                    search_response += f"  {idx+1}. Label: {doc.page_content}\n"
                    search_response += f"     URI: {doc.metadata['uri']}\n"
                    search_response += f"     Description: {doc.metadata['description']}\n"
                    search_response += f"     Similarity Score: {sim_score}\n"
                    
            elif mode == 'PX':
                # Search in examples collection - this is for predicate examples
                results = example_vector_store.similarity_search_with_score(
                    clean_term, 
                    k=3
                )
                search_response += f'Most Similar Search Results for "{clean_term}" - Search Mode [PX]:\n'
                
                for idx, (doc, sim_score) in enumerate(results):
                    search_response += f"  {idx+1}. Label: {doc.metadata['label']}\n"
                    search_response += f"     URI: {doc.metadata['uri']}\n"
                    search_response += f"     Description: {doc.metadata['description']}\n"
                    search_response += f"     Example: {doc.page_content}\n"
                    search_response += f"     Similarity Score: {sim_score}\n"
            
            search_response += "\n"

        state_update = {
            "tool_input": "",
            "last_call": "URI Search Tool - INPUT: " + state["tool_input"],
            "last_response": search_response,
            "call_trace": state.get("call_trace", []) + [(tool_id, tool_input)]
        }

        if state["debug"]:
            state.update(state_update)
            return state, f"Search Results:\n{search_response}"

        return Command(goto="uri_mapping_and_refinement", update=state_update)
        
    except Exception as e:
        error_message = f"SYSTEM MESSAGE: Error occurred in URI search tool: {str(e)}\n{traceback.format_exc()}"
        state_update = {
            "tool_input": "",
            "agent_instruction": error_message,
            "call_trace": state.get("call_trace", []) + [("uri_search_tool", state.get("tool_input", ""))]
        }
        if state["debug"]:
            state.update(state_update)
            return state, f"Error in search tool: {str(e)}"
        return Command(goto="uri_mapping_and_refinement", update=state_update)

