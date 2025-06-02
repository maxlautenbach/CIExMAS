import importlib
import re
import traceback
from typing import Literal

from langgraph.types import Command

from approaches.Supervisor.baseline.setup import cIEState, model, langfuse_handler
from approaches.Supervisor.baseline.prompts import uri_detection_prompt as prompt
import approaches.Supervisor.baseline.prompts
from qdrant_client.http import models
from helper_tools.base_setup import label_vector_store, description_vector_store, example_vector_store
import importlib

importlib.reload(approaches.Supervisor.baseline.prompts)


def agent(state: cIEState) -> Command[Literal["supervisor"]] | tuple[cIEState, str]:
    try:
        search_terms = state["instruction"].split(",")
        search_response = ""
        
        for term in search_terms:
            term = term.strip()
            if not term:
                continue
                
            # Parse search term and mode
            mode = None
            clean_term = term
            
            # Extract mode from term using regex
            if "[" in term and "]" in term:
                mode_match = re.search(r'\[([QPX])\]', term)
                if mode_match:
                    mode = mode_match.group(1)
                    clean_term = re.sub(r'\[([QPX])\]', '', term).strip()
            
            # Default behavior if no mode is specified
            if not mode:
                # Assume it's a label search
                results = label_vector_store.similarity_search_with_score(clean_term, k=3)
                search_response += f'Most Similar Search Results for "{clean_term}" - Default Search Mode (LABEL):\n'
                
                for idx, (doc, sim_score) in enumerate(results):
                    search_response += f"  {idx+1}. Label: {doc.page_content}\n"
                    search_response += f"     URI: {doc.metadata['uri']}\n"
                    search_response += f"     Description: {doc.metadata['description']}\n"
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
                search_response += f'Similar Search Results for "{clean_term}" - Search Mode [{mode}]:\n'
                
                for idx, (doc, sim_score) in enumerate(results):
                    search_response += f"  - Label: {doc.page_content}\n"
                    search_response += f"     URI: {doc.metadata['uri']}\n"
                    search_response += f"     Description: {doc.metadata['description']}\n"
                    if doc.metadata.get("example"):
                        search_response += f"     Example: {doc.metadata['example']}\n"

            elif mode == 'X':
                # Search in examples collection
                results = example_vector_store.similarity_search_with_score(
                    clean_term, 
                    k=3
                )
                search_response += f'Similar Search Results for "{clean_term}" - Search Mode [X]:\n'
                
                for idx, (doc, sim_score) in enumerate(results):
                    search_response += f"  - Label: {doc.metadata['label']}\n"
                    search_response += f"     URI: {doc.metadata['uri']}\n"
                    search_response += f"     Description: {doc.metadata['description']}\n"
                    search_response += f"     Example: {doc.page_content}\n"

            search_response += "\n"

        config = {}

        if state["debug"]:
            config = {"callbacks": [langfuse_handler]}

        chain = prompt | model
        response = chain.invoke({"search_response": search_response}, config=config)

        response = "\n-- URI Detection Agent --\n" + response.content

        if state["debug"]:
            state["messages"].append(response)
            state["instruction"] = ""
            return state, "\n-- Search Response --\n" + search_response + response

        return Command(goto="supervisor", update={"messages": state["messages"] + [response], "instruction": ""})
        
    except Exception as e:
        error_message = f"SYSTEM MESSAGE: Error occurred in URI detection agent: {str(e)}\n{traceback.format_exc()}"
        if state["debug"]:
            return state, f"Error in URI detection agent: {str(e)}"
        return Command(goto="supervisor", update={"messages": state["messages"] + [error_message], "instruction": ""})
