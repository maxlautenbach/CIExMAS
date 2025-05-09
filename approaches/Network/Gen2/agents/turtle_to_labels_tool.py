import re
import traceback
from typing import List, Tuple, Literal
from rdflib import Graph, URIRef
from helper_tools.wikidata_loader import send_query
from langgraph.types import Command
from approaches.Network.Gen2.setup import cIEState

def parse_turtle(turtle_string: str) -> List[Tuple[URIRef, URIRef, URIRef]]:
    """Parse turtle string into list of triples using rdflib."""
    g = Graph()
    try:
        g.parse(data=turtle_string, format="turtle")
        return list(g.triples((None, None, None)))
    except Exception as e:
        raise ValueError(f"Failed to parse turtle string: {str(e)}")

def get_labels_for_uri(uri: str) -> str:
    """Get label for a Wikidata URI using SPARQL."""
    # Use the full URI in the SPARQL query
    query = f"""
    SELECT ?label
    WHERE {{
        <{uri}> rdfs:label ?label .
        FILTER(LANG(?label) = "en")
    }}
    LIMIT 1
    """
    
    response = send_query(query)
    if response and 'results' in response and 'bindings' in response['results']:
        bindings = response['results']['bindings']
        if bindings and 'label' in bindings[0]:
            return bindings[0]['label']['value']
    return uri

def turtle_to_labels(turtle_string: str) -> str:
    """Convert turtle triples to their corresponding labels."""
    triples = parse_turtle(turtle_string)
    labeled_triples = []
    
    for subj, pred, obj in triples:
        subj_label = get_labels_for_uri(str(subj))
        pred_label = get_labels_for_uri(str(pred))
        obj_label = get_labels_for_uri(str(obj))
        
        labeled_triple = f"{subj_label} {pred_label} {obj_label}"
        labeled_triples.append(labeled_triple)
    
    return "\n".join(labeled_triples)

def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
    """Tool function that converts turtle triples to labels."""
    try:
        # Update call_trace with current tool call information
        tool_id = "turtle_to_labels_tool"
        tool_input = state.get("tool_input", "")
        
        # Get the turtle string from tool input
        turtle_string = tool_input.strip()
        if not turtle_string:
            # If no tool input, try to get from messages
            if "messages" in state and state["messages"]:
                for msg in reversed(state["messages"]):
                    if "@prefix" in msg and "wd:" in msg:
                        turtle_string = msg
                        break
        
        if not turtle_string:
            error_message = "No turtle string provided in tool input or messages"
            state_update = {
                "tool_input": "",
                "agent_instruction": f"SYSTEM MESSAGE: {error_message}",
                "call_trace": state.get("call_trace", []) + [(tool_id, tool_input)]
            }
            if state["debug"]:
                state.update(state_update)
                return state, f"Error: {error_message}"
            return Command(goto="validation_and_output", update=state_update)
        
        try:
            labels = turtle_to_labels(turtle_string)
            response = f"Turtle to Labels Tool Output - \n{labels}"
            
            state_update = {
                "tool_input": "",
                "last_call": "Turtle to Labels Tool - INPUT: " + turtle_string,
                "last_response": response,
                "call_trace": state.get("call_trace", []) + [(tool_id, tool_input)]
            }

            if state["debug"]:
                state.update(state_update)
                return state, f"Labels:\n{labels}"

            return Command(goto="validation_and_output", update=state_update)
            
        except Exception as e:
            error_message = f"Failed to process turtle string: {str(e)}"
            state_update = {
                "tool_input": "",
                "agent_instruction": f"SYSTEM MESSAGE: {error_message}",
                "call_trace": state.get("call_trace", []) + [(tool_id, tool_input)]
            }
            if state["debug"]:
                state.update(state_update)
                return state, f"Error: {error_message}"
            return Command(goto="validation_and_output", update=state_update)
            
    except Exception as e:
        error_message = f"Error occurred in turtle to labels tool: {str(e)}\n{traceback.format_exc()}"
        state_update = {
            "tool_input": "",
            "agent_instruction": f"SYSTEM MESSAGE: {error_message}",
            "call_trace": state.get("call_trace", []) + [(tool_id, state.get("tool_input", ""))]
        }
        if state["debug"]:
            state.update(state_update)
            return state, f"Error in turtle to labels tool: {str(e)}"
        return Command(goto="validation_and_output", update=state_update)
