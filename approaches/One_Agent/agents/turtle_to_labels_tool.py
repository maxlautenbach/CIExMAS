import re
import traceback
import html
from typing import List, Tuple, Literal
from rdflib import Graph, URIRef
from helper_tools.wikidata_loader import send_query
from langgraph.types import Command
from approaches.One_Agent.setup import cIEState

def parse_turtle(turtle_string: str) -> List[Tuple[URIRef, URIRef, URIRef]]:
    """Parse turtle string into list of triples using rdflib."""
    g = Graph()
    try:
        # Unescape HTML entities in turtle string
        turtle_string = html.unescape(turtle_string)
        g.parse(data=turtle_string, format="turtle")
        return list(g.triples((None, None, None)))
    except Exception as e:
        raise ValueError(f"Failed to parse turtle string: {str(e)}")

def get_labels_for_uri(uri: str) -> str:
    """Get label for a Wikidata URI using SPARQL."""
    # First check if the URI exists
    existence_query = f"""
    ASK {{
        <{uri}> ?p ?o .
    }}
    """
    
    existence_response = send_query(existence_query)
    if not existence_response or not existence_response.get('boolean', False):
        return f"{uri} (URI does not exist)"
    
    # If URI exists, try to get the label
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
    return f"{uri} (URI exists but has no English label)"

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
        tool_input = state.get("instruction", "")
        
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
                "instruction": "",
                "messages": state.get("messages", []) + [f"Error: {error_message}"]
            }
            if state["debug"]:
                state.update(state_update)
                return state, f"Error: {error_message}"
            return Command(goto="main_agent", update=state_update)
        
        try:
            labels = turtle_to_labels(turtle_string)
            response = f"Turtle to Labels Tool Output - \n{labels}"
            
            state_update = {
                "instruction": "",
                "messages": state.get("messages", []) + [response]
            }

            if state["debug"]:
                state.update(state_update)
                return state, f"Labels:\n{labels}"

            return Command(goto="main_agent", update=state_update)
            
        except Exception as e:
            error_message = f"Failed to process turtle string: {str(e)}"
            state_update = {
                "instruction": "",
                "messages": state.get("messages", []) + [f"Error: {error_message}"]
            }
            if state["debug"]:
                state.update(state_update)
                return state, f"Error: {error_message}"
            return Command(goto="main_agent", update=state_update)
            
    except Exception as e:
        error_message = f"Error occurred in turtle to labels tool: {str(e)}\n{traceback.format_exc()}"
        state_update = {
            "instruction": "",
            "messages": state.get("messages", []) + [f"Error: {error_message}"]
        }
        if state["debug"]:
            state.update(state_update)
            return state, f"Error in turtle to labels tool: {str(e)}"
        return Command(goto="main_agent", update=state_update) 
    

if __name__ == "__main__":
    state = {
        "instruction": "@prefix wd: &lt;http://www.wikidata.org/entity/&gt; . wd:Q7654799 wd:P178 wd:Q631844 . wd:Q7654799 wd:P921 wd:Q210272 . wd:Q7654799 wd:P1056 wd:Q54872 . wd:Q7654799 wd:P366 wd:Q2115 . wd:Q7654799 wd:P366 wd:Q2063 . wd:Q7654799 wd:P366 wd:Q6108942 . wd:Q2115 wd:P941 wd:Q207819",
        "messages": [],
        "call_trace": [],
    }