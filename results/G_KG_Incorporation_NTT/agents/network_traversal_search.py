import re
import traceback
from typing import Literal

from langgraph.types import Command

from results.G_KG_Incorporation_NTT.setup import cIEState, model, langfuse_handler
from helper_tools.base_setup import wikidata_predicate_graph


def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
    try:
        # Get the predicate URI from the instruction
        predicate_uri = state["instruction"].strip()
        
        # Query for super-properties using wikidata_predicate_graph.query()
        super_properties_query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX schema: <http://schema.org/>
        
        SELECT ?superProperty ?superPropertyLabel ?superPropertyDescription
        WHERE {{
          <{predicate_uri}> rdfs:subPropertyOf ?superProperty .
          OPTIONAL {{ ?superProperty rdfs:label ?superPropertyLabel . FILTER(LANG(?superPropertyLabel) = "en") }}
          OPTIONAL {{ ?superProperty rdfs:comment ?superPropertyDescription . FILTER(LANG(?superPropertyDescription) = "en") }}
        }}
        """
        
        # Query for sub-properties using wikidata_predicate_graph.query()
        sub_properties_query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX schema: <http://schema.org/>
        
        SELECT ?subProperty ?subPropertyLabel ?subPropertyDescription
        WHERE {{
          ?subProperty rdfs:subPropertyOf <{predicate_uri}> .
          OPTIONAL {{ ?subProperty rdfs:label ?subPropertyLabel . FILTER(LANG(?subPropertyLabel) = "en") }}
          OPTIONAL {{ ?subProperty rdfs:comment ?subPropertyDescription . FILTER(LANG(?subPropertyDescription) = "en") }}
        }}
        """
        
        # Execute queries using wikidata_predicate_graph.query()
        super_properties_results = wikidata_predicate_graph.query(super_properties_query)
        sub_properties_results = wikidata_predicate_graph.query(sub_properties_query)
        
        # Format the results
        super_properties = []
        for result in super_properties_results:
            prop_uri = str(result.superProperty) if result.superProperty else ""
            prop_label = str(result.superPropertyLabel) if hasattr(result, "superPropertyLabel") and result.superPropertyLabel else "No Label"
            prop_desc = str(result.superPropertyDescription) if hasattr(result, "superPropertyDescription") and result.superPropertyDescription else "No Description"
            super_properties.append({"uri": prop_uri, "label": prop_label, "description": prop_desc})
        
        sub_properties = []
        for result in sub_properties_results:
            prop_uri = str(result.subProperty) if result.subProperty else ""
            prop_label = str(result.subPropertyLabel) if hasattr(result, "subPropertyLabel") and result.subPropertyLabel else "No Label"
            prop_desc = str(result.subPropertyDescription) if hasattr(result, "subPropertyDescription") and result.subPropertyDescription else "No Description"
            sub_properties.append({"uri": prop_uri, "label": prop_label, "description": prop_desc})
        
        response = f"Network traversal results for predicate: {predicate_uri}\n\n"
        response += f"Super-properties ({len(super_properties)}):\n"
        for prop in super_properties:
            response += f"URI: {prop['uri']}\nLabel: {prop['label']}\nDescription: {prop['description']}\n\n"
        
        response += f"Sub-properties ({len(sub_properties)}):\n"
        for prop in sub_properties:
            response += f"URI: {prop['uri']}\nLabel: {prop['label']}\nDescription: {prop['description']}\n\n"
        
        if state["debug"]:
            state["instruction"] = ""
            state["messages"] += [response]
            return state, f"Network Traversal Results:\n{response}"

        return Command(goto="main_agent", update={"messages": state["messages"] + [response], "instruction": ""})
        
    except Exception as e:
        error_message = f"SYSTEM MESSAGE: Error occurred in Network Traversal Search tool: {str(e)}\n{traceback.format_exc()}"
        if state["debug"]:
            state["instruction"] = error_message
            return state, f"Error in Network Traversal Search tool: {str(e)}"
        return Command(goto="main_agent", update={"instruction": error_message})