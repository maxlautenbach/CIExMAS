import re
import traceback
from typing import Literal
from io import StringIO
import rdflib
from rdflib import Graph

from langgraph.types import Command

from approaches.Network.Gen2.setup import cIEState, model, langfuse_handler
from helper_tools.wikidata_loader import send_query


def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
    try:
        # Parse the turtle input using RDFLib
        g = Graph()
        try:
            g.parse(StringIO(state["tool_input"]), format="turtle")
        except Exception as e:
            error_message = f"Failed to parse turtle input: {str(e)}"
            if state["debug"]:
                state["tool_input"] = error_message
                return state, f"Error parsing turtle input: {str(e)}"
            return Command(goto="uri_mapping_and_refinement", update={"agent_instruction": error_message})
        
        all_results = {}
        
        for s, p, o in g:
            predicate_uri = str(p)
            subject_uri = str(s)
            object_uri = str(o)
            
            triple_key = f"{subject_uri}, {predicate_uri}, {object_uri}"
            
            # Updated super-properties query
            super_properties_query = f"""
            SELECT DISTINCT ?superProperty ?superPropertyLabel ?ValueTypeMatched ?SubjectTypeMatched
            WHERE {{
              # Step 1: Super property relationship
              <{predicate_uri}> wdt:P1647 ?superProperty .

              OPTIONAL {{
                ?superProperty p:P2302 ?csVT .
                ?csVT ps:P2302 wd:Q21510865 .  # value type constraint
                ?csVT pq:P2308 ?expectedValueType .
                <{object_uri}> wdt:P31 ?expectedValueType .
              }}

              OPTIONAL {{
                ?superProperty p:P2302 ?csST .
                ?csST ps:P2302 wd:Q21503250 .  # subject type constraint
                ?csST pq:P2308 ?eST .
                <{subject_uri}> wdt:P31 ?eST .
              }}

              BIND(BOUND(?expectedValueType) AS ?ValueTypeMatched)
              BIND(BOUND(?eST) AS ?SubjectTypeMatched)

              SERVICE wikibase:label {{
                bd:serviceParam wikibase:language "en" .
                ?superProperty rdfs:label ?superPropertyLabel .
              }}
            }}
            ORDER BY DESC(?ValueTypeMatched)
            """
            
            # Updated sub-properties query
            sub_properties_query = f"""
            SELECT DISTINCT ?subProperty ?subPropertyLabel ?ValueTypeMatched ?SubjectTypeMatched
            WHERE {{
              # Step 1: Subproperty of
              ?subProperty wdt:P1647 <{predicate_uri}> .

              OPTIONAL {{
                ?superProperty p:P2302 ?csVT .
                ?csVT ps:P2302 wd:Q21510865 .  # value type constraint
                ?csVT pq:P2308 ?expectedValueType .
                <{object_uri}> wdt:P31 ?expectedValueType .
              }}

              OPTIONAL {{
                ?superProperty p:P2302 ?csST .
                ?csST ps:P2302 wd:Q21503250 .  # subject type constraint
                ?csST pq:P2308 ?eST .
                <{subject_uri}> wdt:P31 ?eST .
              }}

              BIND(BOUND(?expectedValueType) AS ?ValueTypeMatched)
              BIND(BOUND(?eST) AS ?SubjectTypeMatched)

              SERVICE wikibase:label {{
                bd:serviceParam wikibase:language "en" .
                ?subProperty rdfs:label ?subPropertyLabel .
              }}
            }}
            ORDER BY DESC(?ValueTypeMatched)
            """
            
            # Execute queries using send_query from wikidata_loader
            super_properties_results = send_query(super_properties_query)
            sub_properties_results = send_query(sub_properties_query)
            
            # Format the results - updated to handle new query response format
            super_properties = []
            try:
                for binding in super_properties_results["results"]["bindings"]:
                    prop_uri = binding["superProperty"]["value"] if "superProperty" in binding else ""
                    prop_label = binding["superPropertyLabel"]["value"] if "superPropertyLabel" in binding else "No Label"
                    subject_type_matched = binding["SubjectTypeMatched"]["value"] if "SubjectTypeMatched" in binding else "false"
                    value_type_matched = binding["ValueTypeMatched"]["value"] if "ValueTypeMatched" in binding else "false"
                    
                    super_properties.append({
                        "uri": prop_uri, 
                        "label": prop_label, 
                        "subject_type_matched": subject_type_matched, 
                        "value_type_matched": value_type_matched
                    })
            except Exception as e:
                if state["debug"]:
                    print(f"Error parsing super properties: {str(e)}")
                super_properties = []
            
            sub_properties = []
            try:
                for binding in sub_properties_results["results"]["bindings"]:
                    prop_uri = binding["subProperty"]["value"] if "subProperty" in binding else ""
                    prop_label = binding["subPropertyLabel"]["value"] if "subPropertyLabel" in binding else "No Label"
                    subject_type_matched = binding["SubjectTypeMatched"]["value"] if "SubjectTypeMatched" in binding else "false"
                    value_type_matched = binding["ValueTypeMatched"]["value"] if "ValueTypeMatched" in binding else "false"
                    
                    sub_properties.append({
                        "uri": prop_uri, 
                        "label": prop_label, 
                        "subject_type_matched": subject_type_matched, 
                        "value_type_matched": value_type_matched
                    })
            except Exception as e:
                if state["debug"]:
                    print(f"Error parsing sub properties: {str(e)}")
                sub_properties = []
            
            all_results[triple_key] = {
                "subject_uri": subject_uri,
                "predicate_uri": predicate_uri,
                "object_uri": object_uri,
                "super_properties": super_properties,
                "sub_properties": sub_properties
            }
        
        # Format the combined response with updated organization by triple
        response = "Possible Predicate Replacements:\n\n"
        
        for triple_key, results in all_results.items():
            # Check if there are any properties with matches
            has_matches = False
            
            # First, check super properties for any matches
            for prop in results['super_properties']:
                if prop['subject_type_matched'] == "true" or prop['value_type_matched'] == "true":
                    has_matches = True
                    break
            
            # Then check sub properties if no matches found yet
            if not has_matches:
                for prop in results['sub_properties']:
                    if prop['subject_type_matched'] == "true" or prop['value_type_matched'] == "true":
                        has_matches = True
                        break
            
            # Filter out properties without matches if there are properties with matches
            if has_matches:
                filtered_super_properties = [prop for prop in results['super_properties'] 
                                           if prop['subject_type_matched'] == "true" or prop['value_type_matched'] == "true"]
                filtered_sub_properties = [prop for prop in results['sub_properties'] 
                                         if prop['subject_type_matched'] == "true" or prop['value_type_matched'] == "true"]
            else:
                filtered_super_properties = results['super_properties']
                filtered_sub_properties = results['sub_properties']
            
            response += f"## Possible Predicate Replacements for Triple: {triple_key}\n\n"
            
            response += f"Super-properties ({len(filtered_super_properties)}):\n"
            for prop in filtered_super_properties:
                response += f"URI: {prop['uri']}\nLabel: {prop['label']}\n"
                response += f"Subject Type Matched: {prop['subject_type_matched']}\n"
                response += f"Value Type Matched: {prop['value_type_matched']}\n\n"
            
            response += f"Sub-properties ({len(filtered_sub_properties)}):\n"
            for prop in filtered_sub_properties:
                response += f"URI: {prop['uri']}\nLabel: {prop['label']}\n"
                response += f"Subject Type Matched: {prop['subject_type_matched']}\n"
                response += f"Value Type Matched: {prop['value_type_matched']}\n\n"
        
        if state["debug"]:
            state["last_call"] = "URI Search Tool - INPUT: " + state["tool_input"],
            state["tool_input"] = ""
            state["last_response"] = response
            return state, f"Network Traversal Results:\n{response}"

        return Command(goto="uri_mapping_and_refinement", update={"last_response": response, "tool_input": "", "last_call": "Network Traversal Search Tool - INPUT: " + state["tool_input"]})
        
    except Exception as e:
        error_message = f"SYSTEM MESSAGE: Error occurred in Network Traversal Search tool: {str(e)}\n{traceback.format_exc()}"
        if state["debug"]:
            state["last_call"] = "URI Search Tool - INPUT: " + state["tool_input"],
            state["tool_input"] = error_message
            return state, f"Error in Network Traversal Search tool: {str(e)}"
        return Command(goto="uri_mapping_and_refinement", update={"agent_instruction": error_message})