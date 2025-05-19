import re
import traceback
from typing import Literal
from io import StringIO
import rdflib
from rdflib import Graph

from langgraph.types import Command

from approaches.Network.Gen2.setup import cIEState, model, langfuse_handler
from helper_tools.wikidata_loader import send_query, get_property_example, get_label


def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
    try:
        # Update call_trace with current tool call information
        tool_id = "network_traversal_search"
        tool_input = state.get("tool_input", "")
        
        # Parse the turtle input using RDFLib
        g = Graph()
        try:
            g.parse(StringIO(state["tool_input"]), format="turtle")
        except Exception as e:
            error_message = f"Failed to parse turtle input: {str(e)}"
            if state["debug"]:
                state["tool_input"] = error_message
                return state, f"Error parsing turtle input: {str(e)}"
            return Command(goto="uri_mapping_and_refinement", update={
                "agent_instruction": error_message,
                "call_trace": state.get("call_trace", []) + [(tool_id, tool_input)]
            })
        
        all_results = {}
        
        for s, p, o in g:
            predicate_uri = str(p)
            subject_uri = str(s)
            object_uri = str(o)
            
            triple_key = f"{subject_uri}, {predicate_uri}, {object_uri}"
            
            # Updated super-properties query
            super_properties_query = f"""
            SELECT DISTINCT ?superProperty ?superPropertyLabel ?ValueTypeMatched ?SubjectTypeMatched ?valueTypeLabel ?subjectTypeLabel
            WHERE {{
              # Step 1: Super property relationship
              <{predicate_uri}> wdt:P1647 ?superProperty .

              OPTIONAL {{
                ?superProperty p:P2302 ?csVT .
                ?csVT ps:P2302 wd:Q21510865 .  # value type constraint
                ?csVT pq:P2308 ?expectedValueType .
                <{object_uri}> wdt:P31|wdt:P31/wdt:P279 ?expectedValueType .
                OPTIONAL {{ ?expectedValueType rdfs:label ?valueTypeLabel . FILTER(LANG(?valueTypeLabel) = "en") }}
              }}

              OPTIONAL {{
                ?superProperty p:P2302 ?csST .
                ?csST ps:P2302 wd:Q21503250 .  # subject type constraint
                ?csST pq:P2308 ?eST .
                <{subject_uri}> wdt:P31|wdt:P31/wdt:P279 ?eST .
                OPTIONAL {{ ?eST rdfs:label ?subjectTypeLabel . FILTER(LANG(?subjectTypeLabel) = "en") }}
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
            SELECT DISTINCT ?subProperty ?subPropertyLabel ?ValueTypeMatched ?SubjectTypeMatched ?valueTypeLabel ?subjectTypeLabel
            WHERE {{
              # Step 1: Subproperty of
              ?subProperty wdt:P1647 <{predicate_uri}> .

              OPTIONAL {{
                ?subProperty p:P2302 ?csVT .
                ?csVT ps:P2302 wd:Q21510865 .  # value type constraint
                ?csVT pq:P2308 ?expectedValueType .
                <{object_uri}> wdt:P31|wdt:P31/wdt:P279 ?expectedValueType .
                OPTIONAL {{ ?expectedValueType rdfs:label ?valueTypeLabel . FILTER(LANG(?valueTypeLabel) = "en") }}
              }}

              OPTIONAL {{
                ?subProperty p:P2302 ?csST .
                ?csST ps:P2302 wd:Q21503250 .  # subject type constraint
                ?csST pq:P2308 ?eST .
                <{subject_uri}> wdt:P31|wdt:P31/wdt:P279 ?eST .
                OPTIONAL {{ ?eST rdfs:label ?subjectTypeLabel . FILTER(LANG(?subjectTypeLabel) = "en") }}
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
                    value_type_label = binding["valueTypeLabel"]["value"] if "valueTypeLabel" in binding else "No Label"
                    subject_type_label = binding["subjectTypeLabel"]["value"] if "subjectTypeLabel" in binding else "No Label"
                    
                    super_properties.append({
                        "uri": prop_uri, 
                        "label": prop_label, 
                        "subject_type_matched": subject_type_matched, 
                        "value_type_matched": value_type_matched,
                        "value_type_label": value_type_label,
                        "subject_type_label": subject_type_label
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
                    value_type_label = binding["valueTypeLabel"]["value"] if "valueTypeLabel" in binding else "No Label"
                    subject_type_label = binding["subjectTypeLabel"]["value"] if "subjectTypeLabel" in binding else "No Label"
                    
                    sub_properties.append({
                        "uri": prop_uri, 
                        "label": prop_label, 
                        "subject_type_matched": subject_type_matched, 
                        "value_type_matched": value_type_matched,
                        "value_type_label": value_type_label,
                        "subject_type_label": subject_type_label
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
            # Order properties based on match criteria - prioritize both matches, then subject matches, then value matches
            def get_priority(prop):
                both_matched = prop['subject_type_matched'] == "true" and prop['value_type_matched'] == "true"
                subject_matched = prop['subject_type_matched'] == "true"
                value_matched = prop['value_type_matched'] == "true"
                
                if both_matched:
                    return 0  # highest priority
                elif subject_matched:
                    return 1
                elif value_matched:
                    return 2
                else:
                    return 3  # lowest priority
            
            # Sort properties by priority
            all_super_properties = sorted(results['super_properties'], key=get_priority)
            all_sub_properties = sorted(results['sub_properties'], key=get_priority)
            
            # Only keep properties with the highest priority available
            if all_super_properties:
                highest_prio_super = get_priority(all_super_properties[0])
                ordered_super_properties = [p for p in all_super_properties if get_priority(p) == highest_prio_super]
            else:
                ordered_super_properties = []
                
            if all_sub_properties:
                highest_prio_sub = get_priority(all_sub_properties[0])
                ordered_sub_properties = [p for p in all_sub_properties if get_priority(p) == highest_prio_sub]
            else:
                ordered_sub_properties = []
            
            response += f"## Possible Predicate Replacements for Triple: {triple_key}\n\n"
            
            # Get an example for the current predicate
            predicate_example = get_property_example(results['predicate_uri'])
            if predicate_example:
                response += f"### Current Predicate Example:\n"
                response += f"Property: {results['predicate_uri']}\n"
                current_pred_label = get_label(results['predicate_uri'])
                response += f"Label: {current_pred_label}\n"
                response += f"Example: {predicate_example['subject_label']}; {current_pred_label}; {predicate_example['object_label']}\n"

            response += f"Super-properties ({len(ordered_super_properties)}):\n\n"
            for prop in ordered_super_properties:
                response += f"URI: {prop['uri']}\nLabel: {prop['label']}\n"
                
                # Get and display example for this super-property
                prop_example = get_property_example(prop['uri'])
                if prop_example:
                    response += f"Example: {prop_example['subject_label']}; {prop['label']}; {prop_example['object_label']}\n"

                # Show matched type labels instead of True/False
                if prop['subject_type_matched'] == "true":
                    subject_type = prop['subject_type_label'] if prop['subject_type_label'] != "No Label" else "Unknown Type"
                    response += f"Subject matches type restriction: {subject_type}\n"
                else:
                    response += f"Subject matches type restriction: None\n"
                    
                if prop['value_type_matched'] == "true":
                    value_type = prop['value_type_label'] if prop['value_type_label'] != "No Label" else "Unknown Type"
                    response += f"Object matches type restriction: {value_type}\n"
                else:
                    response += f"Object matches type restriction: None\n"
                
                response += "\n"
            
            response += f"Sub-properties ({len(ordered_sub_properties)}):\n\n"
            for prop in ordered_sub_properties:
                response += f"URI: {prop['uri']}\nLabel: {prop['label']}\n"
                
                # Get and display example for this sub-property
                prop_example = get_property_example(prop['uri'])
                if prop_example:
                    response += f"Example: {prop_example['subject_label']}; {prop['label']}; {prop_example['object_label']}\n"

                # Show matched type labels instead of True/False
                if prop['subject_type_matched'] == "true":
                    subject_type = prop['subject_type_label'] if prop['subject_type_label'] != "No Label" else "Unknown Type"
                    response += f"Subject matches type restriction: {subject_type}\n"
                else:
                    response += f"Subject matches type restriction: None\n"
                    
                if prop['value_type_matched'] == "true":
                    value_type = prop['value_type_label'] if prop['value_type_label'] != "No Label" else "Unknown Type"
                    response += f"Object matches type restriction: {value_type}\n"
                else:
                    response += f"Object matches type restriction: None\n"
                
                response += "\n"
        
        if state["debug"]:
            state["last_call"] = "Network Traversal Search Tool - INPUT: " + state["tool_input"]
            state["tool_input"] = ""
            state["last_response"] = response
            state["call_trace"] = state.get("call_trace", []) + [(tool_id, tool_input)]
            return state, f"Network Traversal Results:\n{response}"

        return Command(goto="uri_mapping_and_refinement", update={
            "last_response": response, 
            "tool_input": "", 
            "last_call": tool_id,
            "call_trace": state.get("call_trace", []) + [tool_id]
        })
        
    except Exception as e:
        error_message = f"SYSTEM MESSAGE: Error occurred in Network Traversal Search tool: {str(e)}\n{traceback.format_exc()}"
        if state["debug"]:
            state["last_call"] = "Network Traversal Search Tool - INPUT: " + state["tool_input"]
            state["tool_input"] = error_message
            state["call_trace"] = state.get("call_trace", []) + [("network_traversal_search", state.get("tool_input", ""))]
            return state, f"Error in Network Traversal Search tool: {str(e)}"
        return Command(goto="uri_mapping_and_refinement", update={
            "agent_instruction": error_message,
            "call_trace": state.get("call_trace", []) + [("network_traversal_search", state.get("tool_input", ""))]
        })

if __name__ == "__main__":
    res = agent({"tool_input": """@prefix wd: <http://www.wikidata.org/entity/>.
wd:Q567 wd:P463 wd:Q49762.
wd:Q567 wd:P106 wd:Q82955.""", "debug": False})
    print(res)