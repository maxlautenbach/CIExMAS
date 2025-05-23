import traceback
from typing import Literal
from rdflib import Graph
from langgraph.types import Command
from helper_tools.validation import validate_triple, validate_turtle_response
from approaches.Network.Gen2.setup import cIEState

def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
    """
    Tool function that validates if all triples in a Turtle string are semantically correct.
    """
    try:
        # Update call_trace with current tool call information
        tool_id = "semantic_triple_checking_tool"
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

        # First validate if the Turtle string is valid
        is_valid, error_message = validate_turtle_response(turtle_string)
        if not is_valid:
            state_update = {
                "tool_input": "",
                "agent_instruction": f"SYSTEM MESSAGE: Invalid Turtle string: {error_message}",
                "call_trace": state.get("call_trace", []) + [tool_id]
            }
            if state["debug"]:
                state.update(state_update)
                return state, f"Error: Invalid Turtle string: {error_message}"
            return Command(goto="validation_and_output", update=state_update)

        # Parse the Turtle string into a graph
        graph = Graph()
        graph.parse(data=turtle_string, format="turtle")

        # Validate each triple
        results = []
        for subject, predicate, obj in graph:
            subject_restriction_matched, unmatched_subject_constraints, object_restriction_matched, unmatched_object_constraints = validate_triple(
                str(subject), str(predicate), str(obj)
            )
            
            result = f"Triple: {subject} {predicate} {obj}\n"
            result += f"Subject Type Restriction Matched: {subject_restriction_matched}"
            if not subject_restriction_matched and unmatched_subject_constraints:
                result += f" (Unmatched types: {', '.join(unmatched_subject_constraints)})"
            result += f"\nObject Type Restriction Matched: {object_restriction_matched}"
            if not object_restriction_matched and unmatched_object_constraints:
                result += f" (Unmatched types: {', '.join(unmatched_object_constraints)})"
            result += "\n"
            results.append(result)

        response = "Semantic Triple Checking Results:\n\n" + "\n".join(results)
        
        state_update = {
            "tool_input": "",
            "last_response": response,
            "last_call": tool_id,
            "call_trace": state.get("call_trace", []) + [tool_id]
        }

        if state["debug"]:
            state.update(state_update)
            return state, response

        return Command(goto="validation_and_output", update=state_update)
            
    except Exception as e:
        error_message = f"Error occurred in semantic triple checking tool: {str(e)}\n{traceback.format_exc()}"
        state_update = {
            "tool_input": "",
            "agent_instruction": f"SYSTEM MESSAGE: {error_message}",
            "call_trace": state.get("call_trace", []) + [(tool_id, state.get("tool_input", ""))]
        }
        if state["debug"]:
            state.update(state_update)
            return state, f"Error in semantic triple checking tool: {str(e)}"
        return Command(goto="validation_and_output", update=state_update)

if __name__ == "__main__":
    # Example usage
    example_state = {
        "tool_input": """
        @prefix wd: <http://www.wikidata.org/entity/> .
        wd:Q567 wd:P4100 wd:Q1023134 .
        """,
        "debug": True
    }
    result = agent(example_state)
    print(result) 