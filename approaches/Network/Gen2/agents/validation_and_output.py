import importlib
import re
import html
from langgraph.constants import END
from typing import Literal

from langgraph.types import Command

import approaches.Network.Gen2.prompts
from helper_tools.validation import validate_turtle_response

importlib.reload(approaches.Network.Gen2.prompts)

from approaches.Network.Gen2.setup import cIEState, model, langfuse_handler
from approaches.Network.Gen2.prompts import validation_and_output_prompt as prompt


def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
    # Update call_trace with current agent call information
    agent_id = "validation_and_output"
    agent_instruction = state.get("agent_instruction", "")
    
    # Define available gotos for this agent
    available_gotos = ["extractor", "uri_mapping_and_refinement", "END", "validation_and_output", "turtle_to_labels_tool", "semantic_triple_checking_tool"]
    
    # Define available agents (excluding tools and END)
    available_agents = ["extractor", "uri_mapping_and_refinement", "validation_and_output"]
    
    response_chain = prompt | model

    config = {}

    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    response = response_chain.invoke(state, config=config)

    # Extract entities from the response
    content = response.content

    # Extract tool ID and tool input from the response
    goto_match = re.search(r'<goto>(.*?)</goto>', content, re.DOTALL)
    tool_input_match = re.search(r'<tool_input>(.*?)</tool_input>', content, re.DOTALL)
    agent_instruction_match = re.search(r'<agent_instruction>(.*?)</agent_instruction>', content, re.DOTALL)

    # Initialize the update dict with the last agent response
    update = {
        "last_response": content,
        "last_call": agent_id,
        "call_trace": state.get("call_trace", []) + [agent_id]
    }

    # If agent instruction is found, add it to the state
    if agent_instruction_match:
        agent_instruction = agent_instruction_match.group(1).strip()
        update["agent_instruction"] = agent_instruction

    goto = "validation_and_output"

    # If tool ID is found, add it to the state
    if goto_match:
        goto = goto_match.group(1).strip()
        # Check if the specified goto is valid for this agent
        if goto not in available_gotos:
            update["agent_instruction"] = f"\nSYSTEM MESSAGE: The specified goto '{goto}' is not valid for this agent. Valid options are: {', '.join(available_gotos)}. Please provide a valid goto instruction."
            goto = "validation_and_output"
        elif not agent_instruction_match:
            # Clear agent_instruction when calling another agent without new instruction
            update["agent_instruction"] = ""
    else:
        update["agent_instruction"] = "\nSYSTEM MESSAGE: RegEx r'<goto>(.*?)</goto>' doesn't find any match, so the result could not be extracted.\n please fix."
        goto = "validation_and_output"

    # If tool input is found, add it to the state
    if tool_input_match:
        tool_input = tool_input_match.group(1).strip()
        # Unescape HTML entities in tool input
        tool_input = html.unescape(tool_input)
        update["tool_input"] = tool_input

    if goto == "END":
        turtle_match = re.search(r'<ttl>(.*?)</ttl>', content, re.DOTALL)
        if turtle_match:
            turtle_string = turtle_match.group(1)
            # Unescape HTML entities in turtle string
            turtle_string = html.unescape(turtle_string)
            turtle_valid, error_message = validate_turtle_response(turtle_string)
            if turtle_valid:
                update["messages"] = state["messages"] + [turtle_string]
                goto = END
            else:
                update["agent_instruction"] = "\nSYSTEM MESSAGE: " + error_message + "\n please fix."
                goto = "validation_and_output"
        else:
            update["agent_instruction"] = "\nSYSTEM MESSAGE: RegEx r'<ttl>(.*?)</ttl>' doesn't find any match, so the result could not be extracted.\n please fix."
            goto = "validation_and_output"

    if state["debug"]:
        state.update(update)
        return state, f"Response: {content}"

    return Command(
        goto=goto,
        update=update
    )