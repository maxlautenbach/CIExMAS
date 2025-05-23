import importlib
import re
from typing import Literal

from langgraph.types import Command

import approaches.Network.Gen2.prompts
importlib.reload(approaches.Network.Gen2.prompts)

from approaches.Network.Gen2.setup import cIEState, model, langfuse_handler
from approaches.Network.Gen2.prompts import extractor_prompt as prompt


def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
    # Update call_trace with current agent call information
    agent_id = "extractor"
    agent_instruction = state.get("agent_instruction", "")
    
    # Define available gotos for this agent
    available_gotos = ["extractor", "validation_and_output", "uri_mapping_and_refinement"]
    
    # Define available agents (excluding tools)
    available_agents = ["extractor", "validation_and_output", "uri_mapping_and_refinement"]
    
    response_chain = prompt | model

    config = {}

    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    response = response_chain.invoke(state, config=config)

    # Extract entities from the response
    content = response.content
    
    # Extract triples from the response
    triples_match = re.search(r'<triples>(.*?)</triples>', content, re.DOTALL)

    goto_match = re.search(r'<goto>(.*?)</goto>', content, re.DOTALL)
    agent_instruction_match = re.search(r'<agent_instruction>(.*?)</agent_instruction>', content, re.DOTALL)
    
    # Initialize the update dict with the last agent response
    update = {
        "last_response": content,
        "last_call": agent_id,
        "call_trace": state.get("call_trace", []) + [(agent_id, agent_instruction)]
    }
    
    # If triples are found, parse them and add to the state
    if triples_match:
        triples_text = triples_match.group(1).strip()
        # Split the triples text into a list of triples
        triples_list = [triple.strip() for triple in triples_text.split('\n') if triple.strip()]
        update["triples"] = triples_list

    if goto_match:
        goto = goto_match.group(1).strip()
        # Check if the specified goto is valid for this agent
        if goto not in available_gotos:
            update["agent_instruction"] = f"\nSYSTEM MESSAGE: The specified goto '{goto}' is not valid for this agent. Valid options are: {', '.join(available_gotos)}. Please provide a valid goto instruction."
            goto = "extractor"
        elif not agent_instruction_match:
            # Clear agent_instruction when calling another agent without new instruction
            update["agent_instruction"] = ""
    else:
        update["agent_instruction"] = "You did not provide a valid goto instruction. Please provide a valid goto instruction."
        goto = "extractor"

    if state["debug"]:
        state.update(update)
        return state, f"Response: {content}"

    return Command(
        goto=goto,
        update=update
    )