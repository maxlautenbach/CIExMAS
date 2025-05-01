import importlib
import re
from typing import Literal

from langgraph.types import Command

import approaches.Network.Gen2.prompts

importlib.reload(approaches.Network.Gen2.prompts)

from approaches.Network.Gen2.setup import cIEState, model, langfuse_handler
from approaches.Network.Gen2.prompts import uri_mapping_and_refinement_prompt as prompt


def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
    response_chain = prompt | model

    goto="extractor"
    config = {}

    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    response = response_chain.invoke(state, config=config)

    # Extract entities from the response
    content = response.content

    # Extract triples from the response
    triples_match = re.search(r'<triples>(.*?)</triples>', content, re.DOTALL)
    
    # Extract tool ID and tool input from the response
    goto_match = re.search(r'<goto>(.*?)</goto>', content, re.DOTALL)
    tool_input_match = re.search(r'<tool_input>(.*?)</tool_input>', content, re.DOTALL)

    # Initialize the update dict with the last agent response
    update = {
        "last_response": content
    }

    # If triples are found, parse them and add to the state
    if triples_match:
        triples_text = triples_match.group(1).strip()
        # Split the triples text into a list of triples
        triples_list = [triple.strip() for triple in triples_text.split('\n') if triple.strip()]
        update["triples"] = triples_list
    
    # If tool ID is found, add it to the state
    if goto_match:
        goto = goto_match.group(1).strip()
        
    
    # If tool input is found, add it to the state
    if tool_input_match:
        tool_input = tool_input_match.group(1).strip()
        update["tool_input"] = tool_input

    if state["debug"]:
        state.update(update)
        return state, f"Response: {content}"

    return Command(
        goto=goto,
        update=update
    )