import importlib
import re
from langgraph.constants import END
from typing import Literal

from langgraph.types import Command

import approaches.Network.Gen2.prompts
from helper_tools.validation import validate_turtle_response

importlib.reload(approaches.Network.Gen2.prompts)

from approaches.Network.Gen2.setup import cIEState, model, langfuse_handler
from approaches.Network.Gen2.prompts import validation_and_output_prompt as prompt


def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
    response_chain = prompt | model

    config = {}

    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    response = response_chain.invoke(state, config=config)

    # Extract entities from the response
    content = response.content

    goto_match = re.search(r'<goto>(.*?)</goto>', content, re.DOTALL)

    # Initialize the update dict with the last agent response
    update = {
        "last_response": content,
        "last_call": "extractor",
    }
    goto = "validation_and_output"

    # If triples are found, parse them and add to the state
    if goto_match:
        goto = goto_match.group(1).strip()
    else:
        update["agent_instruction"] = "\nSYSTEM MESSAGE: RegEx r'<goto>(.*?)</goto>' doesn't find any match, so the result could not be extracted.\n please fix."


    if goto == "END":
        turtle_match = re.search(r'<ttl>(.*?)</ttl>', content, re.DOTALL)
        if turtle_match:
            turtle_string = turtle_match.group(1)
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