import importlib
import re
from typing import Literal

from langgraph.constants import END
from langgraph.types import Command

from helper_tools.validation import validate_turtle_response
from results.G_KG_Incorporation_NTT.setup import cIEState, model, langfuse_handler
from results.G_KG_Incorporation_NTT.prompts import react_agent_promt as prompt
import results.G_KG_Incorporation_NTT.prompts
importlib.reload(results.G_KG_Incorporation_NTT.prompts)


def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
    response_chain = prompt | model

    config = {}

    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    response = response_chain.invoke(state, config=config).content

    instruction_match = re.search(r'<instruction>(.*?)</instruction>', response, re.DOTALL)
    if instruction_match:
        instruction = instruction_match.group(1)
    else:
        instruction = ""

    id_match = re.search(r'<id>(.*?)</id>', response, re.DOTALL)
    if id_match:
        id = id_match.group(1)
    else:
        id = "main_agent"
        instruction = "\nSYSTEM MESSAGE: RegEx r'<id>(.*?)</id>' doesn't find any match, so the next agent could not be determined.\n please fix."

    turtle_string = ""

    if id == "finish_processing":
        turtle_match = re.search(r'<ttl>(.*?)</ttl>', response, re.DOTALL)
        if turtle_match:
            turtle_string = turtle_match.group(1)
            turtle_valid, error_message = validate_turtle_response(turtle_string)
            if turtle_valid:
                id = END
            else:
                id = "main_agent"
                instruction = "\nSYSTEM MESSAGE: " + error_message + "\n please fix."
        else:
            id = "main_agent"
            instruction = "\nSYSTEM MESSAGE: RegEx r'<ttl>(.*?)</ttl>' doesn't find any match, so the result could not be extracted.\n please fix."



    

    if state["debug"]:
        if id == END:
            state["messages"].append(turtle_string)
        else:
            state["messages"].append(response)
        state["instruction"] = instruction
        return state, response

    if id == END:
        response = turtle_string

    return Command(goto=id, update={"messages": state["messages"] + [response], "instruction": instruction})
