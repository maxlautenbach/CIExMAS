import importlib
from typing import Literal

import re

from langgraph.constants import END
from langgraph.types import Command

from approaches.Supervisor.baseline.setup import cIEState, model, langfuse_handler
from approaches.Supervisor.baseline.prompts import supervisor_prompt as prompt
import approaches.Supervisor.baseline.prompts
from helper_tools.validation import validate_turtle_response

importlib.reload(approaches.Supervisor.baseline.prompts)


def agent(state: cIEState) -> Command[Literal[
    "entity_extraction_agent", "relation_extraction_agent", "uri_detection_agent", END]] | tuple[cIEState, str]:
    config = {}

    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    chain = prompt | model

    response = chain.invoke({"history": state["messages"], "text": state["text"]}, config=config)

    if response.content == "":
        print(state["messages"])
        return Command(goto=END)

    turtle_string = ""

    goto_match = re.search(r'<goto>(.*?)</goto>', response.content, re.DOTALL)
    if goto_match:
        goto = goto_match.group(1)
        if goto == "FINISH":
            ttl_match = re.search(r'<ttl>(.*?)</ttl>', response.content, re.DOTALL)
            if ttl_match:
                turtle_string = ttl_match.group(1)
                turtle_valid, error_message = validate_turtle_response(turtle_string)
                if turtle_valid:
                    goto = END
                else:
                    response += "SYSTEM MESSAGE: " + error_message + "\n please fix."
                    goto = "supervisor"
            else:
                response += "SYSTEM MESSAGE: To finish, please also include the turtle output enclosed in <ttl>INSERT TURTLE OUTPUT HERE</ttl>."
                goto = "supervisor"

    else:
        response += "SYSTEM MESSAGE: You missed to include a goto."
        goto = "supervisor"

    instruction_match = re.search(r'<instruction>(.*?)</instruction>', response.content, re.DOTALL)
    if instruction_match:
        instruction = instruction_match.group(1)
    else:
        instruction = ""

    response = "\n-- Supervisor Agent --\n" + response.content

    if goto != END and goto not in ["entity_extraction_agent", "relation_extraction_agent", "uri_detection_agent", "supervisor"]:
        response += "SYSTEM MESSAGE: The agent you called is non-existant. Please call one of the following: entity_extraction_agent, relation_extraction_agent, uri_detection_agent or FINISH."
        goto = "supervisor"

    if state["debug"]:
        if goto == END:
            state["messages"].append(turtle_string)
        else:
            state["messages"].append("\n-- Supervisor Agent --\n" + response)
        state["instruction"] = instruction
        return state, response

    if goto == END:
        response = turtle_string

    return Command(goto=goto, update={"messages": state["messages"] + [response], "instruction": instruction})
