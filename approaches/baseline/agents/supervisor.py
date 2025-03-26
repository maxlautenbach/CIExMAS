import importlib
from typing import Literal

import re

from langgraph.constants import END
from langgraph.types import Command

from approaches.baseline.setup import cIEState, model, langfuse_handler
from approaches.baseline.prompts import supervisor_prompt as prompt
import approaches.baseline.prompts

importlib.reload(approaches.baseline.prompts)


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

    goto_match = re.search(r'<goto>(.*?)</goto>', response.content, re.DOTALL)
    if goto_match:
        goto = goto_match.group(1)
        if goto == "FINISH":
            goto = END
    else:
        goto = "supervisor"

    instruction_match = re.search(r'<instruction>(.*?)</instruction>', response.content, re.DOTALL)
    if instruction_match:
        instruction = instruction_match.group(1)
    else:
        instruction = ""

    response = "\n-- Supervisor Agent --\n" + response.content

    if state["debug"]:
        state["messages"].append(response)
        state["instruction"] = instruction
        return state, response

    print("SUPERVISOR FINISHED")

    return Command(goto=goto, update={"messages": state["messages"] + [response], "instruction": instruction})
