import importlib
import re
from typing import Literal

from langgraph.types import Command

from approaches.full_sentence.One_Agent.setup import cIEState, model, langfuse_handler
from approaches.full_sentence.One_Agent.prompts import react_agent_promt as prompt
import approaches.full_sentence.One_Agent.prompts
importlib.reload(approaches.full_sentence.One_Agent.prompts)


def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
    response_chain = prompt | model

    config = {}

    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    response = response_chain.invoke(state, config=config)

    instruction_match = re.search(r'<instruction>(.*?)</instruction>', response.content, re.DOTALL)
    if instruction_match:
        instruction = instruction_match.group(1)
    else:
        instruction = ""

    if state["debug"]:
        state["messages"].append("\n-- ReAct Agent --\n" + response.content)
        state["instruction"] = instruction
        return state, response.content

    return Command(goto="main_agent", update={"messages": state["messages"] + ["\n-- ReAct Agent --\n" + response.content], "instruction": instruction})
