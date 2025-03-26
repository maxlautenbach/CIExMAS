import importlib
from typing import Literal

from langgraph.types import Command

from approaches.baseline.setup import cIEState, model, langfuse_handler
from approaches.baseline.prompts import relation_extractor_prompt as prompt
import approaches.baseline.prompts

importlib.reload(approaches.baseline.prompts)


def agent(state: cIEState) -> Command[Literal["supervisor"]] | tuple[cIEState, str]:
    config = {}

    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    chain = prompt | model
    response = chain.invoke({"text": state["text"], "instruction": state["instruction"]}, config=config)

    response = "\n-- Relation Extraction Agent --\n" + response.content

    if state["debug"]:
        state["messages"].append(response)
        state["instruction"] = ""
        return state, response

    print("REL EX FINISHED")

    return Command(goto="supervisor", update={"messages": state["messages"] + [response], "instruction": ""})