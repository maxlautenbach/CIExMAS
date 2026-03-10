import importlib
from typing import Literal

from langgraph.types import Command

from approaches.additional_architectures.Initial_Supervisor.setup import cIEState, model, langfuse_handler
from approaches.additional_architectures.Initial_Supervisor.prompts import relation_extractor_prompt as prompt
import approaches.additional_architectures.Initial_Supervisor.prompts

importlib.reload(approaches.additional_architectures.Initial_Supervisor.prompts)


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

    return Command(goto="supervisor", update={"messages": state["messages"] + [response], "instruction": ""})