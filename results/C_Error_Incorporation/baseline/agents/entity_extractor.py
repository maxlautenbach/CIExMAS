import importlib
from typing import Literal

from langgraph.types import Command

from results.C_Error_Incorporation.baseline.setup import cIEState, model, langfuse_handler
from results.C_Error_Incorporation.baseline.prompts import entity_extractor_prompt as prompt
import results.C_Error_Incorporation.baseline.prompts

importlib.reload(results.C_Error_Incorporation.baseline.prompts)


def agent(state: cIEState) -> Command[Literal["supervisor"]] | tuple[cIEState, str]:
    config = {}

    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    chain = prompt | model
    response = chain.invoke({"text": state["text"], "instruction": state["instruction"]}, config=config)

    response = "\n-- Entity Extraction Agent --\n" + response.content

    if state["debug"]:
        state["messages"].append(response)
        state["instruction"] = ""
        return state, response

    return Command(goto="supervisor", update={"messages": state["messages"] + [response], "instruction": ""})