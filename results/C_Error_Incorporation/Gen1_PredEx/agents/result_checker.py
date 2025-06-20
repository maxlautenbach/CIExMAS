from typing import Literal

from langgraph.types import Command

from results.C_Error_Incorporation.Gen1_PredEx.setup import cIEState, model, langfuse_handler
from results.C_Error_Incorporation.Gen1_PredEx.prompts import result_checker_prompt as prompt
import importlib
import results.C_Error_Incorporation.Gen1_PredEx.prompts
importlib.reload(results.C_Error_Incorporation.Gen1_PredEx.prompts)


def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
    response_chain = prompt | model

    config = {}

    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    response = response_chain.invoke(state, config=config)

    if state["debug"]:
        state["comments"].append("\n-- Result Checker Agent --\n" + response.content)
        return state, response.content

    return Command(goto="planner", update={"comments": state["comments"] + ["\n-- Result Checker Agent --\n" + response.content]})
