import importlib
from typing import Literal

from langgraph.types import Command

from results.B_Task_Modularisation.setup import cIEState, model, langfuse_handler
from results.B_Task_Modularisation.prompts import planner_prompt as prompt
import results.B_Task_Modularisation.prompts
importlib.reload(results.B_Task_Modularisation.prompts)


def agent(state: cIEState) -> Command[Literal["agent_instructor_agent"]] | tuple[cIEState, str]:
    response_chain = prompt | model

    config = {}

    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    response = response_chain.invoke(state, config=config)

    if state["debug"]:
        state["comments"].append("\n-- Planner Agent --\n" + response.content)
        return state, response.content

    return Command(goto="agent_instructor_agent", update={"comments": state["comments"] + ["\n-- Planner Agent --\n" + response.content]})
