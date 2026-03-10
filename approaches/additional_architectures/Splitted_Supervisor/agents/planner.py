import importlib
from typing import Literal

from langgraph.types import Command

from approaches.additional_architectures.Splitted_Supervisor.setup import cIEState, model, langfuse_handler
from approaches.additional_architectures.Splitted_Supervisor.prompts import planner_prompt as prompt
import approaches.additional_architectures.Splitted_Supervisor.prompts
importlib.reload(approaches.additional_architectures.Splitted_Supervisor.prompts)


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
