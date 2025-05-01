
import re
from typing import Literal

from langgraph.types import Command

from approaches.Supervisor.Gen1.setup import cIEState, model, langfuse_handler
from approaches.Supervisor.Gen1.prompts import entity_extractor_prompt as prompt

import importlib
import approaches.Supervisor.Gen1.prompts
importlib.reload(approaches.Supervisor.Gen1.prompts)


def agent(state: cIEState) -> Command[Literal["agent_instructor_agent"]] | tuple[cIEState, str]:
    response_chain = prompt | model

    config = {}

    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    response = response_chain.invoke(state, config=config)

    response = f"""-- Entity Extraction Agent --
    
{response.content} """

    if state["debug"]:
        state["instruction"] = ""
        state["results"] += [response]
        return state, response

    return Command(goto="result_checker_agent", update={"instruction": "", "results": state["results"] + [response]})
