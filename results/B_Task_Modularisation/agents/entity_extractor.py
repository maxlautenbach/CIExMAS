
import re
from typing import Literal

from langgraph.types import Command

from results.B_Task_Modularisation.setup import cIEState, model, langfuse_handler
from results.B_Task_Modularisation.prompts import entity_extractor_prompt as prompt

import importlib
import results.B_Task_Modularisation.prompts
importlib.reload(results.B_Task_Modularisation.prompts)


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
