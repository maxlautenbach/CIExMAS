import re
from typing import Literal

from langgraph.types import Command

from results.C_Error_Incorporation.Gen1_PredEx.setup import cIEState, model, langfuse_handler
from results.C_Error_Incorporation.Gen1_PredEx.prompts import entity_extractor_prompt as prompt

import importlib
import results.C_Error_Incorporation.Gen1_PredEx.prompts
importlib.reload(results.C_Error_Incorporation.Gen1_PredEx.prompts)


def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
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
