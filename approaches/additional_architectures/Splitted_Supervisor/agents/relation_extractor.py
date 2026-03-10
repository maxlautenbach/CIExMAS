import re
from typing import Literal

from langgraph.types import Command

from approaches.additional_architectures.Splitted_Supervisor.setup import cIEState, model, langfuse_handler
from approaches.additional_architectures.Splitted_Supervisor.prompts import relation_extractor_prompt as prompt
import importlib
import approaches.additional_architectures.Splitted_Supervisor.prompts
importlib.reload(approaches.additional_architectures.Splitted_Supervisor.prompts)


def agent(state: cIEState) -> Command[Literal["agent_instructor_agent"]] | tuple[cIEState, str]:
    response_chain = prompt | model

    config = {}

    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    response = response_chain.invoke(state, config=config)

    response = f"""-- Relation Extraction Agent --
    
{response.content} """

    if state["debug"]:
        state["instruction"] = ""
        state["results"] += [response]
        return state, response

    return Command(goto="result_checker_agent", update={"instruction": "", "results": state["results"] + [response]})
