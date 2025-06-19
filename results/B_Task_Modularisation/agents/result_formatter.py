import re
from typing import Literal

from langgraph.constants import END
from langgraph.types import Command

from results.B_Task_Modularisation.setup import cIEState, model, langfuse_handler
from results.B_Task_Modularisation.prompts import result_formatter_prompt as prompt
import importlib
import results.B_Task_Modularisation.prompts
importlib.reload(results.B_Task_Modularisation.prompts)


def agent(state: cIEState) -> Command[Literal[END]] | tuple[cIEState, str]:
    response_chain = prompt | model

    config = {}

    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    response = response_chain.invoke(state, config=config)

    result_match = re.search(r'<ttl>(.*?)</ttl>', response.content, re.DOTALL)

    if result_match:
        result = result_match.group(1)
    else:
        result = ""

    if state["debug"]:
        state["results"].append(result)
        return state, response.content

    return Command(goto=END, update={"results": state["results"] + [result]})
