import re
from typing import Literal

from langgraph.constants import END
from langgraph.types import Command

from approaches.full_sentence.Gen1.setup import cIEState, model, langfuse_handler
from approaches.full_sentence.Gen1.prompts import result_formatter_prompt as prompt
import importlib
import approaches.full_sentence.Gen1.prompts
importlib.reload(approaches.full_sentence.Gen1.prompts)


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
