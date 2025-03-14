
import re
from typing import Literal

from langgraph.types import Command

from approaches.full_sentence.Gen1.setup import cIEState, model, langfuse_handler
from approaches.full_sentence.Gen1.prompts import entity_extractor_prompt as prompt

import importlib
import approaches.full_sentence.Gen1.prompts
importlib.reload(approaches.full_sentence.Gen1.prompts)


def agent(state: cIEState) -> Command[Literal["agent_instructor_agent"]] | tuple[cIEState, str]:
    response_chain = prompt | model

    config = {}

    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    response = response_chain.invoke(state, config=config)

    result_match = re.search(r'<result>(.*?)</result>', response.content, re.DOTALL)
    if result_match:
        result = result_match.group(1)
    else:
        result = ""

    result = f"Output of entity_extraction_agent: {result}"

    if state["debug"]:
        state["instruction"] = ""
        state["results"] += [result]
        return state, response.content

    return Command(goto="result_checker_agent", update={"instruction": "", "results": state["results"] + [result]})
