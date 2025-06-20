import re
from typing import Literal

from langgraph.constants import END
from langgraph.types import Command

from results.C_Error_Incorporation.Gen1_PredEx.setup import cIEState, model, langfuse_handler
from results.C_Error_Incorporation.Gen1_PredEx.prompts import result_formatter_prompt as prompt
import importlib
import results.C_Error_Incorporation.Gen1_PredEx.prompts
from helper_tools.validation import validate_turtle_response

importlib.reload(results.C_Error_Incorporation.Gen1_PredEx.prompts)


def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
    response_chain = prompt | model

    config = {}

    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    response = response_chain.invoke(state, config=config)

    result_match = re.search(r'<ttl>(.*?)</ttl>', response.content, re.DOTALL)

    goto_id = END

    if result_match:
        result = result_match.group(1)
        turtle_valid, error_message = validate_turtle_response(result)
        if not turtle_valid:
            goto_id = "main_agent"
            instruction = "\nSYSTEM MESSAGE: " + error_message + "\n please fix."
        else:
            instruction = ""
    else:
        result = ""
        instruction = "\nSYSTEM MESSAGE: No output could be parsed, as the regex r'<ttl>(.*?)</ttl>' produced no match.\n please fix."

    if state["debug"]:
        state["results"].append(result)
        state["instruction"] = instruction
        return state, response.content

    return Command(goto=goto_id, update={"results": state["results"] + [result], "instruction": instruction})
