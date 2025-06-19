import re
from typing import Literal

from langgraph.types import Command

from results.B_Task_Modularisation.setup import cIEState, model, langfuse_handler
from results.B_Task_Modularisation.prompts import agent_instructor_prompt as prompt
import importlib
import results.B_Task_Modularisation.prompts
importlib.reload(results.B_Task_Modularisation.prompts)



def agent(state: cIEState) -> Command[Literal["agent_instructor_agent"]] | tuple[cIEState, str]:
    response_chain = prompt | model

    config = {}

    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    response = response_chain.invoke(state, config=config)

    agent_id_match = re.search(r'<id>(.*?)</id>', response.content, re.DOTALL)
    if agent_id_match:
        agent_id = agent_id_match.group(1)
    else:
        agent_id = "agent_instructor"

    instruction_match = re.search(r'<instruction>(.*?)</instruction>', response.content, re.DOTALL)
    if instruction_match:
        instruction = instruction_match.group(1)
    else:
        instruction = ""

    if state["debug"]:
        state["instruction"] = instruction
        state["call_trace"] += [(agent_id, instruction)]
        return state, response.content

    return Command(goto=agent_id,
                   update={"instruction": instruction, "call_trace": state["call_trace"] + [(agent_id, instruction)]})
