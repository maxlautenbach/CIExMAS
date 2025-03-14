from typing import Literal

from langgraph.types import Command

from approaches.full_sentence.Gen1.setup import cIEState, model
from approaches.full_sentence.Gen1.prompts import planner_prompt as prompt


def agent(state: cIEState) -> Command[Literal["agent_instructor_agent"]]:
    response_chain = prompt | model

    response = response_chain.invoke(state)

    return Command(goto="agent_instructor_agent", update={"comments": state["comments"] + [response]})