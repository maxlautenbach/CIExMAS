import importlib
import re
from typing import Literal

from langgraph.types import Command

from approaches.full_sentence.Gen1v2.setup import cIEState, model, langfuse_handler
from approaches.full_sentence.Gen1v2.prompts import predicate_extractor_prompt as prompt
import approaches.full_sentence.Gen1v2.prompts
importlib.reload(approaches.full_sentence.Gen1v2.prompts)


def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
    response_chain = prompt | model

    config = {}

    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    response = response_chain.invoke(state, config=config)

    # Extract predicates from the response
    content = response.content
    predicates_match = re.search(r'<predicates>(.*?)</predicates>', content, re.DOTALL)
    new_predicates = set()
    
    if predicates_match:
        predicates_text = predicates_match.group(1)
        new_predicates = {p.strip() for p in predicates_text.split('\n') if p.strip()}
    
    # Update state with new predicates
    update = {
        "predicates": state["predicates"].union(new_predicates),
        "agent_response": content
    }

    if state["debug"]:
        state.update(update)
        return state, f"Response: {content}\nNew Predicates: {new_predicates}\nState Updates: {update}"

    return Command(
        goto="planner",
        update=update
    ) 