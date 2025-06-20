import importlib
import re
from typing import Literal

from langgraph.types import Command

import results.F_URI_Retrieval_Filtering.Gen1v2.prompts
importlib.reload(results.F_URI_Retrieval_Filtering.Gen1v2.prompts)

from results.F_URI_Retrieval_Filtering.Gen1v2.setup import cIEState, model, langfuse_handler
from results.F_URI_Retrieval_Filtering.Gen1v2.prompts import predicate_extractor_prompt as prompt


def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
    response_chain = prompt | model

    config = {}

    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    response = response_chain.invoke(state, config=config)

    # Extract predicates from the response
    content = response.content
    predicates_match = re.search(r'<predicates>(.*?)</predicates>', content, re.DOTALL)
    updated_predicates = set()
    
    if predicates_match:
        predicates_text = predicates_match.group(1)
        updated_predicates = {p.strip() for p in predicates_text.split('\n') if p.strip()}
    
    # Update state with the complete predicates list
    update = {
        "predicates": updated_predicates,
        "agent_response": content
    }

    if state["debug"]:
        state.update(update)
        return state, f"Response: {content}\nUpdated Predicates: {updated_predicates}\nState Updates: {update}"

    return Command(
        goto="planner",
        update=update
    ) 