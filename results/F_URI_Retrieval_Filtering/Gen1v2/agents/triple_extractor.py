import importlib
import re
from typing import Literal

from langgraph.types import Command

import results.F_URI_Retrieval_Filtering.Gen1v2.prompts
importlib.reload(results.F_URI_Retrieval_Filtering.Gen1v2.prompts)

from results.F_URI_Retrieval_Filtering.Gen1v2.setup import cIEState, model, langfuse_handler
from results.F_URI_Retrieval_Filtering.Gen1v2.prompts import triple_extractor_prompt as prompt


def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
    response_chain = prompt | model

    config = {}

    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    response = response_chain.invoke(state, config=config)

    # Extract triples from the response
    content = response.content
    triples_match = re.search(r'<triples>(.*?)</triples>', content, re.DOTALL)
    updated_triples = set()
    
    if triples_match:
        triples_text = triples_match.group(1)
        updated_triples = {t.strip() for t in triples_text.split('\n') if t.strip()}
    
    # Update state with the complete triples list
    update = {
        "triples": updated_triples,
        "agent_response": content
    }

    if state["debug"]:
        state.update(update)
        return state, f"Response: {content}\nUpdated Triples: {updated_triples}\nState Updates: {update}"

    return Command(
        goto="planner",
        update=update
    ) 