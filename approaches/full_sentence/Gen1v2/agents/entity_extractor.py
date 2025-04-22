import importlib
import re
from typing import Literal

from langgraph.types import Command

import approaches.full_sentence.Gen1v2.prompts
importlib.reload(approaches.full_sentence.Gen1v2.prompts)

from approaches.full_sentence.Gen1v2.setup import cIEState, model, langfuse_handler
from approaches.full_sentence.Gen1v2.prompts import entity_extractor_prompt as prompt


def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
    response_chain = prompt | model

    config = {}

    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    response = response_chain.invoke(state, config=config)

    # Extract entities from the response
    content = response.content
    entities_match = re.search(r'<entities>(.*?)</entities>', content, re.DOTALL)
    updated_entities = set()
    
    if entities_match:
        entities_text = entities_match.group(1)
        updated_entities = {e.strip() for e in entities_text.split('\n') if e.strip()}
    
    # Update state with the new entities list
    update = {
        "entities": updated_entities,
        "agent_response": content
    }

    if state["debug"]:
        state.update(update)
        return state, f"Response: {content}\nUpdated Entities: {updated_entities}\nState Updates: {update}"

    return Command(
        goto="planner",
        update=update
    ) 