import importlib
import re
from typing import Literal

from langgraph.types import Command

from approaches.full_sentence.Gen1v2.setup import cIEState, model, langfuse_handler
from approaches.full_sentence.Gen1v2.prompts import entity_extractor_prompt as prompt
import approaches.full_sentence.Gen1v2.prompts
importlib.reload(approaches.full_sentence.Gen1v2.prompts)


def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
    response_chain = prompt | model

    config = {}

    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    response = response_chain.invoke(state, config=config)

    # Extract entities from the response
    content = response.content
    entities_match = re.search(r'<entities>(.*?)</entities>', content, re.DOTALL)
    new_entities = set()
    
    if entities_match:
        entities_text = entities_match.group(1)
        new_entities = {e.strip() for e in entities_text.split('\n') if e.strip()}
    
    # Update state with new entities
    update = {
        "entities": state["entities"].union(new_entities),
        "agent_response": content
    }

    if state["debug"]:
        state.update(update)
        return state, f"Response: {content}\nNew Entities: {new_entities}\nState Updates: {update}"

    return Command(
        goto="planner",
        update=update
    ) 