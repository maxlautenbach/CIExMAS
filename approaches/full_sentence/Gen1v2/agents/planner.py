import importlib
import re
from typing import Literal

from langgraph.types import Command

import approaches.full_sentence.Gen1v2.prompts
importlib.reload(approaches.full_sentence.Gen1v2.prompts)

from approaches.full_sentence.Gen1v2.setup import cIEState, model, langfuse_handler
from approaches.full_sentence.Gen1v2.prompts import planner_prompt as prompt

# Define valid agent IDs
VALID_AGENTS = [
    "planner",
    "entity_extraction_agent",
    "predicate_extraction_agent",
    "triple_extraction_agent",
    "uri_retriever_agent",
    "turtle_extraction_agent"
]

def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
    response_chain = prompt | model

    config = {}

    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    response = response_chain.invoke(state, config=config)

    # Extract the next agent ID and instruction from the response using regex
    content = response.content
    
    next_match = re.search(r'<next>(.*?)</next>', content, re.DOTALL)
    next_section = next_match.group(1) if next_match else ""
    
    id_match = re.search(r'<id>(.*?)</id>', next_section)
    next_agent = id_match.group(1).strip() if id_match else None
    
    instruction_match = re.search(r'<instruction>(.*?)</instruction>', next_section)
    instruction = instruction_match.group(1).strip() if instruction_match else ""

    # Prepare base update with message history
    update = {
        "messages": state["messages"] + ["\n-- Planner Agent --\n" + response.content]
    }

    if not next_agent:
        # Missing agent ID
        update["instruction"] = "The planner response is missing the required <next> section with <id> and <instruction> tags. Please provide a properly formatted response."
        next_agent = "planner"
    elif next_agent not in VALID_AGENTS:
        # Invalid agent ID
        update["instruction"] = f"The agent ID '{next_agent}' is not valid. Please choose from the following valid agent IDs: {', '.join(VALID_AGENTS)}"
        next_agent = "planner"
    else:
        # Valid agent ID
        update["instruction"] = instruction

    if state["debug"]:
        state.update(update)
        return state, f"Response: {response.content}"

    return Command(
        goto=next_agent,
        update=update
    )
