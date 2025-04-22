import importlib
import re
from typing import Literal

from langgraph.constants import END
from langgraph.types import Command
from rdflib import Graph

from approaches.full_sentence.Gen1v2.setup import cIEState, model, langfuse_handler
from approaches.full_sentence.Gen1v2.prompts import turtle_extractor_prompt as prompt
import approaches.full_sentence.Gen1v2.prompts
importlib.reload(approaches.full_sentence.Gen1v2.prompts)


def agent(state: cIEState) -> Command[Literal] | tuple[cIEState, str]:
    response_chain = prompt | model
    config = {}
    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    response = response_chain.invoke(
        {
            "text": state["text"],
            "uri_mapping": state["uri_mapping"],
            "triples": state["triples"]
        },
        config=config
    )

    # Extract turtle output from the response
    content = response.content
    turtle_match = re.search(r'<ttl>(.*?)</ttl>', content, re.DOTALL)
    turtle_output = ""
    instruction = ""
    goto = "planner"
    
    if turtle_match:
        turtle_output = turtle_match.group(1).strip()
        try:
            # Validate turtle syntax using RDFlib
            graph = Graph()
            graph.parse(data=turtle_output, format="turtle")
            goto = END
        except Exception as e:
            instruction = f"\nSYSTEM MESSAGE: Invalid turtle syntax: {str(e)}\nPlease fix the turtle output."
    else:
        instruction = "\nSYSTEM MESSAGE: No turtle output found in the response. Please provide valid turtle output enclosed in <ttl> tags."
    
    # Update state with turtle output
    update = {
        "messages": state["messages"] + [turtle_output],
        "instruction": instruction
    }

    if state["debug"]:
        state.update(update)
        return state, f"Response: {content}"

    return Command(
        goto=goto,
        update=update
    ) 