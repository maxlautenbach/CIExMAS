import importlib
from typing import Literal

from langgraph.types import Command

from approaches.baseline.setup import cIEState, model, langfuse_handler, vector_store
from approaches.baseline.prompts import uri_detection_prompt as prompt
import approaches.baseline.prompts

importlib.reload(approaches.baseline.prompts)


def agent(state: cIEState) -> Command[Literal["supervisor"]] | tuple[cIEState, str]:
    search_terms = state["instruction"].split(",")
    search_response = ""
    for term in search_terms:
        search_response += f'Most Similar Detection Results for {term}:{[{"label": doc.page_content, "uri": doc.metadata["uri"]} for doc in vector_store.similarity_search(term, k=3)]}\n\n'
    search_response = search_response.replace("},", "},\n")

    print("SIM SEARCH FINISHED")

    config = {}

    if state["debug"]:
        config = {"callbacks": [langfuse_handler]}

    chain = prompt | model
    response = chain.invoke({"search_response": search_response}, config=config)

    response = "\n-- URI Detection Agent --\n" + response.content

    if state["debug"]:
        state["messages"].append(response)
        state["instruction"] = ""
        return state, "\n-- Search Response --\n" + search_response + response

    print("URI EX FINISHED")

    return Command(goto="supervisor", update={"messages": state["messages"] + [response], "instruction": ""})
