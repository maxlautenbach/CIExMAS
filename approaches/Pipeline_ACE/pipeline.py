import os
import uuid

from langgraph.graph import StateGraph, START, END

from approaches.Pipeline_ACE.setup import cIEState, langfuse_handler
from approaches.Pipeline_ACE.agents.entity_extractor import agent as entity_extractor
from approaches.Pipeline_ACE.agents.triple_extractor import agent as triple_extractor
from approaches.Pipeline_ACE.tools.uri_retriever import agent as uri_retriever
from approaches.Pipeline_ACE.agents.turtle_generator import agent as turtle_generator


def build_pipeline():
    builder = StateGraph(cIEState)
    builder.add_node("entity_extractor", entity_extractor)
    builder.add_node("triple_extractor", triple_extractor)
    builder.add_node("uri_retriever", uri_retriever)
    builder.add_node("turtle_generator", turtle_generator)
    builder.add_edge(START, "entity_extractor")
    builder.add_edge("entity_extractor", "triple_extractor")
    builder.add_edge("triple_extractor", "uri_retriever")
    builder.add_edge("uri_retriever", "turtle_generator")
    builder.add_edge("turtle_generator", END)
    return builder.compile()


def invoke_pipeline(text: str, instruction: str = "", trace: bool = True) -> dict:
    pipeline = build_pipeline()

    initial_state = {
        "text": text,
        "entities": set(),
        "predicates": set(),
        "triples": set(),
        "uri_candidates": {},
        "turtle": "",
        "instruction": instruction,
    }

    trace_id = str(uuid.uuid4())
    config = {"recursion_limit": 10}
    if trace:
        config["run_id"] = trace_id
        config["callbacks"] = [langfuse_handler]
        config["tags"] = ["Pipeline_ACE", f'{os.getenv("LLM_MODEL_PROVIDER")}-{os.getenv("LLM_MODEL_ID")}']

    result = pipeline.invoke(initial_state, config=config)
    result["trace_id"] = trace_id
    return result
