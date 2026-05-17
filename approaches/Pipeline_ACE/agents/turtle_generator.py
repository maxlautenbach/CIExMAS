from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from rdflib import Graph

from approaches.Pipeline_ACE.setup import cIEState, model
from approaches.Pipeline_ACE.prompts import turtle_generator_prompt as prompt


class TurtleOutput(BaseModel):
    turtle: str = Field(description="Valid Turtle RDF output with @prefix declarations and triple statements")


structured_model = model.with_structured_output(TurtleOutput)


def _build_context(state: cIEState) -> dict:
    """Build context showing triples with URI candidates for each component."""
    triples_with_candidates = []

    for triple in state["triples"]:
        parts = [p.strip() for p in triple.split(";")]
        if len(parts) != 3:
            continue

        subject, predicate, obj = parts
        triples_with_candidates.append({
            "triple": triple,
            "subject": subject,
            "subject_candidates": state["uri_candidates"].get(subject, []),
            "predicate": predicate,
            "predicate_candidates": state["uri_candidates"].get(predicate, []),
            "object": obj,
            "object_candidates": state["uri_candidates"].get(obj, []),
        })

    return {"text": state["text"], "triples_with_candidates": triples_with_candidates}


def agent(state: cIEState, config: RunnableConfig = None) -> dict:
    context = _build_context(state)

    if not context["triples_with_candidates"]:
        return {"turtle": ""}

    response_chain = prompt | structured_model
    response = response_chain.invoke(
        {
            "text": context["text"],
            "triples_with_candidates": context["triples_with_candidates"],
            "instruction": state["instruction"],
        },
        config=config or {}
    )

    turtle = response.turtle.strip() if response else ""

    if turtle:
        try:
            g = Graph()
            g.parse(data=turtle, format="turtle")
        except Exception:
            turtle = ""

    return {"turtle": turtle}
