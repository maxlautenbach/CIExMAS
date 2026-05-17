from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from approaches.Pipeline_ACE.setup import cIEState, model
from approaches.Pipeline_ACE.prompts import triple_extractor_prompt as prompt


class TripleExtractionOutput(BaseModel):
    predicates: list[str] = Field(description="List of predicates (relationships) found in the text")
    triples: list[str] = Field(description="List of triples formatted as 'subject; predicate; object'")


structured_model = model.with_structured_output(TripleExtractionOutput)


def agent(state: cIEState, config: RunnableConfig = None) -> dict:
    response_chain = prompt | structured_model
    response = response_chain.invoke(state, config=config or {})

    predicates = set(response.predicates) if response else set()
    triples = set(response.triples) if response else set()

    return {"predicates": predicates, "triples": triples}
