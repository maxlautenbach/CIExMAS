from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig

from approaches.Pipeline_ACE.setup import cIEState, model
from approaches.Pipeline_ACE.prompts import entity_extractor_prompt as prompt


class EntityExtractionOutput(BaseModel):
    entities: list[str] = Field(description="Complete list of extracted entities")


structured_model = model.with_structured_output(EntityExtractionOutput)


def agent(state: cIEState, config: RunnableConfig = None) -> dict:
    response_chain = prompt | structured_model
    response = response_chain.invoke(state, config=config or {})
    return {"entities": set(response.entities) if response else set()}
