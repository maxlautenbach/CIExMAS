import sys
from typing import TypedDict, List, Tuple

import git

repo = git.Repo(search_parent_directories=True)
sys.path.append(repo.working_dir)
from datetime import datetime
import traceback
from langgraph.types import Command
from typing import Literal, TypedDict
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.prompts import PromptTemplate
import re
from typing import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from helper_tools.evaluation import evaluate_doc, calculate_scores_from_array
from dotenv import load_dotenv
import json
import argparse
import helper_tools.parser as parser
import importlib
import pandas as pd
import warnings
import uuid
from tqdm import tqdm

load_dotenv(repo.working_dir + "/.env", override=True)


# Parse command line arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--split", type=str, required=True, help="Dataset split to use")
arg_parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to process")
arg_parser.add_argument("--dataset", type=str, required=True, help="Dataset to use (e.g., synthie_code, rebel, redfm)")
args = arg_parser.parse_args()

split = args.split
number_of_samples = args.num_samples
dataset = args.dataset

# Load dataset
triple_df, entity_df, docs = parser.unified_parser(dataset, split, number_of_samples)
entity_set = entity_df[['entity', 'entity_uri']].drop_duplicates()
predicate_set_df = triple_df[["predicate", "predicate_uri"]].drop_duplicates()

from helper_tools.base_setup import *

class cIEState(TypedDict):
    text: str
    messages: list[HumanMessage | AIMessage]
    instruction: str

system_prompt = f"""
You are the Supervisor of a conversation among multiple agents.
The conversation is about extracting information (Closed Information Extraction) from a user-provided text. The final output should only contain wikidata URIs instead of the labels of entities and relations. You can provide additional information to the agents using <instruction> tags. I.e. <instruction>Search additional for entities that are not obvious.</instruction>.

Example Output: <relation>Q950380;P361;Q2576666</relation>

Agent Descriptions:
- entity_extraction_agent: Extracts entities from the text. Instructions can change the prompt of the called agent. The agent has only access to your instruction and the text.
- relation_extraction_agent: Extracts relations from the text. Instructions can change the prompt of the called agent and can be used to input already extracted entity labels (i.e. <instruction>Olaf Scholz, Germany, Berlin</instruction>). DO NOT INPUT WIKIDATA URIs like Q950380 or P361. The agent has only access to your instruction and the text.
- uri_detection_agent: Returns possible wikidata URIs for entities and relations based on similarity search. The instruction should be a simple list of search terms, which the uri detection agent is searching for. I.e. "Olaf Scholz, Germany, Berlin". DO NOT INPUT WIKIDATA URIs like Q950380 or P361. The agent has only access to your instruction and the text.

You have two options:
1. Call an agent using <goto>agent_name</goto>. Replace agent_name with either entity_extraction_agent or relation_extraction_agent. I.e. <goto>entity_extraction_agent</goto>.
2. Finish the conversation using <goto>FINISH</goto>. Please output the final relations in <relation> tags alongside with the <goto> tag.


Note:
- Do not provide any information yourself, instead use the agents for this.
- The first <goto> tag in your response will be executed.
- Therefore, do include exact one agent call in your response.
- If you output nothing, this will result in a NoneType Error.


"""

def supervisor(state: cIEState) -> Command[Literal["entity_extraction_agent", "relation_extraction_agent", "uri_detection_agent", END]]:    
    
    response = model.invoke(state["messages"])
    
    print(f"-- START OF OUTPUT (supervisor) --\n\n", response.content, "\n\n-- END OF OUTPUT --\n\n")
        
    goto_match = re.search(r'<goto>(.*?)</goto>', response.content)
    if goto_match:
        goto = goto_match.group(1)
        if goto == "FINISH":
            goto = END
    else:
        goto = "supervisor"
        
    instruction_match = re.search(r'<instruction>(.*?)</instruction>', response.content)
    if instruction_match:
        instruction = instruction_match.group(1)
    else:
        instruction = ""

    return Command(goto=goto, update={"messages": state["messages"] + [response], "instruction": instruction})

def entity_extraction_agent(state: cIEState) -> Command[Literal["supervisor"]]:
    prompt_template = PromptTemplate.from_template("""
    You are an agent tasked with extracting entities from a given text for linking to a knowledge graph. Your job is to capture every entity—both explicit and implicit—and return them as an array. This includes composite entities with modifiers (e.g., "professional football club"). Please output the entities as an array of strings. Do not include any further information or comment in the output.
    
    Example Output: [Olaf Scholz, chancellor, Germany]
    
    Guidelines:
    - An entity is a unique real-world object or concept represented as a node with its properties and relationships.
    - Extract every entity mentioned in the text, including those that are not immediately obvious.
    - For composite entities, include the full descriptive phrase and break it into its core components when appropriate. For example, "chancellor of Germany" should yield [chancellor, Germany] and "professional football club" should capture the descriptive phrase as needed.
    - For composite entities that include a date at the beginning or end, extract the date separately, the entity without the date, and the full composite (e.g., "2022 Winter Olympics" should result in [2022, 2022 Winter Olympics, Winter Olympics]).
    - Also, ensure that dates are extracted as entities.
    
    Instruction: {instruction}
    
    Text: {text}
    """)
    chain = prompt_template | model
    response = chain.invoke({"text": state["text"], "instruction": state["instruction"]})
    
    return Command(goto="supervisor", update={"messages": state["messages"] + [response], "instruction": ""})

def relation_extraction_agent(state: cIEState) -> Command[Literal["supervisor"]]:
    prompt_template = PromptTemplate.from_template(
        """
        You are a relation extraction agent. Your task is to read the text of the user message and extract the relations found in the text. Each relation should be written in this exact format: <relation>subject;predicate;object</relation> (e.g.: <relation>Olaf Scholz;is chancellor of;Germany</relation>). Please return only the relations and no other information.
    
        Note: In addition to the explicit relations mentioned in the text, if an entity is described by a characteristic or category (e.g., renowned film director, prestigious university), you must also extract the corresponding attribute relation automatically. For example, if the text states that "Steven Spielberg is a renowned film director", you should extract: <relation>Steven Spielberg;profession;film director</relation>.
    
        Instruction: {instruction}
    
        Text: {text}
        """
    )
    chain = prompt_template | model
    response = chain.invoke({"text": state["text"], "instruction": state["instruction"]})
    
    return Command(goto="supervisor", update={"messages": state["messages"] + [response], "instruction": ""})

def uri_detection_agent(state):
    search_terms = state["instruction"].split(",")
    response = ""
    for term in search_terms:
        response += f'Search Results for {term}:\n{[{"label": doc.page_content, "uri": doc.metadata["uri"], "similarity_score": score} for doc, score in vector_store.similarity_search_with_score(term, k=3)]}\n\n'
    response = response.replace("},", "},\n")
    return Command(goto="supervisor", update={"messages": state["messages"] + [response], "instruction": ""})
    
builder = StateGraph(cIEState)
builder.add_node(supervisor)
builder.add_node(entity_extraction_agent)
builder.add_node(relation_extraction_agent)
builder.add_node(uri_detection_agent)

builder.add_edge(START, "supervisor")

graph = builder.compile()

evaluation_log = []

for i in tqdm(range(len(docs))):
    target_doc = docs.iloc[i]
    doc_id = target_doc["docid"]
    text = target_doc["text"]
    trace_id = str(uuid.uuid4())
    try:
        response = graph.invoke({"text": target_doc["text"], "messages": [system_prompt, target_doc["text"]], "instruction": ""}, config={"run_id": trace_id, "recursion_limit": 40, "callbacks": [langfuse_handler], "tags":["Initial Baseline", f'{os.getenv("LLM_MODEL_PROVIDER")}-{os.getenv("LLM_MODEL_ID")}']})

        relation_list = [x.split(";") for x in re.findall(r"<relation>(.*?)</relation>", response["messages"][-1].content)]

        turtle_string = "@prefix wd: <http://www.wikidata.org/entity/> .\n"
        for relation in relation_list:
            if len(relation) == 3:
                subject, predicate, obj = relation
                turtle_string += f"wd:{subject} wd:{predicate} wd:{obj} .\n"

        score = calculate_scores_from_array(evaluate_doc(turtle_string=turtle_string, doc_id=doc_id,
                                                         triple_df=triple_df))
        langfuse_client.score(trace_id=trace_id, name="F1-Score", value=score.loc["Triple"]["F1-Score"])
    
    except Exception as e:
        error_msg = f"Error: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        turtle_string = ""
        response = {"messages":[error_msg]}
        langfuse_client.score(trace_id=trace_id, name="F1-Score", value=0)
    
    evaluation_log.append([doc_id, *evaluate_doc(turtle_string, doc_id, triple_df), response["messages"][-1], trace_id])

evaluation_log_df = pd.DataFrame(
    evaluation_log,
    columns=[
        "Doc ID",
        "Correct Triples", "Correct Triples with Parents", "Correct Triples with Related", "Gold Standard Triples", "Total Triples Predicted",
        "Extracted Subjects", "Gold Standard Subjects", "Correct Extracted Subjects",
        "Extracted Predicates", "Gold Standard Predicates", "Correct Extracted Predicates",
        "Detected Predicates Doc Parent", "Detected Predicates Doc Related", 
        "Correct Pred Predicates Parents", "Correct Pred Predicates Related",
        "Extracted Objects", "Gold Standard Objects", "Correct Extracted Objects",
        "Extracted Entities", "Gold Standard Entities", "Correct Extracted Entities", "Result String", "Langfuse Trace ID"
    ]
)

excel_file_path = f"{repo.working_dir}/results/result_evaluation_logs/01_Initial_Baseline.xlsx"
try:
    evaluation_log_df.to_excel(excel_file_path, index=False)
except Exception as e:
    print(excel_file_path)
    print(e)
    evaluation_log_df.to_excel(f"Output.xlsx", index=False)