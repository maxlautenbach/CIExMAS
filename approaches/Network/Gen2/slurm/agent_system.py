import uuid
from datetime import datetime
import os

import git
import sys

repo = git.Repo(search_parent_directories=True)
sys.path.append(repo.working_dir)

import re

from tqdm import tqdm

import helper_tools.parser as parser
import importlib
import pandas as pd
import warnings
from langgraph.graph import StateGraph, START, END
from approaches.Network.Gen2.setup import cIEState
from approaches.Network.Gen2.agents.extractor import agent as extractor
from approaches.Network.Gen2.agents.uri_mapping_and_refinement import agent as uri_mapping_and_refinement
from approaches.Network.Gen2.agents.validation_and_output import agent as validation_and_output
from approaches.Network.Gen2.agents.uri_search_tool import agent as uri_search_tool
from approaches.Network.Gen2.agents.network_traversal_search import agent as network_traversal_search
from approaches.Network.Gen2.agents.turtle_to_labels_tool import agent as turtle_to_labels_tool
from approaches.Network.Gen2.agents.semantic_triple_checking_tool import agent as semantic_triple_checking_tool
from helper_tools.evaluation import evaluate_doc, calculate_scores_from_array
from dotenv import load_dotenv
import json
import argparse

load_dotenv(repo.working_dir + "/.env", override=True)
from helper_tools.base_setup import langfuse_handler, langfuse_client

warnings.filterwarnings("ignore")

importlib.reload(parser)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, required=True, help="Dataset split to use")
parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to process")
parser.add_argument("--dataset", type=str, required=True, help="Dataset to use (e.g., synthie_code, rebel, redfm)")
parser.add_argument("--description", type=str, help="Optional description for the evaluation log")
args = parser.parse_args()

split = args.split
number_of_samples = args.num_samples
dataset = args.dataset
description = args.description

# Load dataset
triple_df, entity_df, docs = parser.unified_parser(dataset, split, number_of_samples)
entity_set = entity_df[['entity', 'entity_uri']].drop_duplicates()
predicate_set_df = triple_df[["predicate", "predicate_uri"]].drop_duplicates()

builder = StateGraph(cIEState)
builder.add_node("extractor", extractor)
builder.add_node("uri_mapping_and_refinement", uri_mapping_and_refinement)
builder.add_node("validation_and_output", validation_and_output)
builder.add_node("uri_search_tool", uri_search_tool)
builder.add_node("network_traversal_tool", network_traversal_search)
builder.add_node("turtle_to_labels_tool", turtle_to_labels_tool)
builder.add_node("semantic_triple_checking_tool", semantic_triple_checking_tool)

builder.add_edge(START, "extractor")

graph = builder.compile()

evaluation_log = []

for i in tqdm(range(len(docs))):
    target_doc = docs.iloc[i]
    doc_id = target_doc["docid"]
    text = target_doc["text"]
    trace_id = str(uuid.uuid4())
    try:
        response = graph.invoke({
            "text": text,
            "triples": {},
            "last_agent_response": "",
            "agent_instruction": "",
            "messages": [],
            "tool_input": "", 
            "debug": False,
            "call_trace": [],
            "uri_mapping": "",
        }, config={"run_id": trace_id, "recursion_limit": 40, "callbacks": [langfuse_handler], "tags":["Gen2", f'{os.getenv("LLM_MODEL_PROVIDER")}-{os.getenv("LLM_MODEL_ID")}']})
        
        # Assuming the final result is in the last message
        if response["messages"]:
            turtle_string = response["messages"][-1]
        else:
            turtle_string = ""
        
        score = calculate_scores_from_array(evaluate_doc(turtle_string=turtle_string, doc_id=doc_id,
                                                         triple_df=triple_df))
        langfuse_client.score(trace_id=trace_id, name="F1-Score", value=score.loc["Triple"]["F1-Score"])
    except Exception as e:
        turtle_string = ""
        response = {"messages":[str(e)]}
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

excel_file_path = f"{repo.working_dir}/approaches/evaluation_logs/Gen2/{dataset}-{split}-{number_of_samples}-evaluation_log-{os.getenv('LLM_MODEL_PROVIDER')}_{os.getenv('LLM_MODEL_ID').replace('/', '-')}-{datetime.now().strftime('%Y-%m-%d-%H%M')}.xlsx"
try:
    evaluation_log_df.to_excel(excel_file_path, index=False)
except Exception as e:
    print(excel_file_path)
    print(e)
    evaluation_log_df.to_excel(f"Output.xlsx", index=False)

if description:
    log_notes_path = f"{repo.working_dir}/approaches/evaluation_logs/log_notes.json"
    try:
        with open(log_notes_path, "r") as log_file:
            log_notes = json.load(log_file)
    except FileNotFoundError:
        log_notes = {}

    # Use just the filename, not the full path
    excel_file_name = os.path.basename(excel_file_path)
    log_notes[excel_file_name] = {"short_description": description}

    with open(log_notes_path, "w") as log_file:
        json.dump(log_notes, log_file, indent=4)