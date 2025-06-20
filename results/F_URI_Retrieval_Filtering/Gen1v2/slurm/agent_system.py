import uuid
from datetime import datetime
import os
import traceback

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
from results.F_URI_Retrieval_Filtering.Gen1v2.setup import cIEState
from results.F_URI_Retrieval_Filtering.Gen1v2.agents.planner import agent as planner
from results.F_URI_Retrieval_Filtering.Gen1v2.agents.entity_extractor import agent as entity_extraction_agent
from results.F_URI_Retrieval_Filtering.Gen1v2.agents.predicate_extractor import agent as predicate_extraction_agent
from results.F_URI_Retrieval_Filtering.Gen1v2.agents.triple_extractor import agent as triple_extraction_agent
from results.F_URI_Retrieval_Filtering.Gen1v2.agents.uri_retriever import agent as uri_retriever_agent
from results.F_URI_Retrieval_Filtering.Gen1v2.agents.turtle_extractor import agent as turtle_extraction_agent
from helper_tools.evaluation import evaluate_doc, calculate_scores_from_array
from dotenv import load_dotenv
import json
import argparse

load_dotenv(repo.working_dir + "/.env", override=True)
from helper_tools.base_setup import langfuse_handler, langfuse_client

warnings.filterwarnings("ignore")

importlib.reload(parser)

# Parse command line arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--split", type=str, required=True, help="Dataset split to use")
arg_parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to process")
arg_parser.add_argument("--dataset", type=str, required=True, help="Dataset to use (e.g., synthie_code, rebel, redfm)")
arg_parser.add_argument("--description", type=str, required=False, help="Description for logging")
args = arg_parser.parse_args()

split = args.split
number_of_samples = args.num_samples
dataset = args.dataset
description = args.description

# Load dataset
relation_df, entity_df, docs = parser.unified_parser(dataset, split, number_of_samples)
entity_set = entity_df[['entity', 'entity_uri']].drop_duplicates()
predicate_set_df = relation_df[["predicate", "predicate_uri"]].drop_duplicates()

builder = StateGraph(cIEState)
builder.add_node("planner", planner)
builder.add_node("entity_extraction_agent", entity_extraction_agent)
builder.add_node("predicate_extraction_agent", predicate_extraction_agent)
builder.add_node("triple_extraction_agent", triple_extraction_agent)
builder.add_node("uri_retriever_agent", uri_retriever_agent)
builder.add_node("turtle_extraction_agent", turtle_extraction_agent)

builder.add_edge(START, "planner")

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
            "messages": [],
            "entities": set(),
            "predicates": set(),
            "triples": set(),
            "uri_mapping": {},
            "agent_response": "",
            "instruction": "",
            "debug": False
        }, config={"run_id": trace_id, "recursion_limit": 70, "callbacks": [langfuse_handler], "tags":["F_URI_Retrieval_Filtering_Gen1v2", f'{os.getenv("LLM_MODEL_PROVIDER")}-{os.getenv("LLM_MODEL_ID")}']})
        
        # Assuming the final result is in the last message
        if response["messages"]:
            turtle_string = response["messages"][-1]
        else:
            turtle_string = ""
        
        score = calculate_scores_from_array(evaluate_doc(turtle_string=turtle_string, doc_id=doc_id,
                                                         triple_df=relation_df))
        langfuse_client.score(trace_id=trace_id, name="F1-Score", value=score.loc["Triple"]["F1-Score"])
    except Exception as e:
        error_msg = f"Error: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        turtle_string = ""
        response = {"messages":[error_msg]}
        langfuse_client.score(trace_id=trace_id, name="F1-Score", value=0)
        
    evaluation_log.append([doc_id, *evaluate_doc(turtle_string, doc_id, relation_df), response["messages"][-1], trace_id])

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

excel_file_path = f"{repo.working_dir}/results/result_evaluation_logs/F_URI_Retrieval_Filtering_Gen1v2.xlsx"
try:
    evaluation_log_df.to_excel(excel_file_path, index=False)
except Exception as e:
    print(excel_file_path)
    print(e)
    evaluation_log_df.to_excel(f"Output.xlsx", index=False)

if description:
    log_notes_path = f"{repo.working_dir}/results/result_evaluation_logs/log_notes.json"
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