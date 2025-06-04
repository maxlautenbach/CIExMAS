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
from approaches.Supervisor.Gen1.setup import cIEState
from approaches.Supervisor.Gen1.agents.agent_instructor import agent as agent_instructor_agent
from approaches.Supervisor.Gen1.agents.entity_extractor import agent as entity_extraction_agent
from approaches.Supervisor.Gen1.agents.relation_extractor import agent as relation_extraction_agent
from approaches.Supervisor.Gen1.agents.uri_detector import agent as uri_detection_agent
from approaches.Supervisor.Gen1.agents.result_checker import agent as result_checker_agent
from approaches.Supervisor.Gen1.agents.result_formatter import agent as result_formatting_agent
from approaches.Supervisor.Gen1.agents.planner import agent as planner
from helper_tools.evaluation import evaluate_doc, calculate_scores_from_array
from dotenv import load_dotenv
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
relation_df, entity_df, docs = parser.unified_parser(dataset, split, number_of_samples)

entity_set = entity_df[['entity', 'entity_uri']].drop_duplicates()
predicate_set_df = relation_df[["predicate", "predicate_uri"]].drop_duplicates()

builder = StateGraph(cIEState)
builder.add_node("planner", planner)
builder.add_node("agent_instructor_agent", agent_instructor_agent)
builder.add_node("entity_extraction_agent", entity_extraction_agent)
builder.add_node("relation_extraction_agent", relation_extraction_agent)
builder.add_node("uri_detection_agent", uri_detection_agent)
builder.add_node("result_checker_agent", result_checker_agent)
builder.add_node("result_formatting_agent", result_formatting_agent)

builder.add_edge(START, "planner")

graph = builder.compile()

evaluation_log = []

for i in tqdm(range(len(docs))):
    target_doc = docs.iloc[i]
    doc_id = target_doc["docid"]
    text = target_doc["text"]
    trace_id = str(uuid.uuid4())
    try:
        response = graph.invoke({"text": text, "results": [], "call_trace": [], "comments": [], "debug": False},
                            config={"run_id": trace_id, "recursion_limit": 70, "callbacks": [langfuse_handler], "tags":["Gen1", f'{os.getenv("LLM_MODEL_PROVIDER")}-{os.getenv("LLM_MODEL_ID")}']})
        final_result = response["results"][-1]
        turtle_string = response["results"][-1]
        score = calculate_scores_from_array(evaluate_doc(turtle_string=turtle_string, doc_id=doc_id,
                                                         triple_df=relation_df))
        langfuse_client.score(trace_id=trace_id, name="F1-Score", value=score.loc["Triple"]["F1-Score"])
    except Exception as e:
        turtle_string = ""
        final_result = e
        langfuse_client.score(trace_id=trace_id, name="F1-Score", value=0)
    evaluation_log.append([doc_id, *evaluate_doc(turtle_string, doc_id, relation_df), final_result, trace_id])

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

excel_file_path = f"{repo.working_dir}/approaches/evaluation_logs/Gen1/{dataset}-{split}-{number_of_samples}-evaluation_log-{os.getenv('LLM_MODEL_PROVIDER')}_{os.getenv('LLM_MODEL_ID').replace('/', '-')}-{datetime.now().strftime('%Y-%m-%d-%H%M')}.xlsx"
try:
    evaluation_log_df.to_excel(excel_file_path, index=False)
except Exception as e:
    print(excel_file_path)
    print(e)
    evaluation_log_df.to_excel(f"Output.xlsx", index=False)

import json
import os.path

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
