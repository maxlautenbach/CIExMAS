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
from approaches.full_sentence.One_Agent.setup import cIEState
from approaches.full_sentence.One_Agent.agents.main_agent import agent as main
from approaches.full_sentence.One_Agent.agents.uri_search_tool import agent as uri_search_tool

from helper_tools.evaluation import evaluate_doc, calculate_scores_from_array
from dotenv import load_dotenv

load_dotenv(repo.working_dir + "/.env", override=True)
from helper_tools.base_setup import langfuse_handler, langfuse_client, session_id

warnings.filterwarnings("ignore")

importlib.reload(parser)

try:
    split = sys.argv[1]
    number_of_samples = int(sys.argv[2])
except IndexError:
    split = "test"
    number_of_samples = 10

triple_df, entity_df, docs = parser.synthie_parser(split, number_of_samples)
entity_set = entity_df[['entity', 'entity_uri']].drop_duplicates()
predicate_set_df = triple_df[["predicate", "predicate_uri"]].drop_duplicates()

builder = StateGraph(cIEState)
builder.add_node("main_agent", main)
builder.add_node("uri_search_tool", uri_search_tool)

builder.add_edge(START, "main_agent")

graph = builder.compile()

evaluation_log = []

for i in tqdm(range(len(docs))):
    target_doc = docs.iloc[i]
    doc_id = target_doc["docid"]
    text = target_doc["text"]
    trace_id = str(uuid.uuid4())
    try:
        # noinspection PyTypeChecker
        response = graph.invoke({"text": text, "messages": [], "instruction": "", "debug": False},
                                config={"run_id": trace_id, "recursion_limit": 70, "callbacks": [langfuse_handler], "tags":["One Agent", f'{os.getenv("LLM_MODEL_PROVIDER")}-{os.getenv("LLM_MODEL_ID")}']})
        turtle_string = response["messages"][-1]
        score = calculate_scores_from_array(evaluate_doc(turtle_string=turtle_string, doc_id=doc_id,
                                           triple_df=triple_df))
        langfuse_client.score(trace_id=trace_id, name="F1-Score", value=score.loc["Triple"]["F1-Score"])
    except Exception as e:
        turtle_string = e
        response = {"messages":[e]}
        langfuse_client.score(trace_id=trace_id, name="F1-Score", value=0)

    evaluation_log.append([doc_id, *evaluate_doc(turtle_string,doc_id, triple_df), response["messages"][-1]])

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
        "Extracted Entities", "Gold Standard Entities", "Correct Extracted Entities", "Result String"
    ]
)

excel_file_path = f"{repo.working_dir}/approaches/evaluation_logs/One_Agent/{split}-{number_of_samples}-evaluation_log-{os.getenv('LLM_MODEL_PROVIDER')}_{os.getenv('LLM_MODEL_ID').replace('/','-')}-{datetime.now().strftime('%Y-%m-%d-%H%M')}.xlsx"
try:
    evaluation_log_df.to_excel(excel_file_path, index=False)
except Exception as e:
    print(excel_file_path)
    print(e)
    evaluation_log_df.to_excel(f"Output.xlsx", index=False)
