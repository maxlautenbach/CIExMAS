import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path

import git

repo = git.Repo(search_parent_directories=True)
sys.path.append(repo.working_dir)

import argparse
import pandas as pd
import warnings
from tqdm import tqdm

from helper_tools.parser import unified_parser
from helper_tools.evaluation import evaluate_doc, calculate_scores_from_array
from approaches.Pipeline_ACE.pipeline import invoke_pipeline
from dotenv import load_dotenv

load_dotenv(repo.working_dir + "/.env", override=True)
from helper_tools.base_setup import langfuse_handler, langfuse_client

warnings.filterwarnings("ignore")

ACE_DIR = Path(repo.working_dir) / "approaches" / "Pipeline_ACE"
GUIDELINES_DIR = ACE_DIR / "guidelines"

arg_parser = argparse.ArgumentParser(description="Pipeline-ACE Evaluation (test split)")
arg_parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
arg_parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to process")
arg_parser.add_argument("--dataset", type=str, default="wiki_cie_text", help="Dataset to use")
arg_parser.add_argument("--description", type=str, help="Optional description for the evaluation log")
args = arg_parser.parse_args()

split = args.split
number_of_samples = args.num_samples
dataset = args.dataset
description = args.description


def load_guidelines(agent_name: str) -> str:
    path = GUIDELINES_DIR / f"{agent_name}.md"
    if path.exists():
        content = path.read_text(encoding="utf-8")
        if content.strip() and "## Rules" in content and len(content.strip().split("\n")) > 2:
            return content
    return ""


# Load guidelines
entity_guidelines = load_guidelines("entity_extractor")
triple_guidelines = load_guidelines("triple_extractor")
guidelines_parts = [g for g in [entity_guidelines, triple_guidelines] if g]
guidelines = "\n\n".join(guidelines_parts)
instruction = f"Follow these guidelines:\n{guidelines}" if guidelines else ""

if guidelines:
    print("Guidelines loaded: entity_extractor.md + triple_extractor.md")
else:
    print("No guidelines found — running without guidelines")

# Load dataset
triple_df, entity_df, docs = unified_parser(dataset, split, number_of_samples)

print(f"Loaded {len(docs)} documents from {dataset}/{split}")

evaluation_log = []

for i in tqdm(range(len(docs))):
    target_doc = docs.iloc[i]
    doc_id = target_doc["docid"]
    text = target_doc["text"]

    try:
        result = invoke_pipeline(text, instruction=instruction, trace=True)
        turtle_string = result.get("turtle", "")
        trace_id = result.get("trace_id")

        score = calculate_scores_from_array(evaluate_doc(turtle_string=turtle_string, doc_id=doc_id, triple_df=triple_df))
        langfuse_client.score(trace_id=trace_id, name="F1-Score", value=score.loc["Triple"]["F1-Score"])
    except Exception as e:
        error_msg = f"Error: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        turtle_string = ""
        trace_id = ""
        langfuse_client.score(trace_id=trace_id, name="F1-Score", value=0)

    evaluation_log.append([doc_id, *evaluate_doc(turtle_string, doc_id, triple_df), turtle_string, trace_id])

evaluation_log_df = pd.DataFrame(
    evaluation_log,
    columns=[
        "Doc ID",
        "Correct Triples", "Correct Triples with Parents", "Correct Triples with Related",
        "Gold Standard Triples", "Total Triples Predicted",
        "Extracted Subjects", "Gold Standard Subjects", "Correct Extracted Subjects",
        "Extracted Predicates", "Gold Standard Predicates", "Correct Extracted Predicates",
        "Detected Predicates Doc Parent", "Detected Predicates Doc Related",
        "Correct Pred Predicates Parents", "Correct Pred Predicates Related",
        "Extracted Objects", "Gold Standard Objects", "Correct Extracted Objects",
        "Extracted Entities", "Gold Standard Entities", "Correct Extracted Entities",
        "Result String", "Langfuse Trace ID",
    ]
)

eval_log_dir = Path(repo.working_dir) / "approaches" / "evaluation_logs" / "Pipeline_ACE"
eval_log_dir.mkdir(parents=True, exist_ok=True)

excel_file_path = str(eval_log_dir / f"{dataset}-{split}-{number_of_samples}-evaluation_log-Pipeline_ACE-{datetime.now().strftime('%Y-%m-%d-%H%M')}.xlsx")
try:
    evaluation_log_df.to_excel(excel_file_path, index=False)
except Exception as e:
    print(excel_file_path)
    print(e)
    evaluation_log_df.to_excel("Output.xlsx", index=False)

if description:
    log_notes_path = f"{repo.working_dir}/approaches/evaluation_logs/log_notes.json"
    try:
        with open(log_notes_path, "r") as log_file:
            log_notes = json.load(log_file)
    except FileNotFoundError:
        log_notes = {}

    excel_file_name = os.path.basename(excel_file_path)
    log_notes[excel_file_name] = {"short_description": description}

    with open(log_notes_path, "w") as log_file:
        json.dump(log_notes, log_file, indent=4)

print(f"\nEvaluation log saved to: {excel_file_path}")
report = calculate_scores_from_array(evaluation_log_df.iloc[:, 1:22].sum().tolist())
print(f"\nMicro-averaged scores:\n{report.to_string()}")
