import importlib
import os
import sys
from copy import deepcopy
import re
from io import BytesIO

import git
import pandas as pd

repo = git.Repo(search_parent_directories=True)
sys.path.append(repo.working_dir)

import helper_tools.evaluation

importlib.reload(helper_tools.evaluation)

from helper_tools import parser
from helper_tools.evaluation import generate_report, calculate_scores_from_array, parse_turtle, get_uri_labels

import streamlit as st

stss = st.session_state

if "dataset_cache" not in stss:
    stss.dataset_cache = dict()

uploaded_file = st.file_uploader(
    "Choose an evaluation log", accept_multiple_files=False
)

if uploaded_file:
    match = re.match(r"(?P<split>\w+)-(?P<num_samples>\d+)-evaluation_log-.*\.xlsx", uploaded_file.name)
    split = match.group("split")
    number_of_samples = int(match.group("num_samples"))
    try:
        stss.relation_df, stss.entity_df, stss.docs = stss.dataset_cache[f"{split}-{number_of_samples}"]
    except KeyError:
        stss.relation_df, stss.entity_df, stss.docs = parser.synthie_parser(split, number_of_samples)
        stss.dataset_cache[f"{split}-{number_of_samples}"] = (stss.relation_df, stss.entity_df, stss.docs)
    stss.entity_set = stss.entity_df[['entity', 'entity_uri']].drop_duplicates()
    stss.predicate_set_df = stss.relation_df[["predicate", "predicate_uri"]].drop_duplicates()
    uploaded_file_io = BytesIO(uploaded_file.read())
    evaluation_log_df = pd.read_excel(uploaded_file_io)
    
    error_patterns = [
        r'^Error',
        r'error',
        r'exception',
        r'failed',
        r'Recursion limit',
        r'Traceback',
        r'not found',
        r'invalid',
        r'crash',
        r'cannot',
        r'undefined',
        r'unsupported',
        r'failure',
    ]
    error_regex = '|'.join(error_patterns)

    # Checkbox to exclude errors
    exclude_errors = st.checkbox("Exclude Errors")
    filtered_df = evaluation_log_df
    if exclude_errors:
        filtered_df = evaluation_log_df[~evaluation_log_df['Result String'].str.contains(error_regex, case=False, regex=True)]

    # Count errors in the evaluation log using a list of common error patterns
    
    error_count = evaluation_log_df['Result String'].str.contains(error_regex, case=False, regex=True).sum()
    st.write(f"Number of errors in evaluation log: {error_count}")
    error_percentage = (error_count / len(evaluation_log_df)) * 100 if len(evaluation_log_df) > 0 else 0
    st.write(f"Percentage of errors in evaluation log: {error_percentage:.1f}%")

    # Schreibe das gefilterte DataFrame korrekt als Excel in einen Buffer
    excel_buffer = BytesIO()
    filtered_df.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)
    report = generate_report(excel_buffer)
    st.write(report)
    for _, row in filtered_df.iterrows():
        with st.container(border=True):
            score = calculate_scores_from_array(row.to_list()[1:(22-len(row))])
            st.write(stss.docs.loc[row["Doc ID"]]["text"])
            st.write(score.loc["Triple"]["F1-Score"])
            result_string = str(row["Result String"])
            turtle_string_match = re.search(r'<ttl>(.*?)</ttl>', result_string, re.DOTALL)
            if turtle_string_match:
                turtle_string = turtle_string_match.group(1)
            else:
                turtle_string = result_string
            result_df, error = parse_turtle(turtle_string)
            if len(result_df) == 0:
                st.error(f"{error}")
            with st.expander("Show Details"):
                st.write("*Turtle String*")
                st.code(turtle_string)
                st.write("*Predicted Triples*")
                st.write(get_uri_labels(result_df)[["subject", "predicate","object"]])
                st.divider()
                st.write("*Actual Triples*")
                st.write(stss.relation_df[stss.relation_df["docid"] == row["Doc ID"]][["subject", "predicate","object"]])
                st.divider()
                st.write("*Score*")
                st.write(score)
                if row.get("Langfuse Trace ID"):
                    st.write(f"[Langfuse Trace: {row['Langfuse Trace ID']}](https://langfuse.webwerkstatt-lautenbach.de/project/cm90430l30006qb07xe6fz6t9/traces?peek={row['Langfuse Trace ID']})")


