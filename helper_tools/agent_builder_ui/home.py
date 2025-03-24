import os
import sys
from cProfile import label
from copy import deepcopy
import re

import git

repo = git.Repo(search_parent_directories=True)
sys.path.append(repo.working_dir)

import streamlit as st
import helper_tools.parser as parser
from helper_tools.evaluation import evaluate, get_uri_labels, parse_turtle
import importlib
import pandas as pd

st.set_page_config(layout="wide")

stss = st.session_state


def reset_state():
    target_doc = stss.docs.iloc[stss.doc_index]
    text = target_doc["text"]
    stss.state = {"text": text, "results": [], "call_trace": [], "comments": [], "debug": True}
    stss.state_history = [deepcopy(stss.state)]
    stss.state_index = 0
    stss.last_answers = dict()
    stss.agent_trace = []


if "relation_df" not in stss:
    stss.relation_df, stss.entity_df, stss.docs = parser.synthie_parser("train")
    stss.entity_set = stss.entity_df[['entity', 'entity_uri']].drop_duplicates()
    stss.predicate_set_df = stss.relation_df[["predicate", "predicate_uri"]].drop_duplicates()

if "doc_index" not in stss:
    stss.doc_index = 0

if "state" not in stss:
    reset_state()


def change_index():
    stss.state_index = stss.state_index_sbox


def update_doc_index():
    stss.doc_index = stss.doc_index_sbox
    reset_state()


st.sidebar.selectbox("State Index", [*range(len(stss.state_history))], key="state_index_sbox", on_change=change_index,
                     index=stss.state_index)

st.header("Current State")
st.subheader("Text:")
st.write(stss.state["text"])

active_state = stss.state_history[stss.state_index_sbox]
with st.expander("Results:"):
    for result in active_state["results"]:
        st.code(result, language=None, wrap_lines=True)
with st.expander("Call Trace:"):
    for call in active_state["call_trace"]:
        st.code(call, language=None, wrap_lines=True)
with st.expander("Comments:"):
    for comment in active_state["comments"]:
        st.code(comment, language=None, wrap_lines=True)


def update_agent_list():
    if stss.option == "Gen1":
        stss.agents = []
        agent_dir = repo.working_dir + "/approaches/full_sentence/Gen1/agents/"
        for agent in [file for file in os.listdir(agent_dir) if ".py" in file]:
            stss.agents.append(import_agent(agent, agent_dir))
        for agent in stss.agents:
            importlib.reload(agent["module"])


st.sidebar.selectbox(
    "Approach",
    ("Gen1"),
    index=None,
    placeholder="Select approach...",
    key="option",
    on_change=update_agent_list
)

st.sidebar.selectbox(
    "Document",
    [*range(len(stss.docs))],
    key="doc_index_sbox",
    on_change=update_doc_index
)


def import_agent(agent_name, agent_dir):
    sys.path.append(agent_dir)
    module = importlib.import_module(agent_name.replace(".py", ""))
    importlib.reload(module)
    return {"id": agent_name.replace(".py", ""), "module": module}


def run_agent(id, module):
    updated_state, response = module.agent(deepcopy(stss.state_history[stss.state_index_sbox]))
    stss.state_history.append(deepcopy(updated_state))
    stss.state_index = len(stss.state_history) - 1
    stss.agent_trace.append(id)
    return response


col1, col2, col3 = st.columns(3)
col1.header("Agents:")
st.write(f"Last 3 agents: {stss.agent_trace[-3:]}")
if col2.button("Refresh Agent List"):
    update_agent_list()
if col3.button("Reset State"):
    reset_state()
    st.rerun()

for agent in stss.get("agents", []):
    with st.container(border=True):
        st.write(agent["id"].capitalize())
        if st.button(f"Run {agent['id'].capitalize()}"):
            stss.last_answers[agent["id"]] = run_agent(agent["id"],agent["module"])
            st.rerun()
        if stss.last_answers.get(agent["id"]):
            with st.expander("Last Answer: "):
                st.write(stss.last_answers[agent["id"]])

st.header("Evaluation")
if st.button("Run Evaluation"):
    last_state = stss.state_history[-1]
    turtle_string = last_state["results"][-1]
    precision, recall, f1_score = evaluate(turtle_string=turtle_string, doc_id=stss.docs.iloc[stss.doc_index]["docid"], relation_df=stss.relation_df)
    st.write(f'Precision: {round(precision,2)*100}%')
    st.write(f'Recall: {round(recall,2)*100}%')
    st.write(f'F1 Score: {round(f1_score,2)*100}%')
    st.write(get_uri_labels(parse_turtle(turtle_string), stss.entity_set, stss.predicate_set_df)[
                 ["subject", "predicate", "object"]])
    st.write(stss.relation_df[stss.relation_df["docid"] == stss.docs.iloc[stss.doc_index]["docid"]][["subject", "predicate", "object"]])
