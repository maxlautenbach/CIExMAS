import os
import sys
from copy import deepcopy

import git

repo = git.Repo(search_parent_directories=True)
sys.path.append(repo.working_dir)

import streamlit as st
import helper_tools.parser as parser
import importlib
import pandas as pd

st.set_page_config(layout="wide")

stss = st.session_state


def reset_state():
    target_doc = stss.docs.iloc[0]
    text = target_doc["text"]
    stss.state = {"text": text, "results": [], "call_trace": [], "comments": [], "debug": True}
    stss.state_history = [deepcopy(stss.state)]
    stss.state_index = 0
    stss.last_answers = dict()


if "relation_df" not in stss:
    stss.relation_df, stss.entity_df, stss.docs = parser.synthie_parser("train")
    stss.entity_set = stss.entity_df[['entity', 'entity_uri']].drop_duplicates()
    stss.predicate_set_df = stss.relation_df[["predicate", "predicate_uri"]].drop_duplicates()

if "state" not in stss:
    reset_state()


def change_index():
    stss.state_index = stss.state_index_sbox


st.sidebar.selectbox("State Index", [*range(len(stss.state_history))], key="state_index_sbox", on_change=change_index,
                     index=stss.state_index)

st.header("Current State")
with st.expander("Expand State:"):
    st.write(stss.state_history[stss.state_index_sbox])


def update_agent_list():
    if stss.option == "Gen1":
        stss.agents = []
        agent_dir = repo.working_dir + "/approaches/full_sentence/Gen1/agents/"
        for agent in [file for file in os.listdir(agent_dir) if ".py" in file]:
            stss.agents.append(import_agent(agent, agent_dir))


st.sidebar.selectbox(
    "Approach",
    ("Gen1"),
    index=None,
    placeholder="Select approach...",
    key="option",
    on_change=update_agent_list
)


def import_agent(agent_name, agent_dir):
    sys.path.append(agent_dir)
    module = importlib.import_module(agent_name.replace(".py", ""))
    importlib.reload(module)
    return {"id": agent_name.replace(".py", ""), "module": module}


def run_agent(module):
    updated_state, response = module.agent(deepcopy(stss.state_history[stss.state_index_sbox]))
    stss.state_history.append(deepcopy(updated_state))
    stss.state_index = len(stss.state_history) - 1
    return response

col1, col2, col3 = st.columns(3)
col1.header("Agents:")
if col2.button("Refresh Agent List"):
    update_agent_list()
if col3.button("Reset State"):
    reset_state()
    st.rerun()

for agent in stss.get("agents", []):
    with st.container(border=True):
        st.write(agent["id"].capitalize())
        if st.button(f"Run {agent['id'].capitalize()}"):
            stss.last_answers[agent["id"]] = run_agent(agent["module"])
            st.rerun()
        if stss.last_answers.get(agent["id"]):
            with st.expander("Last Answer: "):
                st.write(stss.last_answers[agent["id"]])
