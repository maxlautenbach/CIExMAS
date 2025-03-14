import os
import sys
import git

repo = git.Repo(search_parent_directories=True)
sys.path.append(repo.working_dir)

import streamlit as st
import helper_tools.parser as parser
import importlib
import pandas as pd


def import_agent(agent_name, agent_dir):
    sys.path.append(agent_dir)
    return {"id": agent_name.replace(".py", ""), "module": importlib.import_module(agent_name.replace(".py", ""))}


stss = st.session_state

if "relation_df" not in stss:
    stss.relation_df, stss.entity_df, stss.docs = parser.synthie_parser("train")
    stss.entity_set = stss.entity_df[['entity', 'entity_uri']].drop_duplicates()
    stss.predicate_set_df = stss.relation_df[["predicate", "predicate_uri"]].drop_duplicates()

if "state" not in stss:
    target_doc = stss.docs.iloc[0]
    doc_id = target_doc["docid"]
    text = target_doc["text"]
    stss.state = {"text": text, "results": [], "call_trace": [], "comments": []}
    stss.state_history = []

st.header("Current State")
st.write(stss.state)

option = st.sidebar.selectbox(
    "Approach",
    ("Gen1"),
    index=None,
    placeholder="Select approach...",
)

stss.agents = []

if option == "Gen1":
    agent_dir = repo.working_dir + "/approaches/full_sentence/Gen1/agents/"
    print(os.listdir(agent_dir))
    for agent in [file for file in os.listdir(agent_dir) if ".py" in file]:
        stss.agents.append(import_agent(agent, agent_dir))

for agent in stss.agents:
    st.header("Agents:")
    with st.container(border=True):
        st.write(agent["id"])
