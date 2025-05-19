import sys
from typing import TypedDict, List, Tuple

import git

repo = git.Repo(search_parent_directories=True)
sys.path.append(repo.working_dir)

from helper_tools.base_setup import *


class cIEState(TypedDict):
    text: str
    triples: List[str]
    last_call: str  # Only stores agent_id or tool_id
    last_response: str
    agent_instruction: str
    tool_input: str
    messages: List[str]
    debug: bool
    uri_mapping: str
    call_trace: List[str]  # List of agent_ids or tool_ids
