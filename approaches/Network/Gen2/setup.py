import sys
from typing import TypedDict, List, Tuple

import git

repo = git.Repo(search_parent_directories=True)
sys.path.append(repo.working_dir)

from helper_tools.base_setup import *


class cIEState(TypedDict):
    text: str
    triples: List[str]
    last_call: str
    last_response: str
    agent_instruction: str
    tool_input: str
    messages: List[str]
    debug: bool
    call_trace: List[Tuple[str, str]]  # List of (agent_id/tool_id, agent_instruction/tool_input) tuples
