import sys
from typing import TypedDict, List

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
