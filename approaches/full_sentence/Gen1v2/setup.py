import sys
from typing import TypedDict

import git

repo = git.Repo(search_parent_directories=True)
sys.path.append(repo.working_dir)

from helper_tools.base_setup import *


class cIEState(TypedDict):
    text: str
    messages: list[str]
    entities: set[str]
    predicates: set[str]
    triples: set[str]
    uri_mapping: set[tuple[str, str]]
    agent_response: str
    instruction: str
    debug: bool
