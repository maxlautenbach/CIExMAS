import sys
from typing import TypedDict

import git

repo = git.Repo(search_parent_directories=True)
sys.path.append(repo.working_dir)

from helper_tools.base_setup import *


class cIEState(TypedDict):
    text: str
    call_trace: list[tuple[str]]
    results: list[str]
    comments: list[str]
    instruction: str
    debug: bool
