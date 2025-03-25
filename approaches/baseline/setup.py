import sys

import git

repo = git.Repo(search_parent_directories=True)
sys.path.append(repo.working_dir)

from helper_tools.base_setup import *


class cIEState(TypedDict):
    text: str
    messages: list[str]
    instruction: str
    verbose: bool
    debug: bool
