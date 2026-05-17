import sys
from pathlib import Path
from typing import TypedDict

import git

repo = git.Repo(search_parent_directories=True)
sys.path.append(repo.working_dir)

# Add ACE framework to path
ACE_REPO = Path(repo.working_dir).parent / "ace"
if ACE_REPO.exists():
    sys.path.insert(0, str(ACE_REPO))

from helper_tools.base_setup import *


class cIEState(TypedDict):
    text: str
    entities: set[str]
    predicates: set[str]
    triples: set[str]
    uri_candidates: dict[str, list[dict]]
    turtle: str
    instruction: str
