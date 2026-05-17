#!/usr/bin/env python3
"""
Regenerate viewer.html with embedded data from all ACE run folders.

Usage:
    python build_viewer.py
"""

import json
import os
import re
import glob
from pathlib import Path

RESULTS_DIR = Path(__file__).parent
VIEWER_PATH = RESULTS_DIR / "viewer.html"


def collect_run_data():
    runs = {}
    for d in sorted(glob.glob(str(RESULTS_DIR / "ace_*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        run = {}

        for f in ["run_config.json", "initial_test_results.json", "final_test_results.json"]:
            path = os.path.join(d, f)
            if os.path.exists(path):
                with open(path) as fh:
                    run[f] = json.load(fh)

        for pf in ["best_playbook.txt", "final_playbook.txt"]:
            path = os.path.join(d, pf)
            if os.path.exists(path):
                with open(path) as fh:
                    run["playbook"] = fh.read()
                break

        op_path = os.path.join(d, "curator_operations_diff.jsonl")
        if os.path.exists(op_path):
            with open(op_path) as fh:
                run["operations"] = [json.loads(l) for l in fh if l.strip()]

        log_dir = os.path.join(d, "llm_logs")
        run["llmLogs"] = []
        if os.path.isdir(log_dir):
            for lf in sorted(os.listdir(log_dir)):
                if lf.endswith(".json"):
                    with open(os.path.join(log_dir, lf)) as fh:
                        log = json.load(fh)
                        log.pop("prompt", None)
                        log.pop("response", None)
                        log.pop("raw_response", None)
                        run["llmLogs"].append(log)

        runs[name] = run

    return runs


def update_viewer(runs_data):
    with open(VIEWER_PATH, "r") as f:
        html = f.read()

    data_json = json.dumps(runs_data, ensure_ascii=False)
    pattern = r"const RAW_DATA = \{.*?\};"
    replacement = f"const RAW_DATA = {data_json};"
    html = re.sub(pattern, replacement, html, count=1, flags=re.DOTALL)

    with open(VIEWER_PATH, "w") as f:
        f.write(html)

    print(f"Updated viewer.html with {len(runs_data)} runs")
    for name, run in runs_data.items():
        config = run.get("run_config.json", {})
        status = "complete" if "final_test_results.json" in run else "incomplete"
        print(f"  {name}: {config.get('mode','?')} {config.get('num_train','?')}T/{config.get('num_test','?')}E [{status}]")


if __name__ == "__main__":
    runs = collect_run_data()
    if not runs:
        print("No ace_* run folders found.")
    else:
        update_viewer(runs)
