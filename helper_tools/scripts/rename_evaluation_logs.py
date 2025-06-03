import os
import pandas as pd
import json
from pathlib import Path

def rename_evaluation_logs():
    """
    Rename evaluation log files and update their contents according to the new naming scheme:
    - test_text- -> synthie_text-test
    - test- -> synthie_code-test
    - train_text- -> synthie_text-train
    - train- -> synthie_code-train
    """
    # Mapping for file name changes
    name_mapping = {
        'test_text-': 'synthie_text-test-',
        'test-': 'synthie_code-test-',
        'train_text-': 'synthie_text-train-',
        'train-': 'synthie_code-train-'
    }
    
    # Get the project root directory (two levels up from this script)
    project_root = Path(__file__).parent.parent.parent
    
    # Walk through the evaluation logs directory
    eval_logs_dir = project_root / "approaches" / "evaluation_logs"
    for root, dirs, files in os.walk(eval_logs_dir):
        for file in files:
            if file.endswith(".xlsx"):
                old_path = os.path.join(root, file)
                
                # Check if file needs to be renamed
                new_name = file
                for old_prefix, new_prefix in name_mapping.items():
                    if file.startswith(old_prefix):
                        new_name = file.replace(old_prefix, new_prefix)
                        break
                
                if new_name != file:
                    new_path = os.path.join(root, new_name)
                    print(f"Renaming {file} to {new_name}")
                    
                    # Read the Excel file
                    df = pd.read_excel(old_path)
                    
                    # Update the Result String column if it exists
                    if "Result String" in df.columns:
                        for idx, row in df.iterrows():
                            result_string = str(row["Result String"])
                            # Update the split name in the log notes
                            for old_prefix, new_prefix in name_mapping.items():
                                if old_prefix in result_string:
                                    result_string = result_string.replace(old_prefix, new_prefix)
                                    df.at[idx, "Result String"] = result_string
                    
                    # Save the updated file with the new name
                    df.to_excel(new_path, index=False)
                    
                    # Remove the old file
                    os.remove(old_path)
                    print(f"Updated and moved {file} to {new_name}")

def rename_log_notes():
    """
    Update the log_notes.json file to use the new naming scheme:
    - test_text- -> synthie_text-test
    - test- -> synthie_code-test
    - train_text- -> synthie_text-train
    - train- -> synthie_code-train
    """
    # Mapping for file name changes
    name_mapping = {
        'test_text-': 'synthie_text-test-',
        'test-': 'synthie_code-test-',
        'train_text-': 'synthie_text-train-',
        'train-': 'synthie_code-train-'
    }
    
    # Get the project root directory (two levels up from this script)
    project_root = Path(__file__).parent.parent.parent
    
    # Path to log_notes.json
    log_notes_path = project_root / "approaches" / "evaluation_logs" / "log_notes.json"
    
    # Read the current log_notes.json
    with open(log_notes_path, 'r') as f:
        log_notes = json.load(f)
    
    # Create new log_notes with updated names
    new_log_notes = {}
    for old_name, content in log_notes.items():
        new_name = old_name
        for old_prefix, new_prefix in name_mapping.items():
            if old_name.startswith(old_prefix):
                new_name = old_name.replace(old_prefix, new_prefix)
                break
        
        if new_name != old_name:
            print(f"Renaming {old_name} to {new_name}")
            new_log_notes[new_name] = content
        else:
            new_log_notes[old_name] = content
    
    # Write the updated log_notes back to file
    with open(log_notes_path, 'w') as f:
        json.dump(new_log_notes, f, indent=4)
    
    print("Log notes update complete!")


if __name__ == "__main__":
    ##rename_evaluation_logs()
    rename_log_notes() 