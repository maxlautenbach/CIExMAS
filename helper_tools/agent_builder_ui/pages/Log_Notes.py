import streamlit as st
import json
import os
import git
from pathlib import Path
from datetime import datetime

repo = git.Repo(search_parent_directories=True)

# Set page config
st.set_page_config(
    page_title="Log Notes",
    page_icon="📝",
    layout="wide"
)

# Title
st.title("Evaluation Log Notes")

# Path to log_notes.json
NOTES_FILE = Path(repo.working_dir + "/approaches/evaluation_logs/log_notes.json")

# Function to load notes
def load_notes():
    if NOTES_FILE.exists():
        with open(NOTES_FILE, 'r') as f:
            return json.load(f)
    return {}

# Function to save notes
def save_notes(notes):
    with open(NOTES_FILE, 'w') as f:
        json.dump(notes, f, indent=4)

def extract_model_info(file_path):
    """Extract model information from file path."""
    filename = os.path.basename(file_path)
    model_info = filename.split("evaluation_log-")[1].replace(".xlsx", "").replace("-converted", "").replace("_", "-")

    # Split by underscores to get components
    parts = model_info.split("-")[:-4]
    provider = parts[0]
    model_id = "-".join(parts[1:])
    return f"{provider} - {model_id}"

def extract_architecture(file_path):
    """Extract architecture name from the directory path."""
    dir_name = os.path.basename(os.path.dirname(file_path))
    return dir_name

def extract_datasplit(file_path):
    """Extract dataset and split information from file path."""
    filename = os.path.basename(file_path)
    # Extract the split info (e.g., "synthie_code-train-5" from "synthie_code-train-5-evaluation_log-...")
    split_info = filename.split("-evaluation_log-")[0]
    
    # Split by hyphen and take first two parts
    parts = split_info.split("-")
    dataset = parts[0]  # e.g., "synthie_code" or "rebel"
    split = parts[1]    # e.g., "train" or "test"
    
    return dataset, split

def extract_timestamp(file_path):
    """Extract timestamp from filename."""
    filename = os.path.basename(file_path)
    filename = filename.replace('.xlsx', '').replace('-converted', '').replace("_", "-")
    timestamp_str = "-".join(filename.split('-')[-4:])
    return datetime.strptime(timestamp_str, '%Y-%m-%d-%H%M')

def get_note_display_info(log_file):
    """Get display information for a log file."""
    full_path = None
    for root, _, files in os.walk(eval_logs_dir):
        if log_file in files:
            full_path = os.path.join(root, log_file)
            break
    
    if not full_path:
        return None
    
    dataset, split = extract_datasplit(full_path)
    architecture = extract_architecture(full_path)
    model_info = extract_model_info(full_path)
    
    return {
        "dataset": dataset,
        "split": split,
        "architecture": architecture,
        "model_info": model_info
    }

# Load existing notes
notes = load_notes()

# Get list of evaluation logs
eval_logs_dir = Path(repo.working_dir + "/approaches/evaluation_logs")
log_files = []
for root, dirs, files in os.walk(eval_logs_dir):
    for file in files:
        if file.endswith('.xlsx'):
            log_files.append(os.path.join(root, file))

# Organize files by architecture, dataset, split, model, and timestamp
organized_files = {}
for file_path in log_files:
    architecture = extract_architecture(file_path)
    dataset, split = extract_datasplit(file_path)
    model_info = extract_model_info(file_path)
    timestamp = extract_timestamp(file_path)
    
    if architecture not in organized_files:
        organized_files[architecture] = {}
    if dataset not in organized_files[architecture]:
        organized_files[architecture][dataset] = {}
    if split not in organized_files[architecture][dataset]:
        organized_files[architecture][dataset][split] = {}
    if model_info not in organized_files[architecture][dataset][split]:
        organized_files[architecture][dataset][split][model_info] = []
    
    organized_files[architecture][dataset][split][model_info].append((timestamp, file_path))

# Sort timestamps for each model
for arch in organized_files:
    for dataset in organized_files[arch]:
        for split in organized_files[arch][dataset]:
            for model in organized_files[arch][dataset][split]:
                organized_files[arch][dataset][split][model].sort(key=lambda x: x[0])

# Create two columns
col1, col2 = st.columns(2)

# Left column for selecting log file and adding/editing notes
with col1:
    st.subheader("Add/Edit Notes")
    
    # Mode selection (Add new or Edit existing)
    mode = st.radio(
        "Select Mode",
        ["Add New Note", "Edit Existing Note"]
    )
    
    if mode == "Add New Note":
        # Architecture selection
        architectures = sorted(organized_files.keys())
        selected_architecture = st.selectbox(
            "Select Architecture",
            architectures
        )
        
        # Dataset selection
        datasets = sorted(organized_files[selected_architecture].keys())
        selected_dataset = st.selectbox(
            "Select Dataset",
            datasets
        )
        
        # Split selection
        splits = sorted(organized_files[selected_architecture][selected_dataset].keys())
        selected_split = st.selectbox(
            "Select Split",
            splits
        )
        
        # Model selection (Provider + Model ID combined)
        models = sorted(organized_files[selected_architecture][selected_dataset][selected_split].keys())
        selected_model = st.selectbox(
            "Select Model",
            models
        )
        
        # Timestamp selection
        timestamps = organized_files[selected_architecture][selected_dataset][selected_split][selected_model]
        timestamp_options = [ts[0].strftime("%Y-%m-%d %H:%M") for ts in timestamps]
        selected_timestamp = st.selectbox(
            "Select Timestamp",
            timestamp_options
        )
        
        # Get the selected file path
        selected_file = None
        for ts, file_path in timestamps:
            if ts.strftime("%Y-%m-%d %H:%M") == selected_timestamp:
                selected_file = file_path
                break
    else:  # Edit Existing Note
        # Architecture selection
        architectures = sorted(organized_files.keys())
        selected_architecture = st.selectbox(
            "Select Architecture",
            architectures,
            key="edit_architecture"
        )
        
        # Dataset selection
        datasets = sorted(organized_files[selected_architecture].keys())
        selected_dataset = st.selectbox(
            "Select Dataset",
            datasets,
            key="edit_dataset"
        )
        
        # Split selection
        splits = sorted(organized_files[selected_architecture][selected_dataset].keys())
        selected_split = st.selectbox(
            "Select Split",
            splits,
            key="edit_split"
        )
        
        # Model selection
        models = sorted(organized_files[selected_architecture][selected_dataset][selected_split].keys())
        selected_model = st.selectbox(
            "Select Model",
            models,
            key="edit_model"
        )
        
        # Get timestamps for the selected combination
        timestamps = organized_files[selected_architecture][selected_dataset][selected_split][selected_model]
        
        # Filter files_with_notes based on the selected criteria
        filtered_files = []
        for ts, file_path in timestamps:
            if os.path.basename(file_path) in notes:
                filtered_files.append((ts, file_path))
        
        if not filtered_files:
            st.info("No notes exist for the selected combination. Please select a different combination or add a new note.")
            st.stop()
        
        # Show timestamp selection for files that have notes
        timestamp_options = [ts[0].strftime("%Y-%m-%d %H:%M") for ts in filtered_files]
        selected_timestamp = st.selectbox(
            "Select Timestamp",
            timestamp_options,
            key="edit_timestamp"
        )
        
        # Get the selected file path
        selected_file = None
        for ts, file_path in filtered_files:
            if ts.strftime("%Y-%m-%d %H:%M") == selected_timestamp:
                selected_file = file_path
                break
    
    if selected_file:
        # Get existing notes for selected file
        existing_notes = notes.get(os.path.basename(selected_file), {})
        
        # Form for adding/editing notes
        with st.form("notes_form"):
            short_desc = st.text_input(
                "Short Description",
                value=existing_notes.get("short_description", "")
            )
            description = st.text_area(
                "Description",
                value=existing_notes.get("description", "")
            )
            
            submitted = st.form_submit_button("Save Notes")
            
            if submitted:
                # Update notes
                notes[os.path.basename(selected_file)] = {
                    "short_description": short_desc,
                    "description": description
                }
                save_notes(notes)
                st.success("Notes saved successfully!")
                # Clear the form by rerunning the app
                st.rerun()

# Right column for displaying existing notes
with col2:
    st.subheader("Existing Notes")
    
    if not notes:
        st.info("No notes have been added yet.")
    else:
        # Sort notes by dataset, architecture, and model
        sorted_notes = []
        for log_file, note_data in notes.items():
            display_info = get_note_display_info(log_file)
            if display_info:
                sorted_notes.append((
                    display_info["dataset"],
                    display_info["architecture"],
                    display_info["model_info"],
                    log_file,
                    note_data
                ))
        
        # Sort by dataset, architecture, and model
        sorted_notes.sort(key=lambda x: (x[0], x[1], x[2]))
        
        for dataset, architecture, model_info, log_file, note_data in sorted_notes:
            display_title = f"{dataset} - {architecture} - {model_info} - {note_data.get('short_description', 'No short description')}"
            with st.expander(display_title):
                st.write("**Description:**")
                st.write(note_data.get("description", "No description")) 