import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import git
import sys
import json
from collections import defaultdict
import numpy as np
from datetime import datetime

repo = git.Repo(search_parent_directories=True)
sys.path.append(repo.working_dir)

from helper_tools.evaluation import generate_report

# Model ID to shortened name mapping
MODEL_ID_MAPPING = {
    "kosbu-Llama-3.3-70B-Instruct-AWQ": "Llama 3.3 AWQ",
    "ISTA-DASLab-gemma-3-27b-it-GPTQ-4b-128g": "Gemma 3 GPTQ",
    "Meta-Llama-3.3-70B-Instruct": "Llama 3.3",
    "google-gemma-3-27b-it": "Gemma 3",
    "Llama-4-Maverick-17B-128E-Instruct": "Llama 4 Maverick",
    "deepseek-ai-DeepSeek-R1-Distill-Llama-70B": "DeepSeek R1 Dist. Llama",
}

# Set page to full width
st.set_page_config(layout="wide")

def get_evaluation_files():
    """Get all evaluation log files from the evaluation_logs directory."""
    evaluation_files = []
    workspace_root = repo.working_dir
    evaluation_logs_dir = os.path.join(workspace_root,"approaches/evaluation_logs")
    
    if not os.path.exists(evaluation_logs_dir):
        return []
        
    for root, _, files in os.walk(evaluation_logs_dir):
        for file in files:
            if file.endswith(".xlsx"):
                evaluation_files.append(os.path.join(root, file))
    return evaluation_files

def extract_model_info(file_path):
    """Extract model information from file path."""
    # Get the filename without path and extension
    filename = os.path.basename(file_path)
    
    # Remove the evaluation_log- prefix and .xlsx suffix
    model_info = filename.replace("evaluation_log-", "").replace(".xlsx", "").replace("-converted", "").replace("_", "-")
    

    # Split by underscores to get components
    parts = model_info.split("-")[2:-4]
    provider = parts[0]
    model_id = "-".join(parts[1:])
    
    # Get shortened model name if available
    display_model_id = MODEL_ID_MAPPING.get(model_id, model_id)
    
    return provider, model_id, display_model_id

def extract_datasplit(file_path):
    """Extract datasplit from file path."""
    filename = os.path.basename(file_path)
    # Extract the split and number (e.g., "test-5" from "test-5-evaluation_log-...")
    split_info = filename.split("-evaluation_log-")[0]
    return split_info

def extract_architecture(file_path):
    """Extract architecture name from the directory path."""
    dir_name = os.path.basename(os.path.dirname(file_path))
    return dir_name

def extract_timestamp(file_path):
    """Extract timestamp from filename."""
    filename = os.path.basename(file_path)
    # Remove .xlsx and -converted if present
    filename = filename.replace('.xlsx', '').replace('-converted', '').replace("_", "-")
    # Extract the timestamp part (e.g., "2025-04-21-1707")
    timestamp_str = "-".join(filename.split('-')[-4:])
    # Convert to datetime object
    return datetime.strptime(timestamp_str, '%Y-%m-%d-%H%M')

def load_notes():
    """Load notes from log_notes.json."""
    notes_file = os.path.join(repo.working_dir, "approaches/evaluation_logs/log_notes.json")
    if os.path.exists(notes_file):
        with open(notes_file, 'r') as f:
            return json.load(f)
    return {}

def main():
    st.title("Evaluation Results Comparison")
    
    # Get all evaluation files
    evaluation_files = get_evaluation_files()
    
    if not evaluation_files:
        st.error("No evaluation files found in evaluation_logs directory!")
        return
    
    # Extract unique datasplits
    datasplits = sorted(set(extract_datasplit(f) for f in evaluation_files))
    
    # Create dropdown for datasplit selection
    selected_datasplit = st.selectbox(
        "Select Datasplit",
        datasplits
    )
    
    # Filter files for selected datasplit
    filtered_files = [f for f in evaluation_files if extract_datasplit(f) == selected_datasplit]
    
    # Track runs per model with timestamps
    model_runs = defaultdict(list)
    model_names = {}
    
    # Load notes
    notes = load_notes()
    
    # First pass: collect all timestamps for each model
    for file_path in filtered_files:
        provider, model_id, display_model_id = extract_model_info(file_path)
        architecture = extract_architecture(file_path)
        if provider and model_id:
            model_key = f"{provider}_{model_id}_{architecture}"
            timestamp = extract_timestamp(file_path)
            model_runs[model_key].append((timestamp, file_path))
    
    # Second pass: sort by timestamp and assign run numbers with short descriptions
    for model_key in model_runs:
        # Sort by timestamp
        model_runs[model_key].sort(key=lambda x: x[0])
        # Assign run numbers with short descriptions
        for i, (timestamp, file_path) in enumerate(model_runs[model_key], 1):
            provider, model_id, display_model_id = extract_model_info(file_path)
            architecture = extract_architecture(file_path)
            filename = os.path.basename(file_path)
            
            # Get short description if available
            short_desc = notes.get(filename, {}).get("short_description", "")
            run_info = f"Run {i}"
            if short_desc:
                run_info = f"{run_info} - {short_desc}"
            
            model_name = f"{architecture} - {provider} - {display_model_id} ({run_info})"
            model_names[file_path] = model_name
    
    # Generate reports for all files
    reports = []
    for file_path in filtered_files:
        if file_path in model_names:
            report = generate_report(file_path)
            reports.append((model_names[file_path], report))
    
    # Get available metrics from the first report
    available_metrics = reports[0][1].index.tolist() if reports else []
    
    # Create dropdown for metric selection
    selected_metric = st.selectbox(
        "Select Metric",
        available_metrics
    )
    
    # Create dropdown for score selection
    score_metrics = ['F1-Score', 'Precision', 'Recall']
    selected_score = st.selectbox(
        "Select Score for Sorting and Display",
        score_metrics
    )
    
    # Create the plot
    if reports:
        # Extract architectures and scores for sorting
        architectures = [x[0].split(" - ")[0] for x in reports]
        scores = [x[1].loc[selected_metric][selected_score] for x in reports]
        model_names = [x[0] for x in reports]
        
        # Create a DataFrame for sorting
        df_plot = pd.DataFrame({
            'Model': model_names,
            'Architecture': architectures,
            'Score': scores
        })
        
        # Sort by architecture and then by score in descending order
        df_plot = df_plot.sort_values(['Architecture', 'Score'], ascending=[True, False])
        
        # Get the sorted values
        x_labels = df_plot['Model'].tolist()
        y_values = df_plot['Score'].tolist()
        architectures = df_plot['Architecture'].tolist()
        
        unique_architectures = list(set(architectures))
        color_map = plt.get_cmap('tab20', len(unique_architectures))
        colors = [color_map(unique_architectures.index(arch)) for arch in architectures]
        
        fig, ax = plt.subplots(figsize=(12, 6))  # Adjusted for horizontal bars
        bars = ax.barh(range(len(reports)), y_values, color=colors)  # Changed to barh for horizontal bars
        ax.set_yticks(range(len(reports)))
        ax.set_yticklabels(x_labels)
        ax.set_xlabel(selected_score)
        ax.set_xlim(0, 1)  # Set x-axis range from 0 to 1
        ax.set_title(f"{selected_score} for '{selected_metric}' - {selected_datasplit}")
        plt.tight_layout()
        
        # Add value labels on the right of bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{width:.3f}',
                   ha='left', va='center')
        
        # Add legend for architectures
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_map(i)) 
                          for i in range(len(unique_architectures))]
        ax.legend(legend_elements, unique_architectures, 
                 title='Architecture', bbox_to_anchor=(1.05, 1), 
                 loc='upper left')
        
        st.pyplot(fig, bbox_inches='tight')
        
        # Display the data table
        st.subheader("Detailed Results")
        data = []
        for model_name, report in reports:
            row = {"Model": model_name}
            row.update(report.loc[selected_metric].to_dict())
            data.append(row)
        
        df = pd.DataFrame(data)
        # Sort by selected score in descending order
        df = df.sort_values(selected_score, ascending=False)
        # Add rank column starting from 1
        df.insert(0, "Rank", range(1, len(df) + 1))
        st.dataframe(df)

if __name__ == "__main__":
    main() 