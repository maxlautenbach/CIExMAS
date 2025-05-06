#!/usr/bin/env python3

import sys
import git
import pandas as pd
from dotenv import load_dotenv
import importlib

from helper_tools.wikidata_loader import get_types

# Find the root repository directory to properly add it to sys.path
repo = git.Repo(search_parent_directories=True)
sys.path.append(repo.working_dir)

# Load environment variables
load_dotenv(repo.working_dir + "/.env", override=True)

import helper_tools.parser as parser
importlib.reload(parser)

def few_shot_generator(num_docs_to_display=3, target_agent="extractor"):
    """
    Load train dataset with 50 examples, filter for documents with ID >= 10,
    and display text and triples for specified number of documents.
    
    Args:
        num_docs_to_display: Number of documents to display (default: 3)
        target_agent: Type of agent to generate examples for. Options:
                     - "extractor": Display basic triple extraction
                     - "uri_mapping": Display triples with URIs included
    """
    print("Loading training dataset with 50 examples...")
    
    # Parse the SynthIE dataset with 50 examples from training set
    triple_df, entity_df, docs = parser.synthie_parser("train", 50)
    
    # Filter for documents with ID >= 10
    filtered_docs = docs[docs["docid"] >= 10]
    filtered_triple_df = triple_df[triple_df["docid"] >= 10]
    
    # Display information for the specified number of documents
    display_count = min(num_docs_to_display, len(filtered_docs))
    
    print(f"\nDisplaying information for {display_count} documents:\n")
    
    for i, (_, doc) in enumerate(filtered_docs.head(display_count).iterrows()):
        doc_id = doc["docid"]
        text = doc["text"]
        
        # Get all triples for this document
        doc_triples = filtered_triple_df[filtered_triple_df["docid"] == doc_id]
        
        print(f"Text: {text}")
        
        print("<triples>")
        if len(doc_triples) == 0:
            print("  No triples found for this document")
        else:
            for _, triple in doc_triples.iterrows():
                subject_type = get_types(triple['subject_uri'])
                object_type = get_types(triple['object_uri'])
                
                if target_agent == "extractor":
                    print(f"{triple['subject']} (Types: [{','.join(subject_type)}]); {triple['predicate']}; {triple['object']} (Types: [{','.join(object_type)}])")
                elif target_agent == "uri_mapping":
                    print(f"{triple['subject']} (Types: [{','.join(subject_type)}], URI: {triple['subject_uri']}); "
                          f"{triple['predicate']} (URI: {triple['predicate_uri']}); "
                          f"{triple['object']} (Types: [{','.join(object_type)}], URI: {triple['object_uri']})")
        
        print("</triples>\n\n" + "-"*80 + "\n")

if __name__ == "__main__":
    # Default number of documents to display is 3
    # This can be changed by passing a number as a command-line argument
    num_docs = 10
    agent_type = "extractor"  # Default agent type
    
    if len(sys.argv) > 1:
        try:
            num_docs = int(sys.argv[1])
        except ValueError:
            print(f"Invalid argument: {sys.argv[1]}. Using default value of 3.")
    
    # Check if agent type is specified
    if len(sys.argv) > 2:
        agent_type = sys.argv[2]
        if agent_type not in ["extractor", "uri_mapping"]:
            print(f"Invalid agent type: {agent_type}. Using default value 'extractor'.")
            agent_type = "extractor"
    
    few_shot_generator(num_docs, agent_type)