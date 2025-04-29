#!/usr/bin/env python3

import os
import tqdm
import git
from rdflib import Graph, URIRef
from dotenv import load_dotenv

# Import the necessary modules
from helper_tools.qdrant_handler import upload_wikidata_element, upload_wikidata_elements
from helper_tools.redis_handler import get_element_info, element_info_upload

# Find the repository root
repo = git.Repo(search_parent_directories=True)
repo_root = repo.working_dir

# Load environment variables
load_dotenv(repo_root + "/.env", override=True)

def extract_predicate_uri(uri_ref):
    """Extract the Wikidata ID (P-number) from a URI reference."""
    uri = str(uri_ref)
    if "/entity/P" in uri:
        # Extract P-number from URI
        p_id = uri.split("/entity/")[1]
        return uri, p_id
    return None, None

def get_label_from_graph(g, uri):
    """Extract the label for a property directly from the RDF graph."""
    label_predicate = URIRef("http://www.w3.org/2000/01/rdf-schema#label")
    
    # Look for English label (with @en language tag)
    for s, p, o in g.triples((URIRef(uri), label_predicate, None)):
        # Check if this is an English label
        if o.language == "en":
            return str(o)
    
    # If no English label found, return the URI as fallback
    return uri.split("/")[-1]

def get_description_from_graph(g, uri):
    """Extract the description for a property directly from the RDF graph."""
    comment_predicate = URIRef("http://www.w3.org/2000/01/rdf-schema#comment")
    
    # Look for English description (with @en language tag)
    for s, p, o in g.triples((URIRef(uri), comment_predicate, None)):
        # Check if this is an English comment/description
        if o.language == "en":
            return str(o)
    
    # Return empty string if no description found
    return ""

def main():
    print("Loading all_properties.ttl file...")
    
    # Create a new Graph
    g = Graph()
    
    # Parse the TTL file
    ttl_path = os.path.join(repo_root, "infrastructure/all_properties.ttl")
    g.parse(ttl_path, format="ttl")
    
    # Find all predicates (properties) in the graph
    predicates = set()
    
    # Look for subjects with type ObjectProperty or DatatypeProperty
    for s, p, o in g.triples((None, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), URIRef("http://www.w3.org/2002/07/owl#ObjectProperty"))):
        predicates.add(str(s))
            
    for s, p, o in g.triples((None, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), URIRef("http://www.w3.org/2002/07/owl#DatatypeProperty"))):
        predicates.add(str(s))
    
    print(f"Found {len(predicates)} predicates in the TTL file")
    
    # For each predicate, get its label and description directly from the graph
    print("Uploading predicates to Qdrant vector store...")
    already_uploaded = 0
    newly_uploaded = 0

    # Prepare data for bulk upload
    elements_dict = {}

    predicates = sorted(list(predicates))

    for uri in tqdm.tqdm(predicates):
        # Check if the predicate is already uploaded using redis_handler
        existing_info = get_element_info(uri)

        if existing_info:
            # Predicate is already uploaded
            already_uploaded += 1
            continue

        # Get the label and description for this predicate
        label = get_label_from_graph(g, uri)
        description = get_description_from_graph(g, uri)

        # Add to elements dictionary for bulk upload
        elements_dict[uri] = {
            'label': label,
            'description': description
        }

        newly_uploaded += 1

    # Perform the bulk upload if there are any new elements
    if elements_dict:
        print(f"Uploading {len(elements_dict)} predicates in bulk...")
        upload_wikidata_elements(elements_dict)
    
    print(f"Upload complete! {newly_uploaded} predicates newly uploaded, {already_uploaded} predicates were already in the database.")

if __name__ == "__main__":
    main()