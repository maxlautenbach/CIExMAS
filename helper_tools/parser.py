import shutil

import jsonlines
import pandas as pd
import tqdm
from huggingface_hub import snapshot_download
import os
import zipfile
from tqdm import tqdm
import gzip
from dotenv import load_dotenv
import git
from helper_tools.sort_jsonl import sort_jsonl_file

from helper_tools.wikidata_loader import get_property_example

repo = git.Repo(search_parent_directories=True)

load_dotenv(repo.working_dir + "/.env")


def is_jsonl_sorted(filename, doc_id_key="docid", number_of_samples=None):
    """
    Check if a JSONL file is sorted by docid.
    
    Args:
        filename (str): Path to the JSONL file
        doc_id_key (str): Key to use for checking sort order
        number_of_samples (int, optional): Number of samples to check. If None, checks all samples.
        
    Returns:
        bool: True if file is sorted, False otherwise
    """
    last_id = None
    with jsonlines.open(filename) as reader:
        for i, entry in enumerate(reader):
            if number_of_samples is not None and i >= number_of_samples:
                break
            current_id = entry.get(doc_id_key)
            if last_id is not None and current_id < last_id:
                return False
            last_id = current_id
    return True


def add_wikidata_prefix(uri):
    if "^^" not in uri:
        return f"http://www.wikidata.org/entity/{uri}"
    return uri


def upload_parsed_data(relation_df, entity_df):
    """
    Upload parsed entities and predicates to the vector store using bulk upload.
    
    Args:
        relation_df: DataFrame containing relations with predicates
        entity_df: DataFrame containing entities
    """
    if os.getenv("VECTOR_STORE") == "qdrant":
        from helper_tools.qdrant_handler import upload_wikidata_elements
        from helper_tools.redis_handler import get_element_info
        from helper_tools.wikidata_loader import get_description
    else:
        from helper_tools.faiss_handler import upload_wikidata_entity
        return
    
    # Process entities in bulk
    entity_set = entity_df[['entity', 'entity_uri']].drop_duplicates()
    print(f"Preparing entities for bulk upload to {os.getenv('VECTOR_STORE')}...")
    
    # Create a dictionary for bulk upload
    entity_elements = {}
    already_uploaded_entities = 0
    
    # Build dictionary for entities
    for i, row in tqdm(entity_set.iterrows(), total=entity_set.shape[0], desc="Preparing entities"):
        uri = row["entity_uri"]
        label = row["entity"]
        
        # Check if entity is already in Redis
        if get_element_info(uri):
            already_uploaded_entities += 1
            continue
            
        # Get description for the entity
        description = get_description(uri)
        
        # Add to bulk upload dictionary
        entity_elements[uri] = {
            'label': label,
            'description': description
        }
    
    # Upload entities in bulk
    if entity_elements:
        print(f"Uploading {len(entity_elements)} entities to {os.getenv('VECTOR_STORE')} in batches...")
        upload_wikidata_elements(entity_elements, "entity")
        print(f"Entity upload complete. {len(entity_elements)} new entities uploaded, {already_uploaded_entities} entities were already in the database.")
    else:
        print(f"No new entities to upload. {already_uploaded_entities} entities were already in the database.")
    
    # Process predicates in bulk
    predicate_set_df = relation_df[["predicate", "predicate_uri"]].drop_duplicates()
    print(f"Preparing predicates for bulk upload to {os.getenv('VECTOR_STORE')}...")
    
    # Create a dictionary for bulk upload
    predicate_elements = {}
    already_uploaded_predicates = 0
    
    # Build dictionary for predicates
    for i, row in tqdm(predicate_set_df.iterrows(), total=predicate_set_df.shape[0], desc="Preparing predicates"):
        uri = row["predicate_uri"]
        label = row["predicate"]
        
        # Check if predicate is already in Redis
        if get_element_info(uri):
            already_uploaded_predicates += 1
            continue
            
        # Get description for the predicate
        description = get_description(uri)
        property_example = get_property_example(uri)
        example = f"{property_example['subject_label']} {label} {property_example['object_label']}" if property_example else ""
        
        # Add to bulk upload dictionary
        predicate_elements[uri] = {
            'label': label,
            'description': description,
            'example': example
        }
    
    # Upload predicates in bulk
    if predicate_elements:
        print(f"Uploading {len(predicate_elements)} predicates to {os.getenv('VECTOR_STORE')} in batches...")
        upload_wikidata_elements(predicate_elements, "predicates")
        print(f"Predicate upload complete. {len(predicate_elements)} new predicates uploaded, {already_uploaded_predicates} predicates were already in the database.")
    else:
        print(f"No new predicates to upload. {already_uploaded_predicates} predicates were already in the database.")


def babelscape_parser(filename, number_of_samples=10):
    doc_id_key = "docid"
    relation_key = "relations"
    if "rebel" in filename:
        relation_key = "triples"
    elif "sdg_code_davinci_002" in filename or "sdg_text_davinci_003" in filename:
        relation_key = "triplets"
        doc_id_key = "id"
    elif "redfm" in filename:
        relation_key = "relations"

    # Check if file is sorted
    if not is_jsonl_sorted(filename, doc_id_key, number_of_samples):
        print(f"File {filename} is not sorted by {doc_id_key} in the first {number_of_samples} samples. Sorting now...")
        sort_jsonl_file(filename)
        print("Sorting complete. Re-parsing file...")

    data = []

    i = 0

    with jsonlines.open(filename) as reader:
        with tqdm(total=number_of_samples) as pbar:
            for obj in reader:
                data.append(obj)
                i += 1
                pbar.update(1)
                if i == number_of_samples:
                    break

    docs = pd.DataFrame([{
        "docid": datapoint[doc_id_key],
        "text": datapoint["text"]
    }
        for datapoint in data
    ]).drop_duplicates()

    relation_df = pd.DataFrame([
        {
            "docid": datapoint[doc_id_key],
            "subject": triple["subject"]["surfaceform"],
            "subject_uri": add_wikidata_prefix(triple["subject"]["uri"]),
            "predicate": triple["predicate"]["surfaceform"],
            "predicate_uri": add_wikidata_prefix(triple["predicate"]["uri"]),
            "object": triple["object"]["surfaceform"],
            "object_uri": add_wikidata_prefix(triple["object"]["uri"])
        }
        for datapoint in data
        for triple in datapoint[relation_key]
    ])

    entity_df = pd.DataFrame([
        {
            "docid": datapoint[doc_id_key],
            "entity": entity["surfaceform"],
            "entity_uri": add_wikidata_prefix(entity["uri"])
        }
        for datapoint in data
        for entity in datapoint["entities"]
    ])

    upload_parsed_data(relation_df=relation_df, entity_df=entity_df)

    return relation_df, entity_df, docs


def rebel_parser(split, number_of_samples=10):
    file_path = snapshot_download("Babelscape/rebel-dataset", repo_type="dataset", local_dir=os.getenv("DATASET_DIR") + "/REBEL")

    zip_file_path = f"{file_path}/rebel_dataset.zip"
    target_path = f"{file_path}/rebel_dataset"

    if not os.path.exists(target_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(target_path)

    return babelscape_parser(f"{file_path}/rebel_dataset/en_{split}.jsonl", number_of_samples)


def redfm_parser(split, lang="en", number_of_samples=10):
    file_path = snapshot_download("Babelscape/REDFM", repo_type="dataset", local_dir=os.getenv("DATASET_DIR") + "/REDFM")

    return babelscape_parser(f"{file_path}/data/{split}.{lang}.jsonl", number_of_samples)


def synthie_parser(split, number_of_samples=10):
    file_path = snapshot_download("martinjosifoski/SynthIE", repo_type="dataset", local_dir=os.getenv("DATASET_DIR") + "/synthIE")
    
    # Define the file paths based on split type
    if split == "test_text":
        gz_filename = f'{file_path}/sdg_text_davinci_003/test.jsonl.gz'
        filename = f'{file_path}/sdg_text_davinci_003/test.jsonl'
    elif split == "train_text":
        gz_filename = f'{file_path}/sdg_text_davinci_003/val.jsonl.gz'
        filename = f'{file_path}/sdg_text_davinci_003/val.jsonl'
    else:
        gz_filename = f'{file_path}/sdg_code_davinci_002/{split}.jsonl.gz'
        filename = f'{file_path}/sdg_code_davinci_002/{split}.jsonl'
    
    if not os.path.exists(filename):
        with gzip.open(gz_filename, 'rb') as f_in:
            with open(filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    return babelscape_parser(filename, number_of_samples)


if (__name__ == "__main__"):
    relation_df, entity_df, docs = synthie_parser("test_text", 50)
    print("Test Parsing Finished")
