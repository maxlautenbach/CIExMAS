import shutil

import jsonlines
import pandas as pd
import tqdm
from huggingface_hub import snapshot_download
import os
import zipfile
from helper_tools.qdrant_handler import upload_wikidata_entity
from tqdm import tqdm
import gzip

def add_wikidata_prefix(uri):
    if "^^" not in uri:
        return f"http://www.wikidata.org/entity/{uri}"
    return uri


def upload_parsed_data(relation_df, entity_df):
    entity_set = entity_df[['entity', 'entity_uri']].drop_duplicates()
    print("Uploading Entities to Qdrant.")
    for i, row in tqdm(entity_set.iterrows(), total=entity_set.shape[0]):
        upload_wikidata_entity(uri=row["entity_uri"], label=row["entity"])
    print("Uploading Predicates to Qdrant.")
    predicate_set_df = relation_df[["predicate", "predicate_uri"]].drop_duplicates()
    for i, row in tqdm(predicate_set_df.iterrows(), total=predicate_set_df.shape[0]):
        upload_wikidata_entity(uri=row["predicate_uri"], label=row["predicate"])


def babelscape_parser(filename, number_of_samples=10):
    doc_id_key = "docid"
    relation_key = "relations"
    if "rebel" in filename:
        relation_key = "triples"
    elif "sdg_code_davinci_002" in filename:
        relation_key = "triplets"
        doc_id_key = "id"
    elif "redfm" in filename:
        relation_key = "relations"

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
    file_path = snapshot_download("Babelscape/rebel-dataset", repo_type="dataset")

    zip_file_path = f"{file_path}/rebel_dataset.zip"
    target_path = f"{file_path}/rebel_dataset"

    if not os.path.exists(target_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(target_path)

    return babelscape_parser(f"{file_path}/rebel_dataset/en_{split}.jsonl", number_of_samples)


def redfm_parser(split, lang="en", number_of_samples=10):
    file_path = snapshot_download("Babelscape/REDFM", repo_type="dataset")

    return babelscape_parser(f"{file_path}/data/{split}.{lang}.jsonl", number_of_samples)

def synthie_parser(split, number_of_samples=10):
    file_path = snapshot_download("martinjosifoski/SynthIE", repo_type="dataset")
    gz_filename = f'{file_path}/sdg_code_davinci_002/{split}.jsonl.gz'
    filename = f'{file_path}/sdg_code_davinci_002/{split}.jsonl'
    if not os.path.exists(filename):
        with gzip.open(gz_filename, 'rb') as f_in:
            with open(filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    return babelscape_parser(filename, number_of_samples)

if (__name__ == "__main__"):
    relation_df, entity_df, docs = synthie_parser("train", 10)
    print("Test Parsing Finished")
