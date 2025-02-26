import jsonlines
import pandas as pd
import tqdm
from huggingface_hub import snapshot_download
import os
import zipfile

def add_wikidata_prefix(uri):
    if "^^" not in uri:
        return f"http://www.wikidata.org/entity/{uri}"
    return uri

def babelscape_parser(filename, number_of_samples=10):

    if "rebel" in filename:
        relation_key = "triples"
    elif "redfm" in filename:
        relation_key = "relations"
    else:
        relation_key = "relations"

    data = []

    i = 0

    with jsonlines.open(filename) as reader:
        with tqdm.tqdm(total=number_of_samples) as pbar:
            for obj in reader:
                data.append(obj)
                i += 1
                pbar.update(1)
                if i == number_of_samples:
                    break

    docs = pd.DataFrame([{
        "docid": datapoint["docid"],
        "text": datapoint["text"]
    }
        for datapoint in data
    ]).drop_duplicates()

    relation_df = pd.DataFrame([
        {
            "docid": datapoint["docid"],
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
            "docid": datapoint["docid"],
            "entity": entity["surfaceform"],
            "entity_uri": add_wikidata_prefix(entity["uri"])
        }
        for datapoint in data
        for entity in datapoint["entities"]
    ])

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


if (__name__ == "__main__"):
    relation_df, entity_df, docs = rebel_parser("train", 10)
    print("Test")
