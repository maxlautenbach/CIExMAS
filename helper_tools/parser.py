import jsonlines
import pandas as pd
import tqdm


def rebel_parser(path, number_of_samples):
    data = []

    i = 0

    with jsonlines.open(path) as reader:
        with tqdm.tqdm(total=number_of_samples) as pbar:
            for obj in reader:
                data.append(obj)
                i += 1
                pbar.update(1)
                if i == number_of_samples:
                    break

    relation_df = pd.DataFrame([
        {
            "docid": datapoint["docid"],
            "text": datapoint["text"],
            "subject": triple["subject"]["surfaceform"],
            "subject_uri": triple["subject"]["uri"],
            "predicate": triple["predicate"]["surfaceform"],
            "predicate_uri": triple["predicate"]["uri"],
            "object": triple["object"]["surfaceform"],
            "object_uri": triple["object"]["uri"]
        }
        for datapoint in data
        for triple in datapoint["triples"]
    ])

    entity_df = pd.DataFrame([
        {
            "docid": datapoint["docid"],
            "text": datapoint["text"],
            "entity": entity["surfaceform"],
            "entity_uri": entity["uri"]
        }
        for datapoint in data
        for entity in datapoint["entities"]
    ])

    docs = entity_df[["docid", "text"]].drop_duplicates()

    return relation_df, entity_df, docs