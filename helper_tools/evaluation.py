import pandas as pd
from rdflib import Graph, URIRef


def get_uri_labels(df, entity_set, predicate_set_df):
    subjects = []
    predicates = []
    objects = []
    for i, row in df.iterrows():
        try:
            subjects.append(entity_set[entity_set["entity_uri"] == row["subject_uri"]]["entity"].values[0])
        except IndexError:
            subjects.append("Unknown")
        try:
            predicates.append(
                predicate_set_df[predicate_set_df["predicate_uri"] == row["predicate_uri"]]["predicate"].values[0])
        except IndexError:
            predicates.append("Unknown")
        if row["object_uri"] is not None and "^^" in row["object_uri"]:
            objects.append(row["object_uri"])
        else:
            try:
                objects.append(entity_set[entity_set["entity_uri"] == row["object_uri"]]["entity"].values[0])
            except IndexError:
                objects.append("Unknown")
    return pd.concat(
        [df.reset_index(drop=True), pd.DataFrame({"subject": subjects, "predicate": predicates, "object": objects})],
        axis=1)


def parse_turtle(turtle_string):
    # Load the Turtle file into an RDF graph
    result_graph = Graph()
    result_graph.parse(data=turtle_string, format="turtle")

    final_result = []
    for subj, pred, obj in result_graph:
        final_result.append([str(subj), str(pred), str(obj)])

    return pd.DataFrame(final_result,
                                    columns=["subject_uri", "predicate_uri", "object_uri"]).drop_duplicates()


def evaluate(turtle_string, doc_id, relation_df, verbose=False):
    pred_relation_df = parse_turtle(turtle_string)
    doc_relation_df = relation_df[relation_df["docid"] == doc_id][["subject_uri", "predicate_uri", "object_uri"]]
    correct_relation_df = pred_relation_df.merge(doc_relation_df[["subject_uri", "predicate_uri", "object_uri"]],
                                                 on=["subject_uri", "predicate_uri", "object_uri"], how="inner")
    precision = len(correct_relation_df) / len(pred_relation_df)
    recall = len(correct_relation_df) / len(doc_relation_df)
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    if verbose:
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1_score}")

    return precision, recall, f1_score