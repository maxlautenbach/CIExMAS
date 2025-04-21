import os
import re
from pathlib import Path

import pandas as pd
from rdflib import Graph, URIRef
from tqdm import tqdm

from helper_tools import parser
from helper_tools.base_setup import wikidata_predicate_graph
from helper_tools.wikidata_loader import get_label


def get_uri_labels(df):
    subjects = []
    predicates = []
    objects = []
    for i, row in df.iterrows():
        subjects.append(get_label(row["subject_uri"]))
        predicates.append(get_label(row["predicate_uri"]))
        if row["object_uri"] is not None and "^^" in row["object_uri"]:
            objects.append(row["object_uri"])
        else:
            objects.append(get_label(row["object_uri"]))
    return pd.concat(
        [df.reset_index(drop=True), pd.DataFrame({"subject": subjects, "predicate": predicates, "object": objects})],
        axis=1)


def parse_turtle(turtle_string):
    try:
        # Load the Turtle file into an RDF graph
        result_graph = Graph()
        result_graph.parse(data=turtle_string, format="turtle")

        final_result = []
        for subj, pred, obj in result_graph:
            final_result.append([str(subj), str(pred), str(obj)])

        return pd.DataFrame(final_result,
                            columns=["subject_uri", "predicate_uri", "object_uri"]).drop_duplicates(), "Success"
    except Exception as e:
        return pd.DataFrame(columns=["subject_uri", "predicate_uri", "object_uri"]), f"Error: {str(e)}"


def check_inter_predicate_relations(predicate_a, predicate_b):
    inter_predicate_relations = []
    if wikidata_predicate_graph.query(
            f'ASK {{<{predicate_a}> <http://www.w3.org/2000/01/rdf-schema#subPropertyOf>+ <{predicate_b}>.}}').askAnswer:
        inter_predicate_relations.append("subPropertyOf")
    elif wikidata_predicate_graph.query(
            f'ASK {{<{predicate_b}> <http://www.w3.org/2000/01/rdf-schema#subPropertyOf>+ <{predicate_a}>.}}').askAnswer:
        inter_predicate_relations.append("parentPropertyOf")
    return inter_predicate_relations


def evaluate_doc(turtle_string, doc_id, triple_df):
    pred_triple_df, error = parse_turtle(turtle_string)
    doc_triple_df = triple_df[triple_df["docid"] == doc_id][["subject_uri", "predicate_uri", "object_uri"]]
    correct_triple_df = pred_triple_df.merge(doc_triple_df[["subject_uri", "predicate_uri", "object_uri"]],
                                                 on=["subject_uri", "predicate_uri", "object_uri"], how="inner")

    # Subjects
    extracted_subjects = len(set(pred_triple_df["subject_uri"]))
    gold_standard_subjects = len(set(doc_triple_df["subject_uri"]))
    correct_extracted_subjects = len(
        set(pred_triple_df["subject_uri"]).intersection(set(doc_triple_df["subject_uri"])))

    # Predicates
    extracted_predicates = len(set(pred_triple_df["predicate_uri"]))
    gold_standard_predicates = len(set(doc_triple_df["predicate_uri"]))
    correct_extracted_predicates = len(
        set(pred_triple_df["predicate_uri"]).intersection(set(doc_triple_df["predicate_uri"])))

    # Objects
    extracted_objects = len(set(pred_triple_df["object_uri"]))
    gold_standard_objects = len(set(doc_triple_df["object_uri"]))
    correct_extracted_objects = len(
        set(pred_triple_df["object_uri"]).intersection(set(doc_triple_df["object_uri"])))

    # Entities (Subjects + Objects)
    extracted_entities = len(set(pred_triple_df["subject_uri"]).union(set(pred_triple_df["object_uri"])))
    gold_standard_entities = len(set(doc_triple_df["subject_uri"]).union(set(doc_triple_df["object_uri"])))
    correct_extracted_entities = len(
        set(pred_triple_df["subject_uri"]).union(set(pred_triple_df["object_uri"]))
        .intersection(set(doc_triple_df["subject_uri"]).union(set(doc_triple_df["object_uri"])))
    )

    # Triples with Parent and Related Predicates
    correct_triples_df = pred_triple_df.merge(doc_triple_df[["subject_uri", "predicate_uri", "object_uri"]],
                                                on=["subject_uri", "predicate_uri", "object_uri"], how="inner")
    incorrect_triples_df = \
    pred_triple_df.merge(correct_triples_df, how="outer", indicator=True).query('_merge=="left_only"')[
        ["subject_uri", "predicate_uri", "object_uri"]]
    partial_matching_triples_df = incorrect_triples_df.merge(
        doc_triple_df[["subject_uri", "predicate_uri", "object_uri"]], on=["subject_uri", "object_uri"], how="inner")
    correct_triples_with_parent_predicates_df = []
    correct_triples_with_related_predicates_df = []
    for i, row in partial_matching_triples_df.iterrows():
        inter_predicate_relations = check_inter_predicate_relations(row["predicate_uri_x"], row["predicate_uri_y"])
        if "parentPropertyOf" in inter_predicate_relations:
            correct_triples_with_parent_predicates_df.append(row)
        if len(inter_predicate_relations) > 0:
            correct_triples_with_related_predicates_df.append(row)

    correct_triples_with_parent_predicates_df = pd.DataFrame(correct_triples_with_parent_predicates_df).drop(
        "predicate_uri_x", axis=1, errors='ignore').rename(columns={"predicate_uri_y": "predicate_uri"},
                                                           errors='ignore')
    correct_triples_with_parent_predicates_df = pd.concat(
        [correct_triples_with_parent_predicates_df, correct_triples_df]).drop_duplicates()
    correct_triples_with_related_predicates_df = pd.DataFrame(correct_triples_with_related_predicates_df).drop(
        "predicate_uri_x", axis=1, errors='ignore').rename(columns={"predicate_uri_y": "predicate_uri"},
                                                           errors='ignore')
    correct_triples_with_related_predicates_df = pd.concat(
        [correct_triples_with_related_predicates_df, correct_triples_df]).drop_duplicates()

    # Predicates including Parent and Related Predicates
    pred_predicate_set = set(pred_triple_df["predicate_uri"])
    doc_predicate_set = set(doc_triple_df["predicate_uri"])
    correct_predicates_parent = set()
    correct_predicates_related = set()
    for doc_predicate in doc_predicate_set:
        for pred_predicate in pred_predicate_set:
            if pred_predicate == doc_predicate:
                correct_predicates_parent.add(doc_predicate)
                correct_predicates_related.add(doc_predicate)
                break
            inter_predicate_relations = check_inter_predicate_relations(pred_predicate, doc_predicate)
            if "parentPropertyOf" in inter_predicate_relations:
                correct_predicates_parent.add(doc_predicate)
            if len(inter_predicate_relations) > 0:
                correct_predicates_related.add(doc_predicate)
                break
    correct_predicates_with_parent = len(correct_predicates_parent)
    correct_predicates_with_related = len(correct_predicates_related)

    return len(correct_triple_df), len(correct_triples_with_parent_predicates_df), len(
        correct_triples_with_related_predicates_df), len(doc_triple_df), len(
        pred_triple_df), extracted_subjects, gold_standard_subjects, correct_extracted_subjects, extracted_predicates, gold_standard_predicates, correct_extracted_predicates, correct_predicates_with_parent, correct_predicates_with_related, extracted_objects, gold_standard_objects, correct_extracted_objects, extracted_entities, gold_standard_entities, correct_extracted_entities


def generate_pr_f1_score(correct, gold_standard, total_predicted):
    try:
        precision = correct / total_predicted
    except ZeroDivisionError:
        precision = 0
    try:
        recall = correct / gold_standard
    except ZeroDivisionError:
        recall = 0
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1_score


def generate_report(excel_file_path):
    evaluation_log_df = pd.read_excel(excel_file_path)

    # Mapping der Spalten zu den jeweiligen Metriken
    metric_map = {
        "Triple": ["Correct Triples", "Gold Standard Triples", "Total Triples Predicted"],
        "Triple with Parents": ["Correct Triples with Parents", "Gold Standard Triples",
                                  "Total Triples Predicted"],
        "Triple with Related": ["Correct Triples with Related", "Gold Standard Triples",
                                  "Total Triples Predicted"],
        "Subject": ["Correct Extracted Subjects", "Gold Standard Subjects", "Extracted Subjects"],
        "Predicate": ["Correct Extracted Predicates", "Gold Standard Predicates", "Extracted Predicates"],
        "Predicate with Parents": ["Correct Extracted Predicates with Parents", "Gold Standard Predicates",
                                   "Extracted Predicates"],
        "Predicate with Related": ["Correct Extracted Predicates with Related", "Gold Standard Predicates",
                                   "Extracted Predicates"],
        "Object": ["Correct Extracted Objects", "Gold Standard Objects", "Extracted Objects"],
        "Entity": ["Correct Extracted Entities", "Gold Standard Entities", "Extracted Entities"]
    }

    # Dictionaries für das Aufsummieren der Scores
    metric_scores = {metric: {"precision": [], "recall": [], "f1": []} for metric in metric_map}

    # Über alle Dokumente iterieren und Einzelwerte sammeln
    for _, row in evaluation_log_df.iterrows():
        for metric, (correct_col, gold_col, pred_col) in metric_map.items():
            precision, recall, f1 = generate_pr_f1_score(
                row[correct_col], row[gold_col], row[pred_col]
            )
            metric_scores[metric]["precision"].append(precision)
            metric_scores[metric]["recall"].append(recall)
            metric_scores[metric]["f1"].append(f1)

    # Macro Average berechnen
    macro_scores = {
        metric: {
            "Precision": sum(scores["precision"]) / len(scores["precision"]) if scores["precision"] else 0.0,
            "Recall": sum(scores["recall"]) / len(scores["recall"]) if scores["recall"] else 0.0,
            "F1-Score": sum(scores["f1"]) / len(scores["f1"]) if scores["f1"] else 0.0
        }
        for metric, scores in metric_scores.items()
    }

    # DataFrame erstellen
    macro_scores_df = pd.DataFrame.from_dict(macro_scores, orient="index")

    # Ausgabe (optional)
    return macro_scores_df


def calculate_scores_from_array(values_array):
    if len(values_array) != 19:
        raise ValueError(f"Expected 19 values, but got {len(values_array)}")

    (
        correct_triples,
        correct_triples_with_parents,
        correct_triples_with_related,
        gold_triples,
        pred_triples,

        extracted_subjects,
        gold_subjects,
        correct_subjects,

        extracted_predicates,
        gold_predicates,
        correct_predicates,
        correct_predicates_with_parent,
        correct_predicates_with_related,

        extracted_objects,
        gold_objects,
        correct_objects,

        extracted_entities,
        gold_entities,
        correct_entities
    ) = values_array

    result = {}

    # Triple
    precision, recall, f1 = generate_pr_f1_score(correct_triples, gold_triples, pred_triples)
    result["Triple"] = {"Precision": precision, "Recall": recall, "F1-Score": f1}

    # Triple with Parents
    precision, recall, f1 = generate_pr_f1_score(correct_triples_with_parents, gold_triples, pred_triples)
    result["Triple with Parents"] = {"Precision": precision, "Recall": recall, "F1-Score": f1}

    # Triple with Related
    precision, recall, f1 = generate_pr_f1_score(correct_triples_with_related, gold_triples, pred_triples)
    result["Triple with Related"] = {"Precision": precision, "Recall": recall, "F1-Score": f1}

    # Subject
    precision, recall, f1 = generate_pr_f1_score(correct_subjects, gold_subjects, extracted_subjects)
    result["Subject"] = {"Precision": precision, "Recall": recall, "F1-Score": f1}

    # Predicate
    precision, recall, f1 = generate_pr_f1_score(correct_predicates, gold_predicates, extracted_predicates)
    result["Predicate"] = {"Precision": precision, "Recall": recall, "F1-Score": f1}

    # Predicate with Parents
    precision, recall, f1 = generate_pr_f1_score(correct_predicates_with_parent, gold_predicates, extracted_predicates)
    result["Predicate with Parents"] = {"Precision": precision, "Recall": recall, "F1-Score": f1}

    # Predicate with Related
    precision, recall, f1 = generate_pr_f1_score(correct_predicates_with_related, gold_predicates, extracted_predicates)
    result["Predicate with Related"] = {"Precision": precision, "Recall": recall, "F1-Score": f1}

    # Object
    precision, recall, f1 = generate_pr_f1_score(correct_objects, gold_objects, extracted_objects)
    result["Object"] = {"Precision": precision, "Recall": recall, "F1-Score": f1}

    # Entity
    precision, recall, f1 = generate_pr_f1_score(correct_entities, gold_entities, extracted_entities)
    result["Entity"] = {"Precision": precision, "Recall": recall, "F1-Score": f1}

    return pd.DataFrame.from_dict(result, orient="index")


def convert_eval_log(path, dataset_cache):
    match = re.match(r"(?P<split>\w+)-(?P<num_samples>\d+)-evaluation_log-.*\.xlsx", os.path.basename(path))

    if match:
        split = match.group("split")
        number_of_samples = int(match.group("num_samples"))
        try:
            triple_df, entity_df, docs = dataset_cache[f"{split}-{number_of_samples}"]
        except KeyError:
            triple_df, entity_df, docs = parser.synthie_parser(split, number_of_samples)
            dataset_cache[f"{split}-{number_of_samples}"] = (triple_df, entity_df, docs)
    else:
        print("File name does not match with the required format.")
        return

    evaluation_log_df = pd.read_excel(path)
    evaluation_log = []
    for doc_id, row in evaluation_log_df.iterrows():
        result_string = str(row["Result String"])
        turtle_string_match = re.search(r'<ttl>(.*?)</ttl>', result_string, re.DOTALL)
        if turtle_string_match:
            turtle_string = turtle_string_match.group(1)
        else:
            turtle_string = result_string
        evaluation_log.append([doc_id, *evaluate_doc(turtle_string, doc_id, triple_df), result_string])

    evaluation_log_df = pd.DataFrame(
        evaluation_log,
        columns=[
            "Doc ID",
            "Correct Triples", "Correct Triples with Parents", "Correct Triples with Related", "Gold Standard Triples",
            "Total Triples Predicted",
            "Extracted Subjects", "Gold Standard Subjects", "Correct Extracted Subjects",
            "Extracted Predicates", "Gold Standard Predicates", "Correct Extracted Predicates",
            "Correct Extracted Predicates with Parents", "Correct Extracted Predicates with Related",
            "Extracted Objects", "Gold Standard Objects", "Correct Extracted Objects",
            "Extracted Entities", "Gold Standard Entities", "Correct Extracted Entities", "Result String"
        ]
    )
    evaluation_log_df.to_excel(path[:-5] + "-converted.xlsx", index=False)
    return dataset_cache




if __name__ == "__main__":
    dataset_cache = {}
    for root, dirs, files in os.walk("../approaches/evaluation_logs"):
        for file in files:
            if file.endswith(".xlsx") and not file.endswith("-converted.xlsx"):
                file_path = os.path.join(root, file)
                print(f"Converting {file_path}")
                dataset_cache = convert_eval_log(file_path, dataset_cache)
