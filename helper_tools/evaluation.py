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
    try:
        # Load the Turtle file into an RDF graph
        result_graph = Graph()
        result_graph.parse(data=turtle_string, format="turtle")

        final_result = []
        for subj, pred, obj in result_graph:
            final_result.append([str(subj), str(pred), str(obj)])

        return pd.DataFrame(final_result,
                            columns=["subject_uri", "predicate_uri", "object_uri"]).drop_duplicates()
    except Exception:
        return pd.DataFrame(columns=["subject_uri", "predicate_uri", "object_uri"])


def evaluate_doc(turtle_string, doc_id, relation_df):
    pred_relation_df = parse_turtle(turtle_string)
    doc_relation_df = relation_df[relation_df["docid"] == doc_id][["subject_uri", "predicate_uri", "object_uri"]]
    correct_relation_df = pred_relation_df.merge(doc_relation_df[["subject_uri", "predicate_uri", "object_uri"]],
                                                 on=["subject_uri", "predicate_uri", "object_uri"], how="inner")

    # Subjects
    extracted_subjects = len(set(pred_relation_df["subject_uri"]))
    gold_standard_subjects = len(set(doc_relation_df["subject_uri"]))
    correct_extracted_subjects = len(
        set(pred_relation_df["subject_uri"]).intersection(set(doc_relation_df["subject_uri"])))

    # Predicates
    extracted_predicates = len(set(pred_relation_df["predicate_uri"]))
    gold_standard_predicates = len(set(doc_relation_df["predicate_uri"]))
    correct_extracted_predicates = len(
        set(pred_relation_df["predicate_uri"]).intersection(set(doc_relation_df["predicate_uri"])))

    # Objects
    extracted_objects = len(set(pred_relation_df["object_uri"]))
    gold_standard_objects = len(set(doc_relation_df["object_uri"]))
    correct_extracted_objects = len(
        set(pred_relation_df["object_uri"]).intersection(set(doc_relation_df["object_uri"])))

    # Entities (Subjects + Objects)
    extracted_entities = len(set(pred_relation_df["subject_uri"]).union(set(pred_relation_df["object_uri"])))
    gold_standard_entities = len(set(doc_relation_df["subject_uri"]).union(set(doc_relation_df["object_uri"])))
    correct_extracted_entities = len(
        set(pred_relation_df["subject_uri"]).union(set(pred_relation_df["object_uri"]))
        .intersection(set(doc_relation_df["subject_uri"]).union(set(doc_relation_df["object_uri"])))
    )

    return len(correct_relation_df), len(doc_relation_df), len(
        pred_relation_df), extracted_subjects, gold_standard_subjects, correct_extracted_subjects, extracted_predicates, gold_standard_predicates, correct_extracted_predicates, extracted_objects, gold_standard_objects, correct_extracted_objects, extracted_entities, gold_standard_entities, correct_extracted_entities


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
        "Relation": ["Correct Relations", "Gold Standard", "Total Predicted"],
        "Subject": ["Correct Extracted Subjects", "Gold Standard Subjects", "Extracted Subjects"],
        "Predicate": ["Correct Extracted Predicates", "Gold Standard Predicates", "Extracted Predicates"],
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


if __name__ == "__main__":
    print(generate_report("/Users/i538914/Documents/Uni/Masterarbeit/CIExMAS/approaches/evaluation_logs/baseline/evaluation_log-Ollama_llama3.3.xlsx"))
