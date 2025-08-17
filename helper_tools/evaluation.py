import os
import re
import pickle
from pathlib import Path

import pandas as pd
from rdflib import Graph, URIRef
from tqdm import tqdm

from helper_tools import parser
from helper_tools.base_setup import wikidata_predicate_graph
from helper_tools.wikidata_loader import get_label


def get_uri_labels(df):
    """
    Convert URI references in a DataFrame to human-readable labels.
    
    Args:
        df (pd.DataFrame): DataFrame containing subject_uri, predicate_uri, and object_uri columns
        
    Returns:
        pd.DataFrame: Original DataFrame with additional subject, predicate, and object columns containing labels
    """
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
    """
    Parse a Turtle string into a DataFrame of triples.
    
    Args:
        turtle_string (str): Turtle format string to parse
        
    Returns:
        tuple: (DataFrame with triples, "Success" or error message)
    """
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
    """
    Check for hierarchical relationships between two predicates in the Wikidata predicate graph.
    
    Args:
        predicate_a (str): First predicate URI
        predicate_b (str): Second predicate URI
        
    Returns:
        list: List of relationship types found (e.g., ["subPropertyOf", "parentPropertyOf"])
    """
    inter_predicate_relations = []
    if wikidata_predicate_graph.query(
            f'ASK {{<{predicate_a}> <http://www.w3.org/2000/01/rdf-schema#subPropertyOf>+ <{predicate_b}>.}}').askAnswer:
        inter_predicate_relations.append("subPropertyOf")
    elif wikidata_predicate_graph.query(
            f'ASK {{<{predicate_b}> <http://www.w3.org/2000/01/rdf-schema#subPropertyOf>+ <{predicate_a}>.}}').askAnswer:
        inter_predicate_relations.append("parentPropertyOf")
    return inter_predicate_relations


def _calculate_metrics(pred_triple_df, gold_triple_df):
    """
    Helper function to calculate all evaluation metrics from two triple DataFrames.
    
    Args:
        pred_triple_df (pd.DataFrame): DataFrame containing predicted triples
        gold_triple_df (pd.DataFrame): DataFrame containing gold standard triples
        
    Returns:
        tuple: A tuple containing all evaluation metrics
    """
    correct_triple_df = pred_triple_df.merge(gold_triple_df[["subject_uri", "predicate_uri", "object_uri"]],
                                           on=["subject_uri", "predicate_uri", "object_uri"], how="inner")

    # Calculate subject metrics
    extracted_subjects = len(set(pred_triple_df["subject_uri"]))
    gold_standard_subjects = len(set(gold_triple_df["subject_uri"]))
    correct_extracted_subjects = len(
        set(pred_triple_df["subject_uri"]).intersection(set(gold_triple_df["subject_uri"])))

    # Calculate predicate metrics
    extracted_predicates = len(set(pred_triple_df["predicate_uri"]))
    gold_standard_predicates = len(set(gold_triple_df["predicate_uri"]))
    correct_extracted_predicates = len(
        set(pred_triple_df["predicate_uri"]).intersection(set(gold_triple_df["predicate_uri"])))

    # Calculate object metrics
    extracted_objects = len(set(pred_triple_df["object_uri"]))
    gold_standard_objects = len(set(gold_triple_df["object_uri"]))
    correct_extracted_objects = len(
        set(pred_triple_df["object_uri"]).intersection(set(gold_triple_df["object_uri"])))

    # Calculate entity metrics (subjects + objects)
    extracted_entities = len(set(pred_triple_df["subject_uri"]).union(set(pred_triple_df["object_uri"])))
    gold_standard_entities = len(set(gold_triple_df["subject_uri"]).union(set(gold_triple_df["object_uri"])))
    correct_extracted_entities = len(
        set(pred_triple_df["subject_uri"]).union(set(pred_triple_df["object_uri"]))
        .intersection(set(gold_triple_df["subject_uri"]).union(set(gold_triple_df["object_uri"])))
    )

    # Calculate triples with parent and related predicates
    correct_triples_df = pred_triple_df.merge(gold_triple_df[["subject_uri", "predicate_uri", "object_uri"]],
                                            on=["subject_uri", "predicate_uri", "object_uri"], how="inner")
    incorrect_triples_df = \
    pred_triple_df.merge(correct_triples_df, how="outer", indicator=True).query('_merge=="left_only"')[
        ["subject_uri", "predicate_uri", "object_uri"]]
    partial_matching_triples_df = incorrect_triples_df.merge(
        gold_triple_df[["subject_uri", "predicate_uri", "object_uri"]], on=["subject_uri", "object_uri"], how="inner")
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

    # Calculate predicates including parent and related predicates
    pred_predicate_set = set(pred_triple_df["predicate_uri"])
    gold_predicate_set = set(gold_triple_df["predicate_uri"])
    detected_predicates_doc_parent = set()  # For doc predicates detected through parent relationships
    detected_predicates_doc_related = set()  # For doc predicates detected through related relationships
    correct_pred_predicates_parent = set()  # New variable for correct predicates from pred
    correct_pred_predicates_related = set()  # New variable for correct predicates from pred
    
    for gold_predicate in gold_predicate_set:
        for pred_predicate in pred_predicate_set:
            if pred_predicate == gold_predicate:
                # Exact match
                correct_pred_predicates_parent.add(pred_predicate)
                correct_pred_predicates_related.add(pred_predicate)
                detected_predicates_doc_parent.add(gold_predicate)
                detected_predicates_doc_related.add(gold_predicate)
            else:
                inter_predicate_relations = check_inter_predicate_relations(pred_predicate, gold_predicate)
                if "parentPropertyOf" in inter_predicate_relations:
                    correct_pred_predicates_parent.add(pred_predicate)
                    detected_predicates_doc_parent.add(gold_predicate)
                    correct_pred_predicates_related.add(pred_predicate)
                    detected_predicates_doc_related.add(gold_predicate)
                elif len(inter_predicate_relations) > 0:
                    correct_pred_predicates_related.add(pred_predicate)
                    detected_predicates_doc_related.add(gold_predicate)
    
    detected_predicates_doc_parent_count = len(detected_predicates_doc_parent)
    detected_predicates_doc_related_count = len(detected_predicates_doc_related)
    correct_pred_predicates_parent_count = len(correct_pred_predicates_parent)
    correct_pred_predicates_related_count = len(correct_pred_predicates_related)

    if detected_predicates_doc_parent_count < correct_extracted_predicates:
        print("PROBLEM")

    return len(correct_triple_df), len(correct_triples_with_parent_predicates_df), len(
        correct_triples_with_related_predicates_df), len(gold_triple_df), len(
        pred_triple_df), extracted_subjects, gold_standard_subjects, correct_extracted_subjects, extracted_predicates, gold_standard_predicates, correct_extracted_predicates, detected_predicates_doc_parent_count, detected_predicates_doc_related_count, correct_pred_predicates_parent_count, correct_pred_predicates_related_count, extracted_objects, gold_standard_objects, correct_extracted_objects, extracted_entities, gold_standard_entities, correct_extracted_entities


def evaluate_doc(turtle_string, doc_id, triple_df):
    """
    Evaluate a single document by comparing predicted triples with gold standard triples.
    
    Args:
        turtle_string (str): Turtle string containing predicted triples
        doc_id: Document identifier
        triple_df (pd.DataFrame): DataFrame containing gold standard triples
        
    Returns:
        tuple: Evaluation metrics for the document
    """
    pred_triple_df, error = parse_turtle(turtle_string)
    if error != "Success":
        raise ValueError(f"Error parsing turtle string: {error}")
    doc_triple_df = triple_df[triple_df["docid"] == doc_id][["subject_uri", "predicate_uri", "object_uri"]]
    return _calculate_metrics(pred_triple_df, doc_triple_df)


def generate_pr_f1_score(correct, gold_standard, total_predicted):
    """
    Calculate precision, recall, and F1 score from raw counts.
    
    Args:
        correct (int): Number of correct predictions
        gold_standard (int): Number of items in gold standard
        total_predicted (int): Total number of predictions made
        
    Returns:
        tuple: (precision, recall, f1_score)
    """
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


def generate_pr_f1_score_predicates(correct_pred, detected_doc, total_predicted, gold_standard):
    """
    Calculate precision, recall, and F1 score for predicates with special handling.
    
    Args:
        correct_pred (int): Number of correct predicted predicates
        detected_doc (int): Number of document predicates detected
        total_predicted (int): Total number of predicted predicates
        gold_standard (int): Number of predicates in gold standard
        
    Returns:
        tuple: (precision, recall, f1_score)
    """
    try:
        precision = correct_pred / total_predicted
    except ZeroDivisionError:
        precision = 0
    try:
        recall = detected_doc / gold_standard
    except ZeroDivisionError:
        recall = 0
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1_score


def generate_report(excel_file_path, average_type="macro"):
    """
    Generate evaluation report from an Excel file containing evaluation logs.
    
    Args:
        excel_file_path (str): Path to the Excel file containing evaluation logs
        average_type (str): Type of averaging to use ("macro" or "micro")
        
    Returns:
        pd.DataFrame: DataFrame containing precision, recall, and F1 scores for all metrics
    """
    evaluation_log_df = pd.read_excel(excel_file_path)

    # Mapping of columns to their respective metrics
    metric_map = {
        "Triple": ["Correct Triples", "Gold Standard Triples", "Total Triples Predicted"],
        "Triple with Parents": ["Correct Triples with Parents", "Gold Standard Triples",
                                  "Total Triples Predicted"],
        "Triple with Related": ["Correct Triples with Related", "Gold Standard Triples",
                                  "Total Triples Predicted"],
        "Subject": ["Correct Extracted Subjects", "Gold Standard Subjects", "Extracted Subjects"],
        "Predicate": ["Correct Extracted Predicates", "Gold Standard Predicates", "Extracted Predicates"],
        "Predicate with Parents": ["Correct Pred Predicates Parents", "Detected Predicates Doc Parent", "Extracted Predicates", "Gold Standard Predicates"],
        "Predicate with Related": ["Correct Pred Predicates Related", "Detected Predicates Doc Related", "Extracted Predicates", "Gold Standard Predicates"],
        "Object": ["Correct Extracted Objects", "Gold Standard Objects", "Extracted Objects"],
        "Entity": ["Correct Extracted Entities", "Gold Standard Entities", "Extracted Entities"]
    }

    if average_type == "macro":
        # Dictionaries for accumulating scores
        metric_scores = {metric: {"precision": [], "recall": [], "f1": []} for metric in metric_map}

        # Iterate over all documents and collect individual scores
        for _, row in evaluation_log_df.iterrows():
            for metric, columns in metric_map.items():
                if metric in ["Predicate with Parents", "Predicate with Related"]:
                    correct_pred_col, detected_doc_col, pred_col, gold_col = columns
                    precision, recall, f1 = generate_pr_f1_score_predicates(
                        row[correct_pred_col], row[detected_doc_col], row[pred_col], row[gold_col]
                    )
                else:
                    correct_col, gold_col, pred_col = columns
                    precision, recall, f1 = generate_pr_f1_score(
                        row[correct_col], row[gold_col], row[pred_col]
                    )
                metric_scores[metric]["precision"].append(precision)
                metric_scores[metric]["recall"].append(recall)
                metric_scores[metric]["f1"].append(f1)

        # Calculate macro average
        macro_scores = {
            metric: {
                "Precision": sum(scores["precision"]) / len(scores["precision"]) if scores["precision"] else 0.0,
                "Recall": sum(scores["recall"]) / len(scores["recall"]) if scores["recall"] else 0.0,
                "F1-Score": sum(scores["f1"]) / len(scores["f1"]) if scores["f1"] else 0.0
            }
            for metric, scores in metric_scores.items()
        }

        # Create DataFrame
        scores_df = pd.DataFrame.from_dict(macro_scores, orient="index")

    elif average_type == "micro":
        # Initialize sums for micro averaging
        micro_sums = {metric: {"correct": 0, "gold": 0, "pred": 0} for metric in metric_map}
        
        # Sum up all raw counts across documents
        for _, row in evaluation_log_df.iterrows():
            for metric, columns in metric_map.items():
                if metric in ["Predicate with Parents", "Predicate with Related"]:
                    correct_pred_col, detected_doc_col, pred_col, gold_col = columns
                    micro_sums[metric]["correct"] += row[correct_pred_col]
                    micro_sums[metric]["gold"] += row[detected_doc_col]
                    micro_sums[metric]["pred"] += row[pred_col]
                else:
                    correct_col, gold_col, pred_col = columns
                    micro_sums[metric]["correct"] += row[correct_col]
                    micro_sums[metric]["gold"] += row[gold_col]
                    micro_sums[metric]["pred"] += row[pred_col]
        
        # Calculate final metrics using summed values
        micro_scores = {}
        for metric, sums in micro_sums.items():
            if metric in ["Predicate with Parents", "Predicate with Related"]:
                precision, recall, f1 = generate_pr_f1_score_predicates(
                    sums["correct"], sums["gold"], sums["pred"], sums["gold"]
                )
            else:
                precision, recall, f1 = generate_pr_f1_score(
                    sums["correct"], sums["gold"], sums["pred"]
                )
            micro_scores[metric] = {
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1
            }
        
        # Create DataFrame
        scores_df = pd.DataFrame.from_dict(micro_scores, orient="index")
    
    else:
        raise ValueError("average_type must be either 'macro' or 'micro'")

    return scores_df


def calculate_scores_from_array(values_array):
    """
    Calculate evaluation scores from an array of raw metric values.
    
    Args:
        values_array (list): Array of 21 metric values in specific order
        
    Returns:
        pd.DataFrame: DataFrame containing precision, recall, and F1 scores for all metrics
        
    Raises:
        ValueError: If the array doesn't contain exactly 21 values
    """
    if len(values_array) != 21:
        raise ValueError(f"Expected 21 values, but got {len(values_array)}")

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
        detected_predicates_doc_parent,
        detected_predicates_doc_related,
        correct_pred_predicates_parent,
        correct_pred_predicates_related,

        extracted_objects,
        gold_objects,
        correct_objects,

        extracted_entities,
        gold_entities,
        correct_entities
    ) = values_array

    result = {}

    # Calculate triple metrics
    precision, recall, f1 = generate_pr_f1_score(correct_triples, gold_triples, pred_triples)
    result["Triple"] = {"Precision": precision, "Recall": recall, "F1-Score": f1}

    # Calculate triple with parents metrics
    precision, recall, f1 = generate_pr_f1_score(correct_triples_with_parents, gold_triples, pred_triples)
    result["Triple with Parents"] = {"Precision": precision, "Recall": recall, "F1-Score": f1}

    # Calculate triple with related metrics
    precision, recall, f1 = generate_pr_f1_score(correct_triples_with_related, gold_triples, pred_triples)
    result["Triple with Related"] = {"Precision": precision, "Recall": recall, "F1-Score": f1}

    # Calculate subject metrics
    precision, recall, f1 = generate_pr_f1_score(correct_subjects, gold_subjects, extracted_subjects)
    result["Subject"] = {"Precision": precision, "Recall": recall, "F1-Score": f1}

    # Calculate predicate metrics
    precision, recall, f1 = generate_pr_f1_score(correct_predicates, gold_predicates, extracted_predicates)
    result["Predicate"] = {"Precision": precision, "Recall": recall, "F1-Score": f1}

    # Calculate predicate with parents metrics 
    precision, recall, f1 = generate_pr_f1_score_predicates(
        correct_pred_predicates_parent, detected_predicates_doc_parent, extracted_predicates, gold_predicates
    )
    result["Predicate with Parents"] = {"Precision": precision, "Recall": recall, "F1-Score": f1}

    # Calculate predicate with related metrics
    precision, recall, f1 = generate_pr_f1_score_predicates(
        correct_pred_predicates_related, detected_predicates_doc_related, extracted_predicates, gold_predicates
    )
    result["Predicate with Related"] = {"Precision": precision, "Recall": recall, "F1-Score": f1}

    # Calculate object metrics
    precision, recall, f1 = generate_pr_f1_score(correct_objects, gold_objects, extracted_objects)
    result["Object"] = {"Precision": precision, "Recall": recall, "F1-Score": f1}

    # Calculate entity metrics
    precision, recall, f1 = generate_pr_f1_score(correct_entities, gold_entities, extracted_entities)
    result["Entity"] = {"Precision": precision, "Recall": recall, "F1-Score": f1}

    return pd.DataFrame.from_dict(result, orient="index")


def convert_eval_log(path, dataset_cache):
    """
    Convert an evaluation log Excel file by recalculating metrics for each document.
    
    Args:
        path (str): Path to the evaluation log Excel file
        dataset_cache (dict): Cache for dataset loading
        
    Returns:
        dict: Updated dataset cache
    """
    match = re.match(r"(?P<dataset>\w+)-(?P<split>\w+)-(?P<num_samples>\d+)-evaluation_log-.*\.xlsx", os.path.basename(path))

    if match:
        dataset = match.group("dataset")
        split = match.group("split")
        number_of_samples = int(match.group("num_samples"))
        try:
            triple_df, entity_df, docs = dataset_cache[f"{dataset}-{split}-{number_of_samples}"]
        except KeyError:
            triple_df, entity_df, docs = parser.unified_parser(dataset, split, number_of_samples)
            dataset_cache[f"{dataset}-{split}-{number_of_samples}"] = (triple_df, entity_df, docs)
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
            "Detected Predicates Doc Parent", "Detected Predicates Doc Related", 
            "Correct Pred Predicates Parents", "Correct Pred Predicates Related", 
            "Extracted Objects", "Gold Standard Objects", "Correct Extracted Objects",
            "Extracted Entities", "Gold Standard Entities", "Correct Extracted Entities", 
            "Result String"
        ]
    )
    evaluation_log_df.to_excel(path, index=False)
    return dataset_cache


def compare_turtle_strings(predicted_turtle_string, ground_truth_turtle_string):
    """
    Compare two turtle strings directly and return evaluation metrics.
    
    Args:
        predicted_turtle_string (str): The predicted turtle string to evaluate
        ground_truth_turtle_string (str): The ground truth turtle string to compare against
        
    Returns:
        tuple: A tuple containing all evaluation metrics in the same order as evaluate_doc
    """
    pred_triple_df, pred_error = parse_turtle(predicted_turtle_string)
    gold_triple_df, gold_error = parse_turtle(ground_truth_turtle_string)
    
    if pred_error != "Success" or gold_error != "Success":
        raise ValueError(f"Error parsing turtle strings. Predicted error: {pred_error}, Ground truth error: {gold_error}")
    
    return _calculate_metrics(pred_triple_df, gold_triple_df)


def convert_pickle_eval_log(pickle_path, dataset_cache):
    """
    Convert a pickle file containing {docid: turtle_string} dictionary into a proper evaluation log Excel file.
    
    Args:
        pickle_path (str): Path to the pickle file
        dataset_cache (dict): Cache for dataset loading
        
    Returns:
        dict: Updated dataset cache
    """
    # Extract dataset info from filename (similar to convert_eval_log)
    filename = os.path.basename(pickle_path)
    match = re.match(r"(?P<dataset>\w+)-(?P<split>\w+)-(?P<num_samples>\d+)-evaluation_log-.*\.pkl", filename)
    
    if not match:
        print(f"File name {filename} does not match the required format.")
        return dataset_cache
    
    dataset = match.group("dataset")
    split = match.group("split")
    number_of_samples = int(match.group("num_samples"))
    
    # Load dataset
    try:
        triple_df, entity_df, docs = dataset_cache[f"{dataset}-{split}-{number_of_samples}"]
    except KeyError:
        triple_df, entity_df, docs = parser.unified_parser(dataset, split, number_of_samples)
        dataset_cache[f"{dataset}-{split}-{number_of_samples}"] = (triple_df, entity_df, docs)
    
    # Load pickle file
    try:
        with open(pickle_path, 'rb') as f:
            docid_turtle_dict = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file {pickle_path}: {e}")
        return dataset_cache
    
    # Convert to evaluation log format
    evaluation_log = []
    for doc_id, turtle_string in docid_turtle_dict.items():
        turtle_string = turtle_string.replace("wdt:","wd:")
        try:
            metrics = evaluate_doc(turtle_string, doc_id, triple_df)
            evaluation_log.append([doc_id, *metrics, turtle_string])
        except Exception as e:
            print(f"Error evaluating doc {doc_id}: {e}")
            # Add row with zeros for failed evaluations
            zero_metrics = [0] * 21  # 21 metrics from _calculate_metrics
            evaluation_log.append([doc_id, *zero_metrics, turtle_string])
    
    # Create DataFrame
    evaluation_log_df = pd.DataFrame(
        evaluation_log,
        columns=[
            "Doc ID",
            "Correct Triples", "Correct Triples with Parents", "Correct Triples with Related", "Gold Standard Triples",
            "Total Triples Predicted",
            "Extracted Subjects", "Gold Standard Subjects", "Correct Extracted Subjects",
            "Extracted Predicates", "Gold Standard Predicates", "Correct Extracted Predicates",
            "Detected Predicates Doc Parent", "Detected Predicates Doc Related", 
            "Correct Pred Predicates Parents", "Correct Pred Predicates Related", 
            "Extracted Objects", "Gold Standard Objects", "Correct Extracted Objects",
            "Extracted Entities", "Gold Standard Entities", "Correct Extracted Entities", 
            "Result String"
        ]
    )
    
    # Save as Excel file (replace .pkl with .xlsx)
    excel_path = pickle_path.replace('.pkl', '.xlsx')
    evaluation_log_df.to_excel(excel_path, index=False)
    print(f"Converted {pickle_path} to {excel_path}")
    
    return dataset_cache


def convert_all_synthie_text_eval_logs(results_dir="../results/result_evaluation_logs"):
    """
    Convert all evaluation logs for synthie_text dataset with 50 samples and test split.
    
    Args:
        results_dir (str): Path to the results_evaluation_logs directory
        
    Returns:
        dict: Updated dataset cache
    """
    dataset_cache = {}
    dataset = "synthie_text"
    split = "test"
    number_of_samples = 50
    
    # Load dataset once and cache it
    try:
        triple_df, entity_df, docs = dataset_cache[f"{dataset}-{split}-{number_of_samples}"]
    except KeyError:
        triple_df, entity_df, docs = parser.unified_parser(dataset, split, number_of_samples)
        dataset_cache[f"{dataset}-{split}-{number_of_samples}"] = (triple_df, entity_df, docs)
    
    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} not found.")
        return dataset_cache
    
    # Get all Excel files in the directory
    excel_files = [f for f in os.listdir(results_dir) if f.endswith('.xlsx') and not f.startswith('.')]
    
    if not excel_files:
        print(f"No Excel files found in {results_dir}")
        return dataset_cache
    
    print(f"Found {len(excel_files)} Excel files to process:")
    for file in excel_files:
        print(f"  - {file}")
    
    # Convert each file
    for file in excel_files:
        file_path = os.path.join(results_dir, file)
        print(f"\nConverting: {file}")
        try:
            evaluation_log_df = pd.read_excel(file_path)
            evaluation_log = []
            
            for doc_id, row in tqdm(evaluation_log_df.iterrows()):
                result_string = str(row["Result String"])
                turtle_string_match = re.search(r'<ttl>(.*?)</ttl>', result_string, re.DOTALL)
                if turtle_string_match:
                    turtle_string = turtle_string_match.group(1)
                else:
                    turtle_string = result_string
                
                try:
                    metrics = evaluate_doc(turtle_string, doc_id, triple_df)
                    evaluation_log.append([doc_id, *metrics, result_string])
                except Exception as e:
                    print(f"  Error evaluating doc {doc_id}: {e}")
                    # Add row with zeros for failed evaluations
                    zero_metrics = [0] * 21  # 21 metrics from _calculate_metrics
                    evaluation_log.append([doc_id, *zero_metrics, result_string])
            
            # Create DataFrame
            evaluation_log_df = pd.DataFrame(
                evaluation_log,
                columns=[
                    "Doc ID",
                    "Correct Triples", "Correct Triples with Parents", "Correct Triples with Related", "Gold Standard Triples",
                    "Total Triples Predicted",
                    "Extracted Subjects", "Gold Standard Subjects", "Correct Extracted Subjects",
                    "Extracted Predicates", "Gold Standard Predicates", "Correct Extracted Predicates",
                    "Detected Predicates Doc Parent", "Detected Predicates Doc Related", 
                    "Correct Pred Predicates Parents", "Correct Pred Predicates Related", 
                    "Extracted Objects", "Gold Standard Objects", "Correct Extracted Objects",
                    "Extracted Entities", "Gold Standard Entities", "Correct Extracted Entities", 
                    "Result String"
                ]
            )
            
            # Save the converted file
            evaluation_log_df.to_excel(file_path, index=False)
            print(f"  Successfully converted: {file}")
            
        except Exception as e:
            print(f"  Error converting {file}: {e}")
    
    print(f"\nConversion completed for {len(excel_files)} files.")
    return dataset_cache


if __name__ == "__main__":
    # Example usage of the new function
    dataset_cache = convert_all_synthie_text_eval_logs(results_dir="/Users/i538914/Documents/Uni/Masterarbeit/CIExMAS/results/result_evaluation_logs/folder")
    
    # Example of individual file conversion (commented out)
    # dataset_cache = {}
    # convert_eval_log("/Users/i538914/Documents/Uni/Masterarbeit/CIExMAS/results/result_evaluation_logs/synthie_text-test-50-evaluation_log-synthie_large_fe.xlsx", dataset_cache)
    # report = generate_report("/Users/i538914/Documents/Uni/Masterarbeit/CIExMAS/results/result_evaluation_logs/synthie_text-test-50-evaluation_log-synthie_large_fe.xlsx", "macro")
    # print(report.head())
