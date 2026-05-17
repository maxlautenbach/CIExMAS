"""
ACE DataProcessor for the CIExMAS information extraction task.

Converts unified_parser output to ACE sample format and evaluates
predicted Turtle RDF against gold standard triples.
"""

import json
import pandas as pd
from typing import List, Dict, Any

from helper_tools.evaluation import (
    evaluate_doc,
    calculate_scores_from_array,
    parse_turtle,
    generate_pr_f1_score,
)


F1_THRESHOLD = 0.5


class CIExMASDataProcessor:
    """
    DataProcessor for CIExMAS closed information extraction.

    Holds a reference to the gold-standard triple DataFrame so that
    answer_is_correct can evaluate Turtle outputs by doc_id.
    """

    def __init__(self, triple_df: pd.DataFrame):
        """
        Args:
            triple_df: Gold-standard DataFrame with columns
                       [docid, subject_uri, predicate_uri, object_uri]
        """
        self.triple_df = triple_df

    def process_task_data(
        self, docs: pd.DataFrame, triple_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Convert unified_parser output to ACE-compatible sample format.

        Args:
            docs: DataFrame with columns [docid, text]
            triple_df: Gold-standard triple DataFrame

        Returns:
            List of dicts with keys: context, question, target, others
        """
        samples = []
        for _, row in docs.iterrows():
            doc_id = row["docid"]
            text = row["text"]

            target = json.dumps({"doc_id": int(doc_id)})

            sample = {
                "context": text,
                "question": (
                    "Extract all knowledge triples from the given text and produce "
                    "valid Turtle RDF. Map entities and predicates to Wikidata URIs."
                ),
                "target": target,
                "others": {"doc_id": int(doc_id)},
            }
            samples.append(sample)

        return samples

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        Determine whether the predicted Turtle is correct enough.

        Computes Triple-F1 using URI-based evaluation against gold standard.
        Returns True if F1 >= F1_THRESHOLD.

        Args:
            predicted: Turtle string produced by the pipeline
            ground_truth: JSON string with {"doc_id": N}
        """
        f1 = self._compute_triple_f1(predicted, ground_truth)
        return f1 >= F1_THRESHOLD

    def evaluate_accuracy(
        self, predictions: List[str], targets: List[str]
    ) -> float:
        """
        Compute mean Triple-F1 over all samples.

        Args:
            predictions: List of predicted Turtle strings
            targets: List of JSON target strings (each containing doc_id)

        Returns:
            Mean Triple-F1 score (0.0 to 1.0)
        """
        if not predictions:
            return 0.0

        f1_scores = []
        for pred, target in zip(predictions, targets):
            f1 = self._compute_triple_f1(pred, target)
            f1_scores.append(f1)

        return sum(f1_scores) / len(f1_scores)

    def _compute_triple_f1(self, predicted_turtle: str, target_json: str) -> float:
        """
        Compute Triple-F1 for a single sample.

        Args:
            predicted_turtle: Turtle string from the pipeline
            target_json: JSON string with doc_id

        Returns:
            Triple F1 score (0.0 to 1.0)
        """
        try:
            target_data = json.loads(target_json)
            doc_id = target_data["doc_id"]
        except (json.JSONDecodeError, KeyError):
            return 0.0

        if not predicted_turtle or not predicted_turtle.strip():
            return 0.0

        try:
            metrics = evaluate_doc(predicted_turtle, doc_id, self.triple_df)
            scores = calculate_scores_from_array(metrics)
            if "Triple" in scores.index:
                return float(scores.loc["Triple"]["F1-Score"])
            return 0.0
        except Exception:
            return 0.0
