"""
ACE-based optimization runner for the CIExMAS Pipeline.

Uses the real ACE framework (Reflector + Curator) to iteratively improve
a playbook that guides the full extraction pipeline.
The PipelineGenerator wraps the multi-step pipeline as ACE's Generator.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import git

repo = git.Repo(search_parent_directories=True)
sys.path.append(repo.working_dir)

# Add ACE repo root to path (enables both package imports and top-level module imports)
ACE_REPO = Path(repo.working_dir).parent / "ace"
sys.path.insert(0, str(ACE_REPO))

from ace import Reflector, Curator
from playbook_utils import (
    update_bullet_counts,
    extract_playbook_bullets,
    get_playbook_stats,
    extract_json_from_text,
    apply_curator_operations,
)
from logger import log_llm_call, log_bullet_usage
from utils import count_tokens

from helper_tools.parser import unified_parser
from approaches.Pipeline_ACE.ace_clients import (
    initialize_ciexmas_clients,
    get_provider_name,
    get_model_id,
)
from approaches.Pipeline_ACE.ace_generator import PipelineGenerator
from approaches.Pipeline_ACE.ace_data_processor import CIExMASDataProcessor

PIPELINE_ACE_DIR = Path(repo.working_dir) / "approaches" / "Pipeline_ACE"
RESULTS_DIR = PIPELINE_ACE_DIR / "results"

EMPTY_PLAYBOOK = """## ENTITY EXTRACTION STRATEGIES

## TRIPLE FORMATION RULES

## URI MAPPING HEURISTICS

## TURTLE GENERATION PATTERNS

## COMMON MISTAKES TO AVOID

## OTHERS"""


def parse_args():
    parser = argparse.ArgumentParser(description="ACE-optimized CIExMAS Pipeline")

    parser.add_argument(
        "--mode",
        choices=["offline", "online", "eval_only"],
        default="offline",
    )
    parser.add_argument("--dataset", default="wiki_cie_text")
    parser.add_argument("--train-split", default="auto",
                        help="Train split name. 'auto' = use validation split for train+val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--num-train", type=int, default=20)
    parser.add_argument("--num-val", type=int, default=10)
    parser.add_argument("--num-test", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.3,
                        help="Fraction of train data to hold out as validation (when auto-splitting)")

    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--curator-frequency", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--playbook-token-budget", type=int, default=80000)
    parser.add_argument("--test-workers", type=int, default=4)
    parser.add_argument("--online-eval-frequency", type=int, default=5)

    parser.add_argument("--initial-playbook", type=str, default=None)
    parser.add_argument("--save-path", type=str, default=None)

    return parser.parse_args()


def load_initial_playbook(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def evaluate_test_set(generator, data_processor, playbook, test_samples, max_workers=4):
    """Evaluate pipeline on test samples in parallel."""
    correct = 0
    total = 0
    f1_scores = []

    def eval_single(sample):
        turtle, _, _ = generator.generate(
            question=sample["question"],
            playbook=playbook,
            context=sample["context"],
            reflection="(empty)",
        )
        f1 = data_processor._compute_triple_f1(turtle, sample["target"])
        is_correct = f1 >= data_processor.__class__.__dict__.get(
            "F1_THRESHOLD", 0.5
        )
        return f1, is_correct

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(eval_single, s): i for i, s in enumerate(test_samples)}
        for future in as_completed(futures):
            try:
                f1, is_ok = future.result(timeout=300)
                f1_scores.append(f1)
                total += 1
                if is_ok:
                    correct += 1
            except Exception as e:
                f1_scores.append(0.0)
                total += 1
                print(f"  Test sample failed: {e}")

    accuracy = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total, "f1_scores": f1_scores}


def train_single_sample(
    sample,
    generator,
    reflector,
    curator,
    data_processor,
    playbook,
    next_global_id,
    step,
    total_samples,
    max_rounds,
    curator_frequency,
    token_budget,
    log_dir,
    usage_log_path,
):
    """
    Train on a single sample using the ACE loop:
    Generate → Evaluate → Reflect → (Retry) → Curate
    """
    context = sample["context"]
    target = sample["target"]
    question = sample["question"]

    # Step 1: Generate with current playbook
    turtle, bullet_ids, _ = generator.generate(
        question=question,
        playbook=playbook,
        context=context,
        reflection="(empty)",
    )

    is_correct = data_processor.answer_is_correct(turtle, target)
    f1 = data_processor._compute_triple_f1(turtle, target)
    pre_train_f1 = f1

    print(f"    Initial F1={f1:.3f} (correct={is_correct})")

    # Step 2: Reflection
    reflection_content = "(empty)"
    playbook_bullets = extract_playbook_bullets(playbook, bullet_ids)

    if not is_correct:
        # Reflect and retry for incorrect answers
        for round_num in range(max_rounds):
            env_feedback = (
                f"Pipeline produced Turtle with Triple-F1={f1:.3f} (threshold=0.5). "
                f"The output needs improvement."
            )

            reflection_content, bullet_tags, _ = reflector.reflect(
                question=f"Extract knowledge triples from text and produce Turtle RDF.\nText: {context[:500]}...",
                reasoning_trace=f"Pipeline output:\n{turtle[:1000]}",
                predicted_answer=turtle[:500],
                ground_truth=f"Expected doc_id={json.loads(target)['doc_id']} with higher F1",
                environment_feedback=env_feedback,
                bullets_used=playbook_bullets,
                use_ground_truth=True,
                call_id=f"train_reflect_round_{round_num}",
                log_dir=log_dir,
            )

            if bullet_tags:
                playbook = update_bullet_counts(playbook, bullet_tags)

            # Re-generate with reflection
            turtle, bullet_ids, _ = generator.generate(
                question=question,
                playbook=playbook,
                context=context,
                reflection=reflection_content,
            )

            f1 = data_processor._compute_triple_f1(turtle, target)
            is_correct = data_processor.answer_is_correct(turtle, target)

            if is_correct:
                print(f"    Corrected after round {round_num + 1}: F1={f1:.3f}")
                break
    else:
        # Still reflect on correct answers to tag helpful bullets
        env_feedback = (
            f"Pipeline produced Turtle with Triple-F1={f1:.3f}. Good result."
        )

        reflection_content, bullet_tags, _ = reflector.reflect(
            question=f"Extract knowledge triples from text and produce Turtle RDF.\nText: {context[:500]}...",
            reasoning_trace=f"Pipeline output:\n{turtle[:1000]}",
            predicted_answer=turtle[:500],
            ground_truth=f"doc_id={json.loads(target)['doc_id']} - output matches well",
            environment_feedback=env_feedback,
            bullets_used=playbook_bullets,
            use_ground_truth=True,
            call_id="train_reflect_correct",
            log_dir=log_dir,
        )

        if bullet_tags:
            playbook = update_bullet_counts(playbook, bullet_tags)

    # Step 3: Curator
    if step % curator_frequency == 0:
        print(f"    Running Curator at step {step}...")
        stats = get_playbook_stats(playbook)

        playbook, next_global_id, _, _ = curator.curate(
            current_playbook=playbook,
            recent_reflection=reflection_content,
            question_context=context[:500],
            current_step=step,
            total_samples=total_samples,
            token_budget=token_budget,
            playbook_stats=stats,
            use_ground_truth=True,
            call_id=f"train_curate_s_{step}",
            log_dir=log_dir,
            next_global_id=next_global_id,
        )

    post_train_f1 = data_processor._compute_triple_f1(turtle, target)

    return playbook, next_global_id, pre_train_f1, post_train_f1


def run_offline(args, generator, reflector, curator, data_processor, train_samples, val_samples, test_samples, save_path, log_dir):
    """Offline training: train on train set, validate, test at end."""
    playbook = load_initial_playbook(args.initial_playbook) or EMPTY_PLAYBOOK
    next_global_id = 1
    best_accuracy = 0.0
    best_playbook = playbook

    usage_log_path = os.path.join(save_path, "bullet_usage_log.jsonl")
    playbook_dir = os.path.join(save_path, "intermediate_playbooks")
    os.makedirs(playbook_dir, exist_ok=True)

    # Initial test
    if test_samples:
        print("\n--- Initial Test (before training) ---")
        initial_results = evaluate_test_set(generator, data_processor, playbook, test_samples, args.test_workers)
        print(f"Initial Mean F1: {initial_results['accuracy']:.3f}")
        with open(os.path.join(save_path, "initial_test_results.json"), "w") as f:
            json.dump(initial_results, f, indent=2)

    # Training loop
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{args.num_epochs}")
        print(f"{'='*60}")

        pre_f1s, post_f1s = [], []

        for step, sample in enumerate(train_samples, 1):
            print(f"\n  Step {step}/{len(train_samples)}")

            playbook, next_global_id, pre_f1, post_f1 = train_single_sample(
                sample=sample,
                generator=generator,
                reflector=reflector,
                curator=curator,
                data_processor=data_processor,
                playbook=playbook,
                next_global_id=next_global_id,
                step=step,
                total_samples=len(train_samples),
                max_rounds=args.max_rounds,
                curator_frequency=args.curator_frequency,
                token_budget=args.playbook_token_budget,
                log_dir=log_dir,
                usage_log_path=usage_log_path,
            )

            pre_f1s.append(pre_f1)
            post_f1s.append(post_f1)

            if step % args.save_steps == 0:
                pb_path = os.path.join(playbook_dir, f"epoch_{epoch}_step_{step}.txt")
                with open(pb_path, "w") as f:
                    f.write(playbook)

            if step % args.eval_steps == 0 and val_samples:
                print(f"\n  --- Validation at step {step} ---")
                val_results = evaluate_test_set(generator, data_processor, playbook, val_samples, args.test_workers)
                print(f"  Val Mean F1: {val_results['accuracy']:.3f}")

                if val_results["accuracy"] > best_accuracy:
                    best_accuracy = val_results["accuracy"]
                    best_playbook = playbook
                    print(f"  New best: {best_accuracy:.3f}")

        print(f"\nEpoch {epoch} - Pre-train Mean F1: {sum(pre_f1s)/len(pre_f1s):.3f}")
        print(f"Epoch {epoch} - Post-train Mean F1: {sum(post_f1s)/len(post_f1s):.3f}")

    # Save playbooks
    with open(os.path.join(save_path, "final_playbook.txt"), "w") as f:
        f.write(playbook)
    with open(os.path.join(save_path, "best_playbook.txt"), "w") as f:
        f.write(best_playbook)

    # Final test
    if test_samples:
        print("\n--- Final Test (with best playbook) ---")
        final_results = evaluate_test_set(generator, data_processor, best_playbook, test_samples, args.test_workers)
        print(f"Final Mean F1: {final_results['accuracy']:.3f}")
        with open(os.path.join(save_path, "final_test_results.json"), "w") as f:
            json.dump(final_results, f, indent=2)

    return {"best_accuracy": best_accuracy}


def run_online(args, generator, reflector, curator, data_processor, test_samples, save_path, log_dir):
    """Online mode: train and test on the same samples in windows."""
    playbook = load_initial_playbook(args.initial_playbook) or EMPTY_PLAYBOOK
    next_global_id = 1
    window_size = args.online_eval_frequency

    usage_log_path = os.path.join(save_path, "bullet_usage_log.jsonl")
    playbook_dir = os.path.join(save_path, "intermediate_playbooks")
    os.makedirs(playbook_dir, exist_ok=True)

    all_f1s = []
    num_windows = (len(test_samples) + window_size - 1) // window_size

    for w_idx in range(num_windows):
        start = w_idx * window_size
        end = min((w_idx + 1) * window_size, len(test_samples))
        window = test_samples[start:end]

        print(f"\n{'='*60}")
        print(f"WINDOW {w_idx+1}/{num_windows} (samples {start}-{end-1})")
        print(f"{'='*60}")

        # Test window with current playbook
        window_results = evaluate_test_set(generator, data_processor, playbook, window, args.test_workers)
        all_f1s.extend(window_results["f1_scores"])
        print(f"  Window F1: {window_results['accuracy']:.3f}")

        # Train on window
        for step, sample in enumerate(window, 1):
            global_step = start + step
            print(f"\n  Train step {step}/{len(window)} (global {global_step})")

            playbook, next_global_id, pre_f1, post_f1 = train_single_sample(
                sample=sample,
                generator=generator,
                reflector=reflector,
                curator=curator,
                data_processor=data_processor,
                playbook=playbook,
                next_global_id=next_global_id,
                step=global_step,
                total_samples=len(test_samples),
                max_rounds=args.max_rounds,
                curator_frequency=args.curator_frequency,
                token_budget=args.playbook_token_budget,
                log_dir=log_dir,
                usage_log_path=usage_log_path,
            )

        # Save window playbook
        pb_path = os.path.join(playbook_dir, f"window_{w_idx+1}.txt")
        with open(pb_path, "w") as f:
            f.write(playbook)

    # Save final
    with open(os.path.join(save_path, "final_playbook.txt"), "w") as f:
        f.write(playbook)

    final_accuracy = sum(all_f1s) / len(all_f1s) if all_f1s else 0.0
    print(f"\nOnline Final Mean F1: {final_accuracy:.3f}")

    with open(os.path.join(save_path, "online_results.json"), "w") as f:
        json.dump({"accuracy": final_accuracy, "f1_scores": all_f1s}, f, indent=2)

    return {"accuracy": final_accuracy}


def run_eval_only(args, generator, data_processor, test_samples, save_path):
    """Evaluate with a pre-existing playbook."""
    playbook = load_initial_playbook(args.initial_playbook) or EMPTY_PLAYBOOK
    print("\n--- Evaluation Only ---")
    results = evaluate_test_set(generator, data_processor, playbook, test_samples, args.test_workers)
    print(f"Mean F1: {results['accuracy']:.3f} ({results['correct']}/{results['total']} correct)")

    with open(os.path.join(save_path, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    args = parse_args()

    provider = get_provider_name()
    model_id = get_model_id()

    print(f"\n{'='*60}")
    print(f"ACE-OPTIMIZED CIExMAS PIPELINE")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Model: {provider}/{model_id}")
    print(f"Dataset: {args.dataset}")
    print(f"{'='*60}\n")

    # Setup save path
    if args.save_path:
        save_path = args.save_path
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = str(
            RESULTS_DIR / f"ace_{args.mode}_{args.dataset}_{timestamp}"
        )
    os.makedirs(save_path, exist_ok=True)
    log_dir = os.path.join(save_path, "llm_logs")
    os.makedirs(log_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    if args.mode == "offline":
        import pandas as pd

        # Load test set
        test_triple_df, test_entity_df, test_docs = unified_parser(
            args.dataset, args.test_split, args.num_test
        )

        # Load train+val: auto-split a single split if no separate train split exists
        total_needed = args.num_train + args.num_val
        if args.train_split == "auto":
            # Use 'validation' split (or 'train' if it exists) and split into train/val
            try:
                source_triple_df, source_entity_df, source_docs = unified_parser(
                    args.dataset, "train", total_needed
                )
                print(f"  Using 'train' split as source for train+val")
            except Exception:
                source_triple_df, source_entity_df, source_docs = unified_parser(
                    args.dataset, "validation", total_needed
                )
                print(f"  Using 'validation' split as source for train+val (no train split found)")

            # Split by doc_ids
            all_doc_ids = source_docs["docid"].unique()
            n_val = max(1, int(len(all_doc_ids) * args.val_ratio))
            n_train = len(all_doc_ids) - n_val

            train_doc_ids = all_doc_ids[:n_train]
            val_doc_ids = all_doc_ids[n_train:]

            train_docs = source_docs[source_docs["docid"].isin(train_doc_ids)].reset_index(drop=True)
            val_docs = source_docs[source_docs["docid"].isin(val_doc_ids)].reset_index(drop=True)
            train_triple_df = source_triple_df[source_triple_df["docid"].isin(train_doc_ids)].reset_index(drop=True)
            val_triple_df = source_triple_df[source_triple_df["docid"].isin(val_doc_ids)].reset_index(drop=True)
        else:
            # Explicit splits provided
            train_triple_df, _, train_docs = unified_parser(
                args.dataset, args.train_split, args.num_train
            )
            val_triple_df, _, val_docs = unified_parser(
                args.dataset, args.train_split, args.num_train + args.num_val
            )
            # Take last num_val docs as validation
            all_ids = val_docs["docid"].unique()
            val_ids = all_ids[args.num_train:]
            val_docs = val_docs[val_docs["docid"].isin(val_ids)].reset_index(drop=True)
            val_triple_df = val_triple_df[val_triple_df["docid"].isin(val_ids)].reset_index(drop=True)

        # Combined triple_df for evaluation lookups
        all_triple_df = pd.concat([train_triple_df, val_triple_df, test_triple_df]).drop_duplicates()

        data_processor = CIExMASDataProcessor(all_triple_df)
        train_samples = data_processor.process_task_data(train_docs, train_triple_df)
        val_samples = data_processor.process_task_data(val_docs, val_triple_df)
        test_samples = data_processor.process_task_data(test_docs, test_triple_df)

        print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    else:
        test_triple_df, test_entity_df, test_docs = unified_parser(
            args.dataset, args.test_split, args.num_test
        )
        data_processor = CIExMASDataProcessor(test_triple_df)
        test_samples = data_processor.process_task_data(test_docs, test_triple_df)
        train_samples = None
        val_samples = None
        print(f"Test: {len(test_samples)}")

    # Initialize components
    print("Initializing ACE components...")
    generator = PipelineGenerator(model=f"{provider}/{model_id}")

    gen_client, ref_client, cur_client = initialize_ciexmas_clients()
    reflector = Reflector(ref_client, provider.lower(), model_id, args.max_tokens)
    curator = Curator(cur_client, provider.lower(), model_id, args.max_tokens)

    # Save config
    config = {
        "mode": args.mode,
        "dataset": args.dataset,
        "provider": provider,
        "model_id": model_id,
        "train_split": args.train_split,
        "test_split": args.test_split,
        "val_ratio": args.val_ratio,
        "num_train": args.num_train,
        "num_val": args.num_val,
        "num_test": args.num_test,
        "max_rounds": args.max_rounds,
        "curator_frequency": args.curator_frequency,
        "eval_steps": args.eval_steps,
        "playbook_token_budget": args.playbook_token_budget,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(save_path, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Execute
    if args.mode == "offline":
        results = run_offline(
            args, generator, reflector, curator, data_processor,
            train_samples, val_samples, test_samples, save_path, log_dir,
        )
    elif args.mode == "online":
        results = run_online(
            args, generator, reflector, curator, data_processor,
            test_samples, save_path, log_dir,
        )
    else:
        results = run_eval_only(args, generator, data_processor, test_samples, save_path)

    print(f"\n{'='*60}")
    print(f"DONE - Results saved to: {save_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
