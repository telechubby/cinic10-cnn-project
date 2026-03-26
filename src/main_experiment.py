#!/usr/bin/env python3
"""
Main Experiment Script for CINIC-10 CNN Project

Orchestrates the complete experiment pipeline:
  1. Baseline CNN training
  2. Hyperparameter analysis (learning rates, batch sizes, regularization, optimizers)
  3. Data augmentation studies (standard + cutout)
  4. Few-shot learning evaluation
  5. Reduced dataset analysis
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

from datetime import datetime

import numpy as np
import pandas as pd
import torch

from augmentation_studies import compare_augmentation_approaches
from data_preprocessing import create_data_generators
from evaluation import (
    compare_model_performance,
    run_reduced_dataset_experiment,
    plot_reduced_dataset_results,
)
from few_shot_learning import evaluate_few_shot_performance
from hyperparameter_analysis import create_comprehensive_hyperparameter_analysis
from model_architecture import (
    create_baseline_cnn,
    create_cnn_with_regularization,
    create_deep_cnn,
    create_efficient_cnn,
    create_few_shot_cnn,
)
from utils import get_device, set_seeds, train_model

import argparse as _ap, random as _rand
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument("--seed", type=int, default=None)
_args, _ = _parser.parse_known_args()
MASTER_SEED = _args.seed if _args.seed is not None else _rand.randint(10_000, 99_999)
del _ap, _rand, _parser, _args
print(f"Master seed: {MASTER_SEED}  (re-run with --seed {MASTER_SEED} to reproduce)")
set_seeds(MASTER_SEED)

# ── Dataset paths ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR  = os.path.join(_PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "valid")
TEST_DIR  = os.path.join(DATA_DIR, "test")
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
MODELS_DIR  = os.path.join(_PROJECT_ROOT, "models")

# ── Experiment config ─────────────────────────────────────────────────────────
EPOCHS_BASELINE    = 20
EPOCHS_HP_SEARCH   = 5   # shorter sweeps for hyperparameter search
EPOCHS_AUGMENTATION = 5
EPOCHS_FEW_SHOT    = 15
BATCH_SIZE = 32

# CINIC-10 class labels
CINIC_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def setup_project_directories():
    """Create necessary project directories."""
    for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)
    print("Project directories set up successfully")


def run_baseline_experiment():
    """Train baseline CNN and save best weights."""
    print("=" * 60)
    print("RUNNING BASELINE EXPERIMENT")
    print("=" * 60)

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader = create_data_generators(
        TRAIN_DIR, VAL_DIR, batch_size=BATCH_SIZE, augment=False
    )

    model = create_baseline_cnn().to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters())
    checkpoint = os.path.join(MODELS_DIR, "baseline_cnn.pt")

    history = train_model(
        model, train_loader, val_loader, optimizer,
        epochs=EPOCHS_BASELINE, device=device,
        patience=5, checkpoint_path=checkpoint,
    )

    final_val_acc = history["val_accuracy"][-1] if history["val_accuracy"] else 0.0
    print(f"Baseline final val_acc: {final_val_acc:.4f}")
    return model, history


def run_hyperparameter_analysis():
    """Run comprehensive hyperparameter analysis and save results."""
    print("\n" + "=" * 60)
    print("RUNNING HYPERPARAMETER ANALYSIS")
    print("=" * 60)

    results = create_comprehensive_hyperparameter_analysis(
        create_baseline_cnn, TRAIN_DIR, VAL_DIR
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for key, val in results.items():
        pd.DataFrame(val).to_csv(
            os.path.join(RESULTS_DIR, f"hp_{key}.csv"), index=False
        )
        print(f"  Saved: results/hp_{key}.csv")

    print("✓ Hyperparameter analysis complete")
    return results


def run_augmentation_studies():
    """Run data augmentation comparison and save results."""
    print("\n" + "=" * 60)
    print("RUNNING DATA AUGMENTATION STUDIES")
    print("=" * 60)

    results = compare_augmentation_approaches(
        create_baseline_cnn, TRAIN_DIR, VAL_DIR,
        epochs=EPOCHS_AUGMENTATION, batch_size=BATCH_SIZE
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for key, val in results.items():
        pd.DataFrame(val).to_csv(
            os.path.join(RESULTS_DIR, f"aug_{key}.csv"), index=False
        )
        print(f"  Saved: results/aug_{key}.csv")

    print("✓ Augmentation analysis complete")
    return results


def run_few_shot_evaluation():
    """Run few-shot learning evaluation and save results."""
    print("\n" + "=" * 60)
    print("RUNNING FEW-SHOT LEARNING EVALUATION")
    print("=" * 60)

    results = evaluate_few_shot_performance(
        create_few_shot_cnn,
        TRAIN_DIR,
        VAL_DIR,
        few_shot_configs=[1, 5, 10, 50],
        epochs=EPOCHS_FEW_SHOT,
        batch_size=BATCH_SIZE,
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(results["few_shot"]).to_csv(
        os.path.join(RESULTS_DIR, "few_shot_results.csv"), index=False
    )
    print("  Saved: results/few_shot_results.csv")
    print("✓ Few-shot evaluation complete")
    return results


def run_reduced_dataset_analysis():
    """Run reduced training set size analysis and save learning curve."""
    print("\n" + "=" * 60)
    print("RUNNING REDUCED DATASET ANALYSIS")
    print("=" * 60)

    results = run_reduced_dataset_experiment(
        create_baseline_cnn,
        TRAIN_DIR,
        VAL_DIR,
        fractions=[0.1, 0.25, 0.5, 1.0],
        epochs=EPOCHS_AUGMENTATION,
        batch_size=BATCH_SIZE,
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_reduced_dataset_results(
        results,
        save_path=os.path.join(RESULTS_DIR, "reduced_dataset_curve.png"),
    )
    pd.DataFrame(results).to_csv(
        os.path.join(RESULTS_DIR, "reduced_dataset_results.csv"), index=False
    )
    print("  Saved: results/reduced_dataset_results.csv")
    print("  Saved: results/reduced_dataset_curve.png")
    print("✓ Reduced dataset analysis complete")
    return results


def save_experiment_summary(timestamp, results_map):
    """Write a plain-text summary of the experiment run."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_file = os.path.join(RESULTS_DIR, f"experiment_summary_{timestamp}.txt")

    with open(summary_file, "w") as f:
        f.write("CINIC-10 CNN Experiment Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Completed experiments:\n")
        for name, result in results_map.items():
            status = "✓" if result is not None else "✗ (skipped)"
            f.write(f"  {status} {name}\n")

        f.write("\nResults saved to: results/\n")

    print(f"\nExperiment summary saved to {summary_file}")


def run_comprehensive_experiment():
    """Run the complete experimental pipeline."""
    print("Starting Comprehensive CINIC-10 CNN Experiment")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Experiment timestamp: {timestamp}")

    setup_project_directories()

    results_map = {}

    # 1. Baseline
    baseline_model, baseline_history = run_baseline_experiment()
    results_map["baseline"] = baseline_history

    # 2. Hyperparameter analysis
    hp_results = run_hyperparameter_analysis()
    results_map["hyperparameter_analysis"] = hp_results

    # 3. Augmentation studies
    aug_results = run_augmentation_studies()
    results_map["augmentation_studies"] = aug_results

    # 4. Few-shot evaluation
    few_shot_results = run_few_shot_evaluation()
    results_map["few_shot_evaluation"] = few_shot_results

    # 5. Reduced dataset analysis
    reduced_results = run_reduced_dataset_analysis()
    results_map["reduced_dataset_analysis"] = reduced_results

    save_experiment_summary(timestamp, results_map)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return results_map


def main():
    """Entry point."""
    try:
        print("Starting CINIC-10 CNN Deep Learning Project")
        print("===========================================")
        run_comprehensive_experiment()
        print("\nProject execution completed successfully!")
    except Exception as e:
        print(f"Error during experiment execution: {e}")
        raise


if __name__ == "__main__":
    main()
