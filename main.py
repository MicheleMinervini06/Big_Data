import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from time import time
import argparse
import yaml
import numpy as np

from src.data.processing_data import processing_features,processing_features_cv
from src.train.boosting import training_function
from src.predict.evaluate_predict import evaluate
from src.utils.save_load import *
import psutil
import gc
import os

CONFIG_FILE = Path("src/configs/config.yaml")


def run_experiment(name: str, config: dict):
    print(f"\n=== Running {name} ===")
    is_train = config.get("train", True)
    params = config.get("params", {})
    

    folds = processing_features_cv(**params)
    #folds = None
    all_metrics = []

    for i, fold in enumerate(folds, start=1):
        X_train, y_train = fold['X_train'], fold['y_train']
        X_test, y_test = fold['X_test'], fold['y_test']

        if is_train:
            model = training_function(X_train, y_train, fold_index=i, **params)
        else:
            model = load_model(model_name=f"{name}_fold_{i}")

        metrics = evaluate(model, X_train, X_test, y_train, y_test, fold=i)
        all_metrics.append(metrics)

        if is_train:
            save_model(model, model_name=f"{name}_fold_{i}")

        # Log memoria prima della pulizia
        if torch.cuda.is_available():
            print(f"Fold {i} - PRE Cleanup - CUDA Allocated: {torch.cuda.memory_allocated()/(1024*1024):.2f} MB")

        # Pulizia: sposta su CPU, libera memoria e cache
        try:
            model.cpu()
        except Exception:
            pass
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Log memoria dopo la pulizia
        process = psutil.Process(os.getpid())
        cpu_mem = process.memory_info().rss / (1024 * 1024)
        print(f"Fold {i} - Post Cleanup - CPU RAM: {cpu_mem:.2f} MB")
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated()/(1024*1024)
            reserved = torch.cuda.memory_reserved()/(1024*1024)
            print(f"    CUDA Allocated: {alloc:.2f} MB, Reserved: {reserved:.2f} MB")

    # Calcolo e stampa delle metriche medie
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    print("\nAverage metrics across folds:")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")


def parse_args():
    p = argparse.ArgumentParser(description="Run multiple experiments.")
    p.add_argument("--experiments", nargs="*", default=None,
                   help="Lista di esperimenti da eseguire; se non specificato, tutti.")
    p.add_argument("--config", type=str,
                   help="Path a file YAML con definizione esperimenti.")
    return p.parse_args()


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single experiment.")
    parser.add_argument("experiment",
                        help="Nome dell'esperimento da eseguire, definito in config.yaml")
    return parser.parse_args()


def main():
    if not CONFIG_FILE.exists():
        raise FileNotFoundError("Il file config.yaml non è stato trovato. Aggiungilo nella root del progetto.")

    experiments = yaml.safe_load(CONFIG_FILE.read_text())

    args = parse_args()
    name = args.experiment
    if name not in experiments:
        raise KeyError(f"Esperimento '{name}' non definito in config.yaml.")

    run_experiment(name, experiments[name])


if __name__ == "__main__":
    main()