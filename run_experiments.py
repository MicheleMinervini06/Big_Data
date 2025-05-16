from sklearn.ensemble import RandomForestClassifier
import torch
import numpy as np
from src.models.autoencoder import Autoencoder
from src.models.imagerffitterinput import ImageEmbeddingExtractor,  ImageEmbeddingExtractorInput
from sklearn.metrics import classification_report
from src.data.processing_data import processing_features_cv
from src.train.boosting import training_function
from src.predict.evaluate_predict import evaluate
import psutil
import gc
import os
from src.utils.save_load import *
import pandas as pd



def run_boosting(name: str, config: dict):
    print(f"\n=== Running {name} ===")
    is_train = config.get("train", True)
    params = config.get("params", {})
    

    folds = processing_features_cv()
    
    all_metrics = []

    for i, fold in enumerate(folds, start=1):
        X_train, y_train = fold['X_train'], fold['y_train']
        X_test, y_test = fold['X_test'], fold['y_test']
        

        if is_train:
            model = training_function(X_train, y_train, i, params)
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


def run_autoencoder():

    # Genera i folds con dati clinici e immagini
    folds = processing_features_cv()

    # Liste per raccogliere metriche su ogni fold
    # Liste per raccogliere metriche su ogni fold
    tr_accs, tr_prs, tr_recs, tr_f1s = [], [], [], []
    te_accs, te_prs, te_recs, te_f1s = [], [], [], []

    for i, fold in enumerate(folds, start=1):
        # Estrai train/test split per fold
        X_train_clin = fold['X_train']['clinical']
        X_test_clin  = fold['X_test']['clinical']
        X_train_img  = fold['X_train']['images']
        X_test_img   = fold['X_test']['images']
        y_train_fold = fold['y_train']
        y_test_fold  = fold['y_test']

        # --- AUTOENCODER sui dati clinici ---
        ae = Autoencoder(
            input_dim=X_train_clin.shape[1], hidden_dim1=32, bottleneck_dim=16
        )
        ae.fit(X_train_clin, epochs=50)
        # Embedding clinici
        tr_emb_clin = ae.encode(X_train_clin).detach().cpu().numpy()
        te_emb_clin = ae.encode(X_test_clin).detach().cpu().numpy()
        df_tr_clin = pd.DataFrame(tr_emb_clin, index=X_train_clin.index)
        df_te_clin = pd.DataFrame(te_emb_clin, index=X_test_clin.index)

        print("Embeddings clinical - Train:")
        print(df_tr_clin.head())
        print("Embeddings clinical - Test:")
        print(df_te_clin.head())

        # --- CNN per embeddings immagini ---
        # File temporanei distinti per fold
        tmp_file = f"src/data/embed/fold{i}_img_emb.csv"
        tst_file = f"src/data/embed/fold{i}_test_img_emb.csv"
        inp = ImageEmbeddingExtractorInput(
            temp_file=tmp_file, test_file=tst_file
        )
        cnn_ext = ImageEmbeddingExtractor(inp)
        cnn_ext.extract_embeddings(X_train_img)
        cnn_ext.extract_test_embeddings(X_test_img)
        # Leggi e indicizza
        df_tr_img = pd.read_csv(tmp_file, header=None)
        df_tr_img.index = X_train_img.index
        df_te_img = pd.read_csv(tst_file, header=None)
        df_te_img.index = X_test_img.index

        print("Embeddings immagini - Train:")
        print(df_tr_img.head())
        print("Embeddings immagini - Test:")
        print(df_te_img.head())

        # --- Concatenazione embeddings clinici + immagini ---
        X_tr_comb = pd.concat([df_tr_clin, df_tr_img], axis=1)
        X_te_comb = pd.concat([df_te_clin, df_te_img], axis=1)

        print("Concatenated Train DataFrame:")
        print(X_tr_comb.head())
        print("Concatenated Test DataFrame:")
        print(X_te_comb.head())

        X_tr_comb.columns = X_tr_comb.columns.astype(str)
        X_te_comb.columns = X_te_comb.columns.astype(str)

        # --- Classificatore e valutazione sul fold ---
        rfc = RandomForestClassifier(random_state=42)
        rfc.fit(X_tr_comb, y_train_fold)
        # Metriche su train
        y_tr_pred = rfc.predict(X_tr_comb)
        rep_tr = classification_report(y_train_fold, y_tr_pred, output_dict=True)
        acc_tr = rep_tr['accuracy']
        pr_tr  = rep_tr['macro avg']['precision']
        rc_tr  = rep_tr['macro avg']['recall']
        f1_tr  = rep_tr['macro avg']['f1-score']
        tr_accs.append(acc_tr)
        tr_prs.append(pr_tr)
        tr_recs.append(rc_tr)
        tr_f1s.append(f1_tr)

        # Metriche su test
        y_te_pred = rfc.predict(X_te_comb)
        rep_te = classification_report(y_test_fold, y_te_pred, output_dict=True)
        acc_te = rep_te['accuracy']
        pr_te  = rep_te['macro avg']['precision']
        rc_te  = rep_te['macro avg']['recall']
        f1_te  = rep_te['macro avg']['f1-score']
        te_accs.append(acc_te)
        te_prs.append(pr_te)
        te_recs.append(rc_te)
        te_f1s.append(f1_te)

        print(f"Fold {i} - Train: Acc={acc_tr:.4f}, Prec={pr_tr:.4f}, Rec={rc_tr:.4f}, F1={f1_tr:.4f}")
        print(f"Fold {i} - Test : Acc={acc_te:.4f}, Prec={pr_te:.4f}, Rec={rc_te:.4f}, F1={f1_te:.4f}")

    # Calcola e stampa le medie
    print("\n=== Performance medie su tutti i fold ===")
    print(f"Train - Accuracy: {np.mean(tr_accs):.4f}, Precision: {np.mean(tr_prs):.4f}, Recall: {np.mean(tr_recs):.4f}, F1: {np.mean(tr_f1s):.4f}")
    print(f"Test  - Accuracy: {np.mean(te_accs):.4f}, Precision: {np.mean(te_prs):.4f}, Recall: {np.mean(te_recs):.4f}, F1: {np.mean(te_f1s):.4f}")

