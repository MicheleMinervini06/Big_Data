import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from time import time
import pickle
import numpy as np
import mlflow
from src.utils.load_data import *
#from src.data.processing_data import processing_feature
#from src.train.boosting import fit_boosting
#from src.predict.evaluate_predict import evaluate_model
from src.models.autoencoder import Autoencoder
import pandas as pd
import torch
from src.models.imagerffitterinput import ImageEmbeddingExtractor,  ImageEmbeddingExtractorInput
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from src.data.processing_data import *
#from src.train.boosting import training_boosting
#from src.predict.evaluate_predict import evaluate
from src.utils.save_load import *
# legend:dict, local_path:Path = os.path.join(Path(__file__).parent.parent,"data"))

import os
import mlflow

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Experiments-8-fair")

is_Train = True
is_Exp = False

X_mods,X_test,y_train,y_test = processing_feature()


# Lista per raccogliere le metriche di ciascun fold
all_metrics = []


with mlflow.start_run() as run:

    ## Autoencoder --> dati clinici ##
    autoencoder = Autoencoder(input_dim=X_mods["clinical"].shape[1], hidden_dim1=32, bottleneck_dim=16)
    autoencoder.fit(X_mods["clinical"], epochs=50)
    autoencoder.evaluate(X_test["clinical"])

    # Ottieni le embedding dal modello (ritorna un torch.Tensor)
    train_embeddings = autoencoder.encode(X_mods["clinical"])
    test_embeddings = autoencoder.encode(X_test["clinical"])

    # Converto i Tensor in DataFrame Pandas
    train_embeddings_df = pd.DataFrame(train_embeddings.detach().cpu().numpy(), index=X_mods["clinical"].index)
    test_embeddings_df = pd.DataFrame(test_embeddings.detach().cpu().numpy(), index=X_test["clinical"].index)

    # Salvo i DataFrame in CSV e li riletto (facoltativo, per persistentià)
    train_embeddings_df.to_csv("/home/jovyan/veronet_algorithm/src/data/TrainEmbeddings_4.csv", index=True)
    train_embeddings_df = pd.read_csv("/home/jovyan/veronet_algorithm/src/data/TrainEmbeddings_4.csv", index_col=0)
    test_embeddings_df.to_csv("/home/jovyan/veronet_algorithm/src/data/TestEmbeddings_4.csv", index=True)
    test_embeddings_df = pd.read_csv("/home/jovyan/veronet_algorithm/src/data/TestEmbeddings_4.csv", index_col=0)

    print("Embeddings clinical - Train:")
    print(train_embeddings_df.head())
    print("Embeddings clinical - Test:")
    print(test_embeddings_df.head())

    ## CNN --> immagini ##
    inp = ImageEmbeddingExtractorInput(
        temp_file="image_embeddings_4.csv",
        test_file="test_embeddings_4.csv"
    )
    cnn_extractor = ImageEmbeddingExtractor(inp)

    # Calcola embeddings per il training set
    train_embeddings_img = cnn_extractor.extract_embeddings(X_mods["images"])
    # Calcola embeddings per il test set (senza riaddestrare il modello)
    test_embeddings_img = cnn_extractor.extract_test_embeddings(X_test["images"])

    # Leggo i CSV generati dai metodi di estrazione e li setto in base agli indici originali
    train_img = pd.read_csv("image_embeddings.csv", header=None)
    train_img = train_img.set_index(X_mods["images"].index)
    test_img = pd.read_csv("test_embeddings.csv", header=None)
    test_img = test_img.set_index(X_test["images"].index)

    print("Embeddings immagini - Train:")
    print(train_img.head())
    print("Embeddings immagini - Test:")
    print(test_img.head())

    print(f"Dimensioni embeddings immagini Train: {train_img.shape}")
    print(f"Dimensioni embeddings immagini Test: {test_img.shape}")

    ## Concatenazione degli embeddings ##
    concatdf_train = pd.concat([train_embeddings_df, train_img], axis=1)
    concatdf_test = pd.concat([test_embeddings_df, test_img], axis=1)

    print("Concatenated Train DataFrame:")
    print(concatdf_train.head())
    print("Concatenated Test DataFrame:")
    print(concatdf_test.head())

    # --- STEP DI CROSS VALIDATION SUL CLASSIFICATORE --- #
    # Assicurati che le colonne abbiano lo stesso formato (string) per entrambi i DataFrame
    concatdf_train.columns = concatdf_train.columns.astype(str)
    concatdf_test.columns = concatdf_test.columns.astype(str)

    # Impostiamo lo stratified k-fold (qui 5 fold, modificabile in base alle necessità)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold = 1
    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []

    # Liste per accumulare le metriche sui dati di training
    fold_train_accuracies = []
    fold_train_precisions = []
    fold_train_recalls = []
    fold_train_f1s = []

    # La cross validation viene eseguita sulla parte di training: concatdf_train e y_train
    for train_idx, valid_idx in skf.split(concatdf_train, y_train):
        X_train_fold = concatdf_train.iloc[train_idx]
        y_train_fold = y_train.iloc[train_idx]
        X_valid_fold = concatdf_train.iloc[valid_idx]
        y_valid_fold = y_train.iloc[valid_idx]

        # Inizializziamo il classificatore; puoi anche loggare qui i parametri se ne effettui la ricerca
        rfc = RandomForestClassifier(random_state=42)
        rfc.fit(X_train_fold, y_train_fold)

        # Predizione sul fold di validazione
        y_pred_fold = rfc.predict(X_valid_fold)
        report = classification_report(y_valid_fold, y_pred_fold, output_dict=True)
        accuracy = report.get("accuracy")
        precision = report.get("macro avg", {}).get("precision")
        recall = report.get("macro avg", {}).get("recall")
        f1 = report.get("macro avg", {}).get("f1-score")
        fold_accuracies.append(accuracy)
        fold_precisions.append(precision)
        fold_recalls.append(recall)
        fold_f1s.append(f1)

        print(f"Fold {fold} metriche:")
        print(f"  Accuracy: {accuracy}")
        print(f"  Macro Precision: {precision}, Macro Recall: {recall}, Macro F1: {f1}")

        

        # Calcolo delle metriche sui dati di TRAINING per il fold corrente
        y_pred_train = rfc.predict(X_train_fold)
        report_train = classification_report(y_train_fold, y_pred_train, output_dict=True)
        train_accuracy = report_train.get("accuracy")
        train_precision = report_train.get("macro avg", {}).get("precision")
        train_recall = report_train.get("macro avg", {}).get("recall")
        train_f1 = report_train.get("macro avg", {}).get("f1-score")

        # Accumulo metriche training
        fold_train_accuracies.append(train_accuracy)
        fold_train_precisions.append(train_precision)
        fold_train_recalls.append(train_recall)
        fold_train_f1s.append(train_f1)



        fold += 1

    mean_accuracy = np.mean(fold_accuracies)
    mean_precision = np.mean(fold_precisions)
    mean_recall = np.mean(fold_recalls)
    mean_f1 = np.mean(fold_f1s)
    print("Media Cross Validation:")
    print(f"  Accuracy: {mean_accuracy}")
    print(f"  Macro Precision: {mean_precision}")
    print(f"  Macro Recall: {mean_recall}")
    print(f"  Macro F1: {mean_f1}")
    

    # Calcolo delle medie delle metriche sui dati di training
    mean_train_accuracy = np.mean(fold_train_accuracies)
    mean_train_precision = np.mean(fold_train_precisions)
    mean_train_recall = np.mean(fold_train_recalls)
    mean_train_f1 = np.mean(fold_train_f1s)

    print("Media Cross Validation - Training:")
    print(f"  Accuracy: {mean_train_accuracy}")
    print(f"  Macro Precision: {mean_train_precision}")
    print(f"  Macro Recall: {mean_train_recall}")
    print(f"  Macro F1: {mean_train_f1}")

   
 
