from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(model, X_train, X_test, y_train, y_test, fold):
    """
    Calcola metriche di performance per train e test senza dipendenze esterne.
    Restituisce un dizionario con accuracy, precision, recall e f1 score per train e test.
    """
    # Predizioni su train
    train_preds = model.predict(X_train)
    train_accuracy = accuracy_score(y_train.sort_index(), train_preds.sort_index())
    train_precision = precision_score(y_train.sort_index(), train_preds.sort_index(), average='macro', zero_division=0)
    train_recall = recall_score(y_train.sort_index(), train_preds.sort_index(), average='macro', zero_division=0)
    train_f1 = f1_score(y_train.sort_index(), train_preds.sort_index(), average='macro', zero_division=0)

    # Log dei risultati su train
    print(f"Fold {fold} - Train Metrics -> Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f},"
          f" Recall: {train_recall:.4f}, F1: {train_f1:.4f}")

    # Predizioni su test
    test_preds = model.predict(X_test)
    test_accuracy = accuracy_score(y_test.sort_index(), test_preds.sort_index())
    test_precision = precision_score(y_test.sort_index(), test_preds.sort_index(), average='macro', zero_division=0)
    test_recall = recall_score(y_test.sort_index(), test_preds.sort_index(), average='macro', zero_division=0)
    test_f1 = f1_score(y_test.sort_index(), test_preds.sort_index(), average='macro', zero_division=0)

    # Log dei risultati su test
    print(f"Fold {fold} - Test Metrics  -> Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f},"
          f" Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

    # Costruisci dizionario metriche
    return {
        "train_accuracy": train_accuracy,
        "train_precision": train_precision,
        "train_recall": train_recall,
        "train_f1": train_f1,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
    }
