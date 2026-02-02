from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(model, X_train, X_test, y_train, y_test, fold, use_mcdo=False, n_mc_samples=25):
    """
    Calcola metriche di performance per train e test senza dipendenze esterne.
    Restituisce un dizionario con accuracy, precision, recall e f1 score per train e test.
    
    Args:
        model: Trained model
        X_train, X_test: Train/test data
        y_train, y_test: Train/test labels
        fold: Fold number
        use_mcdo: Whether to use MC Dropout for uncertainty quantification (TEST ONLY)
        n_mc_samples: Number of MC samples (default: 25)
    """
    # Predizioni su train (sempre senza MC Dropout)
    train_preds = model.predict(X_train)
    train_accuracy = accuracy_score(y_train.sort_index(), train_preds.sort_index())
    train_precision = precision_score(y_train.sort_index(), train_preds.sort_index(), average='macro', zero_division=0)
    train_recall = recall_score(y_train.sort_index(), train_preds.sort_index(), average='macro', zero_division=0)
    train_f1 = f1_score(y_train.sort_index(), train_preds.sort_index(), average='macro', zero_division=0)

    # Log dei risultati su train
    print(f"Fold {fold} - Train Metrics -> Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f},"
          f" Recall: {train_recall:.4f}, F1: {train_f1:.4f}")

    # Predizioni su test (con o senza MC Dropout)
    confidence = None
    epistemic_uncertainty = None
    
    if use_mcdo and hasattr(model, 'predict_with_mcdo'):
        print(f"Using MC Dropout for test inference (n_mc_samples={n_mc_samples})...")
        mcdo_results = model.predict_with_mcdo(X_test, n_mc_samples=n_mc_samples)
        test_preds = mcdo_results['predictions']
        
        # Estrai metriche di incertezza
        confidence = mcdo_results['confidence'].mean()
        epistemic_uncertainty = mcdo_results['epistemic_uncertainty'].mean()
        print(f"Mean confidence: {confidence:.4f}, Mean epistemic uncertainty: {epistemic_uncertainty:.4f}")
    else:
        test_preds = model.predict(X_test)
    
    test_accuracy = accuracy_score(y_test.sort_index(), test_preds.sort_index())
    test_precision = precision_score(y_test.sort_index(), test_preds.sort_index(), average='macro', zero_division=0)
    test_recall = recall_score(y_test.sort_index(), test_preds.sort_index(), average='macro', zero_division=0)
    test_f1 = f1_score(y_test.sort_index(), test_preds.sort_index(), average='macro', zero_division=0)

    # Log dei risultati su test
    mcdo_tag = " (MC Dropout)" if use_mcdo else ""
    print(f"Fold {fold} - Test Metrics{mcdo_tag}  -> Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f},"
          f" Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

    # Costruisci dizionario metriche
    metrics = {
        "train_accuracy": train_accuracy,
        "train_precision": train_precision,
        "train_recall": train_recall,
        "train_f1": train_f1,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
    }
    
    # Aggiungi metriche MC Dropout se disponibili
    if confidence is not None:
        metrics["confidence"] = confidence
    if epistemic_uncertainty is not None:
        metrics["epistemic_uncertainty"] = epistemic_uncertainty
    
    return metrics
