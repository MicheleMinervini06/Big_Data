#!/usr/bin/env python3
"""
Demo Clinica Semplice: Utilizzo del Modello con Conformal Prediction

Questo script dimostra come utilizzare il modello addestrato con Conformal Prediction
in un contesto clinico reale, utilizzando dati di test reali del modello.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Aggiungi il path del progetto
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.utils.save_load import load_model
from src.data.processing_data import processing_features_cv_with_calibration


def main():
    """
    Demo principale utilizzando dati di test reali.
    """
    print("=" * 80)
    print("DEMO CLINICA: Sistema di Predizione ADNI con Conformal Prediction")
    print("=" * 80)
    print()

    try:
        # 1. Carica il modello addestrato
        print("[1/4] Caricamento modello (exp12_fold_1)...")
        model = load_model("exp12_fold_1")
        print("      Modello caricato con successo\n")

        # 2. Carica dati di test reali dal fold 1
        print("[2/4] Caricamento dati di test dal fold 1...")
        folds = processing_features_cv_with_calibration(use_augmented=False)
        fold_data = folds[0]  # Fold 1

        X_test = fold_data['X_test']
        y_test = fold_data['y_test']
        X_calib = fold_data['X_calib']
        y_calib = fold_data['y_calib']

        print(f"      Dati caricati: {len(y_test)} test, {len(y_calib)} calibrazione")
        print(f"      Features: {X_test['clinical'].shape[1]} cliniche, {len(X_test['images'])} immagini")
        print()

        # 3. Trova indici comuni tra clinical e images
        common_indices = sorted(set(X_test['clinical'].index) & set(X_test['images'].index) if len(X_test['images']) > 0 else X_test['clinical'].index)
        print(f"      Pazienti con dati completi: {len(common_indices)}")

        # Seleziona un campione per trovare casi interessanti
        print("\n[3/4] Analisi prediction sets su campione...")
        np.random.seed(42)
        sample_size = min(50, len(common_indices))  # Campione più grande per trovare diversi casi
        sample_indices = np.random.choice(common_indices, size=sample_size, replace=False)

        # Filtra dati campione
        X_sample = {
            'clinical': X_test['clinical'].loc[sample_indices],
            'images': X_test['images'].loc[sample_indices] if len(X_test['images']) > 0 else pd.DataFrame()
        }
        y_sample = y_test.loc[sample_indices]

        # Fa predizioni sul campione per analizzare i prediction sets
        # Riduci output del modello catturando stdout
        import io
        import sys
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            sample_results = model.predict_with_conformal(
                X_test=X_sample,
                X_calib=X_calib,
                y_calib=y_calib,
                alpha=0.1
            )
        captured_output = f.getvalue()

        # Estrai solo le statistiche essenziali
        lines = captured_output.split('\n')
        for line in lines:
            if 'Singleton sets' in line or 'Mean set size' in line:
                print(f"   {line.strip()}")
        print()

        # Trova pazienti con singleton e double prediction sets
        singleton_indices = []
        double_indices = []

        for i, pred_set in enumerate(sample_results['prediction_sets']):
            if len(pred_set) == 1:
                singleton_indices.append(sample_indices[i])
            elif len(pred_set) == 2:
                double_indices.append(sample_indices[i])

        print(f"      Singleton (sicuri): {len(singleton_indices)} pazienti")
        print(f"      Double (incerti): {len(double_indices)} pazienti")

        # Seleziona 2 singleton e 2 double (o quanti disponibili)
        selected_singleton = singleton_indices[:2] if len(singleton_indices) >= 2 else singleton_indices
        selected_double = double_indices[:2] if len(double_indices) >= 2 else double_indices[:min(2, len(double_indices))]

        selected_from_common = selected_singleton + selected_double
        print(f"\n[4/4] Demo su {len(selected_from_common)} pazienti ({len(selected_singleton)} sicuri + {len(selected_double)} incerti)")
        print("=" * 80)
        print()

        # Usa i risultati del campione invece di rifare le predizioni
        # Crea mapping degli indici selezionati ai risultati del campione
        selected_results = []
        for patient_idx in selected_from_common:
            # Trova l'indice nel campione originale
            sample_pos = np.where(sample_indices == patient_idx)[0][0]
            selected_results.append({
                'prediction': sample_results['predictions'].iloc[sample_pos],
                'confidence': sample_results['confidence'].iloc[sample_pos],
                'prediction_set': sample_results['prediction_sets'][sample_pos],
                'conformal_uncertainty': sample_results['conformal_uncertainty'].iloc[sample_pos]
            })

        # 4. Mostra risultati predizioni
        class_names = {1: "CN", 2: "MCI", 3: "AD"}

        for i, patient_idx in enumerate(selected_from_common):
            print(f"PAZIENTE {i+1}:")
            
            pred_class = selected_results[i]['prediction']
            confidence = selected_results[i]['confidence']
            pred_set = selected_results[i]['prediction_set']

            print(f"  Diagnosi:   {class_names[pred_class]}")
            print(f"  Confidenza: {confidence:.1%}")
            print()

            # Interpretazione clinica basata su prediction set
            if len(pred_set) == 1:
                print("  >> PREDIZIONE SICURA")
                print("     Il modello è confidente nella diagnosi.")
            else:
                print("  >> CONFIDENZA SULLA SICUREZZA BASSA")
                other_classes = [class_names[c] for c in pred_set if c != pred_class]
                print(f"     La classe potrebbe essere anche: {', '.join(other_classes)}")
                print("     Trattare il caso con maggiore attenzione.")
            
            print()

        print("=" * 80)
        print("Demo completata.")
        print("Conformal Prediction garantisce 90% di coverage sulle predizioni.")
        print("=" * 80)

    except Exception as e:
        print(f"\nERRORE: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()