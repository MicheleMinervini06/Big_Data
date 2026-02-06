from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from scipy.stats import spearmanr, pearsonr

def evaluate(model, X_train, X_test, y_train, y_test, fold, X_calib=None, y_calib=None, use_mcdo=False, use_tta=False, use_conformal=False, n_mc_samples=25, n_tta_samples=10, conformal_alpha=0.1):
    """
    Calcola metriche di performance per train e test senza dipendenze esterne.
    Restituisce un dizionario con accuracy, precision, recall e f1 score per train e test.
    
    Args:
        model: Trained model
        X_train, X_test: Train/test data
        y_train, y_test: Train/test labels
        fold: Fold number
        use_mcdo: Whether to use MC Dropout for epistemic uncertainty (TEST ONLY)
        use_tta: Whether to use TTA for aleatoric uncertainty (TEST ONLY)
        n_mc_samples: Number of MC samples (default: 25)
        n_tta_samples: Number of TTA samples (default: 10)
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
    
    # Stampa i pesi alpha delle modalità (ResNet vs RF)
    if hasattr(model, 'modality_weights'):
        print(f"\n{'='*60}")
        print(f"Modality Weights (Alpha) - Fold {fold}")
        print(f"{'='*60}")
        mod_weights = model.modality_weights()
        for modality, row in mod_weights.iterrows():
            alpha_pct = row['alpha'] * 100
            print(f"  {modality:>10s}: {row['alpha']:.6f} ({alpha_pct:>6.2f}%)")
        print(f"{'='*60}\n")

    # Predizioni su test (con o senza MC Dropout)
    confidence = None
    epistemic_uncertainty = None
    unc_correlation_spearman = None
    unc_correlation_pearson = None
    acc_at_rejection_0 = None
    acc_at_rejection_10 = None
    acc_at_rejection_20 = None
    alea_correlation_spearman = None
    alea_correlation_pearson = None
    conformal_uncertainty = None
    conformal_coverage = None
    conf_correlation_spearman = None
    conf_correlation_pearson = None
    
    if use_mcdo and hasattr(model, 'predict_with_mcdo'):
        print(f"Using MC Dropout for test inference (n_mc_samples={n_mc_samples})...")
        mcdo_results = model.predict_with_mcdo(X_test, n_mc_samples=n_mc_samples)
        test_preds = mcdo_results['predictions']
        
        # Estrai metriche di incertezza
        confidence = mcdo_results['confidence'].mean()
        epistemic_uncertainty = mcdo_results['epistemic_uncertainty'].mean()
        print(f"Mean confidence: {confidence:.4f}, Mean epistemic uncertainty: {epistemic_uncertainty:.4f}")
        
        # Identifica top-3 pazienti con maggiore incertezza epistemica
        top_k = 3
        unc_series = mcdo_results['epistemic_uncertainty']
        top_uncertain = unc_series.nlargest(top_k)
        
        print(f"\n🔍 Top-{top_k} pazienti con maggiore incertezza epistemica:")
        print("=" * 80)
        for rank, (patient_id, unc_value) in enumerate(top_uncertain.items(), 1):
            pred_class = mcdo_results['predictions'][patient_id]
            conf_value = mcdo_results['confidence'][patient_id]
            true_class = y_test.loc[patient_id] if patient_id in y_test.index else "N/A"
            correct = "✓" if pred_class == true_class else "✗"
            
            print(f"  #{rank} Patient ID: {patient_id}")
            print(f"      Prediction: {pred_class} | True Label: {true_class} {correct}")
            print(f"      Confidence: {conf_value:.4f} | Epistemic Uncertainty: {unc_value:.4f}")
            print()
        print("=" * 80 + "\n")
        
        #Accuracy-vs-Rejection curve
        print("Validazione Uncertainty Quantification")
        print("=" * 80)
        
        # Allinea predizioni, ground truth e incertezze
        aligned_preds = mcdo_results['predictions'].sort_index()
        aligned_true = y_test.sort_index()
        aligned_unc = mcdo_results['epistemic_uncertainty'].sort_index()
        aligned_conf = mcdo_results['confidence'].sort_index()
        
        # Rimuovi eventuali indici non allineati
        common_idx = aligned_preds.index.intersection(aligned_true.index)
        aligned_preds = aligned_preds[common_idx]
        aligned_true = aligned_true[common_idx]
        aligned_unc = aligned_unc[common_idx]
        aligned_conf = aligned_conf[common_idx]
        
        print(f"\n📊 Analisi UQ su {len(aligned_preds)} campioni del test set\n")
        
        # Ordina campioni per incertezza epistemica (dal più incerto)
        sorted_indices = aligned_unc.sort_values(ascending=False).index
        
        # Calcola accuracy-vs-rejection curve
        rejection_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
        print("\nAccuracy-vs-Rejection Curve:")
        print(f"    {'Rejection %':<15} {'Accuracy':<12} {'Samples Kept':<15}")
        print("    " + "-" * 42)
        
        acc_vs_rejection = {}
        for reject_rate in rejection_rates:
            n_reject = int(len(sorted_indices) * reject_rate)
            kept_indices = sorted_indices[n_reject:]  # Rimuovi i più incerti
            
            preds_kept = aligned_preds[kept_indices]
            true_kept = aligned_true[kept_indices]
            
            acc_kept = accuracy_score(true_kept, preds_kept)
            acc_vs_rejection[reject_rate] = acc_kept
            
            print(f"    {reject_rate*100:>6.0f}%           {acc_kept:>6.4f}       {len(kept_indices):>4}/{len(sorted_indices)}")
        
        # Calcola miglioramento accuracy
        acc_baseline = acc_vs_rejection[0.0]
        acc_at_20 = acc_vs_rejection[0.20]
        improvement = acc_at_20 - acc_baseline
        print(f"\n    ✅ Miglioramento accuracy @ 20% rejection: {improvement:+.4f}")
        if improvement >= 0.05:
            print(f"    🎯 OTTIMO! UQ molto efficace per identificare predizioni dubbie.")
        elif improvement >= 0.02:
            print(f"    ✓ BUONO! UQ utile per filtrare casi incerti.")
        else:
            print(f"    ⚠️  LIMITATO: UQ potrebbe non essere molto discriminativa.")
        
        
        #Correlazione epistemic uncertainty vs errori
        print("\nCorrelazione Uncertainty/Confidence vs Errori:")
        
        # Indicatore di errore binario (1 = errore, 0 = corretto)
        error_indicator = (aligned_preds != aligned_true).astype(int)
        
        # Calcola correlazioni per EPISTEMIC UNCERTAINTY (dovrebbe essere POSITIVA)
        spearman_corr_unc, spearman_p_unc = spearmanr(aligned_unc, error_indicator)
        pearson_corr_unc, pearson_p_unc = pearsonr(aligned_unc, error_indicator)
        
        print(f"\n  📈 Epistemic Uncertainty vs Errori (alta unc → più errori):")
        print(f"    Spearman ρ:  {spearman_corr_unc:>6.4f}  (p={spearman_p_unc:.4f})")
        print(f"    Pearson r:   {pearson_corr_unc:>6.4f}  (p={pearson_p_unc:.4f})")
        
        # Calcola correlazioni per CONFIDENCE (dovrebbe essere NEGATIVA)
        spearman_corr_conf, spearman_p_conf = spearmanr(aligned_conf, error_indicator)
        pearson_corr_conf, pearson_p_conf = pearsonr(aligned_conf, error_indicator)
        
        print(f"\n  📉 Confidence vs Errori (alta conf → meno errori):")
        print(f"    Spearman ρ:  {spearman_corr_conf:>6.4f}  (p={spearman_p_conf:.4f})")
        print(f"    Pearson r:   {pearson_corr_conf:>6.4f}  (p={pearson_p_conf:.4f})")
        
        # Valutazione complessiva (usa il meglio tra uncertainty e confidence)
        abs_corr_unc = abs(spearman_corr_unc)
        abs_corr_conf = abs(spearman_corr_conf)
        best_metric = "Epistemic Uncertainty" if abs_corr_unc > abs_corr_conf else "Confidence"
        best_corr = spearman_corr_unc if abs_corr_unc > abs_corr_conf else spearman_corr_conf
        
        print(f"\n  🎯 Metrica migliore: {best_metric} (|ρ|={abs(best_corr):.4f})")
        
        if abs(best_corr) >= 0.30:
            print(f"    ✅ ECCELLENTE! Correlazione forte - UQ affidabile per applicazioni cliniche.")
        elif abs(best_corr) >= 0.20:
            print(f"    ✓ BUONO! UQ correlata con errori, utile per decision support.")
        elif abs(best_corr) >= 0.10:
            print(f"    ⚠️  MODERATO: UQ mostra correlazione debole.")
        else:
            print(f"    ❌ BASSO: Considerare dropout rate maggiore o metric diversa.")
        
        print("=" * 80 + "\n")
        
        # Salva metriche UQ in variabili temporanee
        unc_correlation_spearman = spearman_corr_unc
        unc_correlation_pearson = pearson_corr_unc
        acc_at_rejection_0 = acc_vs_rejection[0.0]
        acc_at_rejection_10 = acc_vs_rejection[0.10]
        acc_at_rejection_20 = acc_vs_rejection[0.20]
        
    # TTA for Aleatoric Uncertainty (data uncertainty)
    aleatoric_uncertainty = None
    if use_tta and hasattr(model, 'predict_with_tta'):
        print(f"\n{'='*80}")
        print(f"Test-Time Augmentation (TTA) for Aleatoric Uncertainty")
        print(f"{'='*80}\n")
        print(f"Applying TTA on CLINICAL data (n_tta_samples={n_tta_samples})...")
        
        tta_results = model.predict_with_tta(X_test, n_tta_samples=n_tta_samples)
        
        # If MCDO was not used, use TTA predictions
        if not use_mcdo:
            test_preds = tta_results['predictions']
            confidence = tta_results['confidence'].mean()
        
        aleatoric_uncertainty = tta_results['aleatoric_uncertainty'].mean()
        print(f"Mean aleatoric uncertainty: {aleatoric_uncertainty:.4f}")
        
        # Top-3 patients with highest aleatoric uncertainty
        top_k_alea = 3
        alea_series = tta_results['aleatoric_uncertainty']
        top_alea = alea_series.nlargest(top_k_alea)
        
        print(f"\n🔍 Top-{top_k_alea} pazienti con maggiore incertezza aleatoric (data):")
        print("=" * 80)
        for rank, (patient_id, alea_value) in enumerate(top_alea.items(), 1):
            pred_class = tta_results['predictions'][patient_id]
            conf_value = tta_results['confidence'][patient_id]
            true_class = y_test.loc[patient_id] if patient_id in y_test.index else "N/A"
            correct = "✓" if pred_class == true_class else "✗"
            
            print(f"  #{rank} Patient ID: {patient_id}")
            print(f"      Prediction: {pred_class} | True Label: {true_class} {correct}")
            print(f"      Confidence: {conf_value:.4f} | Aleatoric Uncertainty: {alea_value:.4f}")
            print()
        print("=" * 80 + "\n")
        
        # Compute correlation between aleatoric uncertainty and errors
        print(f"📊 Validation Metrics for Aleatoric Uncertainty (TTA)")
        print("=" * 80)
        
        # Align predictions, true labels, and aleatoric uncertainty
        common_idx = tta_results['predictions'].index.intersection(y_test.index)
        aligned_pred_alea = tta_results['predictions'][common_idx]
        aligned_true_alea = y_test[common_idx]
        aligned_alea = tta_results['aleatoric_uncertainty'][common_idx]
        aligned_conf_alea = tta_results['confidence'][common_idx]
        
        # Error indicator (1 if wrong, 0 if correct)
        error_indicator_alea = (aligned_pred_alea != aligned_true_alea).astype(int)
        
        # Calcola correlazioni per ALEATORIC UNCERTAINTY (dovrebbe essere POSITIVA)
        spearman_corr_alea, spearman_p_alea = spearmanr(aligned_alea, error_indicator_alea)
        pearson_corr_alea, pearson_p_alea = pearsonr(aligned_alea, error_indicator_alea)
        
        print(f"\n  📈 Aleatoric Uncertainty vs Errori (alta alea → più errori):")
        print(f"    Spearman ρ:  {spearman_corr_alea:>6.4f}  (p={spearman_p_alea:.4f})")
        print(f"    Pearson r:   {pearson_corr_alea:>6.4f}  (p={pearson_p_alea:.4f})")
        
        # Calcola correlazioni per CONFIDENCE (dovrebbe essere NEGATIVA)
        spearman_corr_conf_alea, spearman_p_conf_alea = spearmanr(aligned_conf_alea, error_indicator_alea)
        pearson_corr_conf_alea, pearson_p_conf_alea = pearsonr(aligned_conf_alea, error_indicator_alea)
        
        print(f"\n  📉 Confidence vs Errori (alta conf → meno errori):")
        print(f"    Spearman ρ:  {spearman_corr_conf_alea:>6.4f}  (p={spearman_p_conf_alea:.4f})")
        print(f"    Pearson r:   {pearson_corr_conf_alea:>6.4f}  (p={pearson_p_conf_alea:.4f})")
        
        # Valutazione complessiva (usa il meglio tra uncertainty e confidence)
        abs_corr_alea = abs(spearman_corr_alea)
        abs_corr_conf_alea = abs(spearman_corr_conf_alea)
        best_metric_alea = "Aleatoric Uncertainty" if abs_corr_alea > abs_corr_conf_alea else "Confidence"
        best_corr_alea = spearman_corr_alea if abs_corr_alea > abs_corr_conf_alea else spearman_corr_conf_alea
        
        print(f"\n  🎯 Metrica migliore: {best_metric_alea} (|ρ|={abs(best_corr_alea):.4f})")
        
        if abs(best_corr_alea) >= 0.30:
            print(f"    ✅ ECCELLENTE! Correlazione forte - Aleatoric UQ affidabile.")
        elif abs(best_corr_alea) >= 0.20:
            print(f"    ✓ BUONO! Aleatoric UQ correlata con errori.")
        elif abs(best_corr_alea) >= 0.10:
            print(f"    ⚠️  MODERATO: Aleatoric UQ mostra correlazione debole.")
        else:
            print(f"    ❌ BASSO: Rumore clinico potrebbe non essere dominante.")
        
        print("=" * 80 + "\n")
        
        # Salva metriche TTA
        alea_correlation_spearman = spearman_corr_alea
        alea_correlation_pearson = pearson_corr_alea
    
    # Conformal Prediction for guaranteed uncertainty quantification
    if use_conformal and X_calib is not None and y_calib is not None:
        print(f"\n{'='*80}")
        print(f"Conformal Prediction (Guaranteed Coverage)")
        print(f"{'='*80}\n")
        
        conf_results = model.predict_with_conformal(
            X_test, X_calib, y_calib, alpha=conformal_alpha
        )
        
        # If MCDO/TTA not used, use conformal predictions
        if not use_mcdo and not use_tta:
            test_preds = conf_results['predictions']
            confidence = conf_results['confidence'].mean()
        
        conformal_uncertainty = conf_results['conformal_uncertainty'].mean()
        conformal_coverage = sum(y_test[idx] in conf_results['prediction_sets'][i] 
                                for i, idx in enumerate(y_test.index)) / len(y_test)
        
        print(f"Mean conformal uncertainty: {conformal_uncertainty:.4f}")
        print(f"Empirical coverage: {conformal_coverage:.2%} (target: {(1-conformal_alpha):.2%})")
        

        # Confronto Accuracy/F1 tra Singleton vs Multi-class Prediction Sets
        print(f"📊 Confronto Performance: Singleton vs Multi-class Prediction Sets")
        print("="*80)

        # Analizza prediction sets
        singleton_indices = []
        multiclass_indices = []

        for i, idx in enumerate(conf_results['predictions'].index):
            pred_set = conf_results['prediction_sets'][i]
            if len(pred_set) == 1:
                singleton_indices.append(idx)
            else:
                multiclass_indices.append(idx)

        print(f"\nDistribuzione Prediction Sets:")
        print(f"  Singleton sets (1 classe): {len(singleton_indices)} ({100*len(singleton_indices)/len(conf_results['predictions']):.1f}%)")
        print(f"  Multi-class sets (2+ classi): {len(multiclass_indices)} ({100*len(multiclass_indices)/len(conf_results['predictions']):.1f}%)")

        if len(singleton_indices) > 0 and len(multiclass_indices) > 0:
            # Metriche per singleton
            preds_singleton = conf_results['predictions'][singleton_indices]
            true_singleton = y_test[singleton_indices]
            acc_singleton = accuracy_score(true_singleton, preds_singleton)
            f1_singleton = f1_score(true_singleton, preds_singleton, average='macro', zero_division=0)

            # Metriche per multi-class
            preds_multiclass = conf_results['predictions'][multiclass_indices]
            true_multiclass = y_test[multiclass_indices]
            acc_multiclass = accuracy_score(true_multiclass, preds_multiclass)
            f1_multiclass = f1_score(true_multiclass, preds_multiclass, average='macro', zero_division=0)

            # Confronto
            delta_acc = acc_singleton - acc_multiclass
            delta_f1 = f1_singleton - f1_multiclass

            print(f"\nConfronto Metriche:")
            print("┌" + "─"*78 + "┐")
            print(f"│ {'Metrica':<25} │ {'Singleton':>12} │ {'Multi-class':>14} │ {'Δ':>10} │")
            print("├" + "─"*78 + "┤")
            print(f"│ {'Accuracy':<25} │ {acc_singleton:>12.4f} │ {acc_multiclass:>14.4f} │ {delta_acc:>+10.4f} │")
            print(f"│ {'F1-macro':<25} │ {f1_singleton:>12.4f} │ {f1_multiclass:>14.4f} │ {delta_f1:>+10.4f} │")
            print("└" + "─"*78 + "┘")

            # Interpretazione
            print(f"\n🔍 Interpretazione:")
            if delta_acc > 0.05:
                print(f"  ✅ Singleton ACCURACY {delta_acc:+.1%} superiore")
                print(f"     → Modello più sicuro per predizioni singleton!")
            elif delta_acc < -0.05:
                print(f"  ⚠️  Multi-class ACCURACY {-delta_acc:+.1%} superiore")
                print(f"     → Incertezza conformal non correlata con errori")
            else:
                print(f"  ≈  Accuracy simile (Δ={delta_acc:+.1%})")

            if delta_f1 > 0.05:
                print(f"  ✅ Singleton F1 {delta_f1:+.1%} superiore")
                print(f"     → Modello più preciso per predizioni singleton!")
            elif delta_f1 < -0.05:
                print(f"  ⚠️  Multi-class F1 {-delta_f1:+.1%} superiore")
            else:
                print(f"  ≈  F1 simile (Δ={delta_f1:+.1%})")

            # Coverage garantita
            coverage_singleton = sum(true_singleton[idx] in conf_results['prediction_sets'][list(conf_results['predictions'].index).index(idx)]
                                   for idx in singleton_indices) / len(singleton_indices)
            coverage_multiclass = sum(true_multiclass[idx] in conf_results['prediction_sets'][list(conf_results['predictions'].index).index(idx)]
                                    for idx in multiclass_indices) / len(multiclass_indices)

            print(f"\n📈 Coverage garantita:")
            print(f"  Singleton: {coverage_singleton:.1%} (target: {(1-conformal_alpha):.1%})")
            print(f"  Multi-class: {coverage_multiclass:.1%} (target: {(1-conformal_alpha):.1%})")

        print("="*80 + "\n")

        # Compute correlation between conformal uncertainty and errors
        print(f"📊 Validation Metrics for Conformal Uncertainty")
        print("="*80)
        
        common_idx = conf_results['predictions'].index.intersection(y_test.index)
        aligned_pred_conf = conf_results['predictions'][common_idx]
        aligned_true_conf = y_test[common_idx]
        aligned_conf_unc = conf_results['conformal_uncertainty'][common_idx]
        
        error_indicator_conf = (aligned_pred_conf != aligned_true_conf).astype(int)
        
        spearman_corr_conf, spearman_p_conf = spearmanr(aligned_conf_unc, error_indicator_conf)
        pearson_corr_conf, pearson_p_conf = pearsonr(aligned_conf_unc, error_indicator_conf)
        
        print(f"\n  📈 Conformal Uncertainty vs Errori (alta unc → più errori):")
        print(f"    Spearman ρ:  {spearman_corr_conf:>6.4f}  (p={spearman_p_conf:.4f})")
        print(f"    Pearson r:   {pearson_corr_conf:>6.4f}  (p={pearson_p_conf:.4f})")
        
        if abs(spearman_corr_conf) >= 0.30:
            print(f"    ✅ ECCELLENTE! Conformal UQ matematicamente garantita.")
        elif abs(spearman_corr_conf) >= 0.20:
            print(f"    ✓ BUONO! Conformal UQ correlata con errori.")
        else:
            print(f"    ⚠️ Coverage garantita: {conformal_coverage:.1%}")
        
        print("="*80 + "\n")
        
        conf_correlation_spearman = spearman_corr_conf
        conf_correlation_pearson = pearson_corr_conf
        
        # If both MCDO and TTA are enabled, compute total uncertainty
        if use_mcdo and epistemic_uncertainty is not None:
            total_uncertainty = np.sqrt(
                mcdo_results['epistemic_uncertainty']**2 + 
                tta_results['aleatoric_uncertainty']**2
            ).mean()
            print(f"📊 Uncertainty Decomposition:")
            print(f"  Epistemic (model):  {epistemic_uncertainty:.4f}")
            print(f"  Aleatoric (data):   {aleatoric_uncertainty:.4f}")
            print(f"  Total (combined):   {total_uncertainty:.4f}")
            print(f"  Ratio (epist/alea): {epistemic_uncertainty/aleatoric_uncertainty if aleatoric_uncertainty > 0 else float('inf'):.2f}")
            print("=" * 80 + "\n")
        
    if not use_mcdo and not use_tta:
        test_preds = model.predict(X_test)
    
    test_accuracy = accuracy_score(y_test.sort_index(), test_preds.sort_index())
    test_precision = precision_score(y_test.sort_index(), test_preds.sort_index(), average='macro', zero_division=0)
    test_recall = recall_score(y_test.sort_index(), test_preds.sort_index(), average='macro', zero_division=0)
    test_f1 = f1_score(y_test.sort_index(), test_preds.sort_index(), average='macro', zero_division=0)

    # Log dei risultati su test
    mcdo_tag = " (MC Dropout)" if use_mcdo else ""
    print(f"Fold {fold} - Test Metrics{mcdo_tag}  -> Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f},"
          f" Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

    # Confronto metriche CON vs SENZA immagini
    if use_mcdo and 'images' in X_test and len(X_test['images']) > 0:
        print("\n" + "="*80)
        print("Confronto Campioni CON vs SENZA Immagini")
        print("="*80)
        
        # Identifica campioni con/senza immagini
        image_indices = set(X_test['images'].index)
        all_test_indices = set(y_test.index)
        
        with_images = sorted(list(image_indices.intersection(all_test_indices)))
        without_images = sorted(list(all_test_indices - image_indices))
        
        print(f"\nCampioni CON immagini: {len(with_images)}")
        print(f"Campioni SENZA immagini: {len(without_images)}\n")
        
        if len(with_images) > 0 and len(without_images) > 0:
            # Metriche per campioni CON immagini
            preds_with = test_preds[with_images]
            true_with = y_test[with_images]
            acc_with = accuracy_score(true_with, preds_with)
            
            if use_mcdo:
                conf_with = mcdo_results['confidence'][with_images].mean()
                unc_with = mcdo_results['epistemic_uncertainty'][with_images].mean()
            else:
                conf_with = None
                unc_with = None
            
            # Metriche per campioni SENZA immagini
            preds_without = test_preds[without_images]
            true_without = y_test[without_images]
            acc_without = accuracy_score(true_without, preds_without)
            
            if use_mcdo:
                conf_without = mcdo_results['confidence'][without_images].mean()
                unc_without = mcdo_results['epistemic_uncertainty'][without_images].mean()
            else:
                conf_without = None
                unc_without = None
            
            # Stampa confronto
            print("┌" + "─"*78 + "┐")
            print(f"│ {'Metrica':<25} │ {'CON immagini':>15} │ {'SENZA immagini':>18} │ {'Δ':>10} │")
            print("├" + "─"*78 + "┤")
            
            delta_acc = acc_with - acc_without
            print(f"│ {'Accuracy':<25} │ {acc_with:>15.4f} │ {acc_without:>18.4f} │ {delta_acc:>+10.4f} │")
            
            if conf_with is not None:
                delta_conf = conf_with - conf_without
                print(f"│ {'Confidence':<25} │ {conf_with:>15.4f} │ {conf_without:>18.4f} │ {delta_conf:>+10.4f} │")
                
                delta_unc = unc_with - unc_without
                print(f"│ {'Epistemic Uncertainty':<25} │ {unc_with:>15.4f} │ {unc_without:>18.4f} │ {delta_unc:>+10.4f} │")
            
            print("└" + "─"*78 + "┘")
            
            # ========================================================================
            # Analisi stratificata per classe
            # ========================================================================
            print("\nAnalisi Stratificata per Classe:")
            print("="*80)
            
            # Distribuzione classi
            classes_with = true_with.value_counts().sort_index()
            classes_without = true_without.value_counts().sort_index()
            all_classes = sorted(set(classes_with.index).union(set(classes_without.index)))
            
            print("\nDistribuzione Classi:")
            print(f"  {'Classe':<10} │ {'CON img':>10} │ {'SENZA img':>12} │ {'% CON img':>12}")
            print("  " + "─"*50)
            
            for cls in all_classes:
                n_with = classes_with.get(cls, 0)
                n_without = classes_without.get(cls, 0)
                pct_with = 100 * n_with / len(with_images) if len(with_images) > 0 else 0
                print(f"  {cls:<10} │ {n_with:>10} │ {n_without:>12} │ {pct_with:>11.1f}%")
            
            # Metriche per classe
            print("\nMetriche per Classe:")
            print("="*80)
            
            for cls in all_classes:
                # Maschera per classe corrente
                mask_with = (true_with == cls)
                mask_without = (true_without == cls)
                
                n_with_cls = mask_with.sum()
                n_without_cls = mask_without.sum()
                
                if n_with_cls == 0 and n_without_cls == 0:
                    continue
                
                print(f"\n🔹 Classe: {cls} (CON img: {n_with_cls}, SENZA img: {n_without_cls})")
                
                if n_with_cls > 0:
                    # Accuracy per classe (recall = samples corretti / totale classe)
                    acc_cls_with = (preds_with[mask_with] == true_with[mask_with]).mean()
                    
                    if use_mcdo:
                        conf_cls_with = mcdo_results['confidence'][with_images][mask_with].mean()
                        unc_cls_with = mcdo_results['epistemic_uncertainty'][with_images][mask_with].mean()
                else:
                    acc_cls_with = None
                    conf_cls_with = None
                    unc_cls_with = None
                
                if n_without_cls > 0:
                    acc_cls_without = (preds_without[mask_without] == true_without[mask_without]).mean()
                    
                    if use_mcdo:
                        conf_cls_without = mcdo_results['confidence'][without_images][mask_without].mean()
                        unc_cls_without = mcdo_results['epistemic_uncertainty'][without_images][mask_without].mean()
                else:
                    acc_cls_without = None
                    conf_cls_without = None
                    unc_cls_without = None
                
                # Stampa confronto per classe
                print(f"  {'Metrica':<25} │ {'CON img':>12} │ {'SENZA img':>12} │ {'Δ':>10}")
                print("  " + "─"*64)
                
                if acc_cls_with is not None and acc_cls_without is not None:
                    delta = acc_cls_with - acc_cls_without
                    print(f"  {'Accuracy':<25} │ {acc_cls_with:>12.4f} │ {acc_cls_without:>12.4f} │ {delta:>+10.4f}")
                elif acc_cls_with is not None:
                    print(f"  {'Accuracy':<25} │ {acc_cls_with:>12.4f} │ {'N/A':>12} │ {'N/A':>10}")
                elif acc_cls_without is not None:
                    print(f"  {'Accuracy':<25} │ {'N/A':>12} │ {acc_cls_without:>12.4f} │ {'N/A':>10}")
                
                if conf_cls_with is not None and conf_cls_without is not None:
                    delta = conf_cls_with - conf_cls_without
                    print(f"  {'Confidence':<25} │ {conf_cls_with:>12.4f} │ {conf_cls_without:>12.4f} │ {delta:>+10.4f}")
                    
                    delta = unc_cls_with - unc_cls_without
                    print(f"  {'Epistemic Unc':<25} │ {unc_cls_with:>12.4f} │ {unc_cls_without:>12.4f} │ {delta:>+10.4f}")
            
            print("\n" + "="*80)
            
            # Interpretazione
            print("\n🔍 Interpretazione:")
            
            if delta_acc > 0.02:
                print(f"  ✅ Accuracy CON immagini è {delta_acc:+.1%} superiore")
                print(f"     → Le immagini migliorano le predizioni!")
                imaging_helps = True
            elif delta_acc < -0.02:
                print(f"  ⚠️  Accuracy SENZA immagini è {-delta_acc:+.1%} superiore")
                print(f"     → Le immagini potrebbero introdurre rumore")
                imaging_helps = False
            else:
                print(f"  ≈  Accuracy simile tra gruppi (Δ={delta_acc:+.1%})")
                print(f"     → Contributo CNN marginale")
                imaging_helps = False
            
            if conf_with is not None:
                if delta_conf > 0.03:
                    print(f"  ✅ Confidence CON immagini è {delta_conf:+.4f} superiore")
                    print(f"     → Modello più sicuro con imaging")
                    imaging_helps = imaging_helps and True
                elif delta_conf < -0.03:
                    print(f"  ⚠️  Confidence SENZA immagini è {-delta_conf:+.4f} superiore")
                    imaging_helps = False
                
                if delta_unc < -0.005:
                    print(f"  ✅ Uncertainty CON immagini è {-delta_unc:+.4f} inferiore")
                    print(f"     → Modello più stabile con imaging")
                    imaging_helps = imaging_helps and True
                elif delta_unc > 0.005:
                    print(f"  ⚠️  Uncertainty SENZA immagini è {delta_unc:+.4f} inferiore")
                    imaging_helps = False
            
            print("="*80 + "\n")

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
    if aleatoric_uncertainty is not None:
        metrics["aleatoric_uncertainty"] = aleatoric_uncertainty
    
    # Aggiungi metriche validazione UQ se disponibili
    if unc_correlation_spearman is not None:
        metrics["unc_correlation_spearman"] = unc_correlation_spearman
        metrics["unc_correlation_pearson"] = unc_correlation_pearson
        metrics["acc_at_rejection_0"] = acc_at_rejection_0
        metrics["acc_at_rejection_10"] = acc_at_rejection_10
        metrics["acc_at_rejection_20"] = acc_at_rejection_20
    
    # Aggiungi metriche validazione TTA se disponibili
    if alea_correlation_spearman is not None:
        metrics["alea_correlation_spearman"] = alea_correlation_spearman
        metrics["alea_correlation_pearson"] = alea_correlation_pearson
    
    # Aggiungi metriche Conformal se disponibili
    if conformal_uncertainty is not None:
        metrics["conformal_uncertainty"] = conformal_uncertainty
        metrics["conformal_coverage"] = conformal_coverage
    if conf_correlation_spearman is not None:
        metrics["conf_correlation_spearman"] = conf_correlation_spearman
        metrics["conf_correlation_pearson"] = conf_correlation_pearson
    
    return metrics
