import numpy as np
import pandas as pd
import os
import copy
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_predict
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import sys
import os

class BoostSH(BaseEstimator, ClassifierMixin):

    def __init__(self, base_estimators:dict, n_iter=10, learning_rate=1.):
        """
            Boost SH : Boosting classification algorithm for multimodal with shared weights
            Greedy approach in which each modality is tested to evaluate the one with larger
            edge

            Arguments:
                base_estimators: dict, {modality:model}
                n_iter {int} -- Number of boosting iterations
                learning_rate {float} --  Learning rate for boosting (default: 1)
        """
        #super(BoostSH, self).__init__()
        super().__init__()
        self.base_estimators = base_estimators 
        self.modalities = {}

        self.models = []
        self.classes = []
        self.alphas = []
        self.modalities_selected = []
        self.weights = []
        self.eps = 10 ** (-6)

        self.n_iter = n_iter
        self.learning_rate = learning_rate

        # # Flag to track if the CNN has been fine-tuned
        # self.cnn_finetuned = False

    def fit(self, X, y, forecast_cv=None, sample_weights=None):
        """
            Fit the model by adding models in an adaboost fashion

            Arguments:
                X {Dict of pd Dataframe} -- Modalities to use for the task
                y {pd Dataframe} -- Labels - Index has to be contained in modality union
                forecast_cv {int} -- Number of fold used to estimate the edge
                    (default: None - Performance are computed on training set)
        """
        self.check_input(X, y)

        self.modalities = copy.deepcopy(X)
        self.classes = np.unique(y)

        index = self.__index_union__(self.modalities)
        y = y.reindex(index)

        # Initialize distribution over weights
        self.initialize_weights(sample_weights, index)
        self.weights = pd.Series(self.weights)

        for t in range(self.n_iter):
            if self.weights.sum() == 0:
                break

            self.weights /= self.weights.sum()

            selected = {'max_edge': -np.inf}
            for m in self.modalities:
                mask = self.modalities[m].index.tolist()
                weak_forecast = self.__compute_weak_forecast__(m, self.modalities[m], y[mask], self.weights[mask], forecast_cv, True, use_mcdo=False)

                tmp = 2 * ((weak_forecast['forecast'] == y[mask].values) - .5)
                edge = (self.weights[mask].values * tmp).sum()
                
                print(f'Modality {m}: edge={edge:.4f}')

                if edge > selected['max_edge']:
                    selected = {'max_edge': edge,
                                'modality': m, 'forecast': weak_forecast['forecast'], 'mask': mask,
                                'model': weak_forecast['model'], 'tmp': tmp}

            if (1 - selected['max_edge']) < self.eps:
                alpha = self.learning_rate * .5 * 10.
            else:
                alpha = self.learning_rate * .5 * np.log((1 + selected['max_edge']) / (1 - selected['max_edge']))

            # Update weights
            self.weights[selected['mask']] *= np.exp(- alpha * selected['tmp'])

            self.models.append(selected['model'])
            self.alphas.append(alpha)
            self.modalities_selected.append(selected['modality'])

            print('Boost.SH iteration ', t)
            print('Winning modality ', selected['modality'])
            print('Edge ', selected['max_edge'])
            print('')

        return
    
    def initialize_weights(self, sample_weights=None, index=None):
        if sample_weights is None:
            # Initialize uniform distribution over weights
            self.weights = pd.Series(1, index=index)

        else:
            # Assign pre-defined distribution over weights
            self.weights = pd.Series(sample_weights, index=index)
    
    
    def __compute_weak_forecast__(self, m:str, data, labels, weights, forecast_cv=None, return_model=True, use_mcdo=False):
        weak_forecast = dict()
        model = universal_clone(self.base_estimators[m])
        print(f'Model architecture: {model.__class__.__name__}.')
        # data, labels, weights = shuffle(data, labels, weights)

        if forecast_cv is None:
            print(f'START Training {m} modality...')
            model.fit(data, labels, sample_weight=weights) #fit a definito modalità per modalità
            print(f'END Training {m} modality...')
            forecast = model.predict(data)

            # print accuracy for random forest (da cancellare in un secondo momento forse)
            if model.__class__.__name__ == 'RandomForestClassifier':
                accuracy = accuracy_score(labels, forecast,sample_weight=weights)
                print(f'RandomForest Training accuracy: {accuracy:.2f}')
            
            # MC Dropout for confidence estimation (only for NeuralNetworkFitter)
            if use_mcdo and hasattr(model, 'predict_proba_mcdo'):
                print(f'Computing MC Dropout confidence for {m} modality...')
                mcdo_results = model.predict_proba_mcdo(data, n_mc_samples=25)
                weak_forecast['confidence'] = mcdo_results['confidence'].cpu().numpy()
                weak_forecast['epistemic_uncertainty'] = mcdo_results['epistemic_uncertainty'].cpu().numpy()
                print(f'Mean confidence: {weak_forecast["confidence"].mean():.4f}')
                print(f'Mean epistemic uncertainty: {weak_forecast["epistemic_uncertainty"].mean():.4f}')
            else:
                # Default confidence = 1.0 for non-MCDO models
                weak_forecast['confidence'] = np.ones(len(data))
                weak_forecast['epistemic_uncertainty'] = np.zeros(len(data))

        else:
            forecast = cross_val_predict(model, data.values, labels.values, cv=forecast_cv,
                                         fit_params={'sample_weight': weights.values})
            # No confidence available in CV mode
            weak_forecast['confidence'] = np.ones(len(forecast))
            weak_forecast['epistemic_uncertainty'] = np.zeros(len(forecast))

        weak_forecast['forecast'] = forecast
        if return_model:
            weak_forecast['model'] = model
        print(f"Forecast: {forecast}")
        return weak_forecast
    
    def predict_proba(self, X):

        index = self.__index_union__(X)
        predictions = pd.DataFrame(0., index=index, columns=self.classes)
        for t in range(len(self.models)):    
            if self.modalities_selected[t] in X.keys():
                X_test = X[self.modalities_selected[t]]
                # print(f"model{t} dataset index:{X_test.index}")
                #int_index = [list(index).index(i) for i in X_test.index]
                test_index = X_test.index
                if self.models[t].__class__.__name__ == 'RandomForestClassifier':
                    probas = self.models[t].predict_proba(X_test)
                else:
                    probas = self.models[t].predict_proba(X_test)
                probas = probas.detach().cpu().numpy() if not isinstance(probas, np.ndarray) else probas
                
                for i, idx in enumerate(test_index):
                    self.alphas[t] = float(self.alphas[t])
                    predictions.loc[idx] += self.alphas[t] * probas[i]
                    
        
        # Normalizzare le previsioni dividendo per la somma su ogni riga
        predictions_normalized = pd.DataFrame(0, columns = predictions.columns, index = predictions.index, dtype="float")
        for i in range(0, len(predictions)):
            if predictions.iloc[i,:].sum() == 0:
                continue
            else:
                predictions_normalized.iloc[i,:] =predictions.iloc[i,:]/predictions.iloc[i,:].sum() 
        return predictions_normalized
        #return predictions.div(predictions.sum(axis=1), axis=0)


    def predict(self, X):
        self.check_X(X)
        assert len(self.models) > 0, 'Model not trained'
        pp = self.predict_proba(X).idxmax(axis=1)
        return pp
    
    def predict_with_mcdo(self, X, n_mc_samples=25):
        """
        Predict with MC Dropout for uncertainty quantification (TEST ONLY).
        Optimized to load data once per modality.
        
        Args:
            X: dict of modality data
            n_mc_samples: number of MC samples
            
        Returns:
            dict with 'predictions', 'confidence', 'epistemic_uncertainty'
        """
        self.check_X(X)
        assert len(self.models) > 0, 'Model not trained'
        
        index = self.__index_union__(X)
        
        # Collect MC samples
        mc_predictions = []
        for mc_iter in range(n_mc_samples):
            np.random.seed(42 + mc_iter)  # For reproducibility

            if mc_iter % 5 == 0:
                print(f"MC Dropout iteration {mc_iter+1}/{n_mc_samples}...")
            
            predictions = pd.DataFrame(0., index=index, columns=self.classes)
            
            for t in range(len(self.models)):
                if self.modalities_selected[t] in X.keys():
                    X_test = X[self.modalities_selected[t]]
                    test_index = X_test.index
                    
                    # Use MC Dropout for NeuralNetworkFitter, tree subsampling for RandomForest
                    if hasattr(self.models[t], 'predict_proba_with_dropout'):
                        # CNN with MC Dropout
                        probas = self.models[t].predict_proba_with_dropout(X_test)
                    # COMMENTED: Tree subsampling approach (replaced by Conformal Prediction)
                    # elif self.models[t].__class__.__name__ == 'RandomForestClassifier':
                    #     # Random Forest: subsample 70% of trees randomly (like dropout p=0.3)
                    #     n_trees = len(self.models[t].estimators_)
                    #     n_sample = int(0.7 * n_trees)  # Use 70% of trees
                    #     sampled_indices = np.random.choice(n_trees, size=n_sample, replace=False)
                    #     sampled_trees = [self.models[t].estimators_[i] for i in sampled_indices]
                    #     
                    #     # Convert DataFrame to numpy to avoid sklearn warning
                    #     X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
                    #     
                    #     # Average predictions from sampled trees
                    #     tree_predictions = np.array([tree.predict_proba(X_test_np) for tree in sampled_trees])
                    #     probas = tree_predictions.mean(axis=0)
                    else:
                        probas = self.models[t].predict_proba(X_test)
                    
                    probas = probas.detach().cpu().numpy() if not isinstance(probas, np.ndarray) else probas
                    
                    for i, idx in enumerate(test_index):
                        predictions.loc[idx] += float(self.alphas[t]) * probas[i]
            
            # Normalize
            predictions_normalized = predictions.div(predictions.sum(axis=1), axis=0).fillna(0)
            mc_predictions.append(predictions_normalized.values)
        
        # Stack MC samples: [n_mc_samples, n_samples, n_classes]
        mc_predictions = np.stack(mc_predictions, axis=0)
        
        # Mean prediction
        mean_probs = mc_predictions.mean(axis=0)
        
        # Confidence (max probability)
        confidence = mean_probs.max(axis=1)
        
        # Epistemic uncertainty (std of max probability across MC samples)
        max_probs_per_sample = mc_predictions.max(axis=2)  # [n_mc_samples, n_samples]
        epistemic_uncertainty = max_probs_per_sample.std(axis=0)  # [n_samples]
        # var_probs = mc_predictions.var(axis=0)
        # epistemic_uncertainty = var_probs.mean(axis=1)
        
        # Final predictions
        final_predictions = pd.Series(self.classes[mean_probs.argmax(axis=1)], index=index)
        
        return {
            'predictions': final_predictions,
            'confidence': pd.Series(confidence, index=index),
            'epistemic_uncertainty': pd.Series(epistemic_uncertainty, index=index)
        }
    
    def predict_with_tta(self, X, n_tta_samples=10, noise_scale=0.05):
        """
        Predict with Test-Time Augmentation for aleatoric uncertainty (data uncertainty).
        Perturbs CLINICAL data only with feature-specific realistic noise levels.
        
        Noise levels based on published clinical variability:
        - MMSE: ±2 points (test-retest variability)
        - ABETA (CSF_Abeta42): ±8% (assay variability)
        - TAU: ±7-10% (measurement error literature)
        - PTAU: ±12% (newer biomarker, more noise)
        - AGE: 0 (stable)
        - APOE4: 0 (genetic, immutable)
        
        Args:
            X: dict of modality data
            n_tta_samples: number of augmented samples per instance
            noise_scale: DEPRECATED - now uses feature-specific noise levels
            
        Returns:
            dict with 'predictions', 'confidence', 'aleatoric_uncertainty'
        """
        self.check_X(X)
        assert len(self.models) > 0, 'Model not trained'
        
        index = self.__index_union__(X)
        
        # Define feature-specific noise levels
        feature_noise_config = {
            # Absolute noise (±N points)
            'MMSE': {'type': 'absolute', 'std': 2.0},
            # Percentage noise (±N% of value)
            'ABETA': {'type': 'percentage', 'std': 0.08},  # ±8%
            'TAU': {'type': 'percentage', 'std': 0.085},   # ±8.5% (midpoint 7-10%)
            'PTAU': {'type': 'percentage', 'std': 0.12},   # ±12%
            # No noise (stable/genetic)
            'AGE': {'type': 'none'},
            'APOE4': {'type': 'none'}
        }
        
        # Print TTA configuration (once)
        if 'clinical' in X:
            print("\n[TTA] Feature-specific noise levels (realistic clinical variability):")
            print("  MMSE:       ±2.0 points (absolute)")
            print("  ABETA:      ±8.0% (assay variability)")
            print("  TAU:        ±8.5% (measurement error)")
            print("  PTAU:       ±12.0% (newer biomarker)")
            print("  AGE/APOE4:  0 (stable/genetic)")
            print("  Others:     ±5.0% (default)")
            print()
        
        # Collect TTA samples
        tta_predictions = []
        for tta_iter in range(n_tta_samples):
            np.random.seed(100 + tta_iter)  # Different seed from MCDO
            
            if tta_iter % 5 == 0:
                print(f"TTA iteration {tta_iter+1}/{n_tta_samples}...")
            
            # Create perturbed data
            X_perturbed = {}
            for modality, data in X.items():
                if modality == 'clinical':
                    # Apply feature-specific noise
                    perturbed = data.copy()
                    
                    for col in data.columns:
                        # Find matching noise config (check if column name contains key)
                        noise_cfg = None
                        for feature_name, cfg in feature_noise_config.items():
                            if feature_name in col.upper():
                                noise_cfg = cfg
                                break
                        
                        if noise_cfg is None:
                            # Default: 5% relative noise for unknown features
                            noise_cfg = {'type': 'percentage', 'std': 0.05}
                        
                        # Apply noise based on type
                        if noise_cfg['type'] == 'absolute':
                            # Absolute noise (e.g., MMSE ±2 points)
                            noise = np.random.normal(0, noise_cfg['std'], size=len(data))
                            perturbed[col] = data[col] + noise
                        elif noise_cfg['type'] == 'percentage':
                            # Percentage noise (e.g., TAU ±8.5%)
                            noise = np.random.normal(0, noise_cfg['std'], size=len(data))
                            perturbed[col] = data[col] * (1 + noise)
                        # elif noise_cfg['type'] == 'none': no perturbation
                    
                    X_perturbed[modality] = perturbed
                else:
                    # Images: no perturbation (already has epistemic via MCDO)
                    X_perturbed[modality] = data
            
            # Standard prediction on perturbed data
            predictions = pd.DataFrame(0., index=index, columns=self.classes)
            
            for t in range(len(self.models)):
                if self.modalities_selected[t] in X_perturbed.keys():
                    X_test = X_perturbed[self.modalities_selected[t]]
                    test_index = X_test.index
                    
                    probas = self.models[t].predict_proba(X_test)
                    probas = probas.detach().cpu().numpy() if not isinstance(probas, np.ndarray) else probas
                    
                    for i, idx in enumerate(test_index):
                        predictions.loc[idx] += float(self.alphas[t]) * probas[i]
            
            # Normalize
            predictions_normalized = predictions.div(predictions.sum(axis=1), axis=0).fillna(0)
            tta_predictions.append(predictions_normalized.values)
        
        # Stack TTA samples: [n_tta_samples, n_samples, n_classes]
        tta_predictions = np.stack(tta_predictions, axis=0)
        
        # Mean prediction
        mean_probs = tta_predictions.mean(axis=0)
        
        # Confidence (max probability)
        confidence = mean_probs.max(axis=1)
        
        # Aleatoric uncertainty (std of max probability across TTA samples)
        max_probs_per_sample = tta_predictions.max(axis=2)
        aleatoric_uncertainty = max_probs_per_sample.std(axis=0)
        
        # Final predictions
        final_predictions = pd.Series(self.classes[mean_probs.argmax(axis=1)], index=index)
        
        return {
            'predictions': final_predictions,
            'confidence': pd.Series(confidence, index=index),
            'aleatoric_uncertainty': pd.Series(aleatoric_uncertainty, index=index)
        }
    
    def predict_proba_with_contributions(self, X):
        """
        Predict probabilities and track contributions from each modality per patient.
        
        This method decomposes the final prediction into modality-specific contributions,
        allowing to understand which modality (clinical vs imaging) was more influential
        for each individual patient's prediction.
        
        Args:
            X: dict of modality data
            
        Returns:
            tuple: (predictions_normalized, contributions) where:
                - predictions_normalized: DataFrame with final probabilities per class
                - contributions: dict with 'clinical' and 'imaging' DataFrames showing
                  the contribution of each modality to the final probability
        """
        self.check_X(X)
        assert len(self.models) > 0, 'Model not trained'
        
        index = self.__index_union__(X)
        predictions = pd.DataFrame(0., index=index, columns=self.classes)
        
        # Track contributions separately by modality
        contrib_clinical = pd.DataFrame(0., index=index, columns=self.classes)
        contrib_imaging = pd.DataFrame(0., index=index, columns=self.classes)
        
        for t in range(len(self.models)):
            if self.modalities_selected[t] in X.keys():
                X_test = X[self.modalities_selected[t]]
                test_index = X_test.index
                
                if self.models[t].__class__.__name__ == 'RandomForestClassifier':
                    probas = self.models[t].predict_proba(X_test)
                else:
                    probas = self.models[t].predict_proba(X_test)
                    
                probas = probas.detach().cpu().numpy() if not isinstance(probas, np.ndarray) else probas
                
                # Weight by alpha
                weighted_proba = self.alphas[t] * probas
                
                for i, idx in enumerate(test_index):
                    predictions.loc[idx, :] += weighted_proba[i, :]
                    
                    # Track contribution by modality
                    if self.modalities_selected[t] == 'clinical':
                        contrib_clinical.loc[idx, :] += weighted_proba[i, :]
                    elif self.modalities_selected[t] == 'images':
                        contrib_imaging.loc[idx, :] += weighted_proba[i, :]
        
        # Normalize predictions
        predictions_normalized = pd.DataFrame(0, columns=predictions.columns, 
                                             index=predictions.index, dtype="float")
        for i in range(len(predictions)):
            if predictions.iloc[i, :].sum() == 0:
                continue
            else:
                predictions_normalized.iloc[i, :] = predictions.iloc[i, :] / predictions.iloc[i, :].sum()
        
        # Normalize contributions (relative to final prediction sum, not independently)
        contrib_clinical_normalized = pd.DataFrame(0, columns=contrib_clinical.columns,
                                                   index=contrib_clinical.index, dtype="float")
        contrib_imaging_normalized = pd.DataFrame(0, columns=contrib_imaging.columns,
                                                  index=contrib_imaging.index, dtype="float")
        
        for i in range(len(predictions)):
            total = predictions.iloc[i, :].sum()
            if total == 0:
                continue
            contrib_clinical_normalized.iloc[i, :] = contrib_clinical.iloc[i, :] / total
            contrib_imaging_normalized.iloc[i, :] = contrib_imaging.iloc[i, :] / total
        
        contributions = {
            'clinical': contrib_clinical_normalized,
            'imaging': contrib_imaging_normalized
        }
        
        return predictions_normalized, contributions
    
    def predict_with_conformal(self, X_test, X_calib, y_calib, alpha=0.1):
        """
        Predict with Conformal Prediction for uncertainty quantification.
        
        Based on Inductive Conformal Prediction:
        - Uses calibration set to compute non-conformity scores
        - Creates prediction sets with guaranteed coverage (1 - alpha)
        - Uncertainty = size of prediction set (larger = more uncertain)
        
        Args:
            X_test: dict of test modalities
            X_calib: dict of calibration modalities (20% of training)
            y_calib: pd.Series of calibration labels
            alpha: miscoverage rate (default 0.1 for 90% coverage)
            
        Returns:
            dict with 'predictions', 'confidence', 'prediction_sets', 'conformal_uncertainty'
        """
        self.check_X(X_test)
        self.check_X(X_calib)
        assert len(self.models) > 0, 'Model not trained'
        
        print(f"\n{'='*80}")
        print(f"Conformal Prediction (Inductive CP)")
        print(f"{'='*80}")
        print(f"Calibration set size: {len(y_calib)}")
        print(f"Miscoverage rate α: {alpha:.2%} (target coverage: {(1-alpha):.2%})")
        print(f"Number of classes: {len(self.classes)}")
        print()
        
        # Step 1: Get calibration probabilities
        print("Step 1: Computing non-conformity scores on calibration set...")
        calib_probs = self.predict_proba(X_calib)
        
        # Step 2: Compute non-conformity scores = 1 - P(true_class)
        non_conformity_scores = []
        for idx in y_calib.index:
            true_class = y_calib[idx]
            prob_true_class = calib_probs.loc[idx, true_class]
            non_conformity = 1 - prob_true_class
            non_conformity_scores.append(non_conformity)
        
        non_conformity_scores = np.array(non_conformity_scores)
        
        # Step 3: Compute quantile threshold
        n_calib = len(non_conformity_scores)
        q_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
        q_level = min(q_level, 1.0)  # Cap at 1.0
        threshold = np.quantile(non_conformity_scores, q_level)
        
        print(f"Non-conformity threshold (q={q_level:.3f}): {threshold:.4f}")
        print(f"  Mean non-conformity: {non_conformity_scores.mean():.4f}")
        print(f"  Median non-conformity: {np.median(non_conformity_scores):.4f}")
        print()
        
        # Step 4: Predict on test set and create prediction sets
        print("Step 2: Creating prediction sets for test samples...")
        test_probs = self.predict_proba(X_test)
        
        prediction_sets = []
        conformal_uncertainties = []
        confidences = []
        final_predictions = []
        
        for idx in test_probs.index:
            # Prediction set = {classes where 1 - P(class) ≤ threshold}
            # Equivalently: {classes where P(class) ≥ 1 - threshold}
            prob_threshold = 1 - threshold
            pred_set = test_probs.columns[test_probs.loc[idx] >= prob_threshold].tolist()
            
            # If empty set (very rare), include top class
            if len(pred_set) == 0:
                pred_set = [test_probs.loc[idx].idxmax()]
            
            prediction_sets.append(pred_set)
            
            # Uncertainty = normalized set size: (|set| - 1) / (n_classes - 1)
            # Range: [0, 1] where 0 = certain (singleton), 1 = completely uncertain (all classes)
            n_classes = len(self.classes)
            uncertainty = (len(pred_set) - 1) / max(n_classes - 1, 1)
            conformal_uncertainties.append(uncertainty)
            
            # Confidence = max probability in set
            max_prob_in_set = test_probs.loc[idx, pred_set].max()
            confidences.append(max_prob_in_set)
            
            # Final prediction = class with highest probability
            final_pred = test_probs.loc[idx].idxmax()
            final_predictions.append(final_pred)
        
        # Convert to Series/DataFrame
        index = test_probs.index
        final_predictions = pd.Series(final_predictions, index=index)
        confidences = pd.Series(confidences, index=index)
        conformal_uncertainties = pd.Series(conformal_uncertainties, index=index)
        
        # Statistics
        set_sizes = [len(s) for s in prediction_sets]
        print(f"Prediction set statistics:")
        print(f"  Mean set size: {np.mean(set_sizes):.2f}")
        print(f"  Median set size: {np.median(set_sizes):.1f}")
        print(f"  Singleton sets (certain): {sum(s == 1 for s in set_sizes)}/{len(set_sizes)} ({sum(s == 1 for s in set_sizes)/len(set_sizes):.1%})")
        print(f"  Full sets (uncertain): {sum(s == n_classes for s in set_sizes)}/{len(set_sizes)} ({sum(s == n_classes for s in set_sizes)/len(set_sizes):.1%})")
        print(f"Mean conformal uncertainty: {conformal_uncertainties.mean():.4f}")
        print("=" * 80 + "\n")
        
        return {
            'predictions': final_predictions,
            'confidence': confidences,
            'prediction_sets': prediction_sets,
            'conformal_uncertainty': conformal_uncertainties,
            'threshold': threshold,
            'alpha': alpha
        }

    def check_input(self, X, y):
        self.check_X(X)
        self.check_y(y)

    def check_X(self, X):
        assert isinstance(X, dict), "Not right format for X"
        for key in X.keys():
            assert isinstance(X[key], pd.DataFrame)
            assert not X[key].empty, "Empty dataframe"

    def check_y(self, y):
        assert isinstance(y, pd.Series), "Not right format for y"
        # assert len(y.unique()) > 1, "One class in data"
        assert not y.empty, "Empty dataframe"

    def __index_union__(self, modalities):
        self.check_X(modalities)
        index = set([])
        for mod in modalities:
            index = index.union(set(modalities[mod].index))

        return list(index)

 


class IRBoostSH(BoostSH):

    def __init__(self, base_estimators:dict, n_iter=10, learning_rate=1., sigma=0.15, gamma=0.3):
        """
            rBoost SH : Boosting classification for multimodal with shared weights.
            Multi-arm bandit approach in which a modality is selected at each iteration

            Arguments:
                base_estimator {sklearn model} -- Base classifier to use on each modality
                n_iter {int} -- Number of boosting iterations
                learning_rate {float} -- Learning rate for boosting (default: 1)
        """
        super().__init__(base_estimators, n_iter=n_iter, learning_rate=learning_rate)
        self.sigma = sigma
        self.gamma = gamma

    def fit(self, X, y,mod, forecast_cv=None, sample_weights=None):
        """
            Fit the model by adding models in a boosting fashion

            Arguments:
                X {Dict of pandas Dataframes/ torch tensors} -- Modalities to use for the task
                y {pandas Series} -- Labels - Index has to be contained in modality union
                forecast_cv {int} -- Number of fold used to estimate the edge
                    (default: None - Performance are computed on training set)
        """
        self.check_input(X, y)
        self.modalities = copy.deepcopy(X)
        self.classes = np.unique(y)
        K = len(self.modalities) 
        possible_modalities = list(self.modalities.keys())

        index = self.__index_union__(self.modalities)
        # print(f"index:{index}")
        # Reorder labels
        y = y.reindex(index)

        # Initialize distribution over weights
        self.initialize_weights(sample_weights, index)
        self.weights = pd.Series(self.weights)

        p_mods = pd.Series(np.exp(self.sigma * self.gamma / 3 * np.sqrt(self.n_iter / K)), index=possible_modalities)
       
        for t in range(self.n_iter):
            
            print('')
            print(f'irBoost.SH training: Iteration {t+1}/{self.n_iter}.')
            print(f'Sample weights: {self.weights}')

            if self.weights.sum() == 0:
                break

            self.weights /= self.weights.sum()

            # Bandit selection of best modality
            q_mods = (1 - self.gamma) * p_mods / p_mods.sum() + self.gamma / K
            print(f"q_mods: {q_mods.to_string()}") ##

            if mod :
                selected_mod = mod
            else:
                selected_mod = np.random.choice(possible_modalities, p=q_mods)
            #selected_mod = 'images'
            print(f'Winning modality: {selected_mod}.')
        
            mask = self.modalities[selected_mod].index.tolist()
            weak_forecast = self.__compute_weak_forecast__(selected_mod, self.modalities[selected_mod], y[mask], self.weights[mask], use_mcdo=False)
            
            # Handle different return types from predict()
            if isinstance(weak_forecast['forecast'], pd.DataFrame):
                forecast_np = weak_forecast['forecast'].values.flatten()
            elif not isinstance(weak_forecast['forecast'], (np.ndarray)):
                forecast_np = weak_forecast['forecast'].detach().cpu().numpy()
            else:
                forecast_np = weak_forecast['forecast']
            target_np = y[mask].values

            # Use this with python version > 3.11
            if (sys.version_info.major, sys.version_info.minor) >= (3, 12):
                tmp = 2 * ((forecast_np == target_np).astype(int) - 0.5)
                
            else:
                tmp = 2 * ((forecast_np == target_np) - 0.5)
            
            ## Use this for python version <= 3.11
            # tmp = 2 * ((weak_forecast['forecast'] == y[mask].values) - .5)

            edge = (self.weights[mask].values * tmp).sum()
            
            print(f'Edge: {edge:.4f}')

            if (1 - edge) < self.eps:
                alpha = self.learning_rate * .5 * 10.
            elif edge <= 0:
                alpha = 0
            else:
                alpha = self.learning_rate * .5 * np.log((1 + edge) / (1 - edge))

            # Update weights
            self.weights[mask] *= np.exp(- alpha * tmp)
            
            # Normalize weights to prevent underflow
            self.weights /= self.weights.sum()
            self.weights *= len(self.weights)  # Scale back to original magnitude

            # Update arm probability
            r_mods = pd.Series(0., index=possible_modalities)
            square = np.sqrt(1 - edge ** 2) if edge < 1 else 0
            r_mods[selected_mod] = (1 - square) / q_mods[selected_mod]
            p_mods *= np.exp(self.gamma / (3 * K) * (r_mods + self.sigma / (q_mods * np.sqrt(self.n_iter * K))))

            self.models.append(weak_forecast['model'])
            self.alphas.append(alpha)
            self.modalities_selected.append(selected_mod)
        
        
            print('')
            print('Iteration ', t)
            print('Winning modality ', selected_mod)
            print(f'Edge: {edge:.4f}')
            print(f'Alpha {alpha}')

        return


    def modality_weights(self):
        """
            Return relative importance of the different modality in the final decision
        """
        assert len(self.models) > 0, 'Model not trained'
        mod_weights = pd.DataFrame({"modality": self.modalities_selected, "alpha": np.abs(self.alphas)})
        return (mod_weights.groupby('modality').sum() / np.sum(np.abs(self.alphas))).sort_values('alpha')


    def save_modalities(self, path):
        df = pd.DataFrame(self.modalities_selected)
        if path.endswith('xlsx'):
            file = path
        else:
            file = os.path.join(path, 'winning modalities.xlsx')
        df.to_excel(file)

        
    # TODO: If needed, add other classification metrics (recall, f1 score, ...) using the same logic as follows.
    def accuracy_score(self, X: dict, y: pd.Series) -> float:
        """
            Calculates the accuracy score of the irBoost.SH model comparing predictions against the true labels.
            This method predicts labels based on the input data `X`, reorders the predictions to match the index of the true labels `y`, and removes any NaN values. It then compares the predictions with the true labels to determine the accuracy score.

            Parameters:
            X (dict): A dictionary of pandas DataFrames, where each key represents a modality/view and the value is the corresponding DataFrame.
            y (pd.Series): A pandas Series containing the true labels.

            Returns:
            float: The accuracy score of the irBoost.SH model predictions.
        """
        # Predict labels based on the input X
        predictions = self.predict(X) 
        # Reorder predictions to match the index of the true labels and remove any NaN values
        predictions_ordered = predictions.reindex(y.index).dropna()
        # Compare the predictions with the true labels to get a boolean series indicating correct predictions
        results_comparison = (predictions_ordered == y[predictions_ordered.index])
        # Calculate the accuracy by summing the correct predictions and dividing by the total number of predictions
        ir_boost_accuracy = sum(results_comparison) / len(results_comparison)
        return ir_boost_accuracy
    

def universal_clone(obj):
    if isinstance(obj, BaseEstimator):
        # Clone per oggetti scikit-learn
        return clone(obj)
    elif isinstance(obj, torch.nn.Module):
        # Deepcopy per oggetti torch
        return copy.deepcopy(obj)
    else:
        # Prova un deepcopy generico
        return copy.deepcopy(obj)