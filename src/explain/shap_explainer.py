"""
SHAP-based explainability for Random Forest models in the boosting ensemble.

SHAP (SHapley Additive exPlanations) provides feature importance values that
explain individual predictions by attributing the contribution of each feature.
"""

import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier


class SHAPExplainer:
    """
    SHAP explainer for Random Forest models in the multimodal ensemble.
    
    This class computes SHAP values for clinical features, showing which
    features contribute most to each patient's prediction.
    """
    
    def __init__(self, model, feature_names: List[str] = None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained IRBoostSH model
            feature_names: List of feature names for clinical data
        """
        self.model = model
        self.feature_names = feature_names
        self.clinical_explainers = []
        self.clinical_model_indices = []
        
        # Extract all clinical RF models from the ensemble
        self._extract_clinical_models()
    
    def _extract_clinical_models(self):
        """
        Extract all RandomForest models that were trained on clinical data.
        """
        for t, (model, modality) in enumerate(zip(self.model.models, 
                                                   self.model.modalities_selected)):
            if modality == 'clinical' and isinstance(model, RandomForestClassifier):
                self.clinical_model_indices.append(t)
                # Create SHAP explainer for this RF using interventional perturbation
                # and avoid strict additivity checks which can fail when the
                # input matrix differs slightly from the training matrix.
                try:
                    explainer = shap.TreeExplainer(model, feature_perturbation='interventional')
                except Exception:
                    explainer = shap.TreeExplainer(model)
                self.clinical_explainers.append(explainer)
    
    def explain_patient(self, clinical_data: pd.DataFrame, patient_idx: int) -> Dict:
        """
        Explain a single patient's prediction using SHAP values.
        
        Args:
            clinical_data: DataFrame with clinical features (single patient)
            patient_idx: Index of the patient
            
        Returns:
            dict with:
                - 'shap_values': SHAP values per class
                - 'base_values': Expected values per class
                - 'feature_contributions': Top contributing features
                - 'aggregated_shap': Weighted average across all clinical models
        """
        if len(self.clinical_explainers) == 0:
            return {
                'error': 'No clinical RF models found in ensemble',
                'shap_values': None
            }
        
        # Save original DataFrame for predict call
        clinical_data_df = clinical_data.copy() if isinstance(clinical_data, pd.DataFrame) else clinical_data
        
        # Compute SHAP values for all clinical models
        # Convert to numpy if DataFrame
        clinical_data_numpy = clinical_data.values if hasattr(clinical_data, 'values') else clinical_data
        
        all_shap_values = []
        all_base_values = []
        
        for explainer, model_idx in zip(self.clinical_explainers, 
                                        self.clinical_model_indices):
            shap_values = explainer.shap_values(clinical_data_numpy, check_additivity=False)
            
            # Handle binary vs multiclass
            if not isinstance(shap_values, list):
                shap_values = [shap_values, -shap_values]
            
            base_values = explainer.expected_value
            if not isinstance(base_values, list):
                base_values = [base_values, -base_values]
            
            # Weight by alpha from boosting
            alpha = self.model.alphas[model_idx]
            
            all_shap_values.append((shap_values, alpha))
            all_base_values.append((base_values, alpha))
        
        # Aggregate SHAP values across models (weighted by alpha)
        total_alpha = sum(alpha for _, alpha in all_shap_values)
        
        # SHAP values is a list [class_0, class_1, class_2]
        n_classes = len(all_shap_values[0][0])
        aggregated_shap = []
        
        for c in range(n_classes):
            class_shap = sum(shap_vals[c] * alpha / total_alpha 
                           for shap_vals, alpha in all_shap_values)
            aggregated_shap.append(class_shap)
        
        # Get predicted class
        predicted_class_idx = self.model.predict(
            {'clinical': clinical_data_df}
        ).iloc[0]
        
        # Convert to class index (assuming classes are 1, 2, 3 → indices 0, 1, 2)
        class_mapping = {cls: idx for idx, cls in enumerate(self.model.classes)}
        pred_idx = class_mapping.get(predicted_class_idx, 0)
        
        # Extract SHAP values for predicted class
        shap_for_prediction = aggregated_shap[pred_idx]
        # Ensure 1D array for single patient
        shap_for_prediction = np.atleast_1d(shap_for_prediction).flatten()
        
        # Get top contributing features
        if self.feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(shap_for_prediction))]
        else:
            # Ensure feature_names matches shap_for_prediction length
            feature_names = self.feature_names[:len(shap_for_prediction)]
        
        # Ensure lengths match
        min_len = min(len(feature_names), len(shap_for_prediction))
        feature_names = feature_names[:min_len]
        shap_for_prediction = shap_for_prediction[:min_len]
        
        # Create feature contributions dataframe
        contributions = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_for_prediction,
            'abs_shap': np.abs(shap_for_prediction)
        }).sort_values('abs_shap', ascending=False)
        
        return {
            'shap_values': aggregated_shap,
            'predicted_class': predicted_class_idx,
            'predicted_class_idx': pred_idx,
            'feature_contributions': contributions,
            'top_5_features': contributions.head(5),
            'aggregated_shap_for_pred': shap_for_prediction
        }
    
    def explain_batch(self, clinical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Explain multiple patients at once.
        
        Args:
            clinical_data: DataFrame with clinical features (multiple patients)
            
        Returns:
            DataFrame with SHAP values for each patient and feature
        """
        results = []
        
        for idx in clinical_data.index:
            patient_data = clinical_data.loc[[idx]]
            explanation = self.explain_patient(patient_data, idx)
            
            if 'error' not in explanation:
                results.append({
                    'patient_id': idx,
                    'predicted_class': explanation['predicted_class'],
                    'top_feature': explanation['top_5_features'].iloc[0]['feature'],
                    'top_shap_value': explanation['top_5_features'].iloc[0]['shap_value']
                })
        
        return pd.DataFrame(results)
    
    def get_global_importance(self, clinical_data: pd.DataFrame, 
                             max_patients: int = None) -> pd.DataFrame:
        """
        Compute global feature importance across all patients using weighted
        aggregation of all clinical RF models in the ensemble.
        
        Args:
            clinical_data: DataFrame with clinical features (all patients)
            max_patients: Maximum number of patients to use (None = all)
            
        Returns:
            DataFrame with mean absolute SHAP values per feature, aggregated
            across all patients and all clinical models (weighted by alpha)
        """
        if len(self.clinical_explainers) == 0:
            return pd.DataFrame()
        
        # Limit number of patients if specified
        if max_patients is not None and len(clinical_data) > max_patients:
            clinical_data = clinical_data.sample(n=max_patients, random_state=42)
        
        print(f"Computing global SHAP importance on {len(clinical_data)} patients...")
        
        # Convert to numpy for SHAP computation
        clinical_data_numpy = clinical_data.values if hasattr(clinical_data, 'values') else clinical_data
        
        # Compute weighted average of SHAP values across all clinical models
        total_alpha = sum(self.model.alphas[idx] for idx in self.clinical_model_indices)
        
        aggregated_shap_per_class = None
        n_classes = None
        
        for explainer, model_idx in zip(self.clinical_explainers, 
                                        self.clinical_model_indices):
            shap_values = explainer.shap_values(clinical_data_numpy, check_additivity=False)
            
            # Handle binary vs multiclass
            if not isinstance(shap_values, list):
                shap_values = [shap_values, -shap_values]
            
            # Get alpha weight for this model
            alpha = self.model.alphas[model_idx]
            weight = alpha / total_alpha
            
            # Initialize aggregated array on first iteration
            if aggregated_shap_per_class is None:
                n_classes = len(shap_values)
                aggregated_shap_per_class = [np.zeros_like(shap_values[c]) 
                                            for c in range(n_classes)]
            
            # Add weighted SHAP values
            for c in range(n_classes):
                aggregated_shap_per_class[c] += shap_values[c] * weight
        
        # Compute mean SHAP (signed) and mean absolute SHAP across classes and patients
        # Shape of each shap_class: (n_patients, n_features)
        mean_shap_per_class = [shap_class.mean(axis=0) for shap_class in aggregated_shap_per_class]
        mean_abs_shap_per_class = [np.abs(shap_class).mean(axis=0) for shap_class in aggregated_shap_per_class]

        # Average across classes to get overall feature importance
        global_mean_shap = np.mean(mean_shap_per_class, axis=0)
        global_mean_abs_shap = np.mean(mean_abs_shap_per_class, axis=0)
        # Ensure numeric 1-D arrays
        try:
            global_mean_shap = np.asarray(global_mean_shap, dtype=float).ravel()
            global_mean_abs_shap = np.asarray(global_mean_abs_shap, dtype=float).ravel()
        except Exception:
            global_mean_shap = np.array([float(x) for x in np.asarray(global_mean_shap).ravel()])
            global_mean_abs_shap = np.array([float(x) for x in np.asarray(global_mean_abs_shap).ravel()])
        
        if self.feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(global_mean_abs_shap))]
        else:
            # Ensure both arrays have same length by truncating to minimum
            min_len = min(len(self.feature_names), len(global_mean_abs_shap))
            if min_len != len(global_mean_abs_shap) or min_len != len(self.feature_names):
                print(f"   ⚠ Adjusting feature length: feature_names={len(self.feature_names)}, global_mean_abs_shap={len(global_mean_abs_shap)}, using min={min_len}")
            feature_names = self.feature_names[:min_len]
            global_mean_abs_shap = global_mean_abs_shap[:min_len]
            global_mean_shap = global_mean_shap[:min_len]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': global_mean_abs_shap,
            'mean_shap_signed': global_mean_shap
        }).sort_values('mean_abs_shap', ascending=False)

        print(f"✓ Global importance computed. Top 5 features (by mean_abs_shap):")
        print(importance_df.head(5)[['feature', 'mean_abs_shap', 'mean_shap_signed']].to_string(index=False))
        
        return importance_df
    
    def compute_shap_summary(self, clinical_data: pd.DataFrame, 
                           max_patients: int = None) -> Dict:
        """
        Compute comprehensive SHAP summary including:
        - Individual SHAP values for all patients
        - Global feature importance
        - Per-class feature importance
        
        Args:
            clinical_data: DataFrame with clinical features
            max_patients: Maximum number of patients to analyze (None = all)
            
        Returns:
            dict with:
                - 'global_importance': DataFrame with overall feature importance
                - 'all_shap_values': Array of SHAP values (n_patients, n_features, n_classes)
                - 'feature_names': List of feature names
                - 'patient_indices': Indices of analyzed patients
        """
        if len(self.clinical_explainers) == 0:
            return {'error': 'No clinical RF models found in ensemble'}
        
        # Sample patients if needed
        if max_patients is not None and len(clinical_data) > max_patients:
            sampled_data = clinical_data.sample(n=max_patients, random_state=42)
        else:
            sampled_data = clinical_data
        
        print(f"\nComputing SHAP summary for {len(sampled_data)} patients...")
        
        # Get global importance
        global_importance = self.get_global_importance(sampled_data, max_patients=None)
        
        # Compute aggregated SHAP values for all patients
        clinical_data_numpy = sampled_data.values if hasattr(sampled_data, 'values') else sampled_data
        total_alpha = sum(self.model.alphas[idx] for idx in self.clinical_model_indices)
        
        all_shap_aggregated = None
        
        for explainer, model_idx in zip(self.clinical_explainers, 
                                        self.clinical_model_indices):
            shap_values_raw = explainer.shap_values(clinical_data_numpy, check_additivity=False)

            # Normalize various SHAP output formats into a list of 2-D arrays
            # Each element should have shape (n_patients, n_features)
            try:
                if isinstance(shap_values_raw, list):
                    # If list contains a single 3-D array (patients, features, classes), split it
                    if len(shap_values_raw) == 1 and isinstance(shap_values_raw[0], np.ndarray) and shap_values_raw[0].ndim == 3:
                        arr = shap_values_raw[0]
                        if arr.shape[0] == clinical_data_numpy.shape[0] and arr.shape[1] == clinical_data_numpy.shape[1]:
                            shap_values = [arr[:, :, c] for c in range(arr.shape[2])]
                        else:
                            # Fallback: attempt to transpose if possible
                            shap_values = [np.asarray(x) for x in shap_values_raw]
                    else:
                        shap_values = []
                        for el in shap_values_raw:
                            a = np.asarray(el)
                            if a.ndim == 3 and a.shape[0] == clinical_data_numpy.shape[0] and a.shape[1] == clinical_data_numpy.shape[1]:
                                # element is (patients, features, classes)
                                shap_values = [a[:, :, c] for c in range(a.shape[2])]
                                break
                            elif a.ndim == 2:
                                shap_values.append(a)
                            else:
                                # try to reshape to (n_patients, n_features)
                                shap_values.append(a.reshape(clinical_data_numpy.shape[0], -1)[:, :clinical_data_numpy.shape[1]])
                elif isinstance(shap_values_raw, np.ndarray):
                    arr = shap_values_raw
                    if arr.ndim == 3:
                        # arr could be (patients, features, classes)
                        if arr.shape[0] == clinical_data_numpy.shape[0] and arr.shape[1] == clinical_data_numpy.shape[1]:
                            shap_values = [arr[:, :, c] for c in range(arr.shape[2])]
                        # or arr could be (classes, patients, features)
                        elif arr.shape[1] == clinical_data_numpy.shape[0] and arr.shape[2] == clinical_data_numpy.shape[1]:
                            shap_values = [arr[c] for c in range(arr.shape[0])]
                        else:
                            # As a last resort, attempt to transpose to (classes, patients, features)
                            shap_values = [np.transpose(arr, (2, 0, 1))[c] for c in range(arr.shape[2])]
                    elif arr.ndim == 2:
                        # Single-array case: create symmetric pair for binary
                        shap_values = [arr, -arr]
                    else:
                        # Unexpected shape -> try to coerce
                        shap_values = [arr.reshape(clinical_data_numpy.shape[0], -1)[:, :clinical_data_numpy.shape[1]]]
                else:
                    shap_values = [np.asarray(shap_values_raw)]
            except Exception as e:
                print(f"   Warning normalizing shap_values: {e}")
                shap_values = [np.asarray(shap_values_raw)]

            # At this point shap_values should be a list of 2-D arrays (n_patients, n_features)
            alpha = self.model.alphas[model_idx]
            weight = alpha / total_alpha

            if all_shap_aggregated is None:
                # Initialize per-class aggregated arrays
                all_shap_aggregated = [np.zeros_like(shap_values[c], dtype=float) for c in range(len(shap_values))]
            # If shape mismatch between existing aggregation and new shap_values, align by truncation
            if len(all_shap_aggregated) != len(shap_values):
                # If existing has different number of classes, expand or truncate to min
                min_c = min(len(all_shap_aggregated), len(shap_values)) if all_shap_aggregated is not None else len(shap_values)
                if min_c == 0:
                    all_shap_aggregated = [np.zeros_like(shap_values[c], dtype=float) for c in range(len(shap_values))]
                else:
                    all_shap_aggregated = [all_shap_aggregated[c] for c in range(min_c)]
                    shap_values = [shap_values[c] for c in range(min_c)]

            # Add weighted SHAP values per class
            for c in range(len(shap_values)):
                try:
                    all_shap_aggregated[c] += shap_values[c].astype(float) * weight
                except Exception:
                    all_shap_aggregated[c] += np.asarray(shap_values[c], dtype=float) * weight
        
        # Convert to numpy array: (n_classes, n_patients, n_features)
        # all_shap_aggregated is a list of arrays, each with shape (n_patients, n_features)
        # Stack them along axis 0 to get (n_classes, n_patients, n_features)
        try:
            shap_array = np.stack(all_shap_aggregated, axis=0).astype(float)
            print(f"   Debug: stacked shap_array shape={shap_array.shape} (n_classes, n_patients, n_features)")
        except Exception as e:
            print(f"   Warning: stack failed ({e}), trying np.array")
            shap_array = np.array(all_shap_aggregated, dtype=float)
            print(f"   Debug: np.array shap_array shape={shap_array.shape}")
        
        # Get feature names
        if self.feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(shap_array.shape[2])]
        else:
            # Ensure both arrays have same length
            min_len = min(len(self.feature_names), shap_array.shape[2])
            if min_len != shap_array.shape[2] or min_len != len(self.feature_names):
                print(f"   ⚠ Adjusting feature length in summary: feature_names={len(self.feature_names)}, shap_features={shap_array.shape[2]}, using min={min_len}")
            feature_names = self.feature_names[:min_len]
            # Also truncate shap_array to match
            shap_array = shap_array[:, :, :min_len]
        
        return {
            'global_importance': global_importance,
            'all_shap_values': shap_array,  # (n_classes, n_patients, n_features)
            'feature_names': feature_names,
            'patient_indices': sampled_data.index.tolist(),
            'n_patients': len(sampled_data),
            'n_features': len(feature_names),
            'n_classes': len(all_shap_aggregated)
        }
    
    def plot_waterfall(self, clinical_data: pd.DataFrame, patient_idx: int, 
                      class_names: Dict = None):
        """
        Create a waterfall plot showing feature contributions for a patient.
        
        Args:
            clinical_data: DataFrame with clinical features (single patient)
            patient_idx: Index of the patient
            class_names: Dict mapping class indices to names (e.g., {0: 'CN', 1: 'MCI', 2: 'AD'})
        """
        explanation = self.explain_patient(clinical_data, patient_idx)
        
        if 'error' in explanation:
            print(f"Error: {explanation['error']}")
            return
        
        # Use shap library's waterfall plot
        pred_class_idx = explanation['predicted_class_idx']
        
        # Get the first explainer for plotting
        if len(self.clinical_explainers) > 0:
            explainer = self.clinical_explainers[0]
            shap_values = explainer.shap_values(clinical_data, check_additivity=False)
            
            # Create explanation object for waterfall plot
            class_name = class_names.get(pred_class_idx, f"Class {pred_class_idx}") if class_names else f"Class {pred_class_idx}"
            
            print(f"\nSHAP Waterfall Plot for Patient {patient_idx}")
            print(f"Predicted Class: {class_name}")
            print("\nTop 5 Contributing Features:")
            print(explanation['top_5_features'][['feature', 'shap_value']].to_string(index=False))
    
    def plot_summary(self, clinical_data: pd.DataFrame, max_patients: int = 100,
                    max_display: int = 20, save_path: str = None):
        """
        Create a SHAP summary plot showing the global feature importance
        across all patients.
        
        Args:
            clinical_data: DataFrame with clinical features
            max_patients: Maximum number of patients to include (for computational efficiency)
            max_display: Maximum number of features to display in the plot
            save_path: Optional path to save the figure (e.g., 'shap_summary.png')
        """
        import matplotlib.pyplot as plt
        
        summary_data = self.compute_shap_summary(clinical_data, max_patients=max_patients)
        
        if 'error' in summary_data:
            print(f"Error: {summary_data['error']}")
            return
        
        # Extract data
        shap_values = summary_data['all_shap_values']  # Should be (n_classes, n_patients, n_features)
        feature_names = summary_data['feature_names']
        
        print(f"   Debug: shap_values shape={shap_values.shape}")
        
        # Ensure correct shape: (n_classes, n_patients, n_features)
        shap_values = np.asarray(shap_values, dtype=float)
        
        if shap_values.ndim != 3:
            raise ValueError(f"shap_values must be 3-D, got shape {shap_values.shape}")
        
        # Detect and fix wrong axis order
        # Expected: (n_classes, n_patients, n_features) - typically n_classes=3, n_patients=100, n_features=46
        # If first dimension is much larger than last, likely (n_patients, n_features, n_classes) - needs transpose
        if shap_values.shape[0] > 10 and shap_values.shape[2] <= 5:
            # Likely (n_patients, n_features, n_classes), transpose to (n_classes, n_patients, n_features)
            shap_values = np.transpose(shap_values, (2, 0, 1))
            print(f"   ⚠ Transposed shap_values from (patients, features, classes) to (classes, patients, features): {shap_values.shape}")
        
        # Average SHAP values across classes for summary plot
        # Compute both signed and absolute averages across classes
        # Shape: (n_patients, n_features)
        shap_signed = np.mean(shap_values, axis=0)
        shap_avg = np.mean(np.abs(shap_values), axis=0)
        print(f"   Debug: shap_signed shape={shap_signed.shape}, shap_avg shape={shap_avg.shape}")
        
        # Create summary plot
        print("\nCreating SHAP summary plot...")
        
        # Use shap's built-in summary plot
        plt.figure(figsize=(10, 8))
        
        # Get the clinical data for the sampled patients
        sampled_indices = summary_data['patient_indices']
        sampled_data = clinical_data.loc[sampled_indices]

        # Ensure sampled_data index aligns with shap_avg rows
        if shap_avg.ndim != 2:
            raise ValueError(f"shap_avg must be 2-D (n_patients, n_features), got shape {shap_avg.shape}")
        
        print(f"   Debug: shap_avg shape={shap_avg.shape}, sampled_data shape={sampled_data.shape}, n_features_names={len(feature_names)}")
        
        if sampled_data.shape[0] != shap_avg.shape[0]:
            # Try resetting index to match ordering used in SHAP computation
            sampled_data = sampled_data.reset_index(drop=True)
            if sampled_data.shape[0] != shap_avg.shape[0]:
                raise ValueError(f"Mismatch between sampled_data rows ({sampled_data.shape[0]}) and shap rows ({shap_avg.shape[0]})")

        try:
            # Use mean absolute SHAP across classes for single summary
            # Build a features DataFrame matching shap_avg shape
            try:
                # Ensure sampled_data columns match feature_names length
                n_features_needed = len(feature_names)
                if sampled_data.shape[1] > n_features_needed:
                    sampled_data = sampled_data.iloc[:, :n_features_needed]
                elif sampled_data.shape[1] < n_features_needed:
                    feature_names = feature_names[:sampled_data.shape[1]]
                    shap_avg = shap_avg[:, :sampled_data.shape[1]]
                
                features_df = pd.DataFrame(sampled_data.values, columns=feature_names)
            except Exception as e:
                print(f"Warning creating features_df: {e}")
                # Fallback: use sampled_data directly
                features_df = sampled_data.copy()
                feature_names = list(features_df.columns)[:shap_avg.shape[1]]
                shap_avg = shap_avg[:, :len(feature_names)]

            # Ensure we plot the same top features as in the global importance table
            importance_df = summary_data.get('global_importance')
            if importance_df is not None and not importance_df.empty:
                top_features = importance_df['feature'].head(max_display).tolist()
                # Find indices of these features in feature_names
                indices = []
                missing = []
                for f in top_features:
                    if f in feature_names:
                        indices.append(feature_names.index(f))
                    else:
                        missing.append(f)

                if len(indices) == 0:
                    print(f"   ⚠ None of the top {max_display} importance features found in sampled data columns. Falling back to full feature set.")
                    selected_shap = shap_signed
                    selected_df = features_df
                    selected_feature_names = feature_names
                else:
                    if missing:
                        print(f"   ⚠ Warning: top features missing from data and will be skipped: {missing}")
                    selected_shap = shap_signed[:, indices]
                    selected_df = features_df.iloc[:, indices]
                    selected_feature_names = [feature_names[i] for i in indices]
            else:
                # No importance table available — plot full set
                selected_shap = shap_signed
                selected_df = features_df
                selected_feature_names = feature_names

            shap.summary_plot(
                selected_shap,
                selected_df,
                max_display=min(max_display, selected_shap.shape[1]),
                show=False,
                feature_names=selected_feature_names,
                sort=False
            )
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                print(f"✓ SHAP summary plot saved to: {save_path}")
            else:
                plt.tight_layout()
                plt.show()
        
        except Exception as e:
            print(f"Could not create summary plot: {e}")
            print("\nAlternative: Top 10 most important features:")
            print(summary_data['global_importance'].head(10).to_string())
        
        finally:
            plt.close()
    
    def save_summary_table(self, clinical_data: pd.DataFrame, 
                          max_patients: int = None,
                          save_path: str = 'shap_global_importance.csv'):
        """
        Compute and save global feature importance table to CSV.
        
        Args:
            clinical_data: DataFrame with clinical features
            max_patients: Maximum number of patients to analyze
            save_path: Path to save the CSV file
        """
        summary_data = self.compute_shap_summary(clinical_data, max_patients=max_patients)
        
        if 'error' not in summary_data:
            importance_df = summary_data['global_importance']
            # Ensure numeric columns
            for col in ['mean_abs_shap', 'mean_shap_signed']:
                if col in importance_df.columns:
                    importance_df[col] = pd.to_numeric(importance_df[col], errors='coerce')

            # Add rank by absolute importance for clarity
            importance_df['rank_abs'] = importance_df['mean_abs_shap'].rank(ascending=False, method='dense').astype(int)

            importance_df.to_csv(save_path, index=False)
            print(f"✓ Global importance table saved to: {save_path}")
            return importance_df
        else:
            print(f"Error: {summary_data['error']}")
            return None
