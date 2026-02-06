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
                # Create SHAP explainer for this RF
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
            shap_values = explainer.shap_values(clinical_data_numpy)
            
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
    
    def get_global_importance(self, clinical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute global feature importance across all patients.
        
        Args:
            clinical_data: DataFrame with clinical features
            
        Returns:
            DataFrame with mean absolute SHAP values per feature
        """
        if len(self.clinical_explainers) == 0:
            return pd.DataFrame()
        
        # Use first explainer to compute SHAP values on all data
        explainer = self.clinical_explainers[0]
        shap_values = explainer.shap_values(clinical_data)
        
        # Average absolute SHAP values across all classes and patients
        mean_abs_shap = []
        for class_shap in shap_values:
            mean_abs_shap.append(np.abs(class_shap).mean(axis=0))
        
        # Average across classes
        global_importance = np.mean(mean_abs_shap, axis=0)
        
        if self.feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(global_importance))]
        else:
            feature_names = self.feature_names
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': global_importance
        }).sort_values('mean_abs_shap', ascending=False)
        
        return importance_df
    
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
            shap_values = explainer.shap_values(clinical_data)
            
            # Create explanation object for waterfall plot
            class_name = class_names.get(pred_class_idx, f"Class {pred_class_idx}") if class_names else f"Class {pred_class_idx}"
            
            print(f"\nSHAP Waterfall Plot for Patient {patient_idx}")
            print(f"Predicted Class: {class_name}")
            print("\nTop 5 Contributing Features:")
            print(explanation['top_5_features'][['feature', 'shap_value']].to_string(index=False))
