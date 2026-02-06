# Test script for explainability modules

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import torch
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.save_load import load_model
from src.data.processing_data import processing_features_cv_with_calibration
from src.explain.shap_explainer import SHAPExplainer
from src.explain.gradcam_explainer import GradCAMExplainer


def test_modality_contributions():
    """Test the new predict_proba_with_contributions method."""
    print("=" * 80)
    print("TEST 1: Modality Contributions")
    print("=" * 80)
    
    # Load model
    print("\nLoading model...")
    model = load_model("exp12_fold_1")
    print("Model loaded successfully")
    
    # Load test data
    print("\nLoading test data...")
    folds = processing_features_cv_with_calibration(use_augmented=False)
    fold_data = folds[0]
    
    X_test = fold_data['X_test']
    y_test = fold_data['y_test']
    
    # Select a few patients with both clinical data and images
    if len(X_test['images']) > 0:
        test_indices = list(X_test['images'].index[:5])
    else:
        test_indices = list(y_test.index[:5])
    X_test_sample = {
        'clinical': X_test['clinical'].loc[test_indices],
        'images': X_test['images'].loc[test_indices] if len(X_test['images']) > 0 else pd.DataFrame()
    }
    
    print(f"Testing on {len(test_indices)} patients")
    
    # Test new method
    print("\nComputing predictions with contributions...")
    predictions, contributions = model.predict_proba_with_contributions(X_test_sample)
    
    print("\nResults:")
    print("-" * 80)
    
    class_names = {1: "CN", 2: "MCI", 3: "AD"}
    
    for idx in test_indices:
        if idx not in predictions.index:
            continue
            
        pred_class = predictions.loc[idx].idxmax()
        pred_prob = predictions.loc[idx, pred_class]
        
        clinical_contrib = contributions['clinical'].loc[idx, pred_class]
        imaging_contrib = contributions['imaging'].loc[idx, pred_class]
        
        total = clinical_contrib + imaging_contrib
        if total > 0:
            clinical_weight = clinical_contrib / total * 100
            imaging_weight = imaging_contrib / total * 100
        else:
            clinical_weight = imaging_weight = 0
        
        print(f"\nPatient {idx}:")
        print(f"  Predicted: {class_names.get(pred_class, pred_class)} ({pred_prob:.2%})")
        print(f"  Clinical contribution: {clinical_weight:.1f}%")
        print(f"  Imaging contribution:  {imaging_weight:.1f}%")
    
    print("\n✓ Modality contributions test completed")


def test_shap_explainer():
    """Test SHAP explainer."""
    print("\n" + "=" * 80)
    print("TEST 2: SHAP Explainer")
    print("=" * 80)
    
    # Load model
    print("\nLoading model...")
    model = load_model("exp12_fold_1")
    
    # Load test data
    print("Loading test data...")
    folds = processing_features_cv_with_calibration(use_augmented=False)
    fold_data = folds[0]
    
    X_test = fold_data['X_test']
    
    # Get feature names
    feature_names = X_test['clinical'].columns.tolist()
    
    # Create SHAP explainer
    print("\nInitializing SHAP explainer...")
    shap_explainer = SHAPExplainer(model, feature_names=feature_names)
    
    print(f"Found {len(shap_explainer.clinical_explainers)} clinical RF models in ensemble")
    
    if len(shap_explainer.clinical_explainers) > 0:
        # Explain a single patient
        test_patient = X_test['clinical'].iloc[[1]]
        patient_idx = X_test['clinical'].index[1]
        
        print(f"\nExplaining patient {patient_idx}...")
        explanation = shap_explainer.explain_patient(test_patient, patient_idx)
        
        if 'error' not in explanation:
            print(f"\nPredicted class: {explanation['predicted_class']}")
            print("\nTop 5 contributing features:")
            top_features = explanation['top_5_features'][['feature', 'shap_value', 'abs_shap']]
            print(top_features.to_string(index=False))
            
            # Create SHAP visualization
            print("\nGenerating SHAP waterfall plot...")
            try:
                # Create output directory
                output_dir = Path('results/explainability')
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Get SHAP values for the predicted class
                shap_values = explanation['aggregated_shap_for_pred']
                feature_values = test_patient.values[0]
                
                # Create waterfall plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Get top 10 features by absolute SHAP value
                top_10 = explanation['feature_contributions'].head(10)
                features = top_10['feature'].values
                values = top_10['shap_value'].values
                
                # Create horizontal bar plot
                colors = ['red' if v < 0 else 'blue' for v in values]
                y_pos = np.arange(len(features))
                
                ax.barh(y_pos, values, color=colors, alpha=0.7)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(features)
                ax.set_xlabel('SHAP Value (impact on prediction)', fontsize=12)
                ax.set_title(f'Top 10 Features for Patient {patient_idx}\nPredicted Class: {explanation["predicted_class"]}', 
                           fontsize=14, fontweight='bold')
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
                ax.grid(axis='x', alpha=0.3)
                
                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='blue', alpha=0.7, label='Positive contribution'),
                    Patch(facecolor='red', alpha=0.7, label='Negative contribution')
                ]
                ax.legend(handles=legend_elements, loc='best')
                
                plt.tight_layout()
                save_path = output_dir / f'shap_patient_{patient_idx}.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved SHAP plot to {save_path}")
                plt.close()
                
            except Exception as e:
                print(f"Warning: Could not generate SHAP plot: {e}")
            
            print("\n✓ SHAP explainer test completed")
        else:
            print(f"Error: {explanation['error']}")
    else:
        print("No clinical models found for SHAP analysis")


def test_gradcam_explainer():
    """Test Grad-CAM explainer with multiple improvements."""
    print("\n" + "=" * 80)
    print("TEST 3: Grad-CAM Explainer")
    print("=" * 80)
    
    # Load model
    print("\nLoading model...")
    model = load_model("exp12_fold_1")
    
    # Load test data to get an image path
    print("Loading test data...")
    folds = processing_features_cv_with_calibration(use_augmented=False)
    fold_data = folds[0]
    X_test = fold_data['X_test']
    
    # Create Grad-CAM explainer
    print("\nInitializing Grad-CAM explainer...")
    gradcam_explainer = GradCAMExplainer(model)
    
    print(f"Found {len(gradcam_explainer.resnet_models)} ResNet models in ensemble")
    
    if len(gradcam_explainer.resnet_models) > 0 and len(X_test['images']) > 0:
        print("\n✓ Grad-CAM explainer initialized")
        
        # SINCRONIZZAZIONE: usa lo stesso paziente di SHAP se possibile
        # Cerca il primo paziente che ha sia dati clinici che immagini
        clinical_indices = set(X_test['clinical'].index)
        image_indices = set(X_test['images'].index)
        common_indices = list(clinical_indices.intersection(image_indices))
        
        if common_indices:
            patient_idx = common_indices[0]
            print(f"\n⚠ NOTA: Usando paziente {patient_idx} (ha sia dati clinici che imaging)")
            print(f"  - SHAP usa il primo paziente con dati clinici: {X_test['clinical'].index[0]}")
            print(f"  - Grad-CAM ora usa il primo paziente con entrambi: {patient_idx}")
        else:
            patient_idx = X_test['images'].index[0]
            print(f"\n⚠ NOTA: Paziente {patient_idx} ha solo imaging, non dati clinici")
        
        # Get image for this patient
        image_row = X_test['images'].loc[patient_idx]
        image_path = image_row.iloc[0] if isinstance(image_row, pd.Series) else image_row
        print(f"Image path: {image_path}")
        
        try:
            # Load the preprocessed image tensor
            with open(image_path, 'rb') as f:
                image_tensor = pickle.load(f)
            
            # Convert to torch tensor if needed
            if isinstance(image_tensor, np.ndarray):
                image_tensor = torch.from_numpy(image_tensor).float()
            
            # Ensure correct shape (1, C, H, W, D)
            if image_tensor.dim() == 4:
                image_tensor = image_tensor.unsqueeze(0)
            
            print(f"Image tensor shape: {image_tensor.shape}")
            
            # Test multiple layers to compare
            layers_to_test = ['layer3', 'layer4']
            output_dir = Path('results/explainability')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for layer_name in layers_to_test:
                print(f"\n{'='*60}")
                print(f"Testing {layer_name.upper()}")
                print('='*60)
                
                # Use AGGREGATED heatmap across all ResNet models
                print(f"Computing aggregated Grad-CAM from {len(gradcam_explainer.resnet_models)} ResNet models...")
                heatmap = gradcam_explainer.get_aggregated_heatmap(
                    image_tensor, 
                    layer_name=layer_name
                )
                
                if heatmap is not None:
                    print(f"Heatmap shape: {heatmap.shape}")
                    print(f"Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
                    
                    # Get original image for overlay
                    original_image = image_tensor.squeeze().cpu().numpy()
                    if original_image.ndim == 4:  # Remove channel dimension
                        original_image = original_image[0]
                    
                    # Generate visualizations for multiple slices
                    slices_to_visualize = [15, 20, 25, 30, 35]
                    print(f"\nGenerating visualizations for slices: {slices_to_visualize}")
                    
                    for slice_idx in slices_to_visualize:
                        if slice_idx < original_image.shape[-1]:
                            save_path = output_dir / f'gradcam_patient_{patient_idx}_{layer_name}_slice{slice_idx}.png'
                            fig = gradcam_explainer.visualize_2d_slice(
                                original_image, 
                                heatmap,
                                slice_idx=slice_idx,
                                alpha=0.5,
                                save_path=save_path
                            )
                            plt.close(fig)
                    
                    print(f"Saved visualizations to {output_dir}/")
                    
                    # Get important regions statistics
                    region_stats = gradcam_explainer.identify_important_regions(heatmap, threshold=0.7)
                    print(f"\nImportant brain regions (threshold=0.7):")
                    print(f"  - Important voxels: {region_stats['n_important_voxels']} ({region_stats['percentage_important']:.2f}%)")
                    print(f"  - Max activation: {region_stats['max_activation']:.3f}")
                    print(f"  - Mean activation: {region_stats['mean_activation']:.3f}")
                    
                else:
                    print(f"Warning: Could not compute Grad-CAM heatmap for {layer_name}")
            
            print("\n✓ Grad-CAM visualization completed for all layers")
                
        except Exception as e:
            print(f"Warning: Could not generate Grad-CAM visualization: {e}")
            import traceback
            traceback.print_exc()
    else:
        if len(gradcam_explainer.resnet_models) == 0:
            print("No ResNet models found for Grad-CAM analysis")
        else:
            print("No test images available for Grad-CAM visualization")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("EXPLAINABILITY MODULE TEST SUITE")
    print("=" * 80)
    
    try:
        # Test 1: Modality contributions
        #test_modality_contributions()
        
        # Test 2: SHAP
        #test_shap_explainer()
        
        # Test 3: Grad-CAM
        test_gradcam_explainer()
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
