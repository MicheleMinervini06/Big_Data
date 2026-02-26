"""
Script to generate SHAP global summary plot and importance table.

This script computes SHAP values across all test patients to identify
the most important clinical features for Alzheimer's disease classification.

Usage:
    python generate_shap_summary.py
    
Output:
    - results/explainability/shap_global_summary.png
    - results/explainability/shap_global_importance.csv
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.utils.save_load import load_model
from src.data.processing_data import processing_features_cv_with_calibration
from src.explain.shap_explainer import SHAPExplainer


def main():
    """Generate SHAP global summary plot and importance table."""
    print("\n" + "=" * 80)
    print("SHAP GLOBAL SUMMARY GENERATOR")
    print("=" * 80)
    
    # Configuration
    MODEL_NAME = "exp12_fold_1"  # Change this to your model name
    MAX_PATIENTS = 100  # Number of patients to analyze (None = all)
    MAX_DISPLAY = 20  # Number of features to display in plot
    OUTPUT_DIR = Path('results/explainability')
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\n1. Loading model: {MODEL_NAME}")
    try:
        model = load_model(MODEL_NAME)
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return
    
    # Load test data
    print("\n2. Loading test data...")
    try:
        folds = processing_features_cv_with_calibration(use_augmented=False)
        fold_data = folds[0]  # Use first fold
        clinical_data = fold_data['X_test']['clinical']
        # Quick fix: drop non-informative identifier columns that skew SHAP (e.g., IMAGEUID)
        if 'IMAGEUID' in clinical_data.columns:
            clinical_data = clinical_data.drop(columns=['IMAGEUID'])
            print("   ✓ Dropped column: IMAGEUID (non-informative identifier)")
        print(f"   ✓ Loaded {len(clinical_data)} test patients")
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return
    
    # Get feature names
    feature_names = clinical_data.columns.tolist()
    print(f"   ✓ Found {len(feature_names)} clinical features")
    
    # Create SHAP explainer
    print("\n3. Initializing SHAP explainer...")
    try:
        shap_explainer = SHAPExplainer(model, feature_names=feature_names)
        n_models = len(shap_explainer.clinical_explainers)
        print(f"   ✓ Found {n_models} clinical RF models in ensemble")
        
        if n_models == 0:
            print("   ✗ No clinical models found. Cannot compute SHAP values.")
            return
    except Exception as e:
        print(f"   ✗ Error initializing SHAP explainer: {e}")
        return
    
    # Generate summary plot
    print("\n4. Generating SHAP global summary plot...")
    print(f"   - Analyzing {MAX_PATIENTS if MAX_PATIENTS else 'all'} patients")
    print(f"   - Displaying top {MAX_DISPLAY} features")
    
    save_path_plot = OUTPUT_DIR / 'shap_global_summary.png'
    try:
        shap_explainer.plot_summary(
            clinical_data, 
            max_patients=MAX_PATIENTS,
            max_display=MAX_DISPLAY,
            save_path=str(save_path_plot)
        )
        print(f"   ✓ Plot saved to: {save_path_plot}")
    except Exception as e:
        print(f"   ⚠ Warning: Could not generate plot: {e}")
        print("   Falling back to table-only output...")
    
    # Save importance table
    print("\n5. Saving feature importance table...")
    save_path_csv = OUTPUT_DIR / 'shap_global_importance.csv'
    try:
        importance_df = shap_explainer.save_summary_table(
            clinical_data,
            max_patients=MAX_PATIENTS,
            save_path=str(save_path_csv)
        )
        
        if importance_df is not None:
            print(f"   ✓ Table saved to: {save_path_csv}")
            print("\n" + "=" * 80)
            print("TOP 15 MOST IMPORTANT FEATURES")
            print("=" * 80)
            print(importance_df.head(15).to_string(index=False))
        else:
            print("   ✗ Could not generate importance table")
    except Exception as e:
        print(f"   ✗ Error saving importance table: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Patients analyzed: {MAX_PATIENTS if MAX_PATIENTS else len(clinical_data)}")
    print(f"Total features: {len(feature_names)}")
    print(f"\nOutput files:")
    print(f"  • {save_path_plot}")
    print(f"  • {save_path_csv}")
    print("\n✓ SHAP summary generation completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
