"""
Run calibration experiments across all folds with comprehensive evaluation.

This script:
1. Trains models on all folds (or loads pre-trained models if train=false)
2. Applies calibration (Temperature Scaling and/or Isotonic Regression)
3. Evaluates cost-sensitive decision making
4. Aggregates results across folds
5. Generates comprehensive reports and visualizations

Usage:
    # Train and calibrate:
    python run_calibration_experiments.py --exp_name exp13 [--augmented] [--output-dir DIR]
    
    # Evaluation only (load pre-trained models):
    python run_calibration_experiments.py --exp_name exp15 [--augmented] [--output-dir DIR]
"""

import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List
import yaml

sys.path.append(str(Path(__file__).parent))

from src.data.processing_data import processing_features_cv_with_calibration
from src.train.boosting import training_function
from src.utils.save_load import save_model, load_model
from src.predict.calibration import (
    calibrate_multimodal_model,
    CostSensitiveDecision,
    CalibrationEvaluator
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)


class CalibrationExperimentRunner:
    """Manages calibration experiments across multiple folds."""
    
    def __init__(self, config: Dict, output_dir: Path, use_augmented: bool = False, 
                 is_train: bool = True, reuse_from: str = None, exp_name: str = None):
        self.config = config
        self.output_dir = output_dir
        self.use_augmented = use_augmented
        self.is_train = is_train
        self.reuse_from = reuse_from
        self.exp_name = exp_name
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'fold_results').mkdir(exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        
        # Results storage
        self.fold_results = []
        self.aggregate_metrics = {}
        
    def run_single_fold(self, fold_idx: int, fold_data: Dict):
        """Run calibration experiment on a single fold."""
        print("\n" + "="*70)
        print(f"FOLD {fold_idx}")
        print("="*70 + "\n")
        
        fold_output_dir = self.output_dir / 'fold_results' / f'fold_{fold_idx}'
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract data
        X_train = fold_data['X_train']
        y_train = fold_data['y_train']
        X_calib = fold_data['X_calib']
        y_calib = fold_data['y_calib']
        X_test = fold_data['X_test']
        y_test = fold_data['y_test']
        
        print(f"Train: {len(y_train)}, Calib: {len(y_calib)}, Test: {len(y_test)}")
        
        # Train or load model
        if self.is_train:
            print("\nTraining model...")
            # Include all config parameters except calibration-specific ones
            # Keep train_cost_sensitive for cost-sensitive training
            params = {k: v for k, v in self.config.items() 
                     if k not in ['apply_temperature_scaling', 'apply_isotonic_regression', 
                                  'use_cost_sensitive', 'aggressive_costs', 'cost_scale', 'name',
                                  'use_conformal', 'conformal_alpha']}
            # Ensure required parameters
            params.update({
                "id": 1,
                "mod": None
            })
            
            model = training_function(X_train, y_train, fold_idx, params)
            print("Training completed!")
            
            # Save trained model
            if self.exp_name:
                model_name = f"{self.exp_name}_fold_{fold_idx}"
                save_model(model, model_name)
                print(f"Model saved as {model_name}")
        else:
            # Load pre-trained model
            model_source = self.reuse_from if self.reuse_from else self.exp_name
            model_name = f"{model_source}_fold_{fold_idx}"
            print(f"\nLoading pre-trained model: {model_name}...")
            model = load_model(model_name)
            print("Model loaded successfully!")
        
        # Apply calibration
        print("\nApplying calibration pipeline...")
        calib_results = calibrate_multimodal_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_calib=X_calib,
            y_calib=y_calib,
            X_test=X_test,
            y_test=y_test,
            apply_temperature_scaling=self.config.get('apply_temperature_scaling', False),
            apply_isotonic_regression=self.config.get('apply_isotonic_regression', False),
            use_cost_sensitive=self.config.get('use_cost_sensitive', True),
            aggressive_costs=self.config.get('aggressive_costs', False),
            cost_scale=self.config.get('cost_scale', 1.0),
            output_dir=fold_output_dir
        )
        
        # Compute metrics
        le = calib_results['label_encoder']
        class_names = le.classes_
        y_test_encoded = le.transform(y_test)
        
        y_pred_standard = calib_results['predictions_standard']
        y_pred_costsens = calib_results['predictions_cost_sensitive']
        
        # Standard metrics
        acc_standard = accuracy_score(y_test_encoded, y_pred_standard)
        prec_std, rec_std, f1_std, _ = precision_recall_fscore_support(
            y_test_encoded, y_pred_standard, average='macro', zero_division=0
        )
        
        # Cost-sensitive metrics
        acc_costsens = accuracy_score(y_test_encoded, y_pred_costsens)
        prec_cs, rec_cs, f1_cs, _ = precision_recall_fscore_support(
            y_test_encoded, y_pred_costsens, average='macro', zero_division=0
        )
        
        # Per-class metrics
        per_class_metrics_std = self._compute_per_class_metrics(
            y_test_encoded, y_pred_standard, class_names
        )
        per_class_metrics_cs = self._compute_per_class_metrics(
            y_test_encoded, y_pred_costsens, class_names
        )
        
        # Calibration metrics
        calib_metrics = calib_results['calibration_metrics']
        ece_before = calib_metrics.loc[calib_metrics['Metric'] == 'ECE', 'Before'].values[0]
        ece_after = calib_metrics.loc[calib_metrics['Metric'] == 'ECE', 'After'].values[0]
        
        # Cost metrics
        if calib_results['cost_evaluation_standard'] is not None:
            cost_std = calib_results['cost_evaluation_standard']['mean_cost_per_sample']
            cost_cs = calib_results['cost_evaluation_cost_sensitive']['mean_cost_per_sample']
            cost_reduction = cost_std - cost_cs
            cost_reduction_pct = (cost_reduction / cost_std) * 100 if cost_std > 0 else 0
        else:
            cost_std = cost_cs = cost_reduction = cost_reduction_pct = None
        
        # Store fold results
        fold_result = {
            'fold': fold_idx,
            # Standard predictions
            'accuracy_standard': acc_standard,
            'precision_standard': prec_std,
            'recall_standard': rec_std,
            'f1_standard': f1_std,
            # Cost-sensitive predictions
            'accuracy_cost_sensitive': acc_costsens,
            'precision_cost_sensitive': prec_cs,
            'recall_cost_sensitive': rec_cs,
            'f1_cost_sensitive': f1_cs,
            # Calibration
            'ece_before': ece_before,
            'ece_after': ece_after,
            'ece_improvement': ece_before - ece_after,
            # Cost metrics
            'cost_standard': cost_std,
            'cost_cost_sensitive': cost_cs,
            'cost_reduction': cost_reduction,
            'cost_reduction_pct': cost_reduction_pct,
            # Per-class metrics
            'per_class_standard': per_class_metrics_std,
            'per_class_cost_sensitive': per_class_metrics_cs,
        }
        
        self.fold_results.append(fold_result)
        
        # Save fold results
        self._save_fold_results(fold_idx, fold_result, fold_output_dir)
        
        return fold_result
    
    def _compute_per_class_metrics(self, y_true, y_pred, class_names):
        """Compute precision, recall, F1 for each class."""
        prec, rec, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics = {}
        for i, class_name in enumerate(class_names):
            metrics[class_name] = {
                'precision': prec[i],
                'recall': rec[i],
                'f1': f1[i],
                'support': support[i]
            }
        
        return metrics
    
    def _save_fold_results(self, fold_idx: int, fold_result: Dict, output_dir: Path):
        """Save results for a single fold."""
        # Save summary
        summary = {k: v for k, v in fold_result.items() 
                  if k not in ['per_class_standard', 'per_class_cost_sensitive']}
        
        with open(output_dir / f'fold_{fold_idx}_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save per-class metrics
        per_class_df = pd.DataFrame({
            'Class': [],
            'Metric': [],
            'Standard': [],
            'Cost_Sensitive': []
        })
        
        for class_name in fold_result['per_class_standard'].keys():
            for metric in ['precision', 'recall', 'f1']:
                per_class_df = pd.concat([per_class_df, pd.DataFrame({
                    'Class': [class_name],
                    'Metric': [metric],
                    'Standard': [fold_result['per_class_standard'][class_name][metric]],
                    'Cost_Sensitive': [fold_result['per_class_cost_sensitive'][class_name][metric]]
                })], ignore_index=True)
        
        per_class_df.to_csv(output_dir / f'fold_{fold_idx}_per_class_metrics.csv', index=False)
    
    def aggregate_results(self):
        """Aggregate results across all folds."""
        print("\n" + "="*70)
        print("AGGREGATING RESULTS ACROSS FOLDS")
        print("="*70 + "\n")
        
        # Extract metrics
        metrics_to_aggregate = [
            'accuracy_standard', 'precision_standard', 'recall_standard', 'f1_standard',
            'accuracy_cost_sensitive', 'precision_cost_sensitive', 'recall_cost_sensitive', 'f1_cost_sensitive',
            'ece_before', 'ece_after', 'ece_improvement',
            'cost_standard', 'cost_cost_sensitive', 'cost_reduction', 'cost_reduction_pct'
        ]
        
        aggregated = {}
        for metric in metrics_to_aggregate:
            values = [r[metric] for r in self.fold_results if r[metric] is not None]
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_min'] = np.min(values)
                aggregated[f'{metric}_max'] = np.max(values)
        
        self.aggregate_metrics = aggregated
        
        # Save aggregated results
        agg_df = pd.DataFrame([
            {
                'Metric': metric.replace('_mean', ''),
                'Mean': aggregated[f'{metric}_mean'],
                'Std': aggregated[f'{metric}_std'],
                'Min': aggregated[f'{metric}_min'],
                'Max': aggregated[f'{metric}_max']
            }
            for metric in metrics_to_aggregate
            if f'{metric}_mean' in aggregated
        ])
        
        agg_path = self.output_dir / 'aggregated_metrics.csv'
        agg_df.to_csv(agg_path, index=False)
        print(f"Aggregated metrics saved to {agg_path}")
        
        return aggregated
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        print("\n" + "="*70)
        print("GENERATING SUMMARY REPORT")
        print("="*70 + "\n")
        
        report_path = self.output_dir / 'summary_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("CALIBRATION EXPERIMENT SUMMARY REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Configuration: {self.config['name']}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of folds: {len(self.fold_results)}\n")
            f.write(f"Data augmentation: {self.use_augmented}\n\n")
            
            f.write("-"*70 + "\n")
            f.write("KEY RESULTS\n")
            f.write("-"*70 + "\n\n")
            
            # Standard predictions
            f.write("Standard Predictions (Argmax):\n")
            f.write(f"  Accuracy:  {self.aggregate_metrics['accuracy_standard_mean']:.4f} ± {self.aggregate_metrics['accuracy_standard_std']:.4f}\n")
            f.write(f"  Precision: {self.aggregate_metrics['precision_standard_mean']:.4f} ± {self.aggregate_metrics['precision_standard_std']:.4f}\n")
            f.write(f"  Recall:    {self.aggregate_metrics['recall_standard_mean']:.4f} ± {self.aggregate_metrics['recall_standard_std']:.4f}\n")
            f.write(f"  F1-Score:  {self.aggregate_metrics['f1_standard_mean']:.4f} ± {self.aggregate_metrics['f1_standard_std']:.4f}\n\n")
            
            # Cost-sensitive predictions
            f.write("Cost-Sensitive Predictions:\n")
            f.write(f"  Accuracy:  {self.aggregate_metrics['accuracy_cost_sensitive_mean']:.4f} ± {self.aggregate_metrics['accuracy_cost_sensitive_std']:.4f}\n")
            f.write(f"  Precision: {self.aggregate_metrics['precision_cost_sensitive_mean']:.4f} ± {self.aggregate_metrics['precision_cost_sensitive_std']:.4f}\n")
            f.write(f"  Recall:    {self.aggregate_metrics['recall_cost_sensitive_mean']:.4f} ± {self.aggregate_metrics['recall_cost_sensitive_std']:.4f}\n")
            f.write(f"  F1-Score:  {self.aggregate_metrics['f1_cost_sensitive_mean']:.4f} ± {self.aggregate_metrics['f1_cost_sensitive_std']:.4f}\n\n")
            
            # Calibration improvement
            f.write("Calibration Improvement:\n")
            f.write(f"  ECE Before: {self.aggregate_metrics['ece_before_mean']:.4f} ± {self.aggregate_metrics['ece_before_std']:.4f}\n")
            f.write(f"  ECE After:  {self.aggregate_metrics['ece_after_mean']:.4f} ± {self.aggregate_metrics['ece_after_std']:.4f}\n")
            f.write(f"  Improvement: {self.aggregate_metrics['ece_improvement_mean']:.4f} ± {self.aggregate_metrics['ece_improvement_std']:.4f}\n\n")
            
            # Cost reduction
            if 'cost_standard_mean' in self.aggregate_metrics:
                f.write("Clinical Cost Analysis:\n")
                f.write(f"  Standard Cost:       {self.aggregate_metrics['cost_standard_mean']:.4f} ± {self.aggregate_metrics['cost_standard_std']:.4f}\n")
                f.write(f"  Cost-Sensitive Cost: {self.aggregate_metrics['cost_cost_sensitive_mean']:.4f} ± {self.aggregate_metrics['cost_cost_sensitive_std']:.4f}\n")
                f.write(f"  Cost Reduction:      {self.aggregate_metrics['cost_reduction_mean']:.4f} ± {self.aggregate_metrics['cost_reduction_std']:.4f}\n")
                f.write(f"  Reduction %:         {self.aggregate_metrics['cost_reduction_pct_mean']:.2f}% ± {self.aggregate_metrics['cost_reduction_pct_std']:.2f}%\n\n")
            
            f.write("="*70 + "\n")
        
        print(f"Summary report saved to {report_path}")
        
        # Print to console
        with open(report_path, 'r') as f:
            print(f.read())
    
    def generate_visualizations(self):
        """Generate aggregate visualizations."""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70 + "\n")
        
        viz_dir = self.output_dir / 'visualizations'
        
        # 1. Metrics comparison across folds
        self._plot_metrics_comparison(viz_dir)
        
        # 2. Cost reduction visualization
        self._plot_cost_reduction(viz_dir)
        
        # 3. Calibration improvement
        self._plot_calibration_improvement(viz_dir)
        
        print(f"Visualizations saved to {viz_dir}")
    
    def _plot_metrics_comparison(self, viz_dir: Path):
        """Plot comparison of metrics across folds."""
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Extract data
            folds = [r['fold'] for r in self.fold_results]
            standard = [r[f'{metric}_standard'] for r in self.fold_results]
            cost_sens = [r[f'{metric}_cost_sensitive'] for r in self.fold_results]
            
            x = np.arange(len(folds))
            width = 0.35
            
            ax.bar(x - width/2, standard, width, label='Standard', alpha=0.8)
            ax.bar(x + width/2, cost_sens, width, label='Cost-Sensitive', alpha=0.8)
            
            ax.set_xlabel('Fold', fontsize=11)
            ax.set_ylabel(metric.capitalize(), fontsize=11)
            ax.set_title(f'{metric.capitalize()} Comparison', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(folds)
            ax.legend()
            ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = viz_dir / 'metrics_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - Metrics comparison saved to {save_path.name}")
    
    def _plot_cost_reduction(self, viz_dir: Path):
        """Plot cost reduction across folds."""
        if None in [r['cost_standard'] for r in self.fold_results]:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        folds = [r['fold'] for r in self.fold_results]
        cost_std = [r['cost_standard'] for r in self.fold_results]
        cost_cs = [r['cost_cost_sensitive'] for r in self.fold_results]
        
        x = np.arange(len(folds))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, cost_std, width, label='Standard', 
                      color='indianred', alpha=0.8)
        bars2 = ax.bar(x + width/2, cost_cs, width, label='Cost-Sensitive',
                      color='seagreen', alpha=0.8)
        
        # Add reduction percentage labels
        for i, (std, cs) in enumerate(zip(cost_std, cost_cs)):
            reduction = ((std - cs) / std) * 100
            sign = '-' if reduction < 0 else '+'
            ax.text(i, max(std, cs) + 0.02, f'{sign}{abs(reduction):.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Fold', fontsize=12)
        ax.set_ylabel('Mean Cost per Sample', fontsize=12)
        ax.set_title('Clinical Cost Reduction: Standard vs Cost-Sensitive',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(folds)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = viz_dir / 'cost_reduction.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - Cost reduction plot saved to {save_path.name}")
    
    def _plot_calibration_improvement(self, viz_dir: Path):
        """Plot calibration improvement across folds."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        folds = [r['fold'] for r in self.fold_results]
        ece_before = [r['ece_before'] for r in self.fold_results]
        ece_after = [r['ece_after'] for r in self.fold_results]
        
        x = np.arange(len(folds))
        width = 0.35
        
        ax.bar(x - width/2, ece_before, width, label='Before Calibration',
              color='coral', alpha=0.8)
        ax.bar(x + width/2, ece_after, width, label='After Calibration',
              color='skyblue', alpha=0.8)
        
        # Add improvement percentage
        for i, (before, after) in enumerate(zip(ece_before, ece_after)):
            if before > 0:
                improvement = ((before - after) / before) * 100
                ax.text(i, max(before, after) + 0.005, f'↓{improvement:.1f}%',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Fold', fontsize=12)
        ax.set_ylabel('Expected Calibration Error (ECE)', fontsize=12)
        ax.set_title('Calibration Improvement via Isotonic Regression',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(folds)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = viz_dir / 'calibration_improvement.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - Calibration improvement plot saved to {save_path.name}")
    
    def run_all_folds(self, n_folds: int = 5):
        """Run experiments on all folds."""
        print("\n" + "="*70)
        print(f"STARTING CALIBRATION EXPERIMENTS: {self.config['name']}")
        print("="*70)
        print(f"Configuration:")
        for key, value in self.config.items():
            if key != 'name':
                print(f"  {key}: {value}")
        print("="*70 + "\n")
        
        # Load data
        print("Loading data with calibration split...")
        folds = processing_features_cv_with_calibration(use_augmented=self.use_augmented)
        print(f"Loaded {len(folds)} folds\n")
        
        # Run each fold
        for fold_idx in range(min(n_folds, len(folds))):
            # Convert to 1-indexed for consistency with other experiments
            fold_num = fold_idx + 1
            try:
                self.run_single_fold(fold_num, folds[fold_idx])
            except Exception as e:
                print(f"\n[ERROR] Fold {fold_num} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Aggregate and visualize
        if self.fold_results:
            self.aggregate_results()
            self.generate_summary_report()
            self.generate_visualizations()
            
            print("\n" + "="*70)
            print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"\nResults saved to: {self.output_dir.absolute()}")
        else:
            print("\n[WARNING] No successful fold results to aggregate!")


def main():
    # Load configurations from config.yaml
    config_path = Path(__file__).parent / 'src' / 'configs' / 'config.yaml'
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    
    parser = argparse.ArgumentParser(
        description='Run calibration experiments across all folds'
    )
    parser.add_argument('--exp_name', type=str, default='exp13',
                       help='Experiment configuration to use')
    parser.add_argument('--augmented', action='store_true',
                       help='Use data augmentation (default: False)')
    parser.add_argument('--output-dir', type=str, 
                       default='results/calibration_experiments',
                       help='Output directory for results')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of folds to run')
    
    args = parser.parse_args()
    
    # Get experiment configuration
    exp_config = full_config[args.exp_name]
    config = exp_config['params']
    
    # Check if training or evaluation only
    is_train = exp_config.get('train', True)
    reuse_from = exp_config.get('reuse_from', None)
    
    if not is_train and reuse_from:
        print(f"\n⚠️  Running in EVALUATION-ONLY mode")
        print(f"   Loading models from: {reuse_from}")
    
    # Add name for display
    config['name'] = f"Experiment {args.exp_name}"
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.exp_name}_{timestamp}"
    
    # Run experiments
    runner = CalibrationExperimentRunner(
        config=config,
        output_dir=output_dir,
        use_augmented=args.augmented,
        is_train=is_train,
        reuse_from=reuse_from,
        exp_name=args.exp_name
    )
    
    runner.run_all_folds(n_folds=args.n_folds)


if __name__ == "__main__":
    main()
