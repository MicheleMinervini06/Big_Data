"""
Probability Calibration and Cost-Sensitive Decision Making for Clinical Prediction

This module implements:
1. Temperature Scaling (TS) for calibrating neural network probabilities
2. Isotonic Regression (IR) for calibrating ensemble probabilities
3. Cost-sensitive Bayesian decision rule for clinical predictions
4. Calibration evaluation metrics (ECE, reliability diagrams, etc.)

Clinical Context:
    The cost matrix reflects clinical severity of misdiagnosis:
    - AD→CN (missed dementia): highest cost (1.0)
    - CN→AD (false positive dementia): very high cost (0.9)
    - Within-impaired errors (AD↔MCI): moderate cost (0.7-0.8)
    - MCI↔CN errors: lower but non-trivial cost (0.3-0.5)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class TemperatureScaling:
    """
    Temperature Scaling for calibrating neural network probabilities.
    
    Learns a single scalar parameter T (temperature) to scale logits before softmax:
        p_calibrated = softmax(logits / T)
    
    Reference:
        Guo et al. "On Calibration of Modern Neural Networks", ICML 2017
    
    Usage:
        ts = TemperatureScaling()
        ts.fit(logits_calib, y_calib)
        probs_calibrated = ts.transform(logits_test)
    """
    
    def __init__(self, device='cpu'):
        """
        Args:
            device: torch device ('cpu' or 'cuda')
        """
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # Initialize T=1.5
        self.device = device
        self.is_fitted = False
        
    def fit(self, logits: np.ndarray, y_true: np.ndarray, 
            max_iter: int = 50, lr: float = 0.01):
        """
        Fit temperature parameter using calibration set.
        
        Args:
            logits: Raw logits from neural network, shape (n_samples, n_classes)
            y_true: True labels, shape (n_samples,)
            max_iter: Maximum optimization iterations
            lr: Learning rate for LBFGS optimizer
        """
        # Convert to torch tensors
        logits_tensor = torch.FloatTensor(logits).to(self.device)
        labels_tensor = torch.LongTensor(y_true).to(self.device)
        
        # Reset temperature
        self.temperature = nn.Parameter(torch.ones(1, device=self.device) * 1.5)
        
        # Optimize temperature using NLL loss
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(logits_tensor / self.temperature, labels_tensor)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        self.is_fitted = True
        print(f"Temperature Scaling fitted: T = {self.temperature.item():.4f}")
        
    def transform(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Raw logits, shape (n_samples, n_classes)
            
        Returns:
            Calibrated probabilities, shape (n_samples, n_classes)
        """
        assert self.is_fitted, "Must call fit() before transform()"
        
        with torch.no_grad():
            logits_tensor = torch.FloatTensor(logits).to(self.device)
            scaled_probs = torch.softmax(logits_tensor / self.temperature, dim=1)
            
        return scaled_probs.cpu().numpy()
    
    def fit_transform(self, logits_calib: np.ndarray, y_calib: np.ndarray,
                      logits_test: np.ndarray) -> np.ndarray:
        """
        Fit on calibration set and transform test set.
        
        Args:
            logits_calib: Calibration logits
            y_calib: Calibration labels
            logits_test: Test logits
            
        Returns:
            Calibrated test probabilities
        """
        self.fit(logits_calib, y_calib)
        return self.transform(logits_test)


class IsotonicRegressionCalibrator:
    """
    Isotonic Regression for calibrating ensemble probabilities.
    
    Fits a monotonic mapping for each class to improve calibration
    of final ensemble predictions.
    
    Reference:
        Zadrozny & Elkan. "Transforming Classifier Scores into Accurate 
        Multiclass Probability Estimates", KDD 2002
    
    Usage:
        ir = IsotonicRegressionCalibrator()
        ir.fit(probs_calib, y_calib)
        probs_calibrated = ir.transform(probs_test)
    """
    
    def __init__(self):
        self.calibrators = {}  # One isotonic regressor per class
        self.classes = None
        self.is_fitted = False
        
    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        """
        Fit isotonic regression for each class.
        
        Args:
            probs: Predicted probabilities, shape (n_samples, n_classes)
            y_true: True labels, shape (n_samples,)
        """
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
            
        self.classes = np.unique(y_true)
        n_classes = len(self.classes)
        
        # Fit one isotonic regressor per class
        for i, cls in enumerate(self.classes):
            # Binary target: 1 if true class equals cls, 0 otherwise
            y_binary = (y_true == cls).astype(int)
            
            # Get predicted probability for this class
            p_class = probs[:, i] if probs.ndim > 1 else probs
            
            # Fit isotonic regression
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(p_class, y_binary)
            
            self.calibrators[cls] = ir
            
        self.is_fitted = True
        print(f"Isotonic Regression fitted for {n_classes} classes")
        
    def transform(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply isotonic regression to probabilities.
        
        Args:
            probs: Predicted probabilities, shape (n_samples, n_classes)
            
        Returns:
            Calibrated probabilities, shape (n_samples, n_classes)
        """
        assert self.is_fitted, "Must call fit() before transform()"
        
        n_samples = probs.shape[0]
        n_classes = len(self.classes)
        
        # Apply isotonic regression to each class
        calibrated_probs = np.zeros((n_samples, n_classes))
        for i, cls in enumerate(self.classes):
            p_class = probs[:, i]
            calibrated_probs[:, i] = self.calibrators[cls].transform(p_class)
            
        # Normalize to ensure probabilities sum to 1
        row_sums = calibrated_probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
        calibrated_probs = calibrated_probs / row_sums
        
        return calibrated_probs
    
    def fit_transform(self, probs_calib: np.ndarray, y_calib: np.ndarray,
                      probs_test: np.ndarray) -> np.ndarray:
        """
        Fit on calibration set and transform test set.
        """
        self.fit(probs_calib, y_calib)
        return self.transform(probs_test)


class CostSensitiveDecision:
    """
    Cost-sensitive Bayesian decision rule for clinical predictions.
    
    Given calibrated probabilities p(y=i|x) and cost matrix C(i,j),
    selects the class that minimizes expected cost:
        
        ŷ(x) = argmin_j Σ_i C(i,j) * p(y=i|x)
    
    Clinical cost matrix for CN/MCI/AD:
        Rows = true class (CN, MCI, AD)
        Cols = predicted class (CN, MCI, AD)
        
        C = [[0.0, 0.3, 0.9],    # True CN
             [0.5, 0.0, 0.7],    # True MCI
             [1.0, 0.8, 0.0]]    # True AD
             
    Interpretation:
        - AD→CN (1.0): Missed dementia - most severe error
        - CN→AD (0.9): False positive dementia - very serious
        - AD→MCI (0.8): Underestimating AD severity
        - MCI→AD (0.7): Overestimating MCI to AD
        - CN→MCI (0.3): False positive impairment
        - MCI→CN (0.5): Missed impairment
    """
    
    def __init__(self, cost_matrix: Optional[np.ndarray] = None, 
                 class_names: Optional[List[str]] = None,
                 aggressive_mode: bool = False,
                 cost_scale: float = 1.0):
        """
        Args:
            cost_matrix: Cost matrix C(i,j), shape (n_classes, n_classes)
                        If None, uses default clinical cost matrix for CN/MCI/AD
            class_names: Names of classes (e.g., ['CN', 'MCI', 'AD'])
            aggressive_mode: If True, uses 5x more aggressive costs
            cost_scale: Multiplicative scaling factor for all costs (default: 1.0)
        """
        if cost_matrix is None:
            if aggressive_mode:
                # Aggressive clinical cost matrix (5x stronger)
                self.cost_matrix = np.array([
                    [0.0, 1.5, 5.0],   # True CN: CN→AD very expensive
                    [2.5, 0.0, 3.5],   # True MCI: both errors costly
                    [10.0, 5.0, 0.0]   # True AD: AD→CN CRITICAL
                ])
                print("[COST-SENSITIVE] Using AGGRESSIVE cost matrix (5x)")
            else:
                # Default clinical cost matrix for CN/MCI/AD
                self.cost_matrix = np.array([
                    [0.0, 0.3, 0.9],  # True CN
                    [0.5, 0.0, 0.7],  # True MCI
                    [1.0, 0.8, 0.0]   # True AD
                ])
                print("[COST-SENSITIVE] Using DEFAULT cost matrix")
            
            # Apply additional scaling if specified
            if cost_scale != 1.0:
                self.cost_matrix = self.cost_matrix * cost_scale
                print(f"[COST-SENSITIVE] Applying cost_scale={cost_scale}x")
            
            self.class_names = ['CN', 'MCI', 'AD']
        else:
            self.cost_matrix = np.array(cost_matrix)
            self.class_names = class_names or [f'Class_{i}' for i in range(len(cost_matrix))]
            
        self._validate_cost_matrix()
        
    def _validate_cost_matrix(self):
        """Validate cost matrix properties."""
        assert self.cost_matrix.ndim == 2, "Cost matrix must be 2D"
        assert self.cost_matrix.shape[0] == self.cost_matrix.shape[1], \
            "Cost matrix must be square"
        assert np.allclose(np.diag(self.cost_matrix), 0), \
            "Diagonal of cost matrix (correct predictions) must be zero"
            
    def predict(self, probs: np.ndarray) -> np.ndarray:
        """
        Make cost-sensitive predictions.
        
        Args:
            probs: Calibrated probabilities, shape (n_samples, n_classes)
            
        Returns:
            Predicted class indices, shape (n_samples,)
        """
        # Compute expected cost for each possible prediction
        # expected_cost[i, j] = cost of predicting class j for sample i
        expected_costs = probs @ self.cost_matrix.T
        
        # Select class that minimizes expected cost
        predictions = np.argmin(expected_costs, axis=1)
        
        return predictions
    
    def predict_with_costs(self, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions and return expected costs.
        
        Returns:
            predictions: Predicted class indices
            expected_costs: Expected cost for each sample's prediction
        """
        expected_costs_matrix = probs @ self.cost_matrix.T
        predictions = np.argmin(expected_costs_matrix, axis=1)
        
        # Get the expected cost of the chosen prediction for each sample
        expected_costs = expected_costs_matrix[np.arange(len(predictions)), predictions]
        
        return predictions, expected_costs
    
    def evaluate_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Evaluate total and per-class cost of predictions.
        
        Args:
            y_true: True class indices
            y_pred: Predicted class indices
            
        Returns:
            Dictionary with cost metrics
        """
        # Total cost
        costs = self.cost_matrix[y_true, y_pred]
        total_cost = costs.sum()
        mean_cost = costs.mean()
        
        # Per-class costs
        per_class_costs = {}
        for i, class_name in enumerate(self.class_names):
            mask = y_true == i
            if mask.sum() > 0:
                class_cost = costs[mask].mean()
                per_class_costs[class_name] = class_cost
        
        return {
            'total_cost': total_cost,
            'mean_cost_per_sample': mean_cost,
            'per_class_mean_cost': per_class_costs,
            'cost_std': costs.std()
        }
    
    def visualize_cost_matrix(self, save_path: Optional[str] = None):
        """
        Visualize the cost matrix as a heatmap.
        
        Args:
            save_path: Path to save the figure (optional)
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(self.cost_matrix, 
                   annot=True, 
                   fmt='.2f',
                   cmap='YlOrRd',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Cost'},
                   ax=ax)
        
        ax.set_xlabel('Predicted Class', fontsize=12)
        ax.set_ylabel('True Class', fontsize=12)
        ax.set_title('Clinical Cost Matrix for Misclassification', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cost matrix saved to {save_path}")
        
        return fig
    
    def sensitivity_analysis(self, probs: np.ndarray, 
                            perturbation_range: np.ndarray = np.linspace(0.8, 1.2, 5)) -> Dict:
        """
        Analyze sensitivity of decisions to cost matrix perturbations.
        
        Args:
            probs: Calibrated probabilities
            perturbation_range: Multipliers to apply to cost matrix
            
        Returns:
            Dictionary with sensitivity metrics
        """
        base_predictions = self.predict(probs)
        
        results = {
            'perturbation_factors': [],
            'prediction_stability': [],  # % of predictions unchanged
        }
        
        for factor in perturbation_range:
            # Perturb cost matrix (keep diagonal at 0)
            perturbed_costs = self.cost_matrix * factor
            np.fill_diagonal(perturbed_costs, 0)
            
            # Temporarily use perturbed costs
            original_costs = self.cost_matrix.copy()
            self.cost_matrix = perturbed_costs
            
            perturbed_predictions = self.predict(probs)
            stability = (perturbed_predictions == base_predictions).mean()
            
            # Restore original costs
            self.cost_matrix = original_costs
            
            results['perturbation_factors'].append(factor)
            results['prediction_stability'].append(stability)
        
        return results


class CalibrationEvaluator:
    """
    Evaluate calibration quality of probabilistic predictions.
    
    Metrics:
        - Expected Calibration Error (ECE)
        - Maximum Calibration Error (MCE)
        - Brier Score
        - Negative Log-Likelihood
        - Reliability diagrams
    """
    
    @staticmethod
    def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, 
                                   n_bins: int = 10) -> Tuple[float, Dict]:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE measures the difference between confidence and accuracy:
            ECE = Σ_b (n_b / n) * |acc_b - conf_b|
            
        Args:
            y_true: True class labels
            probs: Predicted probabilities, shape (n_samples, n_classes)
            n_bins: Number of bins for confidence intervals
            
        Returns:
            ece: Expected calibration error
            bin_data: Dictionary with per-bin statistics
        """
        # Get confidence (max probability) and predictions
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = (predictions == y_true).astype(float)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        bin_data = {
            'bin_centers': [],
            'bin_accuracies': [],
            'bin_confidences': [],
            'bin_sizes': []
        }
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            bin_size = in_bin.sum()
            
            if bin_size > 0:
                bin_accuracy = accuracies[in_bin].mean()
                bin_confidence = confidences[in_bin].mean()
                
                ece += (bin_size / len(y_true)) * abs(bin_accuracy - bin_confidence)
                
                bin_data['bin_centers'].append((bin_lower + bin_upper) / 2)
                bin_data['bin_accuracies'].append(bin_accuracy)
                bin_data['bin_confidences'].append(bin_confidence)
                bin_data['bin_sizes'].append(bin_size)
        
        return ece, bin_data
    
    @staticmethod
    def maximum_calibration_error(y_true: np.ndarray, probs: np.ndarray,
                                  n_bins: int = 10) -> float:
        """
        Compute Maximum Calibration Error (MCE).
        
        MCE is the maximum deviation between confidence and accuracy:
            MCE = max_b |acc_b - conf_b|
        """
        _, bin_data = CalibrationEvaluator.expected_calibration_error(
            y_true, probs, n_bins
        )
        
        if not bin_data['bin_accuracies']:
            return 0.0
        
        bin_errors = [abs(acc - conf) for acc, conf in 
                     zip(bin_data['bin_accuracies'], bin_data['bin_confidences'])]
        
        return max(bin_errors)
    
    @staticmethod
    def plot_reliability_diagram(y_true: np.ndarray, probs: np.ndarray,
                                n_bins: int = 10, save_path: Optional[str] = None):
        """
        Plot reliability diagram (calibration curve).
        
        Args:
            y_true: True class labels
            probs: Predicted probabilities
            n_bins: Number of confidence bins
            save_path: Path to save figure (optional)
        """
        ece, bin_data = CalibrationEvaluator.expected_calibration_error(
            y_true, probs, n_bins
        )
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        
        # Plot actual calibration
        if bin_data['bin_centers']:
            ax.plot(bin_data['bin_confidences'], bin_data['bin_accuracies'], 
                   'o-', markersize=8, linewidth=2, label=f'Model (ECE={ece:.3f})')
            
            # Add bar chart showing sample distribution
            ax2 = ax.twinx()
            ax2.bar(bin_data['bin_centers'], bin_data['bin_sizes'], 
                   width=1.0/n_bins, alpha=0.3, color='gray', label='Sample Count')
            ax2.set_ylabel('Number of Samples', fontsize=11)
            ax2.legend(loc='upper left')
        
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Reliability Diagram', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Reliability diagram saved to {save_path}")
        
        return fig
    
    @staticmethod
    def comprehensive_evaluation(y_true: np.ndarray, probs_before: np.ndarray,
                                probs_after: np.ndarray, 
                                method_name: str = "Calibration") -> pd.DataFrame:
        """
        Compare calibration metrics before and after calibration.
        
        Returns:
            DataFrame with before/after comparison
        """
        # Compute metrics for uncalibrated probabilities
        ece_before, _ = CalibrationEvaluator.expected_calibration_error(y_true, probs_before)
        mce_before = CalibrationEvaluator.maximum_calibration_error(y_true, probs_before)
        
        # Convert to one-hot for Brier score
        y_true_oh = np.eye(probs_before.shape[1])[y_true]
        brier_before = np.mean((probs_before - y_true_oh) ** 2)
        
        # NLL (handle log(0) with clipping)
        probs_clipped = np.clip(probs_before, 1e-10, 1.0)
        nll_before = -np.mean(np.log(probs_clipped[np.arange(len(y_true)), y_true]))
        
        # Compute metrics for calibrated probabilities
        ece_after, _ = CalibrationEvaluator.expected_calibration_error(y_true, probs_after)
        mce_after = CalibrationEvaluator.maximum_calibration_error(y_true, probs_after)
        
        y_true_oh = np.eye(probs_after.shape[1])[y_true]
        brier_after = np.mean((probs_after - y_true_oh) ** 2)
        
        probs_clipped = np.clip(probs_after, 1e-10, 1.0)
        nll_after = -np.mean(np.log(probs_clipped[np.arange(len(y_true)), y_true]))
        
        # Create comparison DataFrame
        results = pd.DataFrame({
            'Metric': ['ECE', 'MCE', 'Brier Score', 'NLL'],
            'Before': [ece_before, mce_before, brier_before, nll_before],
            'After': [ece_after, mce_after, brier_after, nll_after],
        })
        
        results['Improvement'] = results['Before'] - results['After']
        results['Improvement %'] = (results['Improvement'] / results['Before'] * 100).round(2)
        
        print(f"\n{'='*60}")
        print(f"Calibration Evaluation: {method_name}")
        print(f"{'='*60}")
        print(results.to_string(index=False))
        print(f"{'='*60}\n")
        
        return results


# Convenience function for full calibration pipeline
def calibrate_multimodal_model(model, X_train, y_train, X_calib, y_calib, X_test, y_test,
                               apply_temperature_scaling: bool = True,
                               apply_isotonic_regression: bool = True,
                               use_cost_sensitive: bool = True,
                               aggressive_costs: bool = False,
                               cost_scale: float = 1.0,
                               output_dir: Optional[str] = None) -> Dict:
    """
    Full calibration pipeline for multimodal clinical prediction model.
    
    Args:
        model: Trained IRBoostSH model
        X_train, y_train: Training data
        X_calib, y_calib: Calibration data (for TS/IR fitting)
        X_test, y_test: Test data
        apply_temperature_scaling: Whether to apply TS to individual models
        apply_isotonic_regression: Whether to apply IR to final ensemble
        use_cost_sensitive: Whether to use cost-sensitive decision rule
        aggressive_costs: Whether to use aggressive cost matrix
        cost_scale: Additional scaling factor for cost matrix (default: 1.0)
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary with calibrated predictions and evaluation metrics
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("MULTIMODAL MODEL CALIBRATION PIPELINE")
    print("="*70 + "\n")
    
    # Step 1: Get uncalibrated probabilities
    print("Step 1: Getting uncalibrated probabilities...")
    probs_calib_uncal = model.predict_proba(X_calib).values
    probs_test_uncal = model.predict_proba(X_test).values
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_calib_encoded = le.fit_transform(y_calib)
    y_test_encoded = le.transform(y_test)
    
    # Step 2: Apply Isotonic Regression (on ensemble output)
    if apply_isotonic_regression:
        print("\nStep 2: Applying Isotonic Regression to ensemble probabilities...")
        ir_calibrator = IsotonicRegressionCalibrator()
        probs_test_calibrated = ir_calibrator.fit_transform(
            probs_calib_uncal, y_calib_encoded, probs_test_uncal
        )
    else:
        print("\nStep 2: Skipping Isotonic Regression")
        probs_test_calibrated = probs_test_uncal
    
    # Step 3: Evaluate calibration improvement
    print("\nStep 3: Evaluating calibration quality...")
    calib_results = CalibrationEvaluator.comprehensive_evaluation(
        y_test_encoded, probs_test_uncal, probs_test_calibrated,
        method_name="Isotonic Regression"
    )
    
    # Step 4: Visualize reliability diagram
    if output_dir:
        print("\nStep 4: Creating reliability diagrams...")
        CalibrationEvaluator.plot_reliability_diagram(
            y_test_encoded, probs_test_uncal, n_bins=10,
            save_path=output_dir / "reliability_uncalibrated.png"
        )
        CalibrationEvaluator.plot_reliability_diagram(
            y_test_encoded, probs_test_calibrated, n_bins=10,
            save_path=output_dir / "reliability_calibrated.png"
        )
    
    # Step 5: Always create cost evaluator for clinical cost analysis
    print("\nStep 5: Setting up clinical cost evaluator...")
    cost_decision = CostSensitiveDecision(
        aggressive_mode=aggressive_costs,
        cost_scale=cost_scale
    )
    
    # Standard predictions (argmax) - always computed
    y_pred_standard = probs_test_calibrated.argmax(axis=1)
    
    # ALWAYS evaluate standard (argmax) clinical costs for comparison
    cost_eval_standard = cost_decision.evaluate_cost(y_test_encoded, y_pred_standard)
    print(f"\nStandard (argmax) clinical cost: {cost_eval_standard['mean_cost_per_sample']:.4f}")
    
    # Cost-sensitive decision rule (optional)
    if use_cost_sensitive:
        print("\nApplying cost-sensitive Bayesian decision rule...")
        
        # Visualize cost matrix
        if output_dir:
            cost_decision.visualize_cost_matrix(
                save_path=output_dir / "cost_matrix.png"
            )
        
        # Cost-sensitive predictions
        y_pred_costsens, expected_costs = cost_decision.predict_with_costs(
            probs_test_calibrated
        )
        
        # Evaluate cost-sensitive costs
        cost_eval_costsens = cost_decision.evaluate_cost(y_test_encoded, y_pred_costsens)
        
        print("\n" + "-"*60)
        print("Cost Comparison: Standard vs Cost-Sensitive")
        print("-"*60)
        print(f"Standard (argmax) mean cost:     {cost_eval_standard['mean_cost_per_sample']:.4f}")
        print(f"Cost-sensitive mean cost:        {cost_eval_costsens['mean_cost_per_sample']:.4f}")
        print(f"Cost reduction:                  {cost_eval_standard['mean_cost_per_sample'] - cost_eval_costsens['mean_cost_per_sample']:.4f}")
        print(f"Relative improvement:            {(1 - cost_eval_costsens['mean_cost_per_sample']/cost_eval_standard['mean_cost_per_sample'])*100:.2f}%")
        print("-"*60 + "\n")
        
        # Sensitivity analysis
        print("Step 6: Sensitivity analysis of cost matrix...")
        sensitivity = cost_decision.sensitivity_analysis(probs_test_calibrated)
        
        print("Prediction stability under cost matrix perturbations:")
        for factor, stability in zip(sensitivity['perturbation_factors'], 
                                     sensitivity['prediction_stability']):
            print(f"  Factor {factor:.2f}x: {stability*100:.1f}% predictions unchanged")
    
    else:
        print("\nCost-sensitive decision DISABLED - using standard argmax predictions")
        y_pred_costsens = y_pred_standard  # Same as standard
        cost_eval_costsens = cost_eval_standard  # Same cost since same predictions
    
    # Return results
    results = {
        'probs_uncalibrated': probs_test_uncal,
        'probs_calibrated': probs_test_calibrated,
        'predictions_standard': probs_test_calibrated.argmax(axis=1),
        'predictions_cost_sensitive': y_pred_costsens,
        'calibration_metrics': calib_results,
        'cost_evaluation_standard': cost_eval_standard,
        'cost_evaluation_cost_sensitive': cost_eval_costsens,
        'label_encoder': le
    }
    
    print("\n" + "="*70)
    print("CALIBRATION PIPELINE COMPLETED")
    print("="*70 + "\n")
    
    return results
