"""
Explainability module for ADNI multimodal model.

This module provides interpretability tools for the boosting ensemble model:
- SHAP: Feature-level explanations for Random Forest models
- Grad-CAM: Pixel-level explanations for ResNet imaging models
- Modality contributions: Decomposition of predictions by modality
"""

from .shap_explainer import SHAPExplainer
from .gradcam_explainer import GradCAMExplainer

__all__ = ['SHAPExplainer', 'GradCAMExplainer']
