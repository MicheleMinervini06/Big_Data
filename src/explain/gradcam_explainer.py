"""
Grad-CAM (Gradient-weighted Class Activation Mapping) explainability for ResNet models.

Grad-CAM visualizes which regions of the brain scan are most important for
the model's prediction by computing gradients with respect to feature maps.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from pathlib import Path


class GradCAMExplainer:
    """
    Grad-CAM explainer for ResNet models in the multimodal ensemble.
    
    This class generates heatmaps showing which brain regions are most influential
    for the model's prediction.
    """
    
    def __init__(self, model, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize Grad-CAM explainer.
        
        Args:
            model: Trained IRBoostSH model
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.resnet_models = []
        self.resnet_model_indices = []
        
        # Extract all ResNet models from the ensemble
        self._extract_resnet_models()
    
    def _extract_resnet_models(self):
        """
        Extract all ResNet (VeroResNet) models from the ensemble.
        """
        for t, (wrapper, modality) in enumerate(zip(self.model.models, 
                                                     self.model.modalities_selected)):
            if modality == 'images':
                # For NeuralNetworkFitter: wrapper.inp_list.model or wrapper.model
                resnet = None
                if hasattr(wrapper, 'model'):
                    resnet = wrapper.model
                elif hasattr(wrapper, 'inp_list') and hasattr(wrapper.inp_list, 'model'):
                    resnet = wrapper.inp_list.model
                
                # Check if it's a VeroResNet
                if resnet is not None and hasattr(resnet, 'res'):  # VeroResNet has 'res' attribute
                    self.resnet_model_indices.append(t)
                    self.resnet_models.append(resnet)
                    resnet.eval()  # Set to evaluation mode
    
    def _get_target_layer(self, resnet_model, layer_name='layer4'):
        """
        Get the target layer for Grad-CAM (typically the last conv layer).
        
        Args:
            resnet_model: VeroResNet model
            layer_name: Name of the layer to use ('layer3' or 'layer4')
            
        Returns:
            Target layer
        """
        # VeroResNet has 'res' attribute which is the MONAI ResNet
        if hasattr(resnet_model, 'res'):
            # MONAI ResNet structure
            if hasattr(resnet_model.res, layer_name):
                return getattr(resnet_model.res, layer_name)
        return None
    
    def compute_gradcam(self, image_tensor: torch.Tensor, target_class: int = None,
                       model_idx: int = 0, layer_name: str = 'layer4') -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a single image.
        
        Args:
            image_tensor: Input brain scan tensor (1, C, H, W, D) or (C, H, W, D)
            target_class: Target class for Grad-CAM (if None, uses predicted class)
            model_idx: Index of ResNet model to use (default: 0, first ResNet)
            layer_name: Target layer name ('layer3' or 'layer4')
            
        Returns:
            Heatmap as numpy array (H, W, D) or average over depth dimension
        """
        if len(self.resnet_models) == 0:
            print("No ResNet models found in ensemble")
            return None
        
        if model_idx >= len(self.resnet_models):
            model_idx = 0
        
        resnet = self.resnet_models[model_idx]
        target_layer = self._get_target_layer(resnet, layer_name)
        
        if target_layer is None:
            print("Could not find target layer for Grad-CAM")
            return None
        
        # Ensure tensor has batch dimension
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.unsqueeze(0)
        
        image_tensor = image_tensor.to(self.device)
        image_tensor.requires_grad = True
        
        # Forward pass with hooks
        activations = []
        gradients = []
        
        def forward_hook(module, input, output):
            activations.append(output)
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        # Register hooks
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        
        # Forward pass
        resnet.eval()
        output = resnet(image_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        resnet.zero_grad()
        target_output = output[0, target_class]
        target_output.backward()
        
        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()
        
        # Compute Grad-CAM
        if len(activations) > 0 and len(gradients) > 0:
            activation = activations[0]  # (1, C, H, W, D)
            gradient = gradients[0]      # (1, C, H, W, D)
            
            # Global average pooling on gradients
            weights = gradient.mean(dim=(2, 3, 4), keepdim=True)  # (1, C, 1, 1, 1)
            
            # Weighted combination of activation maps
            cam = (weights * activation).sum(dim=1, keepdim=True)  # (1, 1, H, W, D)
            cam = F.relu(cam)  # ReLU to keep positive contributions
            
            # Convert to numpy
            cam = cam.squeeze().detach().cpu().numpy()
            
            # Normalize to [0, 1]
            if cam.max() > 0:
                cam = cam / cam.max()
            
            # Upsample to match input image size
            # Input shape: (1, 1, H, W, D) -> we need cam to be (H, W, D)
            input_shape = image_tensor.shape[2:]  # (H, W, D)
            if cam.shape != input_shape:
                from scipy.ndimage import zoom
                zoom_factors = tuple(input_shape[i] / cam.shape[i] for i in range(3))
                cam = zoom(cam, zoom_factors, order=1)  # Bilinear interpolation
            
            return cam
        
        return None
    
    def visualize_2d_slice(self, image: np.ndarray, heatmap: np.ndarray, 
                          slice_idx: int = None, alpha: float = 0.4,
                          save_path: Optional[Path] = None) -> plt.Figure:
        """
        Visualize Grad-CAM heatmap overlaid on a 2D slice of the brain scan.
        
        Args:
            image: Original brain scan (H, W, D) or (1, H, W, D)
            heatmap: Grad-CAM heatmap (H, W, D)
            slice_idx: Index of slice to visualize (if None, uses middle slice)
            alpha: Transparency of heatmap overlay
            save_path: Path to save the figure (if None, displays it)
            
        Returns:
            matplotlib Figure
        """
        # Remove channel dimension if present
        if image.ndim == 4:
            image = image[0]
        
        # Select middle slice if not specified
        if slice_idx is None:
            slice_idx = image.shape[-1] // 2
        
        # Extract 2D slices
        image_slice = image[:, :, slice_idx]
        heatmap_slice = heatmap[:, :, slice_idx] if heatmap.ndim == 3 else heatmap
        
        # Resize heatmap to match image size if needed
        if heatmap_slice.shape != image_slice.shape:
            from scipy.ndimage import zoom
            zoom_factors = (image_slice.shape[0] / heatmap_slice.shape[0],
                          image_slice.shape[1] / heatmap_slice.shape[1])
            heatmap_slice = zoom(heatmap_slice, zoom_factors, order=1)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image_slice, cmap='gray')
        axes[0].set_title('Original Brain Scan')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap_slice, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(image_slice, cmap='gray')
        axes[2].imshow(heatmap_slice, cmap='jet', alpha=alpha)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        return fig
    
    def explain_patient(self, image_path: str, target_class: int = None,
                       slice_indices: List[int] = None,
                       save_dir: Optional[Path] = None) -> Dict:
        """
        Generate Grad-CAM explanation for a patient's brain scan.
        
        Args:
            image_path: Path to the brain scan image
            target_class: Target class for explanation
            slice_indices: List of slice indices to visualize
            save_dir: Directory to save visualizations
            
        Returns:
            dict with heatmap and visualizations
        """
        # Load image (implement based on your data format)
        # This is a placeholder - you need to implement image loading
        print(f"Loading image from {image_path}...")
        
        # For now, return a placeholder
        return {
            'error': 'Image loading not implemented yet',
            'heatmap': None,
            'message': 'Implement image loading based on your ADNI data format'
        }
    
    def get_aggregated_heatmap(self, image_tensor: torch.Tensor, 
                              target_class: int = None, layer_name: str = 'layer4') -> np.ndarray:
        """
        Compute aggregated Grad-CAM across all ResNet models in the ensemble.
        
        Args:
            image_tensor: Input brain scan
            target_class: Target class for Grad-CAM
            layer_name: Target layer name ('layer3' or 'layer4')
            
        Returns:
            Averaged heatmap weighted by model alphas
        """
        if len(self.resnet_models) == 0:
            return None
        
        heatmaps = []
        weights = []
        
        for i, model_idx in enumerate(self.resnet_model_indices):
            heatmap = self.compute_gradcam(image_tensor, target_class, model_idx=i, layer_name=layer_name)
            if heatmap is not None:
                alpha = self.model.alphas[model_idx]
                heatmaps.append(heatmap)
                weights.append(alpha)
        
        if len(heatmaps) == 0:
            return None
        
        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        aggregated = sum(h * w for h, w in zip(heatmaps, weights))
        
        # Normalize
        if aggregated.max() > 0:
            aggregated = aggregated / aggregated.max()
        
        return aggregated
    
    def identify_important_regions(self, heatmap: np.ndarray, 
                                  threshold: float = 0.7) -> Dict:
        """
        Identify brain regions with high activation in the heatmap.
        
        Args:
            heatmap: Grad-CAM heatmap
            threshold: Threshold for considering a region "important"
            
        Returns:
            dict with statistics about important regions
        """
        important_mask = heatmap > threshold
        n_important_voxels = important_mask.sum()
        total_voxels = heatmap.size
        
        return {
            'threshold': threshold,
            'n_important_voxels': int(n_important_voxels),
            'percentage_important': float(n_important_voxels / total_voxels * 100),
            'max_activation': float(heatmap.max()),
            'mean_activation': float(heatmap.mean()),
            'mean_important_activation': float(heatmap[important_mask].mean()) if n_important_voxels > 0 else 0.0
        }
