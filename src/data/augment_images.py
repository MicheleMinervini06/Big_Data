"""
Data Augmentation per immagini MRI (Pre-augmentation - Opzione 2)

Genera variazioni augmented delle immagini e le salva su disco per training efficiente.
Le augmentations sono clinicamente sensate per brain MRI.
"""

import albumentations as A
import numpy as np
import pickle
import os
from pathlib import Path
from tqdm import tqdm
import cv2

# Importing config file
os.sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import IMAGE_FOLDER, IMAGE_AUGMENTED_FOLDER

class MRIAugmenter:
    """
    Clinically-grounded augmentations per brain MRI
    
    Basato su:
    - Rotation: ±10° (patient head movement during scan)
    - Elastic deform: σ=30 (scanner variability)
    - Gaussian noise: σ=0.01 (SNR variability)
    - Contrast scaling: ±10% (scanner calibration variation)
    """
    
    def __init__(self, p_augment=1.0):
        """
        Args:
            p_augment: probability of applying augmentation (1.0 per pre-augmentation)
        """
        self.p_augment = p_augment
        
        # Define augmentation pipeline
        self.transform = A.Compose([
            
            # Spatial: Head movement during scan
            A.Rotate(
                limit=(-10, 10),  # ±10 degrees
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT,
                p=0.6
            ),
            
            # Spatial: Small translations (patient positioning)
            A.Affine(
                translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},  # ±5% of image
                scale=(0.95, 1.05),  # ±5% zoom
                rotate=(-5, 5),
                interpolation=cv2.INTER_LINEAR,
                p=0.4
            ),
            
            # Elastic deformation (scanner/field inhomogeneity)
            A.ElasticTransform(
                alpha=30,
                sigma=5,
                p=0.3
            ),
            
            # Pixel-level: Gaussian noise (SNR degradation)
            A.GaussNoise(
                var_limit=(0.005, 0.015),  # Small noise for 0-1 normalized
                p=0.3
            ),
            
            # Pixel-level: Contrast variation (scanner calibration)
            A.RandomBrightnessContrast(
                brightness_limit=0.1,  # ±10%
                contrast_limit=0.1,
                p=0.4
            ),
            
            # Pixel-level: Gamma correction (non-linear intensity)
            A.RandomGamma(
                gamma_limit=(90, 110),
                p=0.3
            ),
            
        ], p=self.p_augment)
    
    def augment(self, image_array, seed=None):
        """
        Apply augmentation to MRI volume
        
        Args:
            image_array: (1, D, H, W) or (D, H, W) or (H, W) numpy array, float32
            seed: random seed per reproducibilità
        
        Returns:
            augmented: same shape as input
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Convert to numpy if torch.Tensor
        import torch
        if isinstance(image_array, torch.Tensor):
            image_array = image_array.cpu().numpy()
        
        # Ensure float32 and normalized [0, 1]
        if image_array.dtype != np.float32:
            image_array = image_array.astype(np.float32)
        
        # Normalize if needed
        if image_array.max() > 1.0:
            image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)
        
        # Ensure 2D for albumentations (works per slice if 3D)
        if image_array.ndim == 4 and image_array.shape[0] == 1:
            image_array = image_array[0]  # Remove channel dimension if single channel
        
        if image_array.ndim == 3:
            # 3D volume: augment each slice independently
            augmented = np.zeros_like(image_array)
            for z in range(image_array.shape[0]):
                slice_2d = image_array[z, :, :]
                transformed = self.transform(image=slice_2d)['image']
                augmented[z, :, :] = transformed
            return augmented
        elif image_array.ndim == 2:
            # 2D image
            return self.transform(image=image_array)['image']
        else:
            raise ValueError(f"Unsupported image shape: {image_array.shape}")


def generate_augmented_dataset(
    input_dir: str,
    output_dir: str,
    n_augmentations: int = 5,
    p_augment: float = 1.0
):
    """
    Genera dataset augmented da immagini pickle esistenti.
    
    Args:
        input_dir: directory con immagini pickle originali
        output_dir: directory dove salvare augmented images
        n_augmentations: numero di varianti augmented per immagine
        p_augment: probabilità di applicare augmentation (1.0 = sempre)
    
    Output structure:
        output_dir/
            original_id_0.pkl  (originale copiato)
            original_id_1.pkl  (augmented variant 1)
            original_id_2.pkl  (augmented variant 2)
            ...
    """
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all pickle files
    pkl_files = list(input_path.glob("*.pkl"))
    print(f"Found {len(pkl_files)} pickle files in {input_dir}")
    
    if len(pkl_files) == 0:
        print("❌ No pickle files found!")
        return
    
    augmenter = MRIAugmenter(p_augment=p_augment)
    
    print(f"Generating {n_augmentations} augmentations per image...")
    print(f"Output directory: {output_dir}")
    
    total_files = 0
    
    for pkl_file in tqdm(pkl_files, desc="Processing images"):
        # Load original image
        with open(pkl_file, 'rb') as f:
            img_original = pickle.load(f)
        
        # Extract filename without extension
        base_name = pkl_file.stem  # e.g., "12345"
        
        # Save original as variant 0
        original_out_path = output_path / f"{base_name}_aug0.pkl"
        with open(original_out_path, 'wb') as f:
            pickle.dump(img_original, f)
        total_files += 1
        
        # Generate augmented variants
        for aug_idx in range(1, n_augmentations + 1):
            # Use deterministic seed for reproducibility
            seed = hash(f"{base_name}_{aug_idx}") % (2**32)
            
            img_augmented = augmenter.augment(img_original, seed=seed)
            
            # Save augmented variant
            aug_out_path = output_path / f"{base_name}_aug{aug_idx}.pkl"
            with open(aug_out_path, 'wb') as f:
                pickle.dump(img_augmented, f)
            total_files += 1
    
    print(f"\n✅ Augmentation complete!")
    print(f"   Original images: {len(pkl_files)}")
    print(f"   Augmented variants per image: {n_augmentations}")
    print(f"   Total files generated: {total_files}")
    print(f"   Storage location: {output_dir}")
    
    # Estimate storage size
    import shutil
    total_size_bytes = sum(f.stat().st_size for f in output_path.glob("*.pkl"))
    total_size_mb = total_size_bytes / (1024 ** 2)
    print(f"   Total storage used: {total_size_mb:.1f} MB")


def load_augmented_paths(augmented_dir: str, original_legend: dict):
    """
    Crea DataFrame con percorsi alle immagini augmented.
    
    Args:
        augmented_dir: directory con immagini augmented
        original_legend: legenda originale con mapping image_id -> metadata
    
    Returns:
        DataFrame con colonne: ['image_path', 'Patient ID', 'VISCODE', 'original_image_id']
    """
    import pandas as pd
    
    augmented_path = Path(augmented_dir)
    pkl_files = list(augmented_path.glob("*.pkl"))
    
    print(f"Found {len(pkl_files)} augmented images in {augmented_dir}")
    
    # Parse filenames: "processed_002_S_1070_I30862_aug0.pkl" -> imageuid=30862, aug_idx=0
    records = []
    for pkl_file in pkl_files:
        filename = pkl_file.stem  # "processed_002_S_1070_I30862_aug0"
        parts = filename.split('_aug')
        
        if len(parts) != 2:
            print(f"Warning: Skipping malformed filename {pkl_file.name}")
            continue
        
        original_id_str = parts[0]  # "processed_002_S_1070_I30862"
        aug_idx = int(parts[1])
        
        # Extract IMAGEUID from format: processed_002_S_1070_I30862 -> 30862
        if "_I" not in original_id_str:
            print(f"Warning: Cannot find '_I' in filename {filename}")
            continue
        
        try:
            imageuid_parts = original_id_str.split("_I")
            imageuid_str = imageuid_parts[1]  # "30862"
            original_id = int(imageuid_str)
        except (ValueError, IndexError):
            continue
        
        # Find metadata from original legend
        try:
            idx_in_legend = original_legend['Image Data ID'].index(original_id)
            patient_id = original_legend['Subject'][idx_in_legend]
            # Use 'Visit' if available, otherwise default to 'bl' (baseline)
            viscode = original_legend.get('Visit', [None] * len(original_legend['Image Data ID']))[idx_in_legend]
            if viscode is None:
                viscode = "bl"  # default baseline
        except (ValueError, IndexError, KeyError):
            # Image not in legend, skip
            continue
        
        records.append({
            'image_path': str(pkl_file.absolute()),
            'Patient ID': patient_id,
            'VISCODE': viscode,
            'original_image_id': original_id,
            'aug_index': aug_idx
        })
    
    df_augmented = pd.DataFrame(records)
    print(f"Successfully mapped {len(df_augmented)} augmented images to metadata")
    
    return df_augmented


if __name__ == "__main__":
    """
    Script per generare dataset augmented.
    
    Usage:
        python -m src.data.augment_images
    """
    
    # Configurazione
    INPUT_DIR = IMAGE_FOLDER  # Cartella con immagini originali
    OUTPUT_DIR = IMAGE_AUGMENTED_FOLDER  # Cartella per immagini augmented
    N_AUGMENTATIONS = 5  # Numero di varianti per immagine (default 5)
    
    print("="*80)
    print("MRI Data Augmentation - Pre-augmentation Pipeline")
    print("="*80)
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Augmentations per image: {N_AUGMENTATIONS}")
    print("="*80 + "\n")
    
    # Genera augmented dataset
    try:
        generate_augmented_dataset(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            n_augmentations=N_AUGMENTATIONS,
            p_augment=1.0
        )
    except Exception as e:
        print(f"❌ Error during augmentation: {e}")    
