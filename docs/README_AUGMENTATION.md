# Data Augmentation for MRI Images - FASE 2

## Overview
This module implements **pre-computed data augmentation** for MRI brain images to improve CNN performance in the ensemble model.

### Current Problem
- CNN alpha contribution: **0.001** (0.1%) - RF dominates
- Root cause: Poor CNN training (high loss 5-11)
- Solution: Expand training dataset with augmented images → improve CNN generalization → increase alpha

### Strategy: Pre-Augmentation (Opzione 2)
- **Advantages**: Faster training, reproducible augmentations, no runtime overhead
- **Storage**: 330MB (original) → ~1.65GB (5x augmentation) ✅ Acceptable
- **N augmentations**: 5 variants per image + 1 original = 6x dataset expansion

---

## Files

### `augment_images.py`
Main augmentation module with:
- **`MRIAugmenter`**: Augmentation pipeline using albumentations library
- **`generate_augmented_dataset()`**: Pre-computes and saves augmented images
- **`load_augmented_paths()`**: Loads augmented dataset with metadata mapping

### Augmentation Transformations
Optimized for medical imaging:
- **Rotate**: ±10° (p=0.6)
- **ShiftScaleRotate**: ±5% shift/scale, ±5° rotation (p=0.4)
- **ElasticTransform**: Simulates anatomical variations (p=0.3)
- **GaussNoise**: Simulates acquisition noise (p=0.3)
- **RandomBrightnessContrast**: ±10% (p=0.4)
- **RandomGamma**: 90-110% (p=0.3)

---

## Usage

### Step 1: Generate Augmented Dataset

Run from project root in WSL:

```bash
cd '/mnt/c/Users/mikim/Desktop/Uni/Big Data/Big_Data'
source venv_wsl/bin/activate
python -m src.data.augment_images
```

**Expected Output:**
- Input: `src/data/images_pre/` (~86 images, 330MB)
- Output: `src/data/images_augmented/` (~516 files = 86×6, ~1.65GB)
- Naming: `{image_id}_aug0.pkl` (original), `{image_id}_aug1-5.pkl` (augmented)
- Progress: tqdm progress bar

**Verification:**
```python
import os
augmented_dir = "src/data/images_augmented"
print(f"Total files: {len(os.listdir(augmented_dir))}")
# Expected: ~516 files
```

### Step 2: Train with Augmented Data

**Config: exp9**
```yaml
exp9:
  id: 9
  train: true
  params:
    epochs: 50
    mb_size_train: 8
    n_iteration: 10
    frezze_layer: 2
    mod: null
    use_augmented: true  # ← Enable augmentation
```

**Run Training:**
```bash
python main.py exp9
```

### Step 3: Compare Results

Compare exp9 (augmented) vs exp1 (baseline):

```bash
python run_experiment_suite.py baseline_vs_augmented
```

**Key Metrics to Check:**
- **CNN alpha**: Target >0.05 (5% contribution, vs 0.1% currently)
- **CNN loss**: Should decrease compared to exp1
- **Overall accuracy**: Should not decrease (augmentation helps or neutral)

---

## Integration Details

### `processing_data.py`
Added function `processing_features_cv_augmented(use_augmented=True)`:
```python
if use_augmented:
    from src.data.augment_images import load_augmented_paths
    df_paths = load_augmented_paths("src/data/images_augmented", legend)
```

### `run_experiments.py`
Modified `run_boosting()` to handle `use_augmented` flag:
```python
use_augmented = params.get("use_augmented", False)
if use_augmented:
    folds = processing_features_cv_augmented(use_augmented=True)
else:
    folds = processing_features_cv()
```

---

## Technical Details

### Deterministic Augmentation
Each augmentation uses deterministic seeding for reproducibility:
```python
seed = hash(f"{base_name}_{aug_idx}") % (2**32)
np.random.seed(seed)
```

### Memory Management
Augmented images saved as pickled numpy arrays (same format as originals) for efficient loading during training.

### Backward Compatibility
- Original experiments (exp1-8) unaffected
- `processing_features_cv()` still loads only original images
- `processing_features_cv_augmented()` loads 6x expanded dataset

---

## Expected Outcomes

### Success Criteria
- ✅ CNN alpha increases from 0.001 to >0.05 (50x improvement)
- ✅ CNN loss decreases during training (better generalization)
- ✅ Overall test accuracy >= exp1 (no degradation)

### If CNN Alpha < 0.05 (Augmentation Insufficient)
Possible next steps:
1. Increase augmentation factor (10x instead of 5x)
2. Unfreeze more layers (frezze_layer: 1 or 0)
3. Increase learning rate
4. Add more aggressive augmentations (e.g., GridDistortion)
5. Accept RF-dominant ensemble (not necessarily bad)

### If CNN Alpha > 0.1 (Success!)
Proceed to **FASE 3**: Test-Time Augmentation (TTA) for aleatoric uncertainty
- TTA on imaging: 10 augmented versions at test time
- TTA on clinical: Feature noise (MMSE ±2, biomarkers ±5-8%)
- Total uncertainty = epistemic (MC Dropout) + aleatoric (TTA)

---

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'albumentations'`
```bash
pip install albumentations
```

**Issue**: Out of memory during augmentation
- Reduce batch processing in `generate_augmented_dataset()`
- Process images sequentially instead of batched

**Issue**: Training slower with augmented data
- Expected: 6x more samples → longer epochs
- Mitigation: Already using pre-augmentation (no runtime overhead per sample)

**Issue**: CNN alpha still low after exp9
- Check CNN loss trend: Should decrease compared to exp1
- If loss still high: Try unfreezing more layers or increasing epochs

---

## Notes

- **Albumentations library**: Medical-imaging optimized (better than torchvision for MRI)
- **6x expansion**: 86 images → 516 samples (5 augmentations + 1 original)
- **Training time**: Expect ~6x longer per epoch (proportional to dataset size)
- **Storage**: 1.65GB is acceptable for 5x augmentation (images_pre/ is only 330MB)
- **Reproducibility**: Deterministic seeding ensures same augmentations across runs

---

## References

- albumentations documentation: https://albumentations.ai/
- Medical image augmentation best practices: [Shorten & Khoshgoftaar, 2019]
- Ensemble learning with boosting: [Friedman, 2001]
