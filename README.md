# Veronet Project

A multimodal deep-learning framework for medical image analysis that combines clinical data, 3D image embeddings, and boosting-based classifiers.


## 1  Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python ≥ 3.10** | Development and testing done on  3.10 and 3.11. |
| **GPU** | Recommended for training the CNN and boosting models. |
| **Git** | Optional, but handy for cloning the repository. |


## 2  Setup

### 2.0 Create data folders and download data

In the src/data folder, create the following folders:

```text
src/
├── data/
│   ├── embed/             # Cached image embeddings
|   |-─ images_pre/        # Raw images (.nii)
|   |-─ images_post/       # Processed images (.pkl)
|   └── tables/            # Clinical CSVs & legends
```

Then, download the data and place it in the appropriate folders.


### 2.1 Create & activate a virtual environment

<details>
<summary>Windows (PowerShell)</summary>

```powershell
# From the repository root
cd \path\to\veronet_paper

python -m venv .venv          # Create the venv
.\.venv\Scripts\Activate.ps1  # Activate it
````

</details>

<details>
<summary>Linux / macOS</summary>

```bash
# From the repository root
cd /path/to/veronet_paper

python3 -m venv .venv         # Create the venv
source .venv/bin/activate     # Activate it
```

</details>

### 2.2 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```



## 3  Configuration

All experiment hyper-parameters live in **`src/configs/config.yaml`**.
Feel free to copy the file and tweak epochs, learning rates, paths, or experiment lists as needed.

---

## 4  Project Layout

```text
veronet/
├── main.py                    # CLI entry-point to launch experiments
├── run_experiments.py         # Orchestrates training and evaluation loops
├── src/
│   ├── configs/
│   │   ├── config.yaml        # Default experiment definitions
│   │   └── config.py          # Python constants for common paths
│   ├── data/
│   │   ├── processing_data.py # Modality merging & CV splits
│   │   ├── embed/             # Cached image embeddings
│   │   ├── images_pre/        # Raw images
│   │   ├── images_post/       # Processed images
│   │   └── tables/            # Clinical CSVs & legends
│   ├── models/
│   │   ├── autoencoder.py          # Embeds clinical tables
│   │   ├── custom_base_estimators.py# NN & RF wrappers
│   │   ├── imagerffitterinput.py   # 3D ResNet feature extractor
│   │   └── lutech_models.py        # BoostSH / IRBoostSH
│   ├── predict/
│   │   └── evaluate_predict.py # Accuracy / precision / recall / F1
│   ├── train/
│   │   └── boosting.py        # Training logic for multimodal boosting
│   └── utils/
│       ├── save_load.py       # Model I/O helpers
│       └── load_data.py       # NIfTI + clinical ingestion
|
└── requirements.txt
```



## 5  Available Experiments

| ID       | Description                                                            |
| -------- | ---------------------------------------------------------------------- |
| **exp1** | Full multimodal training with CNN frezzed layers. |
| **exp2** | Full multimodal training without CNN frezzed layers.                                              |
| **exp3** | Clinical-only baseline.                                                |
| **exp4** | Image-only baseline with CNN frezzed layers.                                                   |
| **exp5** | Image-only baselin without CNN frezzed layers.                                 |
| **exp6** | Random-Forest baseline.                                                |
| **exp7** | Autoencoder (clinical) + CNN (images) concatenated for classification. |

The active parameters (epochs, learning rate, batch size, etc.) for each experiment are defined in `src/configs/config.yaml`.


## 6  Running Experiments

```bash
# Syntax
python main.py <experiment_id>

# Examples
python main.py exp3       # Clinical-only
python main.py exp7       # AE + CNN multimodal pipeline
```

---

## 7  Deactivating the Environment

```bash
deactivate
```

---

### Support

Please open an issue or submit a pull request if you hit any problems or have suggestions to improve the project.






