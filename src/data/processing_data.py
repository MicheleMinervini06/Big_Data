from src.utils.load_data import ImagePreprocessor, retrieve_object_from_minio, treat_labels, treat_clinical_data, merge_modalities
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import StratifiedKFold

LOCAL_DATA_DIR =  "src/data"
IMAGE_FOLDER = LOCAL_DATA_DIR / "images"
LABELS_FILE = LOCAL_DATA_DIR / "ADNIMERGE_29Nov2024.csv"
LEGEND_FILE = LOCAL_DATA_DIR / "ADNI_T1_MPRAGE_12_17_2024.csv"
IMG_PREFIX = "processed_MRI"
TEST_PERC = 0.2
N_SPLITS = 5
ADNI_WASTE = [
    "ABETA", "update_stamp", "TAU", "PTAU", "FSVERSION", "RID", "CDRSB", "EcoPtMem",
    "EXAMDATE", "SITE", "COLPROT", "ORIGPROT", "PTETHCAT", 'PTRACCAT', "APOE4",
    "FDG", "PIB", "AV45", "Hippocampus", "WholeBrain", "Enthorhinal", "Fusiform",
    "MidTemp", "ICV", "M"
]


def processing_features_cv():
    # Scarica i dati da MinIO e posizionali in data/ (manuale)
    labels_df = treat_labels(LABELS_FILE)
    clinical_df = treat_clinical_data(LABELS_FILE, unwanted_cols=ADNI_WASTE, threshold=0.9)
    legend = pd.read_csv(LEGEND_FILE, on_bad_lines='skip').to_dict(orient='list')

    img_pre = ImagePreprocessor(img_input_size=(1,128,128,50), legend=legend)
    paths_df = img_pre.get_images_paths_df(LOCAL_DATA_DIR / IMG_PREFIX)

    merged = merge_modalities([clinical_df, paths_df], labels_df)
    merged = merged.drop(columns=['Patient ID','VISCODE'], errors='ignore').reset_index(drop=True)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    folds = []
    for train_idx, test_idx in skf.split(merged, merged['Label']):
        tr, te = merged.loc[train_idx], merged.loc[test_idx]
        folds.append({
            'X_train': {'clinical': tr.drop(['Label'], axis=1).drop(columns=['image_path']),
                        'images': tr[['image_path']]},
            'y_train': tr['Label'],
            'X_test':  {'clinical': te.drop(['Label'], axis=1).drop(columns=['image_path']),
                        'images': te[['image_path']]},
            'y_test': te['Label']
        })
    return folds
