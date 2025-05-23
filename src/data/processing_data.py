from src.utils.load_data import ImagePreprocessor, retrieve_object_from_minio, treat_labels, treat_clinical_data, merge_modalities
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src.configs.config import *


def processing_feature():
    # 1) Carica label e dati clinici
    df_labels   = treat_labels(LABELS_FILE)
    print(f"[DEBUG] labels_df shape: {df_labels.shape}")
    df_clinical = treat_clinical_data(LABELS_FILE, unwanted_cols=ADNI_WASTE, threshold=0.9)
    print(f"[DEBUG] clinical_df shape: {df_clinical.shape}")

    # 2) Carica legenda
    df_legend = pd.read_csv(LEGEND_FILE, on_bad_lines='skip')
    legend = df_legend.to_dict(orient='list')
    print(f"[DEBUG] legend keys: {list(legend.keys())}")
    sample_id = legend['Image Data ID'][0]
    print(f"[DEBUG] sample Image Data ID: {sample_id}")

    # 3) Pre-processore immagini
    img_pre = ImagePreprocessor(
        img_input_size=(1,128,128,50),
        legend=legend
    )
    df_paths = img_pre.get_images_paths_df()
    print(f"[DEBUG] paths_df shape: {df_paths.shape}")
    print(f"[DEBUG] paths_df headD:{df_paths.head()}")

    #Mergio le modalità in un unico dataset
    # Merge modalities
    df_merge = merge_modalities([df_clinical,df_paths],df_labels)
    df_merge = df_merge.drop(columns=['Patient ID'])
    df_merge = df_merge.drop(columns=['VISCODE'])
    df_merge = df_merge.reset_index(drop=True)

    #train-test-split
    x_train, x_test = train_test_split(df_merge, test_size=TEST_PERC, random_state=42)

    clinical_cols = df_clinical.columns.tolist()
    clinical_cols.remove('Patient ID')
    clinical_cols.remove('VISCODE')

    X_mod1_train = x_train[clinical_cols]
    X_mod3_train = x_train[["image_path"]]

    X_mod1_test = x_test[clinical_cols]
    X_mod3_test = x_test[["image_path"]]

    

    X_mods = {'clinical': X_mod1_train, 'images': X_mod3_train.dropna()}

    X_test = {'clinical': X_mod1_test,  'images': X_mod3_test.dropna()}
    y_train = x_train["Label"]
    y_test = x_test["Label"]

    return X_mods,X_test,y_train,y_test


def processing_features_cv():
    # Scarica i dati da MinIO e posizionali in data/ (manuale)
    
    # 1) Carica label e dati clinici
    df_labels   = treat_labels(LABELS_FILE)
    print(f"[DEBUG] labels_df shape: {df_labels.shape}")
    df_clinical = treat_clinical_data(LABELS_FILE, unwanted_cols=ADNI_WASTE, threshold=0.9)
    print(f"[DEBUG] clinical_df shape: {df_clinical.shape}")

    # 2) Carica legenda
    df_legend = pd.read_csv(LEGEND_FILE, on_bad_lines='skip')
    legend = df_legend.to_dict(orient='list')
    print(f"[DEBUG] legend keys: {list(legend.keys())}")
    sample_id = legend['Image Data ID'][0]
    print(f"[DEBUG] sample Image Data ID: {sample_id}")

    # 3) Pre-processore immagini
    img_pre = ImagePreprocessor(
        img_input_size=(1,128,128,50),
        legend=legend
    )
    df_paths = img_pre.get_images_paths_df()
    #Mergio le modalità in un unico dataset
    # Merge modalities
    df_merge = merge_modalities([df_clinical,df_paths],df_labels)
    df_merge = df_merge.drop(columns=['Patient ID'])
    df_merge = df_merge.drop(columns=['VISCODE'])
    df_merge = df_merge.reset_index(drop=True)

    clinical_cols = df_clinical.columns.tolist()
    clinical_cols.remove('Patient ID')
    clinical_cols.remove('VISCODE')

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df_merge, df_merge["Label"])):
        train_df = df_merge.loc[train_idx].reset_index(drop=True)
        test_df = df_merge.loc[test_idx].reset_index(drop=True)

        fold_data = {
            'X_train': {
                'clinical': train_df[clinical_cols],
                'images': train_df[['image_path']].dropna()
            },
            'y_train': train_df["Label"],
            'X_test': {
                'clinical': test_df[clinical_cols],
                'images': test_df[['image_path']].dropna()
            },
            'y_test': test_df["Label"]
        }

        folds.append(fold_data)

    return folds

