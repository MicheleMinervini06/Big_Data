from src.utils.load_data import ImagePreprocessor, retrieve_object_from_minio, treat_labels, treat_clinical_data, merge_modalities
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
import pandas as pd
from src.configs.config import *
from pathlib import Path


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


def processing_features_cv_augmented(use_augmented=True):
    """
    Versione con supporto per immagini augmented.
    
    Args:
        use_augmented: Se True, usa immagini da images_augmented/, altrimenti images_pre/
    
    Returns:
        folds: lista di fold con X_train, y_train, X_test, y_test
    """
    # 1) Carica label e dati clinici
    df_labels   = treat_labels(LABELS_FILE)
    print(f"[DEBUG] labels_df shape: {df_labels.shape}")
    df_clinical = treat_clinical_data(LABELS_FILE, unwanted_cols=ADNI_WASTE, threshold=0.9)
    print(f"[DEBUG] clinical_df shape: {df_clinical.shape}")

    # 2) Carica legenda
    df_legend = pd.read_csv(LEGEND_FILE, on_bad_lines='skip')
    legend = df_legend.to_dict(orient='list')
    print(f"[DEBUG] legend keys: {list(legend.keys())}")

    # 3) Pre-processore immagini (originali o augmented)
    if use_augmented:
        images_dir = "src/data/images_augmented"
        print(f"[DEBUG] Using AUGMENTED images from {images_dir}")
        
        # Load augmented paths
        from src.data.augment_images import load_augmented_paths
        df_paths = load_augmented_paths(images_dir, legend)
    else:
        print(f"[DEBUG] Using ORIGINAL images")
        img_pre = ImagePreprocessor(
            img_input_size=(1,128,128,50),
            legend=legend
        )
        df_paths = img_pre.get_images_paths_df()
    
    print(f"[DEBUG] paths_df shape: {df_paths.shape}")
    
    # Merge modalities
    df_merge = merge_modalities([df_clinical, df_paths], df_labels)
    
    # CRITICAL: Prevent data leakage with augmented images
    # Keep original_image_id for grouping before dropping columns
    if use_augmented and 'original_image_id' in df_paths.columns:
        # Map image_path to original_image_id for grouping
        image_to_group = df_paths.set_index('image_path')['original_image_id'].to_dict()
        df_merge['group_id'] = df_merge['image_path'].map(image_to_group).fillna(-1).astype(int)
        print(f"[DEBUG] Group-based split enabled: {df_merge['group_id'].nunique()} unique image groups")
        print(f"[DEBUG] Total samples: {len(df_merge)}, Augmentation factor: {len(df_merge) / df_merge['group_id'].nunique():.1f}x")
    else:
        # For non-augmented data, each row is independent
        df_merge['group_id'] = range(len(df_merge))
    
    df_merge = df_merge.drop(columns=['Patient ID'])
    df_merge = df_merge.drop(columns=['VISCODE'])
    df_merge = df_merge.reset_index(drop=True)

    clinical_cols = df_clinical.columns.tolist()
    clinical_cols.remove('Patient ID')
    clinical_cols.remove('VISCODE')

    # Use GroupKFold if augmented to prevent leakage
    if use_augmented and 'group_id' in df_merge.columns:
        print(f"[ANTI-LEAKAGE] Using GroupKFold to keep same-image augmentations together")
        groups = df_merge['group_id']
        kf = GroupKFold(n_splits=N_SPLITS)
        split_iter = kf.split(df_merge, df_merge["Label"], groups=groups)
    else:
        print(f"[DEBUG] Using StratifiedKFold (standard split)")
        kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        split_iter = kf.split(df_merge, df_merge["Label"])
    
    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
        train_df = df_merge.loc[train_idx].reset_index(drop=True)
        test_df = df_merge.loc[test_idx].reset_index(drop=True)
        
        # Drop group_id from data (not a feature)
        if 'group_id' in train_df.columns:
            train_df = train_df.drop(columns=['group_id'])
        if 'group_id' in test_df.columns:
            test_df = test_df.drop(columns=['group_id'])

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



def processing_features_cv_with_calibration(use_augmented=False):
    """
    Versione con calibration set per Conformal Prediction.
    Split: 60% train, 20% calibration, 20% test
    """
    df_labels = treat_labels(LABELS_FILE)
    df_clinical = treat_clinical_data(LABELS_FILE, unwanted_cols=ADNI_WASTE, threshold=0.9)
    df_legend = pd.read_csv(LEGEND_FILE, on_bad_lines='skip')
    legend = df_legend.to_dict(orient='list')

    if use_augmented:
        from src.data.augment_images import load_augmented_paths
        df_paths = load_augmented_paths("src/data/images_augmented", legend)
    else:
        img_pre = ImagePreprocessor(img_input_size=(1,128,128,50), legend=legend)
        df_paths = img_pre.get_images_paths_df()
    
    df_merge = merge_modalities([df_clinical, df_paths], df_labels)
    
    if use_augmented and 'original_image_id' in df_paths.columns:
        image_to_group = df_paths.set_index('image_path')['original_image_id'].to_dict()
        df_merge['group_id'] = df_merge['image_path'].map(image_to_group).fillna(-1).astype(int)
    else:
        df_merge['group_id'] = range(len(df_merge))
    
    df_merge = df_merge.drop(columns=['Patient ID', 'VISCODE']).reset_index(drop=True)
    clinical_cols = [c for c in df_clinical.columns if c not in ['Patient ID', 'VISCODE']]

    if use_augmented:
        groups = df_merge['group_id']
        kf_outer = GroupKFold(n_splits=N_SPLITS)
        outer_split = kf_outer.split(df_merge, df_merge["Label"], groups=groups)
    else:
        kf_outer = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        outer_split = kf_outer.split(df_merge, df_merge["Label"])
    
    folds = []
    for fold_idx, (train_calib_idx, test_idx) in enumerate(outer_split):
        train_calib_df = df_merge.loc[train_calib_idx].reset_index(drop=True)
        test_df = df_merge.loc[test_idx].reset_index(drop=True)
        
        if use_augmented:
            groups_tc = train_calib_df['group_id']
            kf_inner = GroupKFold(n_splits=4)
            train_idx_inner, calib_idx_inner = next(kf_inner.split(
                train_calib_df, train_calib_df["Label"], groups=groups_tc
            ))
        else:
            kf_inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=42 + fold_idx)
            train_idx_inner, calib_idx_inner = next(kf_inner.split(
                train_calib_df, train_calib_df["Label"]
            ))
        
        train_df = train_calib_df.loc[train_idx_inner].reset_index(drop=True)
        calib_df = train_calib_df.loc[calib_idx_inner].reset_index(drop=True)
        
        for df in [train_df, calib_df, test_df]:
            if 'group_id' in df.columns:
                df.drop(columns=['group_id'], inplace=True)
        
        fold_data = {
            'X_train': {'clinical': train_df[clinical_cols], 'images': train_df[['image_path']].dropna()},
            'y_train': train_df["Label"],
            'X_calib': {'clinical': calib_df[clinical_cols], 'images': calib_df[['image_path']].dropna()},
            'y_calib': calib_df["Label"],
            'X_test': {'clinical': test_df[clinical_cols], 'images': test_df[['image_path']].dropna()},
            'y_test': test_df["Label"]
        }
        folds.append(fold_data)

    return folds
