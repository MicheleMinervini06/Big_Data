from src.utils.load_data import ImagePreprocessor, retrieve_object_from_minio, treat_labels, treat_clinical_data, merge_modalities
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from src.configs.config import *


def processing_feature():
    # Scarica i dati da MinIO e posizionali in data/ (manuale)
    labels_df = treat_labels(LABELS_FILE)
    clinical_df = treat_clinical_data(LABELS_FILE, unwanted_cols=ADNI_WASTE, threshold=0.9)
    legend = pd.read_csv(LEGEND_FILE, on_bad_lines='skip').to_dict(orient='list')

    img_pre = ImagePreprocessor(img_input_size=(1,128,128,50), legend=legend)
    paths_df = img_pre.get_images_paths_df(LOCAL_DATA_DIR / IMG_PREFIX)

    df_merge = merge_modalities([clinical_df, paths_df], labels_df) # unisce dati clinici, path immagini e etichette in un unico dataFrame
    df_merge = df_merge.drop(columns=['Patient ID', 'VISCODE'], errors='ignore').reset_index(drop=True) # rimuove eventuali colonne non utili ('Patient ID', 'VISCODE')


    # Prepare clinical columns
    clinical_cols = [col for col in clinical_df.columns if col not in ['Patient ID', 'VISCODE']] # crea una lista di tutte le colonne che contengono solo i dati clinici (escludendo 'Patient ID'e'VISCODE')
    print(f"\n✅ Colonne cliniche selezionate: {clinical_cols}")
    print (f"\n Dimensione dati clinici: {len(clinical_cols)} ")
    
    # Imputazione NaN con la media
    # df_merge[clinical_cols] = df_merge[clinical_cols].fillna(df_merge[clinical_cols].mean()) # sostituisce NaN nei dati clinici con la media della rispettiva colonna
    df_merge[clinical_cols] = df_merge[clinical_cols].fillna(0)
    
    # Normalizzazione
    scaler = StandardScaler() # porta tutte le feature cliniche su una scala comune (media 0, deviazione standard 1) --> (valore-media)/devstd
    df_merge[clinical_cols] = scaler.fit_transform(df_merge[clinical_cols])


    ## Train-test split
    x_train, x_test = train_test_split(df_merge, test_size=TEST_PERC, random_state=42)

    X_mod1_train = x_train[clinical_cols]
    X_mod3_train = x_train[["image_path"]]

    X_mod1_test = x_test[clinical_cols]
    X_mod3_test = x_test[["image_path"]]

    

    X_mods = {'clinical': X_mod1_train, 'images': X_mod3_train.dropna()}
    X_test = {'clinical': X_mod1_test, 'images': X_mod3_test.dropna()}
    y_train = x_train["Label"]
    print(f"y_train: {y_train.shape}")
    y_test = x_test["Label"]
    print(f"y_test: {y_test.shape}")

    return X_mods,X_test,y_train,y_test


def processing_features_cv():
    # Scarica i dati da MinIO e posizionali in data/ (manuale)
    
    # 1) Carica label e dati clinici
    labels_df   = treat_labels(LABELS_FILE)
    print(f"[DEBUG] labels_df shape: {labels_df.shape}")
    clinical_df = treat_clinical_data(LABELS_FILE, unwanted_cols=ADNI_WASTE, threshold=0.9)
    print(f"[DEBUG] clinical_df shape: {clinical_df.shape}")

    # 2) Carica legenda
    legend_df = pd.read_csv(LEGEND_FILE, on_bad_lines='skip')
    legend = legend_df.to_dict(orient='list')
    print(f"[DEBUG] legend keys: {list(legend.keys())}")
    sample_id = legend['Image Data ID'][0]
    print(f"[DEBUG] sample Image Data ID: {sample_id}")

    # 3) Pre-processore immagini
    img_pre = ImagePreprocessor(
        img_input_size=(1,128,128,50),
        legend=legend
    )
    paths_df = img_pre.get_images_paths_df()
    print(f"[DEBUG] paths_df shape: {paths_df.shape}")
    print(f"[DEBUG] paths_df headD:{paths_df.head()}")

    merged = merge_modalities([clinical_df, paths_df], labels_df)
    print(f"[DEBUG] merged shape before drop: {merged.shape}")
    merged = merged.drop(columns=['Patient ID','VISCODE'], errors='ignore').reset_index(drop=True)
    print(f"[DEBUG] merged shape after drop: {merged.shape}")
    print(f"[DEBUG] merged head:{merged.head()}")

    clinical_cols = [col for col in clinical_df.columns if col not in ['Patient ID', 'VISCODE']] # crea una lista di tutte le colonne che contengono solo i dati clinici (escludendo 'Patient ID'e'VISCODE')
    print(f"\n✅ Colonne cliniche selezionate: {clinical_cols}")
    print (f"\n Dimensione dati clinici: {len(clinical_cols)} ")

    merged[clinical_cols] = merged[clinical_cols].fillna(0)

    # Normalizzazione
    scaler = StandardScaler() # porta tutte le feature cliniche su una scala comune (media 0, deviazione standard 1) --> (valore-media)/devstd
    merged[clinical_cols] = scaler.fit_transform(merged[clinical_cols])

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(merged, merged["Label"])):
        train_df = merged.loc[train_idx].reset_index(drop=True)
        test_df = merged.loc[test_idx].reset_index(drop=True)

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

