"""
Handles data genearation or loading and preparation for training
"""

from io import BytesIO
from typing import List

import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
from Crypto.SelfTest.Cipher.test_SIV import transform
from minio import Minio
import nibabel as nib
from pathlib import Path
from src.data.processing_data import IMAGE_FOLDER
import os
import pickle
import nibabel as nib

from minio.datatypes import Object

pd.set_option('future.no_silent_downcasting', True)

class ImagePreprocessor:
    """
    Gestisce il caricamento e preprocessing delle immagini da cartella locale.
    """
    def __init__(self, img_input_size: tuple[int, int, int, int], legend: dict, cache_dir: Path = IMAGE_FOLDER):
        self.img_input_size = img_input_size
        self.legend = legend
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_images_paths_df(self, image_folder: Path) -> pd.DataFrame:
        """
        Scansiona la cartella locale per file .nii o simili,
        li pre-processa e salva in cache come pickle.
        """
        records = []
        for nifti_path in image_folder.rglob("*.nii*"):
            key = nifti_path.stem
            cache_path = self.cache_dir / f"{key}.pkl"
            if not cache_path.exists():
                img = self._load_nifti(nifti_path)
                img_rs = self.reshape_img(img, self.img_input_size)
                img_sc = self.min_max_scale_image(img_rs)
                with open(cache_path, "wb") as f:
                    pickle.dump(img_sc, f)
            # recupera ID paziente e VISCODE da legenda
            id_idx = self.legend["Image Data ID"].index(key)
            pid = self.legend["Subject"][id_idx]
            vis = self.legend["Visit"][id_idx]
            records.append({"image_path": str(cache_path), "Patient ID": pid, "VISCODE": vis})
        return pd.DataFrame(records)

    def _load_nifti(self, path: Path) -> torch.Tensor:
        """Carica file NIfTI da disco e converte in tensor."""
        
        img = nib.load(str(path))
        arr = img.get_fdata()
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

    @staticmethod
    def reshape_img(img: torch.tensor, desired_shape: [int, int, int, int]) -> torch.tensor:
        x_start = int(list(img.shape)[1] / 2) - int(desired_shape[1] / 2)
        x_end = x_start + desired_shape[1]
        y_start = int(list(img.shape)[2] / 2) - int(desired_shape[2] / 2)
        y_end = y_start + desired_shape[2]
        z_start = int(list(img.shape)[3] / 2) - int(desired_shape[3] / 2)
        z_end = z_start + desired_shape[3]
        rs = img[:, x_start:x_end, y_start:y_end, z_start:z_end]
        return rs

    @staticmethod
    def min_max_scale_image(image:torch.tensor) -> torch.tensor:
        eps = torch.max(torch.tensor([1e-8]), torch.max(image) - torch.min(image))
        pp = (image - torch.min(image)) / eps
        return pp


def dataframe_to_tensor(data: pd.DataFrame):
    x_list = []
    for index in data.index:
        x_list.append(data.images[index])

    return torch.stack(x_list, dim=0)

def merge_modalities(different_modalities_dfs: list, labels_df: pd.DataFrame):
    """
    Merges data from different modalities in the format required by veronet
    inputs:
        different_modalities_dfs: list of pandas DataFrames containing the data for each of the modalities, each one must have a 
        "Patient ID" and a "VISCODE" (optional) column to do the merging
        labels_df: a pandas DataFrame containing the labels and a column called "Patient ID" to do the merging
    returns:
        out_dataframe: a pandas DataFrame containing all the data
    """
    out_dataframe = labels_df
    for modality in different_modalities_dfs:
        if "VISCODE" in modality.columns:
            out_dataframe = pd.merge(out_dataframe, modality, left_on=['Patient ID','VISCODE'], right_on=['Patient ID','VISCODE'], how='left')
        else:
            out_dataframe = pd.merge(out_dataframe, modality, left_on=['Patient ID'], right_on=['Patient ID'], how='left')
    return out_dataframe.drop_duplicates()


def nifti_df_from_local(paths_df: pd.DataFrame):
    imgs_list = []
    for obj in paths_df["image_path"]:
        with open(obj, "rb") as fh:
            pp = pickle.load(fh)
            imgs_list.append(pp)
    out_df = pd.DataFrame({"images": imgs_list})
    return out_df


def retrieve_object_from_minio(client: Minio, bucket:str, url:str):
    response = client.get_object(bucket, url)
    out = BytesIO(response.read())
    return out


def treat_labels(raw_labels_file:BytesIO):
    complete_df = pd.read_csv(raw_labels_file, on_bad_lines='skip',sep=',')
    labels_df = complete_df[["PTID", "DX", "VISCODE"]]
    labels_df = labels_df.replace("Dementia", "AD")
    labels_df = labels_df.rename({"PTID": "Patient ID", "DX": "Label"}, axis=1)
    labels_df = labels_df.replace({"CN": 1, "MCI": 2, "AD": 3}).infer_objects(copy=False)
    labels_df = labels_df.dropna(axis=0)
    return labels_df


def treat_clinical_data(raw_labels_file: BytesIO, unwanted_cols : list[str], threshold=0.8):
    complete_df = pd.read_csv(raw_labels_file, on_bad_lines='skip',sep=',')
    clinical_df = complete_df.drop("DX", axis=1)
    ptid = clinical_df["PTID"]
    viscode = clinical_df["VISCODE"]
    clinical_df = clinical_df.drop("VISCODE", axis=1)
    clinical_df = clinical_df.drop("PTID", axis=1)
    #select only numerical types
    clinical_df = get_rid_of_unwanted_cols(clinical_df, unwanted_cols)
    clinical_df = treat_missing_clinical(clinical_df, threshold)

    df_numeric = clinical_df.select_dtypes(include = 'number')
    df_categorical = clinical_df.select_dtypes(exclude = 'number')
    ohe = sklearn.preprocessing.OneHotEncoder(sparse_output=False).set_output(transform="pandas")
    df_ohe = ohe.fit_transform(df_categorical)
    df_total = pd.concat([df_numeric, df_ohe], axis=1)
    df_total['Patient ID'] = ptid
    df_total["VISCODE"] = viscode
    return df_total

    
def get_rid_of_unwanted_cols(clinical_df, unwanted_cols):
    """
    Since many columns are not compliant in the adnimerge file, some of them must be discarded
    """
    for i in unwanted_cols:
        if i in clinical_df.columns:
            clinical_df = clinical_df.drop(i, axis=1)
    for i in clinical_df.columns:
        if "_bl" in i or "_BL" in i:
            clinical_df = clinical_df.drop(i, axis = 1)
        
    return clinical_df
    


def treat_missing_clinical(df, alpha):
    """
    Filter the colu,ns based on percentage of null values.
    :param df: input dataframe full of NaNs
    :param alpha: Threshold of percentage of missing values to exclude (eg. 0.9 for 90%)
    :return: filtered DataFrame
    """
    # Soglia assoluta di valori nulli consentiti per colonna
    threshold = alpha * len(df)
    # Filtra le colonne con valori nulli minori della soglia
    filtered_df = df.loc[:, df.isnull().sum() < threshold]
    return filtered_df
