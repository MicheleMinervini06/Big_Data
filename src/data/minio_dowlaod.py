import os
from pathlib import Path
from minio import Minio
from minio.error import S3Error

# --- CONFIGURAZIONE DA AMBIENTE ---
MINIO_URL      = "kubeflow-minio.lutechdigitale.it"      
MINIO_USERNAME = "minio" # la tua access key
MINIO_PASSWORD = "minio123" # la tua secret key

if not all([MINIO_URL, MINIO_USERNAME, MINIO_PASSWORD]):
    raise ValueError("Assicurati di aver impostato MINIO_URL, MINIO_USERNAME e MINIO_PASSWORD")

# Istanza client MinIO
minio_client = Minio(
    MINIO_URL,
    access_key=MINIO_USERNAME,
    secret_key=MINIO_PASSWORD,
    secure=True
)

# Bucket e prefix da esplorare
BUCKET_NAME = "fair"
PREFIX      = "multimodal_fullsample/processed_MRI/"

# Directory locale dove salvare i .nii
DOWNLOAD_DIR = Path("./downloaded_nii")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

def download_all_nii():
    """
    Scorre ricorsivamente tutti gli oggetti sotto PREFIX
    e scarica quelli che finiscono con .nii
    """
    try:
        for obj in minio_client.list_objects(bucket_name=BUCKET_NAME,
                                             prefix=PREFIX,
                                             recursive=True):
            if obj.object_name.lower().endswith(".nii"):
                local_path = DOWNLOAD_DIR / Path(obj.object_name).name
                print(f"Scarico {obj.object_name} → {local_path}")
                minio_client.fget_object(
                    bucket_name=BUCKET_NAME,
                    object_name=obj.object_name,
                    file_path=str(local_path)
                )
        print("Download completato.")
    except S3Error as err:
        print(f"Errore durante l'accesso a MinIO: {err}")

if __name__ == "__main__":
    download_all_nii()
