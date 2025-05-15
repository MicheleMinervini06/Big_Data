import os
import torch
import monai
import pandas as pd
from src.utils.load_data import nifti_df_from_local, dataframe_to_tensor
from dataclasses import dataclass

@dataclass
class ImageEmbeddingExtractorInput:
    temp_file: str = "image_embeddings.csv"  # File per il training set
    test_file: str = "test_embeddings.csv"   # File per il test set

class ImageEmbeddingExtractor:
    def __init__(self, inp_list: ImageEmbeddingExtractorInput):
        self.inp_list = inp_list
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Carica ResNet18 3D pre-addestrata senza l'ultimo layer fully connected
        self.resnet = monai.networks.nets.resnet18(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=512,  # Otteniamo embedding di dimensione 512
            pretrained=True,
            feed_forward=False,
            bias_downsample=True,
            shortcut_type="A"
        )
        self.resnet.eval().to(self.device)

    def _compute_embeddings(self, data: pd.DataFrame, file_path: str, use_cache=True):
        """
        Calcola e salva embeddings per un set di immagini NIfTI.
        """
        if use_cache and os.path.exists(file_path):
            print(f"📂 Carico embeddings da {file_path}...")
            return pd.read_csv(file_path, header=None).values

        print(f"⚙️  Calcolo embeddings per {file_path}...")
        embeddings_list = []

        with torch.no_grad():
            for idx in range(len(data)):
                x = data.iloc[idx:idx+1]  # Prendi una riga alla volta
                x_df = nifti_df_from_local(paths_df=x)
                x_tensor = dataframe_to_tensor(x_df).to(self.device)

                embedding = self.resnet.forward(x_tensor)  # Estrai embedding
                embeddings_list.append(embedding.cpu().numpy().flatten())

        # Salva embeddings in CSV
        all_embeddings = pd.DataFrame(embeddings_list)
        all_embeddings.to_csv(file_path, index=False, header=None)

        print(f"✅ Embeddings salvati in {file_path}")
        return all_embeddings.values

    def extract_embeddings(self, data: pd.DataFrame, use_cache=True):
        """
        Estrai gli embeddings per il training set.
        """
        return self._compute_embeddings(data, self.inp_list.temp_file, use_cache)

    def extract_test_embeddings(self, data_test: pd.DataFrame, use_cache=True):
        """
        Estrai gli embeddings per il test set senza riaddestrare il modello.
        """
        return self._compute_embeddings(data_test, self.inp_list.test_file, use_cache)
