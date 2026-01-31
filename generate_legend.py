"""
Script per generare il file legenda ADNI dalle immagini pkl
"""
import pandas as pd
import os
from pathlib import Path

# Path alle immagini pkl
images_dir = r"c:\Users\mikim\Desktop\wetransfer_fair_images-1-zip_2025-12-23_1103\images"
output_file = r"c:\Users\mikim\Desktop\Uni\Big Data\Big_Data\src\data\tables\ADNI_T1_MPRAGE_12_17_2024.csv"

print("Generazione file legenda da immagini pkl...")

# Raccoglie i file pkl
pkl_files = list(Path(images_dir).glob("processed_*.pkl"))
print(f"Trovati {len(pkl_files)} file pkl")

# Estrae informazioni dai nomi file
# Formato: processed_{PTID}_I{IMAGEUID}.pkl
# Esempio: processed_002_S_0619_I57662.pkl

data = []
for pkl_file in pkl_files:
    name = pkl_file.stem  # nome senza estensione
    # Remove "processed_" prefix
    name = name.replace("processed_", "")
    
    # Split by "_I" to separate PTID and IMAGEUID
    if "_I" in name:
        ptid_part, imageuid = name.split("_I")
        imageuid = int(imageuid)
        
        data.append({
            "Image Data ID": imageuid,
            "Subject": ptid_part,
            "Acq Date": "2024-01-01",  # Data fittizia
            "Description": "MPRAGE"
        })

# Crea DataFrame
df_legend = pd.DataFrame(data)
print(f"\nCreato DataFrame con {len(df_legend)} righe")
print("\nPrime 5 righe:")
print(df_legend.head())

# Salva CSV
os.makedirs(os.path.dirname(output_file), exist_ok=True)
df_legend.to_csv(output_file, index=False)
print(f"\n✅ File legenda salvato in: {output_file}")
