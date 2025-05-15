from pathlib import Path

# Punto alla root del progetto src/
CONFIG_DIR = Path(__file__).parent
DATA_DIR   = CONFIG_DIR.parent / "data"

IMAGE_FOLDER = DATA_DIR / "images_post"
LABELS_FILE  = DATA_DIR / "tables/ADNIMERGE_29Nov2024.csv"
LEGEND_FILE  = DATA_DIR / "tables/ADNI_T1_MPRAGE_12_17_2024.csv"
IMG_PREFIX = DATA_DIR / "images_pre"

TEST_PERC = 0.2
N_SPLITS = 5
ADNI_WASTE = [
    "ABETA", "update_stamp", "TAU", "PTAU", "FSVERSION", "RID", "CDRSB", "EcoPtMem",
    "EXAMDATE", "SITE", "COLPROT", "ORIGPROT", "PTETHCAT", 'PTRACCAT', "APOE4",
    "FDG", "PIB", "AV45", "Hippocampus", "WholeBrain", "Enthorhinal", "Fusiform",
    "MidTemp", "ICV", "M"
]