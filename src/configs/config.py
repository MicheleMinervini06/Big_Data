LOCAL_DATA_DIR =  "src/data"
IMAGE_FOLDER = LOCAL_DATA_DIR + "/images"
LABELS_FILE = LOCAL_DATA_DIR + "/tables/ADNIMERGE_29Nov2024.csv"
LEGEND_FILE = LOCAL_DATA_DIR + "/tabels/ADNI_T1_MPRAGE_12_17_2024.csv"
IMG_PREFIX = "processed_MRI"
TEST_PERC = 0.2
N_SPLITS = 5
ADNI_WASTE = [
    "ABETA", "update_stamp", "TAU", "PTAU", "FSVERSION", "RID", "CDRSB", "EcoPtMem",
    "EXAMDATE", "SITE", "COLPROT", "ORIGPROT", "PTETHCAT", 'PTRACCAT', "APOE4",
    "FDG", "PIB", "AV45", "Hippocampus", "WholeBrain", "Enthorhinal", "Fusiform",
    "MidTemp", "ICV", "M"
]