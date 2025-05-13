import pickle
from pathlib import Path

# Directory per salvare i modelli
MODEL_DIR = Path("src/models/models_saved")
MODEL_DIR.mkdir(exist_ok=True)

def save_model(model, model_name: str):
    """
    Salva il modello tramite pickle nella cartella models/ con nome <model_name>.pkl
    """
    path = MODEL_DIR / f"{model_name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def load_model(model_name: str):
    """
    Carica il modello pickle da models/<model_name>.pkl
    """
    path = MODEL_DIR / f"{model_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model file {path} not found")
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {path}")
    return model