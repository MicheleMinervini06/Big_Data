from pathlib import Path
import argparse
import yaml
from run_experiments import run_autoencoder,run_boosting

CONFIG_FILE = Path("src/configs/config.yaml")

def parse_args():
    p = argparse.ArgumentParser(description="Run multiple experiments.")
    p.add_argument("--experiments", nargs="*", default=None,
                   help="Lista di esperimenti da eseguire; se non specificato, tutti.")
    p.add_argument("--config", type=str,
                   help="Path a file YAML con definizione esperimenti.")
    return p.parse_args()


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single experiment.")
    parser.add_argument("experiment",
                        help="Nome dell'esperimento da eseguire, definito in config.yaml")
    return parser.parse_args()


def main():

    if not CONFIG_FILE.exists():
        raise FileNotFoundError("Il file config.yaml non è stato trovato. Aggiungilo nella root del progetto.")

    experiments = yaml.safe_load(CONFIG_FILE.read_text())

    args = parse_args()
    name = args.experiment
    if name not in experiments:
        raise KeyError(f"Esperimento '{name}' non definito in config.yaml.")

    if name == 'exp7':
        run_autoencoder()
    else:
        run_boosting(name, experiments[name])
    #run_experiment(name, experiments[name])


if __name__ == "__main__":
    main()