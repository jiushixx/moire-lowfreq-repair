"""Explicit execution entry point; importing modules never starts a run."""
import argparse, importlib, json
from pathlib import Path

def main():
    entries = {
        "evaluate": ("evaluation", "run_frozen_test"),
        "train-repair": ("repair", "run_independent"),
        "control-gate": ("spatial", "run_spatial"),
        "control-calibration": ("calibration", "run_calibration"),
        "control-joint": ("joint", "run_joint"),
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=entries)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    name, entry = entries[args.experiment]
    module = importlib.import_module("moire_project." + name)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key, value in config.items():
        if not key.isupper() or not hasattr(module, key):
            raise ValueError("Unknown configuration: " + key)
        if isinstance(getattr(module, key), Path):
            value = Path(value)
        setattr(module, key, value)
    getattr(module, entry)()

if __name__ == "__main__":
    main()
