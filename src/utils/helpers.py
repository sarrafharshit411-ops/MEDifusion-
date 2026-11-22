"""
MediFusion AI - Utility Functions
Common helpers used across the project.
"""

import os
import json
import random
import logging
import yaml
import numpy as np
import torch


def load_config(config_path: str = None) -> dict:
    """Load YAML configuration file with path expansion."""
    if config_path is None:
        # Default: look relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(project_root, "configs", "config.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    if "dataset" in config:
        for k, v in config["dataset"].items():
            if isinstance(v, str) and v.startswith("~"):
                config["dataset"][k] = os.path.expanduser(v)
    return config


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ensure_dir(path: str):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_metrics(metrics: dict, filepath: str):
    """Save metrics dictionary to JSON file."""
    ensure_dir(os.path.dirname(filepath))
    # Convert numpy types to Python native types for JSON serialization
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            clean[k] = float(v)
        elif isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        else:
            clean[k] = v
    with open(filepath, "w") as f:
        json.dump(clean, f, indent=2)
    logging.info(f"Metrics saved to {filepath}")


def load_metrics(filepath: str) -> dict:
    """Load metrics from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def setup_logging(level=logging.INFO):
    """Configure logging for the project with stdout stream and immediate flush."""
    import sys
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
