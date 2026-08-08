import yaml
import logging
import hashlib
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """
    Create and configure a logger for consistent project-wide logging.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# ---------------------------------------------------------
# Config Loader
# ---------------------------------------------------------
def load_config(path: str) -> dict:
    """
    Load YAML configuration file.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ---------------------------------------------------------
# Data Hashing (for MLflow logging)
# ---------------------------------------------------------
def get_file_hash(path: str) -> str:
    """
    Compute a SHA256 hash of a file to track data version.
    Useful for MLflow logging.
    """
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


# ---------------------------------------------------------
# Safe Data Loader
# ---------------------------------------------------------
def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV file with basic validation.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"Dataset at {path} is empty.")

    return df


# ---------------------------------------------------------
# Train/Test Split Helper
# ---------------------------------------------------------
def split_features_target(df: pd.DataFrame, target: str):
    """
    Split dataframe into X and y.
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataset.")

    X = df.drop(columns=[target])
    y = df[target]

    return X, y

