"""Data loader module for AI4I 2020 Predictive Maintenance dataset."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, ROOT_DIR
from src.data.generate_synthetic import generate_synthetic_ai4i_dataset
from src.utils import setup_logger

logger = setup_logger("load_data")

# Recognized filenames for real dataset
REAL_DATASET_NAMES = [
    "ai4i_2020.csv",
    "ai4i2020.csv",
    "AI4I_2020.csv",
    "ai4i_data.csv",
    "ai4i2020_predictive_maintenance.csv",
]

# Standard column mapping dictionary
COLUMN_RENAME_MAP = {
    "Type": "type",
    "type": "type",
    "Air temperature [K]": "air_temp",
    "air temperature [k]": "air_temp",
    "air_temp": "air_temp",
    "Process temperature [K]": "process_temp",
    "process temperature [k]": "process_temp",
    "process_temp": "process_temp",
    "Rotational speed [rpm]": "rotational_speed",
    "rotational speed [rpm]": "rotational_speed",
    "rotational_speed": "rotational_speed",
    "Torque [Nm]": "torque",
    "torque [nm]": "torque",
    "torque": "torque",
    "Tool wear [min]": "tool_wear",
    "tool wear [min]": "tool_wear",
    "tool_wear": "tool_wear",
    "Target": "machine_failure",
    "target": "machine_failure",
    "Machine failure": "machine_failure",
    "machine failure": "machine_failure",
    "machine_failure": "machine_failure",
}

CANONICAL_COLUMNS = [
    "type",
    "air_temp",
    "process_temp",
    "rotational_speed",
    "torque",
    "tool_wear",
    "machine_failure",
]


def find_raw_dataset(data_dir: Path = DATA_RAW_DIR) -> Tuple[Optional[Path], str]:
    """Searches for real AI4I dataset in data/raw; returns (path, source_type)."""
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    # Check for known real file names
    for name in REAL_DATASET_NAMES:
        candidate = data_dir / name
        if candidate.is_file():
            logger.info(f"Discovered real AI4I dataset file: {candidate}")
            return candidate, "real"

    # Check any .csv in raw dir that is not synthetic
    for candidate in data_dir.glob("*.csv"):
        if "synthetic" not in candidate.name.lower():
            logger.info(f"Found non-synthetic raw candidate file: {candidate}")
            return candidate, "real"

    # Check if synthetic file already exists
    synthetic_candidate = data_dir / "synthetic_ai4i_smoke_test.csv"
    if synthetic_candidate.is_file():
        logger.info(f"Discovered existing synthetic smoke-test dataset: {synthetic_candidate}")
        return synthetic_candidate, "synthetic"

    return None, "none"


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes columns, cleans failure flags, and drops IDs."""
    clean_df = df.copy()

    # Normalize column names for flexible matching
    column_mapping = {}
    for col in clean_df.columns:
        if col in COLUMN_RENAME_MAP:
            column_mapping[col] = COLUMN_RENAME_MAP[col]
        elif col.lower().strip() in COLUMN_RENAME_MAP:
            column_mapping[col] = COLUMN_RENAME_MAP[col.lower().strip()]

    clean_df = clean_df.rename(columns=column_mapping)

    # Create machine_failure if not present but failure modes exist
    if "machine_failure" not in clean_df.columns:
        failure_modes = [col for col in ["TWF", "HDF", "PWF", "OSF", "RNF"] if col in clean_df.columns]
        if failure_modes:
            logger.info(f"Synthesizing machine_failure from failure modes: {failure_modes}")
            clean_df["machine_failure"] = (clean_df[failure_modes].sum(axis=1) > 0).astype(int)
        elif "Target" in clean_df.columns:
            clean_df["machine_failure"] = clean_df["Target"].astype(int)
        elif "Machine failure" in clean_df.columns:
            clean_df["machine_failure"] = clean_df["Machine failure"].astype(int)

    # Validate target presence
    if "machine_failure" not in clean_df.columns:
        raise ValueError("Could not find or construct 'machine_failure' target column.")

    # Drop extraneous metadata/identifiers
    cols_to_drop = [
        col
        for col in clean_df.columns
        if col not in CANONICAL_COLUMNS
    ]
    if cols_to_drop:
        logger.info(f"Dropping extraneous columns: {cols_to_drop}")
        clean_df = clean_df.drop(columns=cols_to_drop)

    # Ensure canonical column order
    clean_df = clean_df[CANONICAL_COLUMNS]

    # Type casting
    clean_df["type"] = clean_df["type"].astype(str).str.strip().str.upper()
    for col in ["air_temp", "process_temp", "rotational_speed", "torque", "tool_wear"]:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
    clean_df["machine_failure"] = clean_df["machine_failure"].astype(int)

    return clean_df


def save_metadata(
    df: pd.DataFrame,
    source: str,
    output_dir: Path = DATA_PROCESSED_DIR,
) -> Path:
    """Saves dataset metadata to data/processed/metadata.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"

    target_dist = df["machine_failure"].value_counts().to_dict()
    # Format target_dist keys as string for clean JSON serialization
    target_dist_str = {str(k): int(v) for k, v in target_dist.items()}

    missing_counts = df.isnull().sum().to_dict()

    metadata = {
        "dataset_source": source,
        "number_of_rows": int(len(df)),
        "number_of_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "target_distribution": target_dist_str,
        "target_positive_percentage": round(float(df["machine_failure"].mean() * 100), 2),
        "missing_value_count": missing_counts,
        "created_timestamp": datetime.utcnow().isoformat() + "Z",
        "warning": (
            "WARNING: Pipeline is utilizing synthetic smoke-test data. Replace with real AI4I 2020 dataset for final submission."
            if source == "synthetic"
            else "Real dataset loaded successfully."
        ),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Saved dataset metadata to {metadata_path}")
    return metadata_path


def load_ai4i_data(
    data_dir: Path = DATA_RAW_DIR,
    auto_generate_synthetic: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """Loads and standardizes AI4I dataset, falling back to synthetic generator if necessary."""
    file_path, source = find_raw_dataset(data_dir)

    if file_path is None:
        if auto_generate_synthetic:
            logger.warning("No dataset located in data/raw. Generating synthetic AI4I smoke-test dataset...")
            synthetic_df = generate_synthetic_ai4i_dataset()
            file_path = data_dir / "synthetic_ai4i_smoke_test.csv"
            source = "synthetic"
        else:
            raise FileNotFoundError(f"No raw dataset located in {data_dir}")

    logger.info(f"Loading raw dataset from {file_path} (source: {source})...")
    raw_df = pd.read_csv(file_path)

    clean_df = standardize_dataframe(raw_df)
    logger.info(f"Standardized dataset successfully: shape={clean_df.shape}")

    # Manage synthetic data warning file
    warning_path = ROOT_DIR / "USE_REAL_DATA.md"
    if source == "real" and warning_path.exists():
        warning_path.unlink()
        logger.info(f"Removed synthetic warning file {warning_path} since real data is active.")
    elif source == "synthetic" and not warning_path.exists():
        warning_content = (
            "# Synthetic Data Warning\n\n"
            "This pipeline is currently using synthetic data for smoke testing only. "
            "Replace data/raw/ with the real AI4I 2020 dataset before final submission.\n\n"
            "See `DOWNLOAD_INSTRUCTIONS.md` for steps on downloading the official dataset from UCI or Kaggle.\n"
        )
        warning_path.write_text(warning_content, encoding="utf-8")

    # Save metadata
    save_metadata(clean_df, source)

    return clean_df, source


if __name__ == "__main__":
    df, src = load_ai4i_data()
    print(f"Data successfully loaded from {src}. Shape: {df.shape}")
    print(df.head())
