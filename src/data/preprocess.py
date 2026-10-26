"""Preprocessing module creating split datasets for Manual ML and AutoML pipelines."""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PROCESSED_DIR,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.utils import set_random_seed, setup_logger

logger = setup_logger("preprocess")


def stratified_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
    target_col: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits dataset into stratified train and test sets using fixed random seed."""
    set_random_seed(seed)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[target_col],
    )
    logger.info(
        f"Stratified split completed: Train shape={train_df.shape}, "
        f"Test shape={test_df.shape}, Test size={test_size}"
    )
    return train_df.copy(), test_df.copy()


def prepare_manual_ml_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocesses data for manual ML:
    - Numeric features: missing values imputed with training median.
    - Categorical features: missing values imputed with training mode.
    - One-hot encodes 'type'.
    - Target column remains machine_failure.
    """
    train_proc = train_df.copy()
    test_proc = test_df.copy()

    # 1. Impute numeric features using training median
    for col in NUMERIC_FEATURES:
        if col in train_proc.columns:
            median_val = train_proc[col].median()
            train_proc[col] = train_proc[col].fillna(median_val)
            test_proc[col] = test_proc[col].fillna(median_val)

    # 2. Impute categorical features using training mode
    if "type" in train_proc.columns:
        mode_val = train_proc["type"].mode()[0] if not train_proc["type"].empty else "L"
        train_proc["type"] = train_proc["type"].fillna(mode_val)
        test_proc["type"] = test_proc["type"].fillna(mode_val)

    # 3. One-hot encode 'type'
    expected_categories = ["H", "L", "M"]
    train_encoded = pd.get_dummies(train_proc, columns=["type"], prefix="type", dtype=int)
    test_encoded = pd.get_dummies(test_proc, columns=["type"], prefix="type", dtype=int)

    # Ensure all dummy columns exist in both train and test
    for cat in expected_categories:
        col_name = f"type_{cat}"
        if col_name not in train_encoded.columns:
            train_encoded[col_name] = 0
        if col_name not in test_encoded.columns:
            test_encoded[col_name] = 0

    # Align columns, placing target at the end
    feature_cols = [c for c in train_encoded.columns if c != target_col]
    feature_cols.sort()
    aligned_cols = feature_cols + [target_col]

    train_proc_final = train_encoded[aligned_cols]
    test_proc_final = test_encoded[aligned_cols]

    logger.info(
        f"Manual ML datasets prepared: Train columns={list(train_proc_final.columns)}, "
        f"Train shape={train_proc_final.shape}, Test shape={test_proc_final.shape}"
    )
    return train_proc_final, test_proc_final


def prepare_automl_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocesses data for AutoML:
    - Retains numeric features.
    - Retains 'type' as categorical string (no one-hot encoding).
    - Preserves target column.
    """
    train_proc = train_df.copy()
    test_proc = test_df.copy()

    # Ensure type is clean string
    train_proc["type"] = train_proc["type"].astype(str)
    test_proc["type"] = test_proc["type"].astype(str)

    logger.info(
        f"AutoML datasets prepared: Train shape={train_proc.shape}, "
        f"Test shape={test_proc.shape}"
    )
    return train_proc, test_proc


def preprocess_and_save_all(
    df: pd.DataFrame,
    output_dir: Path = DATA_PROCESSED_DIR,
) -> Dict[str, Path]:
    """Executes full preprocessing pipeline and saves CSV files to data/processed/."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Stratified split
    train_raw, test_raw = stratified_split(df)

    # 2. Manual ML datasets
    train_manual, test_manual = prepare_manual_ml_data(train_raw, test_raw)
    path_train_manual = output_dir / "train_manual.csv"
    path_test_manual = output_dir / "test_manual.csv"
    train_manual.to_csv(path_train_manual, index=False)
    test_manual.to_csv(path_test_manual, index=False)
    logger.info(f"Saved manual ML sets: {path_train_manual}, {path_test_manual}")

    # 3. AutoML datasets
    train_automl, test_automl = prepare_automl_data(train_raw, test_raw)
    path_train_automl = output_dir / "train_automl.csv"
    path_test_automl = output_dir / "test_automl.csv"
    train_automl.to_csv(path_train_automl, index=False)
    test_automl.to_csv(path_test_automl, index=False)
    logger.info(f"Saved AutoML sets: {path_train_automl}, {path_test_automl}")

    return {
        "train_manual": path_train_manual,
        "test_manual": path_test_manual,
        "train_automl": path_train_automl,
        "test_automl": path_test_automl,
    }


if __name__ == "__main__":
    from src.data.load_data import load_ai4i_data

    clean_df, _ = load_ai4i_data()
    saved = preprocess_and_save_all(clean_df)
    print("Preprocessed files successfully created:")
    for k, v in saved.items():
        print(f" - {k}: {v}")
