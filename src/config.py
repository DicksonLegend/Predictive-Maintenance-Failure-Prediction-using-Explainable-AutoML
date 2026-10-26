"""Global configuration and project paths for Predictive Maintenance AutoML & XAI."""

from pathlib import Path

# Fixed random seed for reproducibility
RANDOM_SEED: int = 42

# Target variable definition
TARGET_COLUMN: str = "machine_failure"

# Train/Test split configuration
TEST_SIZE: float = 0.2

# Evaluation metrics
PRIMARY_METRIC: str = "f1"
SECONDARY_METRICS: list[str] = ["roc_auc", "precision", "recall", "accuracy"]

# AutoML time budget configurations (in seconds)
AUTOML_TIME_BUDGET_SECONDS_AGENT: int = 60
AUTOML_TIME_BUDGET_SECONDS_FINAL: int = 900

# Canonical feature definitions
FEATURE_COLUMNS: list[str] = [
    "type",
    "air_temp",
    "process_temp",
    "rotational_speed",
    "torque",
    "tool_wear",
]

NUMERIC_FEATURES: list[str] = [
    "air_temp",
    "process_temp",
    "rotational_speed",
    "torque",
    "tool_wear",
]

CATEGORICAL_FEATURES: list[str] = ["type"]
MACHINE_TYPES: list[str] = ["L", "M", "H"]

# Project Directory Paths
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
RESULTS_DIR: Path = ROOT_DIR / "results"
PLOTS_DIR: Path = ROOT_DIR / "plots"
MODELS_DIR: Path = ROOT_DIR / "models"
REPORTS_DIR: Path = ROOT_DIR / "reports"
SCREENSHOTS_DIR: Path = REPORTS_DIR / "screenshots"
NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"
APP_DIR: Path = ROOT_DIR / "app"
SLIDES_DIR: Path = ROOT_DIR / "slides"

# Ensure all essential directories exist
ALL_DIRS: list[Path] = [
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    RESULTS_DIR,
    PLOTS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    SCREENSHOTS_DIR,
    NOTEBOOKS_DIR,
    APP_DIR,
    SLIDES_DIR,
]


def init_directories() -> None:
    """Create all core directories if they do not exist."""
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    init_directories()
    print(f"Project Root initialized at: {ROOT_DIR}")
