"""Synthetic data generator for AI4I 2020 Predictive Maintenance smoke testing."""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from src.config import DATA_RAW_DIR, RANDOM_SEED, ROOT_DIR
from src.utils import set_random_seed, setup_logger

logger = setup_logger("generate_synthetic")


def generate_synthetic_ai4i_dataset(
    n_samples: int = 10000,
    seed: int = RANDOM_SEED,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Generates synthetic predictive maintenance dataset matching the AI4I 2020 schema.

    Target imbalance is calibrated to ~3.5% - 5.0% positive failures.
    """
    set_random_seed(seed)
    rng = np.random.RandomState(seed)

    logger.info(f"Generating {n_samples} synthetic AI4I smoke-test records with seed {seed}...")

    # Type distribution: L (60%), M (30%), H (10%)
    types = rng.choice(["L", "M", "H"], size=n_samples, p=[0.60, 0.30, 0.10])

    # Air temperature [K]: 295 to 305 Kelvin
    air_temp = np.round(rng.normal(loc=300.0, scale=2.0, size=n_samples), 1)
    air_temp = np.clip(air_temp, 295.3, 304.5)

    # Process temperature [K]: 305 to 315 Kelvin (strongly correlated with air_temp)
    temp_diff = rng.normal(loc=10.0, scale=1.0, size=n_samples)
    process_temp = np.round(air_temp + temp_diff, 1)
    process_temp = np.clip(process_temp, 305.7, 313.8)

    # Rotational speed [rpm]: 1000 to 3000 rpm (typical center around 1530)
    rotational_speed = rng.normal(loc=1530, scale=180, size=n_samples).astype(int)
    rotational_speed = np.clip(rotational_speed, 1168, 2886)

    # Torque [Nm]: 10 to 80 Nm (inversely correlated with speed)
    # Power = torque * angular_speed -> torque ~ Power / speed
    expected_torque = 40.0 - 0.025 * (rotational_speed - 1530)
    torque_noise = rng.normal(loc=0.0, scale=7.0, size=n_samples)
    torque = np.round(expected_torque + torque_noise, 1)
    torque = np.clip(torque, 10.0, 76.6)

    # Tool wear [min]: 0 to 250 minutes
    tool_wear = rng.randint(0, 253, size=n_samples)

    # Physics-based failure logic:
    # 1. Tool Wear Failure (TWF): wear > 200 min
    p_twf = np.where(tool_wear > 200, (tool_wear - 200) / 100.0 * 0.15, 0.001)
    twf = (rng.uniform(0, 1, size=n_samples) < p_twf).astype(int)

    # 2. Heat Dissipation Failure (HDF): temp difference < 8.6 K and speed < 1380 rpm
    temp_margin = process_temp - air_temp
    hdf_cond = (temp_margin < 8.6) & (rotational_speed < 1380)
    p_hdf = np.where(hdf_cond, 0.70, 0.001)
    hdf = (rng.uniform(0, 1, size=n_samples) < p_hdf).astype(int)

    # 3. Power Failure (PWF): mechanical power = torque * speed (rad/s) is out of bounds
    power_watts = torque * (rotational_speed * (2 * np.pi / 60))
    pwf_cond = (power_watts < 3500) | (power_watts > 9000)
    p_pwf = np.where(pwf_cond, 0.60, 0.001)
    pwf = (rng.uniform(0, 1, size=n_samples) < p_pwf).astype(int)

    # 4. Overstrain Failure (OSF): product of tool wear and torque exceeds thresholds
    osf_metric = tool_wear * torque
    osf_thresh = np.where(types == "L", 11000, np.where(types == "M", 12000, 13000))
    p_osf = np.where(osf_metric > osf_thresh, 0.65, 0.001)
    osf = (rng.uniform(0, 1, size=n_samples) < p_osf).astype(int)

    # 5. Random Network/Sensor Failure (RNF): ~0.1% random failure
    rnf = (rng.uniform(0, 1, size=n_samples) < 0.0019).astype(int)

    # Combined Machine failure indicator
    machine_failure = ((twf == 1) | (hdf == 1) | (pwf == 1) | (osf == 1) | (rnf == 1)).astype(int)

    # Generate synthetic UDI and Product ID
    udi = np.arange(1, n_samples + 1)
    product_ids = [f"{t}{10000 + i}" for i, t in enumerate(types)]

    df = pd.DataFrame(
        {
            "UDI": udi,
            "Product ID": product_ids,
            "Type": types,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "Target": machine_failure,
            "TWF": twf,
            "HDF": hdf,
            "PWF": pwf,
            "OSF": osf,
            "RNF": rnf,
        }
    )

    failure_rate = df["Target"].mean() * 100
    logger.info(
        f"Synthetic dataset generated: shape={df.shape}, "
        f"failures={df['Target'].sum()} ({failure_rate:.2f}%)"
    )

    if output_path is None:
        output_path = DATA_RAW_DIR / "synthetic_ai4i_smoke_test.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved synthetic dataset to {output_path}")

    # Create warning file USE_REAL_DATA.md
    warning_path = ROOT_DIR / "USE_REAL_DATA.md"
    warning_content = (
        "# Synthetic Data Warning\n\n"
        "This pipeline is currently using synthetic data for smoke testing only. "
        "Replace data/raw/ with the real AI4I 2020 dataset before final submission.\n\n"
        "See `DOWNLOAD_INSTRUCTIONS.md` for steps on downloading the official dataset from UCI or Kaggle.\n"
    )
    warning_path.write_text(warning_content, encoding="utf-8")
    logger.info(f"Created warning banner at {warning_path}")

    return df


if __name__ == "__main__":
    generate_synthetic_ai4i_dataset()
