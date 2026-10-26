# Dataset Download Instructions: AI4I 2020 Predictive Maintenance

To execute final benchmarking and produce submission-ready models and reports, download the official **AI4I 2020 Predictive Maintenance Dataset** and place it in the `data/raw/` directory.

---

## 1. Dataset Overview
- **Name:** AI4I 2020 Predictive Maintenance Dataset
- **Records:** 10,000 instances
- **Features:** 14 attributes including machine type, ambient air temp, process temp, rotational velocity, torque, tool wear, and failure mode flags.

---

## 2. Official Sources

### Option A: UCI Machine Learning Repository (Recommended)
1. Navigate to the official UCI repository page:  
   [UCI Machine Learning Repository - AI4I 2020](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
2. Click **Download** or fetch the dataset archive.
3. Extract `ai4i2020.csv`.

### Option B: Kaggle
1. Navigate to the Kaggle dataset page:  
   [Kaggle - AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)
2. Download `ai4i2020.csv`.

---

## 3. Placement in Project Directory
Place the downloaded CSV into:
```
predictive_maintenance_xai_automl/
└── data/
    └── raw/
        └── ai4i2020.csv   <-- Place file here
```

Accepted file names in `data/raw/`:
- `ai4i2020.csv`
- `ai4i_2020.csv`
- `AI4I_2020.csv`
- `ai4i_data.csv`

---

## 4. Re-running the Pipeline with Real Data
Once the real dataset is placed into `data/raw/`:
1. Activate your virtual environment:
   ```powershell
   .venv\Scripts\activate
   ```
2. Re-run Phase 1:
   ```powershell
   python scripts/run_phase1.py
   ```
3. Verify that `data/processed/metadata.json` reflects `"dataset_source": "real"`, and remove `USE_REAL_DATA.md` if present.
