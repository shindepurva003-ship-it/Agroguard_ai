from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REQUIRED = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]

def make_demo_data(seed=42, per_crop=90):
    """Illustrative demo data only; do not use to report research performance."""
    rng = np.random.default_rng(seed)
    centers = {
        "rice": [80, 45, 40, 25, 82, 6.2, 220],
        "wheat": [70, 45, 40, 20, 55, 6.8, 85],
        "maize": [75, 45, 20, 24, 60, 6.5, 90],
        "cotton": [120, 45, 20, 27, 55, 7.0, 65],
        "chickpea": [40, 65, 80, 20, 35, 7.2, 65],
        "mungbean": [30, 55, 25, 28, 65, 6.8, 60],
        "banana": [100, 80, 50, 27, 78, 6.3, 180],
        "pomegranate": [50, 25, 35, 27, 45, 7.2, 55],
    }
    rows = []
    for label, center in centers.items():
        arr = rng.normal(center, [12, 10, 10, 3, 10, .45, 25], size=(per_crop, 7))
        arr[:, 0:3] = np.clip(arr[:, 0:3], 0, 140)
        arr[:, 3] = np.clip(arr[:, 3], 8, 42)
        arr[:, 4] = np.clip(arr[:, 4], 15, 100)
        arr[:, 5] = np.clip(arr[:, 5], 4, 9)
        arr[:, 6] = np.clip(arr[:, 6], 10, 350)
        for row in arr:
            rows.append(list(row) + [label])
    return pd.DataFrame(rows, columns=REQUIRED)

def load_crop_data():
    real_path = DATA / "Crop_recommendation.csv"
    if real_path.exists():
        df = pd.read_csv(real_path)
        df.columns = [c.strip() for c in df.columns]
        # Normalize common spelling/case variants.
        aliases = {"ph": "ph", "label": "label", "N": "N", "P": "P", "K": "K"}
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")
        df = df[REQUIRED].copy()
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        for c in REQUIRED[:-1]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna().drop_duplicates()
        return df, "External labelled dataset: data/Crop_recommendation.csv"
    return make_demo_data(), "ILLUSTRATIVE DEMO DATA — replace before research evaluation"

def sample_validation_template():
    return pd.DataFrame(columns=[
        "observation_id", "date", "state", "district", "crop",
        "soil_N", "soil_P", "soil_K", "soil_pH", "soil_moisture_pct",
        "temperature_C", "relative_humidity_pct", "rainfall_mm",
        "observed_stress", "notes"
    ])
