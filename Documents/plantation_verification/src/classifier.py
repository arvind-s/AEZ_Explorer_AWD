from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.harmonize import CLASS_NAMES

BAND_COLS = [f"A{i:02d}" for i in range(64)]
TREES_CLASS_ID = 1


def train(
    parquet_path: Path,
    model_dir: Path,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, float]:
    """
    Train XGBoost on the training Parquet and save artifacts.
    Returns {"macro_f1": float, "trees_f1": float} on the validation split.
    """
    df = pd.read_parquet(parquet_path)
    X = df[BAND_COLS].values.astype(np.float32)
    y = df["class_id"].values.astype(int)

    # Class weights: inverse frequency
    classes, counts = np.unique(y, return_counts=True)
    weight_map = {c: len(y) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    sample_weights = np.array([weight_map[c] for c in y])

    # Geography-aware split: if "lon" column present, assign each pixel to
    # a 10°×10° grid cell and use cell ID as stratification group so adjacent
    # pixels don't leak across splits. Falls back to class-stratified random
    # split when coordinates are absent or too few unique groups.
    use_geo_split = False
    if "lon" in df.columns:
        lon_bin = (df["lon"].values / 10).astype(int)
        lat_bin = (df["lat"].values / 10).astype(int)
        geo_group = lon_bin * 1000 + lat_bin
        n_groups = len(np.unique(geo_group))
        # Only use geo-aware split if we have enough groups
        if n_groups > 1:
            use_geo_split = True
            from sklearn.model_selection import GroupShuffleSplit
            gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
            train_idx, val_idx = next(gss.split(X, y, groups=geo_group))
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            w_train = sample_weights[train_idx]

    if not use_geo_split:
        X_train, X_val, y_train, y_val, w_train, _ = train_test_split(
            X, y, sample_weights,
            test_size=val_fraction,
            stratify=y,
            random_state=seed,
        )

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        early_stopping_rounds=20,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred = model.predict(X_val)
    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    trees_f1 = f1_score(
        y_val, y_pred,
        labels=[TREES_CLASS_ID],
        average="macro",
        zero_division=0,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir / "model.ubj"))

    with open(model_dir / "class_map.json", "w") as f:
        json.dump({str(k): v for k, v in CLASS_NAMES.items()}, f, indent=2)

    with open(model_dir / "feature_names.json", "w") as f:
        json.dump(BAND_COLS, f)

    print(f"macro-F1={macro_f1:.3f}  trees-F1={trees_f1:.3f}")
    return {"macro_f1": float(macro_f1), "trees_f1": float(trees_f1)}


def load_model(
    model_dir: Path,
) -> tuple[XGBClassifier, dict[int, str], list[str]]:
    """Load model + metadata from model_dir. Returns (model, class_map, feature_names)."""
    model = XGBClassifier()
    model.load_model(str(model_dir / "model.ubj"))

    with open(model_dir / "class_map.json") as f:
        class_map = {int(k): v for k, v in json.load(f).items()}

    with open(model_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    return model, class_map, feature_names
