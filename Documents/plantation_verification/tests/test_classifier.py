import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from src.classifier import train, load_model

BAND_COLS = [f"A{i:02d}" for i in range(64)]

def _make_parquet(tmp_path: Path, n_per_class: int = 200) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    for class_id in range(7):
        embs = rng.normal(class_id * 2, 0.5, (n_per_class, 64)).astype(np.float32)
        for emb in embs:
            row = {"source": "test", "year": 2021, "class_id": class_id,
                   "class_name": str(class_id), "lon": 0.0, "lat": 0.0}
            row.update(dict(zip(BAND_COLS, emb.tolist())))
            rows.append(row)
    df = pd.DataFrame(rows)
    p = tmp_path / "training.parquet"
    df.to_parquet(p, index=False)
    return p

def test_train_returns_metrics(tmp_path):
    parquet_path = _make_parquet(tmp_path)
    metrics = train(parquet_path, tmp_path / "models")
    assert "macro_f1" in metrics
    assert "trees_f1" in metrics
    assert 0.0 <= metrics["macro_f1"] <= 1.0

def test_train_saves_artifacts(tmp_path):
    parquet_path = _make_parquet(tmp_path)
    model_dir = tmp_path / "models"
    train(parquet_path, model_dir)
    assert (model_dir / "model.ubj").exists()
    assert (model_dir / "class_map.json").exists()
    assert (model_dir / "feature_names.json").exists()

def test_load_model_returns_correct_types(tmp_path):
    parquet_path = _make_parquet(tmp_path)
    model_dir = tmp_path / "models"
    train(parquet_path, model_dir)
    model, class_map, feature_names = load_model(model_dir)
    assert hasattr(model, "predict_proba")
    assert isinstance(class_map, dict)
    assert len(feature_names) == 64

def test_load_model_predicts_correct_shape(tmp_path):
    parquet_path = _make_parquet(tmp_path)
    model_dir = tmp_path / "models"
    train(parquet_path, model_dir)
    model, class_map, _ = load_model(model_dir)
    X = np.random.default_rng(0).normal(0, 1, (10, 64)).astype(np.float32)
    probs = model.predict_proba(X)
    assert probs.shape == (10, 7)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

def test_train_high_f1_on_separable_data(tmp_path):
    # Well-separated clusters → should achieve high F1
    parquet_path = _make_parquet(tmp_path, n_per_class=300)
    metrics = train(parquet_path, tmp_path / "models")
    assert metrics["macro_f1"] > 0.80, f"macro_f1={metrics['macro_f1']:.3f} below 0.80"
