from __future__ import annotations
from pathlib import Path
import pandas as pd

FOLDER_TO_LABEL: dict[str, str] = {
    "is_blur": "blur",
    "is_clear": "accepted",
    "is_proper_doc": "accepted",
    "is_proper_concent_form": "accepted",
    "is_land_record": "accepted",
    "is_full_concent_form": "accepted",
    "is_not_document": "not_document",
    "is_not_full_document(cut)": "cut",
    "is_not_full_doc(cut)": "cut",
    "is_not_proper_doc(cut)": "cut",
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".pdf"}


def load_ground_truth(test_data_dir: Path) -> pd.DataFrame:
    rows = []
    for doc_type_dir in sorted(test_data_dir.iterdir()):
        if not doc_type_dir.is_dir():
            continue
        for state_dir in sorted(doc_type_dir.iterdir()):
            if not state_dir.is_dir():
                continue
            for category_dir in sorted(state_dir.rglob("*")):
                if not category_dir.is_dir():
                    continue
                label = FOLDER_TO_LABEL.get(category_dir.name)
                if label is None:
                    continue
                for f in sorted(category_dir.rglob("*")):
                    if f.is_file() and f.suffix.lower() in EXTENSIONS:
                        rows.append({
                            "file_path": f,
                            "doc_type": doc_type_dir.name,
                            "state": state_dir.name,
                            "expected_label": label,
                        })
    return pd.DataFrame(rows).drop_duplicates(subset=["file_path"]).reset_index(drop=True)


def make_subset(
    gt_df: pd.DataFrame,
    n_per_class: int = 18,
    seed: int = 42,
) -> pd.DataFrame:
    parts = []
    for label, grp in gt_df.groupby("expected_label"):
        n = min(n_per_class, len(grp))
        parts.append(grp.sample(n=n, random_state=seed))
    return pd.concat(parts).reset_index(drop=True)
