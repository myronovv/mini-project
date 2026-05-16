"""Pipeline обробки даних: нормалізація, train/test split, завантаження."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLUMNS


@dataclass
class ProcessedData:
    X_train: np.ndarray
    y_train: np.ndarray | None
    X_test: np.ndarray
    y_test: np.ndarray | None
    feature_names: list[str]


class DataPipeline:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.feature_names = list(FEATURE_COLUMNS)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_names].values.astype(np.float32)
        return self.scaler.fit_transform(X).astype(np.float32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_names].values.astype(np.float32)
        return self.scaler.transform(X).astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path)
        meta = {"feature_names": self.feature_names}
        path.with_suffix(".meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "DataPipeline":
        pipe = cls()
        pipe.scaler = joblib.load(path)
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            pipe.feature_names = meta["feature_names"]
        return pipe


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def prepare_from_files(
    train_path: Path,
    test_path: Path | None = None,
) -> tuple[ProcessedData, "DataPipeline"]:
    train_df = load_csv(train_path)
    pipeline = DataPipeline()
    X_train = pipeline.fit_transform(train_df)

    y_train = None
    if "is_anomaly" in train_df.columns:
        y_train = train_df["is_anomaly"].values.astype(int)

    if test_path and test_path.exists():
        test_df = load_csv(test_path)
        X_test = pipeline.transform(test_df)
        y_test = (
            test_df["is_anomaly"].values.astype(int)
            if "is_anomaly" in test_df.columns
            else None
        )
    else:
        X_test = np.empty((0, X_train.shape[1]), dtype=np.float32)
        y_test = None

    return ProcessedData(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=pipeline.feature_names,
    ), pipeline
