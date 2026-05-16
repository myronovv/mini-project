"""Підготовка даних і моделі перед запуском UI."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from src.config import DATA_DIR, EPOCHS, MODELS_DIR, REPORTS_DIR
from src.data_generator import save_datasets
from src.metrics import evaluate
from src.model import AnomalyDetector
from src.pipeline import DataPipeline, prepare_from_files

TRAIN_CSV = DATA_DIR / "train_normal.csv"
TEST_CSV = DATA_DIR / "test_with_anomalies.csv"
MODEL_META = MODELS_DIR / "meta.json"
SCALER_PATH = MODELS_DIR / "scaler.joblib"


def models_ready() -> bool:
    return MODEL_META.exists() and SCALER_PATH.exists()


def data_ready() -> bool:
    return TRAIN_CSV.exists() and TEST_CSV.exists()


def ensure_data(on_progress: Callable[[str], None] | None = None) -> None:
    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not data_ready():
        log("Генерація синтетичних даних…")
        save_datasets(TRAIN_CSV, TEST_CSV)
    else:
        log("Дані вже існують.")


def ensure_model(
    epochs: int = EPOCHS,
    force: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    ensure_data(on_progress)

    if models_ready() and not force:
        log("Модель вже навчена.")
        return json.loads(MODEL_META.read_text(encoding="utf-8"))

    log(f"Навчання autoencoder ({epochs} епох)…")
    data, pipeline = prepare_from_files(TRAIN_CSV, TEST_CSV)
    detector = AnomalyDetector(data.X_train.shape[1])
    losses = detector.train(data.X_train, epochs=epochs)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    detector.save(MODELS_DIR)
    pipeline.save(SCALER_PATH)

    metrics = {}
    if data.y_test is not None and len(data.y_test):
        start = time.perf_counter()
        pred, _ = detector.predict(data.X_test)
        elapsed_ms = (time.perf_counter() - start) * 1000
        result = evaluate(data.y_test, pred, elapsed_ms, len(data.y_test))
        metrics = result.as_dict()
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "evaluation.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    meta = json.loads(MODEL_META.read_text(encoding="utf-8"))
    meta["last_loss"] = losses[-1] if losses else None
    meta["metrics"] = metrics
    return meta


def ensure_ready(
    epochs: int = EPOCHS,
    force_train: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    return ensure_model(epochs=epochs, force=force_train, on_progress=on_progress)


def load_agent_stack() -> tuple[AnomalyDetector, DataPipeline]:
    if not models_ready():
        ensure_ready()
    return AnomalyDetector.load(MODELS_DIR), DataPipeline.load(SCALER_PATH)
