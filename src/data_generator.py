"""Синтетичні дані сенсорів енергомережі з мітками аномалій."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, RANDOM_SEED


def _normal_segment(n: int, rng: np.random.Generator) -> pd.DataFrame:
    voltage = rng.normal(110.0, 0.8, n)
    current = rng.normal(420.0, 12.0, n)
    frequency = rng.normal(50.0, 0.02, n)
    active_power = voltage * current * 0.001 * rng.uniform(0.92, 0.98, n)
    reactive = active_power * rng.uniform(0.08, 0.15, n)
    temp = rng.normal(62.0, 3.0, n)
    load = rng.normal(68.0, 5.0, n).clip(20, 95)
    return pd.DataFrame(
        {
            "voltage_kv": voltage,
            "current_a": current,
            "frequency_hz": frequency,
            "active_power_mw": active_power,
            "reactive_power_mvar": reactive,
            "transformer_temp_c": temp,
            "line_load_pct": load,
        }
    )


def _inject_anomalies(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    anomaly_mask = np.zeros(n, dtype=bool)
    kinds: list[str] = [""] * n

    events = [
        ("voltage_sag", lambda i: _voltage_sag(out, i)),
        ("frequency_drift", lambda i: _frequency_drift(out, i)),
        ("overload", lambda i: _overload(out, i)),
        ("thermal_fault", lambda i: _thermal_fault(out, i)),
    ]

    for _ in range(max(8, n // 500)):
        start = rng.integers(0, max(1, n - 30))
        length = rng.integers(8, 25)
        kind, fn = events[rng.integers(0, len(events))]
        for i in range(start, min(start + length, n)):
            fn(i)
            anomaly_mask[i] = True
            kinds[i] = kind

    out["is_anomaly"] = anomaly_mask.astype(int)
    out["anomaly_type"] = kinds
    return out


def _voltage_sag(df: pd.DataFrame, i: int) -> None:
    df.at[i, "voltage_kv"] *= 0.78
    df.at[i, "current_a"] *= 1.25
    df.at[i, "line_load_pct"] = min(99, df.at[i, "line_load_pct"] * 1.15)


def _frequency_drift(df: pd.DataFrame, i: int) -> None:
    df.at[i, "frequency_hz"] += 0.35
    df.at[i, "reactive_power_mvar"] *= 1.4


def _overload(df: pd.DataFrame, i: int) -> None:
    df.at[i, "line_load_pct"] = 98.0
    df.at[i, "current_a"] *= 1.35
    df.at[i, "active_power_mw"] *= 1.3
    df.at[i, "transformer_temp_c"] += 18


def _thermal_fault(df: pd.DataFrame, i: int) -> None:
    df.at[i, "transformer_temp_c"] += 28
    df.at[i, "reactive_power_mvar"] *= 1.2


def generate_dataset(
    n_normal: int = 8000,
    n_test: int = 2000,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    normal = _normal_segment(n_normal, rng)
    normal["is_anomaly"] = 0
    normal["anomaly_type"] = ""

    test = _normal_segment(n_test, rng)
    test = _inject_anomalies(test, rng)

    timestamps = pd.date_range("2025-01-01", periods=len(normal), freq="min")
    normal.insert(0, "timestamp", timestamps)

    test_ts = pd.date_range(
        normal["timestamp"].iloc[-1] + pd.Timedelta(minutes=1),
        periods=len(test),
        freq="min",
    )
    test.insert(0, "timestamp", test_ts)

    cols = ["timestamp"] + FEATURE_COLUMNS + ["is_anomaly", "anomaly_type"]
    return normal[cols], test[cols]


def save_datasets(train_path, test_path, **kwargs) -> None:
    train, test = generate_dataset(**kwargs)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
