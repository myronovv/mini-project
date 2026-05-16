"""Агентна логіка: політика рішень за рівнем аномалії."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from src.model import AnomalyDetector
from src.pipeline import DataPipeline


class AlertLevel(str, Enum):
    NORMAL = "НОРМА"
    WARNING = "ПОПЕРЕДЖЕННЯ"
    CRITICAL = "АВАРІЙНИЙ СТАН"


@dataclass
class AgentDecision:
    timestamp: str
    alert_level: AlertLevel
    reconstruction_error: float
    threshold: float
    is_anomaly: bool
    recommended_action: str
    anomaly_hint: str


class NetworkAnomalyAgent:
    """Агент виявлення аварійних станів у енергомережі."""

    def __init__(self, detector: AnomalyDetector, pipeline: DataPipeline) -> None:
        self.detector = detector
        self.pipeline = pipeline
        self.threshold = detector.threshold or 0.0

    def _severity(self, error: float) -> AlertLevel:
        if error <= self.threshold:
            return AlertLevel.NORMAL
        ratio = error / self.threshold
        if ratio >= 2.5:
            return AlertLevel.CRITICAL
        return AlertLevel.WARNING

    def _action(self, level: AlertLevel, row: pd.Series) -> tuple[str, str]:
        hints = []
        if row.get("voltage_kv", 110) < 95:
            hints.append("просідання напруги")
        if row.get("frequency_hz", 50) > 50.15 or row.get("frequency_hz", 50) < 49.85:
            hints.append("відхилення частоти")
        if row.get("line_load_pct", 0) > 90:
            hints.append("перевантаження лінії")
        if row.get("transformer_temp_c", 0) > 85:
            hints.append("перегрів трансформатора")

        hint = ", ".join(hints) if hints else "невідомий патерн відхилення"

        actions = {
            AlertLevel.NORMAL: "Продовжити штатний моніторинг.",
            AlertLevel.WARNING: (
                "Посилити моніторинг вузла; перевірити навантаження та температуру; "
                "підготувати оператора до можливого втручання."
            ),
            AlertLevel.CRITICAL: (
                "НЕГАЙНО: ізолювати ділянку мережі, знизити навантаження, "
                "сповістити диспетчера; за потреби — перемикання резерву."
            ),
        }
        return actions[level], hint

    def analyze(self, df: pd.DataFrame) -> list[AgentDecision]:
        X = self.pipeline.transform(df)
        flags, errors = self.detector.predict(X)
        decisions: list[AgentDecision] = []

        for i, row in df.iterrows():
            idx = df.index.get_loc(i)
            level = self._severity(float(errors[idx]))
            action, hint = self._action(level, row)
            ts = str(row["timestamp"]) if "timestamp" in row else f"row_{idx}"
            decisions.append(
                AgentDecision(
                    timestamp=ts,
                    alert_level=level,
                    reconstruction_error=float(errors[idx]),
                    threshold=self.threshold,
                    is_anomaly=bool(flags[idx]),
                    recommended_action=action,
                    anomaly_hint=hint,
                )
            )
        return decisions

    def summarize(self, decisions: list[AgentDecision]) -> dict:
        total = len(decisions)
        anomalies = sum(1 for d in decisions if d.is_anomaly)
        by_level = {lvl.value: 0 for lvl in AlertLevel}
        for d in decisions:
            by_level[d.alert_level.value] += 1
        return {
            "total_records": total,
            "detected_anomalies": anomalies,
            "anomaly_rate_pct": round(100 * anomalies / max(total, 1), 2),
            "by_alert_level": by_level,
        }
