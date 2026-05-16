#!/usr/bin/env python3
"""CLI агента виявлення аномалій у енергомережі (варіант 10, без API)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tabulate import tabulate

from src.agent import AlertLevel, NetworkAnomalyAgent
from src.config import DATA_DIR, MODELS_DIR, REPORTS_DIR
from src.data_generator import save_datasets
from src.metrics import evaluate
from src.model import AnomalyDetector
from src.pipeline import DataPipeline, load_csv, prepare_from_files


def cmd_generate(_: argparse.Namespace) -> None:
    train = DATA_DIR / "train_normal.csv"
    test = DATA_DIR / "test_with_anomalies.csv"
    save_datasets(train, test)
    print(f"Згенеровано:\n  {train}\n  {test}")


def cmd_train(args: argparse.Namespace) -> None:
    train_path = Path(args.train)
    test_path = Path(args.test) if args.test else None

    data, pipeline = prepare_from_files(train_path, test_path)
    detector = AnomalyDetector(data.X_train.shape[1])
    losses = detector.train(data.X_train, epochs=args.epochs)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    detector.save(MODELS_DIR)
    pipeline.save(MODELS_DIR / "scaler.joblib")

    print(f"Навчання завершено. Поріг аномалії: {detector.threshold:.6f}")
    print(f"Остання loss: {losses[-1]:.6f}")

    if data.y_test is not None and len(data.y_test):
        start = time.perf_counter()
        pred, _ = detector.predict(data.X_test)
        elapsed_ms = (time.perf_counter() - start) * 1000
        result = evaluate(data.y_test, pred, elapsed_ms, len(data.y_test))
        report_path = REPORTS_DIR / "train_metrics.json"
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(result.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("Метрики на тестовому наборі:")
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))


def cmd_detect(args: argparse.Namespace) -> None:
    df = load_csv(Path(args.file))
    detector = AnomalyDetector.load(MODELS_DIR)
    pipeline = DataPipeline.load(MODELS_DIR / "scaler.joblib")
    agent = NetworkAnomalyAgent(detector, pipeline)
    decisions = agent.analyze(df)

    if args.summary_only:
        print(json.dumps(agent.summarize(decisions), ensure_ascii=False, indent=2))
        return

    rows = []
    for d in decisions:
        if args.anomalies_only and not d.is_anomaly:
            continue
        rows.append(
            [
                d.timestamp,
                d.alert_level.value,
                f"{d.reconstruction_error:.5f}",
                "ТАК" if d.is_anomaly else "ні",
                d.anomaly_hint[:40],
            ]
        )

    print(
        tabulate(
            rows,
            headers=["Час", "Рівень", "Помилка", "Аномалія", "Ознаки"],
            tablefmt="simple",
        )
    )

    critical = [d for d in decisions if d.alert_level == AlertLevel.CRITICAL]
    if critical:
        print("\n--- Критичні рішення агента ---")
        for d in critical[: args.limit]:
            print(f"\n[{d.timestamp}] {d.recommended_action}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    test_path = Path(args.test)
    data, _ = prepare_from_files(
        DATA_DIR / "train_normal.csv",
        test_path,
    )
    detector = AnomalyDetector.load(MODELS_DIR)
    start = time.perf_counter()
    pred, errors = detector.predict(data.X_test)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if data.y_test is None:
        print("У файлі немає міток is_anomaly.", file=sys.stderr)
        sys.exit(1)

    result = evaluate(data.y_test, pred, elapsed_ms, len(data.y_test))
    out = REPORTS_DIR / "evaluation.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = result.as_dict()
    payload["mean_reconstruction_error"] = float(errors.mean())
    payload["max_reconstruction_error"] = float(errors.max())
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nЗбережено: {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Агент виявлення аномалій у енергомережі (autoencoder, варіант 10)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="Згенерувати синтетичні CSV-дані")
    p_gen.set_defaults(func=cmd_generate)

    p_train = sub.add_parser("train", help="Навчити autoencoder на нормальних даних")
    p_train.add_argument(
        "--train",
        default=str(DATA_DIR / "train_normal.csv"),
        help="CSV з нормальним режимом",
    )
    p_train.add_argument(
        "--test",
        default=str(DATA_DIR / "test_with_anomalies.csv"),
        help="CSV з аномаліями для валідації",
    )
    p_train.add_argument("--epochs", type=int, default=80)
    p_train.set_defaults(func=cmd_train)

    p_det = sub.add_parser("detect", help="Аналіз файлу та рішення агента")
    p_det.add_argument("--file", required=True, help="Шлях до CSV")
    p_det.add_argument("--anomalies-only", action="store_true")
    p_det.add_argument("--summary-only", action="store_true")
    p_det.add_argument("--limit", type=int, default=5)
    p_det.set_defaults(func=cmd_detect)

    p_eval = sub.add_parser("evaluate", help="Розрахувати F1, precision, recall")
    p_eval.add_argument(
        "--test",
        default=str(DATA_DIR / "test_with_anomalies.csv"),
    )
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
