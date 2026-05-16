#!/usr/bin/env python3
"""Одна команда: дані → навчання → Streamlit (без pip install)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    sys.path.insert(0, str(ROOT))

    from src.bootstrap import ensure_ready

    steps: list[str] = []

    def on_progress(msg: str) -> None:
        print(f"  • {msg}")
        steps.append(msg)

    print("Підготовка проєкту…")
    ensure_ready(on_progress=on_progress)
    print("\nЗапуск інтерфейсу: http://localhost:8501\n")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ROOT / "app.py"),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
