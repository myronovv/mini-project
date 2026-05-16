"""Streamlit UI — агент виявлення аномалій у енергомережі."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.agent import AlertLevel, NetworkAnomalyAgent
from src.bootstrap import TEST_CSV, ensure_model, load_agent_stack
from src.config import FEATURE_COLUMNS, REPORTS_DIR
from src.pipeline import load_csv

st.set_page_config(
    page_title="Агент аномалій енергомережі",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

LEVEL_COLORS = {
    AlertLevel.NORMAL.value: "#22c55e",
    AlertLevel.WARNING.value: "#f59e0b",
    AlertLevel.CRITICAL.value: "#ef4444",
}


@st.cache_resource
def get_agent() -> NetworkAnomalyAgent:
    detector, pipeline = load_agent_stack()
    return NetworkAnomalyAgent(detector, pipeline)


def load_metrics() -> dict | None:
    path = REPORTS_DIR / "evaluation.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def render_metrics_cards(metrics: dict) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
    c2.metric("Precision", f"{metrics.get('precision', 0):.1%}")
    c3.metric("Recall", f"{metrics.get('recall', 0):.1%}")
    c4.metric("F1", f"{metrics.get('f1', 0):.2f}")
    c5.metric("Latency", f"{metrics.get('latency_ms_per_sample', 0):.3f} мс")


def plot_signals(df: pd.DataFrame, anomalies: pd.Series | None = None) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    fig.patch.set_facecolor("#0e1117")

    axes[0].plot(df["timestamp"], df["voltage_kv"], color="#60a5fa", linewidth=0.9)
    axes[0].set_ylabel("кВ")
    axes[0].set_title("Напруга", color="white")
    axes[1].plot(df["timestamp"], df["frequency_hz"], color="#34d399", linewidth=0.9)
    axes[1].set_ylabel("Гц")
    axes[1].set_title("Частота", color="white")
    axes[2].plot(df["timestamp"], df["line_load_pct"], color="#fbbf24", linewidth=0.9)
    axes[2].set_ylabel("%")
    axes[2].set_title("Завантаження лінії", color="white")

    if anomalies is not None:
        mask = anomalies.astype(bool).values
        if mask.any():
            for ax, col in zip(axes, ["voltage_kv", "frequency_hz", "line_load_pct"]):
                ax.scatter(
                    df["timestamp"].values[mask],
                    df[col].values[mask],
                    color="#ef4444",
                    s=14,
                    zorder=5,
                    label="аномалія",
                )

    for ax in axes:
        ax.set_facecolor("#1a1f2e")
        ax.tick_params(colors="#94a3b8")
        ax.title.set_color("#e2e8f0")
        ax.yaxis.label.set_color("#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def page_monitor(agent: NetworkAnomalyAgent) -> None:
    st.subheader("Моніторинг мережі")

    source = st.radio(
        "Джерело даних",
        ["Тестовий набір (з аномаліями)", "Завантажити CSV"],
        horizontal=True,
    )

    if source.startswith("Тестовий"):
        df = load_csv(TEST_CSV)
    else:
        uploaded = st.file_uploader("CSV з сенсорами", type=["csv"])
        if uploaded is None:
            st.info("Завантажте файл або оберіть тестовий набір.")
            return
        df = pd.read_csv(uploaded)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

    max_rows = st.slider("Кількість записів для аналізу", 100, len(df), min(500, len(df)))
    df = df.head(max_rows).copy()

    only_anomalies = st.checkbox("Показувати лише аномалії", value=False)

    with st.spinner("Аналіз агентом…"):
        decisions = agent.analyze(df)
        summary = agent.summarize(decisions)

    err_series = pd.Series([d.reconstruction_error for d in decisions])
    flag_series = pd.Series([d.is_anomaly for d in decisions])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Записів", summary["total_records"])
    m2.metric("Аномалій", summary["detected_anomalies"])
    m3.metric("Частка аномалій", f"{summary['anomaly_rate_pct']}%")
    m4.metric("Критичних", summary["by_alert_level"].get(AlertLevel.CRITICAL.value, 0))

    plot_signals(df, flag_series)

    st.markdown("#### Помилка реконструкції (autoencoder)")
    chart_df = pd.DataFrame(
        {"timestamp": df["timestamp"], "error": err_series, "поріг": agent.threshold}
    ).set_index("timestamp")
    st.line_chart(chart_df, color=["#f87171", "#64748b"])

    rows = []
    for d in decisions:
        if only_anomalies and not d.is_anomaly:
            continue
        rows.append(
            {
                "Час": d.timestamp,
                "Рівень": d.alert_level.value,
                "Помилка": round(d.reconstruction_error, 6),
                "Аномалія": "Так" if d.is_anomaly else "Ні",
                "Ознаки": d.anomaly_hint,
                "Дія": d.recommended_action,
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
    )

    critical = [d for d in decisions if d.alert_level == AlertLevel.CRITICAL]
    if critical:
        st.error(f"Виявлено {len(critical)} критичних станів")
        for d in critical[:5]:
            st.markdown(f"**{d.timestamp}** — {d.recommended_action}")


def page_metrics() -> None:
    st.subheader("Метрики якості")
    metrics = load_metrics()
    if not metrics:
        st.warning("Файл метрик відсутній. Перезапустіть навчання з бічної панелі.")
        return
    render_metrics_cards(metrics)
    st.json(metrics)


def page_about() -> None:
    st.subheader("Про агента")
    st.markdown(
        """
**Варіант 10** — виявлення аномалій у енергомережі.

| Етап | Опис |
|------|------|
| Дані | Синтетичні сенсори: напруга, струм, частота, потужність, температура, навантаження |
| Модель | Autoencoder (навчання лише на нормальному режимі) |
| Агент | Класифікація: НОРМА / ПОПЕРЕДЖЕННЯ / АВАРІЙНИЙ СТАН + рекомендації |
| Інтерфейс | Streamlit (локально, без зовнішніх API) |
        """
    )


def main() -> None:
    st.title("⚡ Агент виявлення аномалій у енергомережі")
    st.caption("Autoencoder · варіант 10 · локальний запуск")

    with st.sidebar:
        st.header("Керування")
        epochs = st.slider("Епохи навчання", 20, 120, 80, step=10)
        if st.button("Перегенерувати дані та навчити модель", width="stretch"):
            progress = st.empty()

            def log(msg: str) -> None:
                progress.info(msg)

            with st.spinner("Підготовка…"):
                from src.data_generator import save_datasets
                from src.bootstrap import TRAIN_CSV, TEST_CSV

                save_datasets(TRAIN_CSV, TEST_CSV)
                ensure_model(epochs=epochs, force=True, on_progress=log)
            get_agent.clear()
            st.success("Готово!")
            st.rerun()

        st.divider()
        meta_path = Path(__file__).parent / "models" / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            st.metric("Поріг аномалії", f"{meta.get('threshold', 0):.6f}")

        metrics = load_metrics()
        if metrics:
            st.metric("F1-score", f"{metrics.get('f1', 0):.2f}")

    agent = get_agent()
    tab1, tab2, tab3 = st.tabs(["Моніторинг", "Метрики", "Довідка"])
    with tab1:
        page_monitor(agent)
    with tab2:
        page_metrics()
    with tab3:
        page_about()


if __name__ == "__main__":
    main()
