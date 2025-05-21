import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Статистика", page_icon="📊", layout="wide")


class Statistics:
    def __init__(self):
        self.stats_dir = Path("stats")

    def load_trades_history(self) -> pd.DataFrame:
        """Загрузка истории сделок"""
        try:
            trades_file = self.stats_dir / "trades_history.json"
            if trades_file.exists():
                with open(trades_file, "r") as f:
                    data = json.load(f)
                return pd.DataFrame(data)
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Ошибка загрузки истории сделок: {e}")
            return pd.DataFrame()

    def load_performance_metrics(self) -> dict:
        """Загрузка метрик производительности"""
        try:
            metrics_file = self.stats_dir / "performance_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            st.error(f"Ошибка загрузки метрик: {e}")
            return {}


def main():
    stats = Statistics()

    st.title("📊 Детальная статистика")

    # Загрузка данных
    trades_df = stats.load_trades_history()
    metrics = stats.load_performance_metrics()

    if trades_df.empty:
        st.warning("Нет данных для отображения")
        return

    # Фильтры
    col1, col2, col3 = st.columns(3)
    with col1:
        pairs = st.multiselect(
            "Торговые пары", options=trades_df["pair"].unique(), default=trades_df["pair"].unique()
        )
    with col2:
        date_range = st.date_input(
            "Период", value=(trades_df["timestamp"].min(), trades_df["timestamp"].max())
        )
    with col3:
        strategy = st.multiselect(
            "Стратегии",
            options=trades_df["strategy"].unique(),
            default=trades_df["strategy"].unique(),
        )

    # Применение фильтров
    filtered_df = trades_df[
        (trades_df["pair"].isin(pairs))
        & (trades_df["timestamp"].dt.date.between(*date_range))
        & (trades_df["strategy"].isin(strategy))
    ]

    # Основные метрики
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_trades = len(filtered_df)
        win_trades = len(filtered_df[filtered_df["pnl"] > 0])
        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
        st.metric("Win Rate", f"{win_rate:.2f}%")

    with col2:
        total_pnl = filtered_df["pnl"].sum()
        avg_pnl = filtered_df["pnl"].mean()
        st.metric("Средний PNL", f"{avg_pnl:.2f}%")

    with col3:
        max_drawdown = filtered_df["pnl"].min()
        st.metric("Макс. просадка", f"{max_drawdown:.2f}%")

    with col4:
        sharpe = metrics.get("sharpe_ratio", 0)
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")

    # Графики
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Кривая доходности")
        cumulative_pnl = filtered_df["pnl"].cumsum()
        fig = px.line(x=filtered_df["timestamp"], y=cumulative_pnl, title="Кумулятивный PNL")
        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Распределение PNL")
        fig = px.histogram(filtered_df, x="pnl", nbins=50, title="Распределение прибыли/убытков")
        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Статистика по парам
    st.markdown("### Статистика по парам")
    pair_stats = (
        filtered_df.groupby("pair")
        .agg({"pnl": ["count", "mean", "sum"], "timestamp": "count"})
        .round(2)
    )

    pair_stats.columns = ["Торгов", "Средний PNL", "Общий PNL"]
    st.dataframe(
        pair_stats.style.background_gradient(subset=["Средний PNL", "Общий PNL"], cmap="RdYlGn"),
        use_container_width=True,
    )

    # Статистика по стратегиям
    st.markdown("### Статистика по стратегиям")
    strategy_stats = (
        filtered_df.groupby("strategy")
        .agg({"pnl": ["count", "mean", "sum"], "timestamp": "count"})
        .round(2)
    )

    strategy_stats.columns = ["Торгов", "Средний PNL", "Общий PNL"]
    st.dataframe(
        strategy_stats.style.background_gradient(
            subset=["Средний PNL", "Общий PNL"], cmap="RdYlGn"
        ),
        use_container_width=True,
    )

    # История сделок
    st.markdown("### История сделок")
    st.dataframe(filtered_df.sort_values("timestamp", ascending=False), use_container_width=True)


if __name__ == "__main__":
    main()
