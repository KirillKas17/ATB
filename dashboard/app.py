import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Настройка страницы
st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Стили
st.markdown(
    """
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        border: none;
        padding: 10px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive {
        color: #4CAF50;
    }
    .negative {
        color: #F44336;
    }
</style>
""",
    unsafe_allow_html=True,
)


class Dashboard:
    def __init__(self):
        self.data_dir = Path("data")
        self.stats_dir = Path("stats")
        self.config_dir = Path("config")

    def load_stats(self) -> Dict:
        """Загрузка статистики"""
        try:
            stats_file = self.stats_dir / "trading_stats.json"
            if stats_file.exists():
                with open(stats_file, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            st.error(f"Ошибка загрузки статистики: {e}")
            return {}

    def load_positions(self) -> List[Dict]:
        """Загрузка открытых позиций"""
        try:
            positions_file = self.data_dir / "positions.json"
            if positions_file.exists():
                with open(positions_file, "r") as f:
                    return json.load(f)
            return []
        except Exception as e:
            st.error(f"Ошибка загрузки позиций: {e}")
            return []

    def load_pairs_stats(self) -> Dict:
        """Загрузка статистики по парам"""
        try:
            pairs_file = self.stats_dir / "pairs_stats.json"
            if pairs_file.exists():
                with open(pairs_file, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            st.error(f"Ошибка загрузки статистики пар: {e}")
            return {}


def main():
    dashboard = Dashboard()

    # Сайдбар
    with st.sidebar:
        st.title("🤖 Trading Bot")

        # Выбор пары
        pairs_stats = dashboard.load_pairs_stats()
        selected_pair = st.selectbox(
            "Торговая пара", options=list(pairs_stats.keys()) if pairs_stats else ["BTCUSDT"]
        )

        # Кнопки управления
        st.markdown("### Управление")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Запустить"):
                st.info("Запуск бота...")
        with col2:
            if st.button("⏹ Остановить"):
                st.info("Остановка бота...")

        st.markdown("### Действия")
        if st.button("🔄 Тренировка"):
            st.info("Запуск тренировки...")
        if st.button("📊 Бектестинг"):
            st.info("Запуск бектестинга...")

        # Статистика выбранной пары
        if selected_pair in pairs_stats:
            st.markdown("### Статистика пары")
            pair_stats = pairs_stats[selected_pair]
            st.metric("PNL", f"{pair_stats.get('pnl', 0):.2f}%")
            st.metric("Win Rate", f"{pair_stats.get('win_rate', 0):.2f}%")
            st.metric("Торгов", pair_stats.get("trades", 0))

    # Основной контент
    stats = dashboard.load_stats()
    positions = dashboard.load_positions()

    # Верхняя панель с основными метриками
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "PNL (24ч)",
            f"{stats.get('pnl_24h', 0):.2f}%",
            delta=f"{stats.get('pnl_24h_change', 0):.2f}%",
        )
    with col2:
        st.metric(
            "Win Rate",
            f"{stats.get('win_rate', 0):.2f}%",
            delta=f"{stats.get('win_rate_change', 0):.2f}%",
        )
    with col3:
        st.metric("Торгов", stats.get("total_trades", 0), delta=stats.get("trades_24h", 0))
    with col4:
        st.metric("Активных позиций", len(positions), delta=stats.get("positions_change", 0))

    # Графики
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### PNL по периодам")
        periods = {
            "День": stats.get("pnl_24h", 0),
            "Неделя": stats.get("pnl_7d", 0),
            "Месяц": stats.get("pnl_30d", 0),
            "3 месяца": stats.get("pnl_90d", 0),
        }
        fig = px.bar(
            x=list(periods.keys()),
            y=list(periods.values()),
            color=list(periods.values()),
            color_continuous_scale=["red", "green"],
        )
        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Распределение прибыли")
        if "pnl_distribution" in stats:
            fig = px.histogram(
                x=stats["pnl_distribution"], nbins=50, color_discrete_sequence=["#1E88E5"]
            )
            fig.update_layout(
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Открытые позиции
    st.markdown("### Открытые позиции")
    if positions:
        df = pd.DataFrame(positions)
        st.dataframe(
            df.style.background_gradient(subset=["pnl"], cmap="RdYlGn"), use_container_width=True
        )
    else:
        st.info("Нет открытых позиций")

    # Текущая задача
    st.markdown("### Текущая задача")
    if "current_task" in stats:
        task = stats["current_task"]
        st.markdown(
            f"""
        <div class="metric-card">
            <h4>{task['name']}</h4>
            <p>{task['description']}</p>
            <div class="progress">
                <div class="progress-bar" role="progressbar" 
                     style="width: {task['progress']}%">
                    {task['progress']}%
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Нет активных задач")


if __name__ == "__main__":
    main()
