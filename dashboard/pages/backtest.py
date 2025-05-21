import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Бектестинг", page_icon="📊", layout="wide")


class Backtest:
    def __init__(self):
        self.data_dir = Path("data")
        self.results_dir = Path("results")

    def load_results(self) -> Dict:
        """Загрузка результатов бектестинга"""
        try:
            results_file = self.results_dir / "backtest_results.json"
            if results_file.exists():
                with open(results_file, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            st.error(f"Ошибка загрузки результатов: {e}")
            return {}

    def load_trades(self) -> pd.DataFrame:
        """Загрузка сделок бектестинга"""
        try:
            trades_file = self.results_dir / "backtest_trades.json"
            if trades_file.exists():
                with open(trades_file, "r") as f:
                    data = json.load(f)
                return pd.DataFrame(data)
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Ошибка загрузки сделок: {e}")
            return pd.DataFrame()


def main():
    backtest = Backtest()

    st.title("📊 Бектестинг")

    # Загрузка данных
    results = backtest.load_results()
    trades_df = backtest.load_trades()

    # Настройки бектестинга
    st.markdown("### Настройки")

    col1, col2, col3 = st.columns(3)

    with col1:
        pair = st.selectbox("Торговая пара", options=["BTCUSDT", "ETHUSDT", "BNBUSDT"])

        timeframe = st.selectbox("Таймфрейм", options=["1m", "5m", "15m", "1h", "4h", "1d"])

    with col2:
        start_date = st.date_input("Начало периода", value=datetime.now() - timedelta(days=30))

        end_date = st.date_input("Конец периода", value=datetime.now())

    with col3:
        strategy = st.selectbox("Стратегия", options=["trend", "mean_reversion", "breakout"])

        initial_balance = st.number_input(
            "Начальный баланс", min_value=100.0, max_value=1000000.0, value=10000.0
        )

    # Запуск бектестинга
    if st.button("▶️ Запустить бектестинг"):
        st.info("Запуск бектестинга...")

        # Здесь будет код запуска бектестинга

        st.success("Бектестинг завершен")

    # Результаты
    if results:
        st.markdown("### Результаты")

        # Основные метрики
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Общий PNL",
                f"{results.get('total_pnl', 0):.2f}%",
                delta=f"{results.get('pnl_change', 0):.2f}%",
            )

        with col2:
            st.metric("Win Rate", f"{results.get('win_rate', 0):.2f}%")

        with col3:
            st.metric("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}")

        with col4:
            st.metric("Макс. просадка", f"{results.get('max_drawdown', 0):.2f}%")

        # Графики
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Кривая доходности")
            if "equity_curve" in results:
                fig = px.line(
                    x=results["equity_curve"]["dates"],
                    y=results["equity_curve"]["values"],
                    title="Кривая доходности",
                )
                fig.update_layout(
                    showlegend=False,
                    margin=dict(l=0, r=0, t=30, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Распределение PNL")
            if "pnl_distribution" in results:
                fig = px.histogram(
                    x=results["pnl_distribution"], nbins=50, title="Распределение прибыли/убытков"
                )
                fig.update_layout(
                    showlegend=False,
                    margin=dict(l=0, r=0, t=30, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

        # Детальная статистика
        st.markdown("### Детальная статистика")

        if not trades_df.empty:
            # Статистика по месяцам
            trades_df["month"] = pd.to_datetime(trades_df["timestamp"]).dt.strftime("%Y-%m")
            monthly_stats = (
                trades_df.groupby("month")
                .agg({"pnl": ["count", "mean", "sum"], "timestamp": "count"})
                .round(2)
            )

            monthly_stats.columns = ["Торгов", "Средний PNL", "Общий PNL"]
            st.dataframe(
                monthly_stats.style.background_gradient(
                    subset=["Средний PNL", "Общий PNL"], cmap="RdYlGn"
                ),
                use_container_width=True,
            )

            # История сделок
            st.markdown("### История сделок")
            st.dataframe(
                trades_df.sort_values("timestamp", ascending=False), use_container_width=True
            )
        else:
            st.info("Нет данных о сделках")


if __name__ == "__main__":
    main()
