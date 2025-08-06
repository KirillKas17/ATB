import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
import yaml

st.set_page_config(page_title="Управление", page_icon="🎮", layout="wide")


class BotControl:
    def __init__(self) -> None:
        self.config_dir = Path("config")
        self.data_dir = Path("data")

    def load_config(self) -> Dict:
        """Загрузка конфигурации"""
        try:
            config_file = self.config_dir / "config.yaml"
            if config_file.exists():
                with open(config_file, "r") as f:
                    config_data = yaml.safe_load(f)
                    return config_data if isinstance(config_data, dict) else {}
            return {}
        except Exception as e:
            st.error(f"Ошибка загрузки конфигурации: {e}")
            return {}

    def save_config(self, config: Dict[Any, Any]) -> None:
        """Сохранение конфигурации"""
        try:
            config_file = self.config_dir / "config.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config, f)
            st.success("Конфигурация сохранена")
        except Exception as e:
            st.error(f"Ошибка сохранения конфигурации: {e}")

    def load_tasks(self) -> List[Dict]:
        """Загрузка задач"""
        try:
            tasks_file = self.data_dir / "tasks.json"
            if tasks_file.exists():
                with open(tasks_file, "r") as f:
                    tasks_data = json.load(f)
                    return tasks_data if isinstance(tasks_data, list) else []
            return []
        except Exception as e:
            st.error(f"Ошибка загрузки задач: {e}")
            return []


def main() -> None:
    control = BotControl()

    st.title("🎮 Управление ботом")

    # Загрузка данных
    config = control.load_config()
    tasks = control.load_tasks()

    # Основные настройки
    st.markdown("### Основные настройки")

    col1, col2 = st.columns(2)

    with col1:
        # Настройки торговли
        st.markdown("#### Настройки торговли")
        config["trading"] = {
            "max_positions": st.number_input(
                "Макс. количество позиций",
                min_value=1,
                max_value=100,
                value=config.get("trading", {}).get("max_positions", 5),
            ),
            "position_size": st.number_input(
                "Размер позиции (%)",
                min_value=0.1,
                max_value=100.0,
                value=config.get("trading", {}).get("position_size", 2.0),
            ),
            "stop_loss": st.number_input(
                "Stop Loss (%)",
                min_value=0.1,
                max_value=100.0,
                value=config.get("trading", {}).get("stop_loss", 2.0),
            ),
            "take_profit": st.number_input(
                "Take Profit (%)",
                min_value=0.1,
                max_value=100.0,
                value=config.get("trading", {}).get("take_profit", 4.0),
            ),
        }

        # Настройки рисков
        st.markdown("#### Управление рисками")
        config["risk"] = {
            "max_daily_loss": st.number_input(
                "Макс. дневной убыток (%)",
                min_value=0.1,
                max_value=100.0,
                value=config.get("risk", {}).get("max_daily_loss", 5.0),
            ),
            "max_drawdown": st.number_input(
                "Макс. просадка (%)",
                min_value=0.1,
                max_value=100.0,
                value=config.get("risk", {}).get("max_drawdown", 20.0),
            ),
            "risk_reward": st.number_input(
                "Risk/Reward",
                min_value=0.1,
                max_value=10.0,
                value=config.get("risk", {}).get("risk_reward", 2.0),
            ),
        }

    with col2:
        # Настройки стратегий
        st.markdown("#### Стратегии")
        strategies = config.get("strategies", {})

        for strategy in ["trend", "mean_reversion", "breakout"]:
            st.markdown(f"##### {strategy.title()}")
            strategies[strategy] = {
                "enabled": st.checkbox(
                    f"Включить {strategy}",
                    value=strategies.get(strategy, {}).get("enabled", False),
                ),
                "weight": st.slider(
                    f"Вес {strategy}",
                    min_value=0.0,
                    max_value=1.0,
                    value=strategies.get(strategy, {}).get("weight", 0.33),
                ),
            }

        config["strategies"] = strategies

        # Настройки уведомлений
        st.markdown("#### Уведомления")
        config["notifications"] = {
            "telegram": st.checkbox(
                "Telegram", value=config.get("notifications", {}).get("telegram", False)
            ),
            "email": st.checkbox(
                "Email", value=config.get("notifications", {}).get("email", False)
            ),
            "desktop": st.checkbox(
                "Desktop", value=config.get("notifications", {}).get("desktop", True)
            ),
        }

    # Сохранение конфигурации
    if st.button("💾 Сохранить настройки"):
        control.save_config(config)

    # Управление задачами
    st.markdown("### Управление задачами")

    if tasks:
        tasks_df = pd.DataFrame(tasks)
        st.dataframe(tasks_df, use_container_width=True)

        # Действия с задачами
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("▶️ Запустить все"):
                st.info("Запуск всех задач...")

        with col2:
            if st.button("⏹ Остановить все"):
                st.info("Остановка всех задач...")

        with col3:
            if st.button("🔄 Сбросить все"):
                st.info("Сброс всех задач...")
    else:
        st.info("Нет активных задач")

    # Логи
    st.markdown("### Логи")

    log_file = Path("logs/bot.log")
    if log_file.exists():
        with open(log_file, "r") as f:
            logs = f.readlines()[-100:]  # Последние 100 строк

        st.text_area("Последние логи", value="".join(logs), height=200)
    else:
        st.info("Логи не найдены")


if __name__ == "__main__":
    main()
