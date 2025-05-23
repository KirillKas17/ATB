import json
import os
from typing import Any, Dict

from loguru import logger


class ConfigManager:
    """Менеджер конфигурации."""

    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.load_config()

    def load_config(self) -> None:
        """Загрузка конфигурации из файла."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    self.config = json.load(f)
            else:
                self.config = self._get_default_config()
                self.save_config()

        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.config = self._get_default_config()

    def save_config(self) -> None:
        """Сохранение конфигурации в файл."""
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=4)

        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def get_config(self) -> Dict[str, Any]:
        """Получение текущей конфигурации."""
        return self.config

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Обновление конфигурации."""
        try:
            self.config.update(new_config)
            self.save_config()

        except Exception as e:
            logger.error(f"Error updating config: {e}")

    def get_value(self, key: str, default: Any = None) -> Any:
        """Получение значения по ключу."""
        return self.config.get(key, default)

    def set_value(self, key: str, value: Any) -> None:
        """Установка значения по ключу."""
        try:
            self.config[key] = value
            self.save_config()

        except Exception as e:
            logger.error(f"Error setting config value: {e}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Получение конфигурации по умолчанию."""
        return {
            "trading": {
                "mode": "paper",  # paper/live
                "max_positions": 5,
                "max_leverage": 5,
                "min_risk_reward": 2.0,
                "max_daily_loss": 0.02,  # 2%
                "max_position_size": 0.1,  # 10% от баланса
                "default_timeframe": "1h",
                "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
            },
            "risk_management": {
                "max_drawdown": 0.1,  # 10%
                "stop_loss_atr_multiplier": 2.0,
                "take_profit_atr_multiplier": 3.0,
                "trailing_stop_activation": 0.02,  # 2%
                "trailing_stop_distance": 0.01,  # 1%
                "max_correlation": 0.7,
            },
            "indicators": {
                "sma_periods": [20, 50, 200],
                "ema_periods": [20, 50, 200],
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "bollinger_period": 20,
                "bollinger_std": 2,
                "atr_period": 14,
            },
            "signals": {
                "min_confidence": 0.7,
                "signal_weights": {
                    "market_regime": 0.3,
                    "risk": 0.2,
                    "whales": 0.2,
                    "news": 0.15,
                    "market_maker": 0.15,
                },
                "max_signal_history": 100,
            },
            "logging": {
                "level": "INFO",
                "file": "trading_bot.log",
                "max_size": 10485760,  # 10MB
                "backup_count": 5,
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "trading_bot",
                "user": "postgres",
                "password": "",
            },
            "api": {
                "exchange": "binance",
                "api_key": "",
                "api_secret": "",
                "testnet": True,
            },
        }

    def validate_config(self) -> bool:
        """Проверка валидности конфигурации."""
        try:
            required_sections = [
                "trading",
                "risk_management",
                "indicators",
                "signals",
                "logging",
                "database",
                "api",
            ]

            # Проверка наличия всех секций
            if not all(section in self.config for section in required_sections):
                return False

            # Проверка торговых параметров
            trading = self.config["trading"]
            if not all(
                key in trading
                for key in [
                    "mode",
                    "max_positions",
                    "max_leverage",
                    "min_risk_reward",
                    "max_daily_loss",
                ]
            ):
                return False

            # Проверка управления рисками
            risk = self.config["risk_management"]
            if not all(
                key in risk
                for key in [
                    "max_drawdown",
                    "stop_loss_atr_multiplier",
                    "take_profit_atr_multiplier",
                ]
            ):
                return False

            # Проверка индикаторов
            indicators = self.config["indicators"]
            if not all(
                key in indicators
                for key in [
                    "sma_periods",
                    "ema_periods",
                    "rsi_period",
                    "macd_fast",
                    "macd_slow",
                    "macd_signal",
                ]
            ):
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating config: {e}")
            return False

    def reset_to_default(self) -> None:
        """Сброс конфигурации к значениям по умолчанию."""
        try:
            self.config = self._get_default_config()
            self.save_config()

        except Exception as e:
            logger.error(f"Error resetting config: {e}")

    def backup_config(self) -> None:
        """Создание резервной копии конфигурации."""
        try:
            backup_path = f"{self.config_path}.backup"
            with open(backup_path, "w") as f:
                json.dump(self.config, f, indent=4)

        except Exception as e:
            logger.error(f"Error backing up config: {e}")

    def restore_from_backup(self) -> bool:
        """Восстановление конфигурации из резервной копии."""
        try:
            backup_path = f"{self.config_path}.backup"
            if not os.path.exists(backup_path):
                return False

            with open(backup_path, "r") as f:
                self.config = json.load(f)
            self.save_config()
            return True

        except Exception as e:
            logger.error(f"Error restoring from backup: {e}")
            return False
