"""
Улучшенная система конфигурации для Syntra.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .exceptions import ConfigurationError
from .logging import get_logger

logger = get_logger(__name__)


class SecretManager:
    """
    Архитектурный слой для работы с секретами: шифрование, ротация, безопасное хранение.
    Для промышленного использования интегрировать с HashiCorp Vault, AWS Secrets Manager и т.д.
    """

    @staticmethod
    def get_secret(name: str, default: Optional[str] = None) -> str:
        value = os.getenv(name, default)
        if not value:
            raise ConfigurationError(
                f"Secret {name} is required and must be set in environment variables."
            )
        # Здесь можно добавить дешифрование, если секреты зашифрованы
        return value


@dataclass
class DatabaseConfig:
    """Конфигурация базы данных."""

    host: str = "localhost"
    port: int = 5432
    database: str = "syntra"
    username: str = "syntra_user"
    password: str = field(
        default_factory=lambda: SecretManager.get_secret("DB_PASS", "")
    )
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False


@dataclass
class ExchangeConfig:
    """Конфигурация биржи."""

    name: str = "bybit"
    api_key: str = field(
        default_factory=lambda: SecretManager.get_secret("EXCHANGE_API_KEY", "")
    )
    api_secret: str = field(
        default_factory=lambda: SecretManager.get_secret("EXCHANGE_API_SECRET", "")
    )
    testnet: bool = True
    rate_limit: int = 100
    timeout: int = 30
    retry_attempts: int = 3


@dataclass
class RiskConfig:
    """Конфигурация риск-менеджмента."""

    max_position_size: float = 0.1
    max_daily_loss: float = 0.05
    max_portfolio_risk: float = 0.02
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    max_leverage: float = 1.0
    max_drawdown: float = 0.15
    position_sizing_method: str = "kelly"


@dataclass
class TradingConfig:
    """Конфигурация торговли."""

    enabled: bool = True
    max_orders: int = 10
    order_timeout: int = 60
    min_order_size: float = 0.001
    max_order_size: float = 1.0
    allowed_pairs: List[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    trading_hours: Dict[str, str] = field(
        default_factory=lambda: {"start": "00:00", "end": "23:59"}
    )


@dataclass
class MLConfig:
    """Конфигурация машинного обучения."""

    enabled: bool = True
    model_path: str = "models/"
    training_data_path: str = "data/training/"
    prediction_interval: int = 60
    confidence_threshold: float = 0.7
    retrain_interval: int = 86400  # 24 часа
    max_models: int = 10


@dataclass
class LoggingConfig:
    """Конфигурация логирования."""

    level: str = "INFO"
    file_path: str = "logs/"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    loggers: Dict[str, str] = field(
        default_factory=lambda: {
            "trading": "INFO",
            "risk": "INFO",
            "ml": "INFO",
            "exchange": "INFO",
        }
    )


@dataclass
class DashboardConfig:
    """Конфигурация дашборда."""

    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    reload: bool = True
    workers: int = 1


@dataclass
class CacheConfig:
    """Конфигурация кэширования."""

    enabled: bool = True
    default_ttl: int = 300
    max_size: int = 1000
    cleanup_interval: int = 3600


class SyntraConfig:
    """Основная конфигурация Syntra."""

    # Основные настройки
    environment: str = "development"
    debug: bool = False
    version: str = "1.0.0"
    # Компоненты конфигурации
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    # Дополнительные настройки
    custom_settings: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Валидация конфигурации после инициализации."""
        self._validate()

    def _validate(self):
        """Валидация конфигурации."""
        errors = []
        # Проверка базы данных
        if not self.database.host:
            errors.append("Database host is required")
        # Проверка биржи
        if not self.exchange.api_key:
            errors.append("Exchange API key is required")
        if not self.exchange.api_secret:
            errors.append("Exchange API secret is required")
        # Проверка риск-менеджмента
        if self.risk.max_position_size <= 0 or self.risk.max_position_size > 1:
            errors.append("max_position_size must be between 0 and 1")
        if self.risk.max_daily_loss <= 0 or self.risk.max_daily_loss > 1:
            errors.append("max_daily_loss must be between 0 and 1")
        # Проверка торговли
        if self.trading.min_order_size <= 0:
            errors.append("min_order_size must be positive")
        if self.trading.max_order_size <= self.trading.min_order_size:
            errors.append("max_order_size must be greater than min_order_size")
        if errors:
            raise ConfigurationError(
                f"Configuration validation failed: {'; '.join(errors)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование конфигурации в словарь."""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "version": self.version,
            "database": self.database.__dict__,
            "exchange": self.exchange.__dict__,
            "risk": self.risk.__dict__,
            "trading": self.trading.__dict__,
            "ml": self.ml.__dict__,
            "logging": self.logging.__dict__,
            "dashboard": self.dashboard.__dict__,
            "cache": self.cache.__dict__,
            "custom_settings": self.custom_settings,
        }

    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Обновление конфигурации из словаря с безопасным преобразованием типов."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                attr = getattr(self, key)
                if isinstance(value, dict) and hasattr(attr, "__dict__"):
                    # Обновляем вложенные объекты
                    for sub_key, sub_value in value.items():
                        if hasattr(attr, sub_key):
                            current_type = type(getattr(attr, sub_key))
                            if sub_value is not None and current_type in [
                                int,
                                float,
                                bool,
                                str,
                            ]:
                                try:
                                    if current_type is bool:
                                        casted = (
                                            bool(sub_value)
                                            if isinstance(sub_value, bool)
                                            else str(sub_value).lower() == "true"
                                        )
                                    else:
                                        casted = current_type(sub_value)
                                    setattr(attr, sub_key, casted)
                                except Exception as e:
                                    logger.error(
                                        f"Type conversion error for {key}.{sub_key}: {e}"
                                    )
                            else:
                                setattr(attr, sub_key, sub_value)
                else:
                    # FIXED: TECHNICAL_AUDIT_REPORT.md
                    current_type = type(attr)
                    if value is not None and current_type in [int, float, bool, str]:
                        try:
                            if current_type is bool:
                                casted = (
                                    bool(value)
                                    if isinstance(value, bool)
                                    else str(value).lower() == "true"
                                )
                            else:
                                casted = current_type(value)
                            setattr(self, key, casted)
                        except Exception as e:
                            logger.error(f"Type conversion error for {key}: {e}")
                    else:
                        setattr(self, key, value)
        # Перевалидируем
        self._validate()


class ConfigManager:
    """Менеджер конфигурации."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/config.yaml"
        self.config: Optional[SyntraConfig] = None
        self._load_environment_variables()

    def _load_environment_variables(self):
        """Загрузка переменных окружения."""
        # База данных
        os.environ.setdefault("DB_HOST", "localhost")
        os.environ.setdefault("DB_PORT", "5432")
        os.environ.setdefault("DB_NAME", "syntra")
        os.environ.setdefault("DB_USER", "syntra_user")
        os.environ.setdefault("DB_PASS", "")
        # Биржа
        os.environ.setdefault("EXCHANGE_API_KEY", "")
        os.environ.setdefault("EXCHANGE_API_SECRET", "")
        os.environ.setdefault("EXCHANGE_TESTNET", "true")
        # Окружение
        os.environ.setdefault("ENVIRONMENT", "development")
        os.environ.setdefault("DEBUG", "false")

    def load_config(self) -> SyntraConfig:
        """Загрузка конфигурации."""
        if self.config is not None:
            return self.config
        try:
            # Создаем базовую конфигурацию
            config = SyntraConfig()
            # Загружаем из файла если существует
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, "r", encoding="utf-8") as f:
                    file_config = yaml.safe_load(f)
                    config.update_from_dict(file_config)
            # Применяем переменные окружения
            self._apply_environment_variables(config)
            # Валидируем
            config._validate()
            self.config = config
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}")

    def _apply_environment_variables(self, config: SyntraConfig):
        """Применение переменных окружения к конфигурации."""
        # Основные настройки
        config.environment = os.getenv("ENVIRONMENT", config.environment)
        config.debug = os.getenv("DEBUG", "false").lower() == "true"
        # База данных
        config.database.host = os.getenv("DB_HOST", config.database.host)
        config.database.port = int(os.getenv("DB_PORT", str(config.database.port)))
        config.database.database = os.getenv("DB_NAME", config.database.database)
        config.database.username = os.getenv("DB_USER", config.database.username)
        config.database.password = os.getenv("DB_PASS", config.database.password)
        # Биржа
        config.exchange.api_key = os.getenv("EXCHANGE_API_KEY", config.exchange.api_key)
        config.exchange.api_secret = os.getenv(
            "EXCHANGE_API_SECRET", config.exchange.api_secret
        )
        config.exchange.testnet = (
            os.getenv("EXCHANGE_TESTNET", "true").lower() == "true"
        )
        # Риск-менеджмент
        if os.getenv("MAX_POSITION_SIZE"):
            max_position_size = os.getenv("MAX_POSITION_SIZE")
            if max_position_size is not None:
                config.risk.max_position_size = float(max_position_size)
        if os.getenv("MAX_DAILY_LOSS"):
            max_daily_loss = os.getenv("MAX_DAILY_LOSS")
            if max_daily_loss is not None:
                config.risk.max_daily_loss = float(max_daily_loss)
        if os.getenv("STOP_LOSS_PCT"):
            stop_loss_pct = os.getenv("STOP_LOSS_PCT")
            if stop_loss_pct is not None:
                config.risk.stop_loss_pct = float(stop_loss_pct)

    def save_config(self, config: SyntraConfig, path: Optional[str] = None):
        """Сохранение конфигурации в файл."""
        try:
            save_path = path or self.config_path
            config_file = Path(save_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.dump(
                    config.to_dict(), f, default_flow_style=False, allow_unicode=True
                )
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise ConfigurationError(f"Configuration saving failed: {e}")

    def reload_config(self) -> SyntraConfig:
        """Перезагрузка конфигурации."""
        self.config = None
        return self.load_config()

    def get_config(self) -> SyntraConfig:
        """Получение конфигурации."""
        if self.config is None:
            return self.load_config()
        return self.config


# Глобальный менеджер конфигурации
config_manager = ConfigManager()


@lru_cache(maxsize=1)
def get_config() -> SyntraConfig:
    """Получение конфигурации (с кэшированием)."""
    return config_manager.get_config()


def update_config(updates: Dict[str, Any]):
    """Обновление конфигурации."""
    config = get_config()
    config.update_from_dict(updates)
    config_manager.save_config(config)


def reload_config() -> SyntraConfig:
    """Перезагрузка конфигурации."""
    get_config.cache_clear()  # Очищаем кэш
    return config_manager.reload_config()
