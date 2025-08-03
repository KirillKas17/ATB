"""
Система валидации конфигурации для ATB.
Обеспечивает проверку конфигурационных файлов,
валидацию схем и автоматическое исправление ошибок.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

import yaml
from pydantic import BaseModel, Field, ValidationError, validator


class ConfigSeverity(Enum):
    """Уровни серьёзности проблем конфигурации."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ConfigIssue:
    """Проблема конфигурации."""

    severity: ConfigSeverity
    message: str
    field_path: str
    current_value: Any
    expected_value: Optional[Any] = None
    suggestion: Optional[str] = None
    line_number: Optional[int] = None


class BaseConfigValidator:
    """Базовый класс для валидаторов конфигурации."""

    def __init__(self) -> None:
        self.issues: List[ConfigIssue] = []
        self.auto_fix_enabled = True
        self.validators: Dict[str, Type[BaseModel]] = {}  # Добавляем атрибут validators

    def validate(self, config: Dict[str, Any]) -> bool:
        """
        Валидация конфигурации.
        Args:
            config: Конфигурация для валидации
        Returns:
            True если конфигурация валидна
        """
        try:
            # Очищаем предыдущие проблемы
            self.issues.clear()
            # Базовая валидация структуры
            if not isinstance(config, dict):
                self.add_issue(
                    ConfigSeverity.CRITICAL,
                    "Configuration must be a dictionary",
                    "root",
                    type(config).__name__,
                    "dict",
                )
                return False
            # Валидация обязательных секций
            required_sections = ["application", "trading"]
            for section in required_sections:
                if section not in config:
                    self.add_issue(
                        ConfigSeverity.CRITICAL,
                        f"Required section '{section}' is missing",
                        section,
                        None,
                        "present",
                    )
            # Валидация каждой секции
            for section_name, section_data in config.items():
                if not isinstance(section_data, dict):
                    self.add_issue(
                        ConfigSeverity.ERROR,
                        f"Section '{section_name}' must be a dictionary",
                        section_name,
                        type(section_data).__name__,
                        "dict",
                    )
                    continue
                # Валидация через Pydantic модели
                if section_name in self.validators:
                    try:
                        validator_class = self.validators[section_name]
                        validator_class(**section_data)
                    except ValidationError as e:
                        for error in e.errors():
                            field_path = f"{section_name}.{'.'.join(str(loc) for loc in error['loc'])}"
                            self.add_issue(
                                ConfigSeverity.ERROR,
                                error["msg"],
                                field_path,
                                error.get("input"),
                                error.get("ctx", {}).get("expected"),
                            )
            # Проверка критических проблем
            if self.has_critical_issues():
                return False
            return len(self.issues) == 0
        except Exception as e:
            self.add_issue(
                ConfigSeverity.CRITICAL,
                f"Validation error: {str(e)}",
                "root",
                None,
                None,
            )
            return False

    def get_issues(self) -> List[ConfigIssue]:
        """Получить список проблем."""
        return self.issues

    def has_critical_issues(self) -> bool:
        """Проверить наличие критических проблем."""
        return any(issue.severity == ConfigSeverity.CRITICAL for issue in self.issues)

    def add_issue(
        self,
        severity: ConfigSeverity,
        message: str,
        field_path: str,
        current_value: Any,
        expected_value: Optional[Any] = None,
        suggestion: Optional[str] = None,
        line_number: Optional[int] = None,
    ) -> None:
        """Добавить проблему."""
        issue = ConfigIssue(
            severity=severity,
            message=message,
            field_path=field_path,
            current_value=current_value,
            expected_value=expected_value,
            suggestion=suggestion,
            line_number=line_number,
        )
        self.issues.append(issue)


class ApplicationConfig(BaseModel):
    """Схема конфигурации приложения."""

    # Основные настройки
    app_name: str = Field(..., description="Название приложения")
    version: str = Field(..., description="Версия приложения")
    environment: str = Field(default="development", description="Окружение")
    debug: bool = Field(default=False, description="Режим отладки")
    # Настройки логирования
    logging_level: str = Field(default="INFO", description="Уровень логирования")
    log_file: Optional[str] = Field(None, description="Файл логов")
    log_rotation: str = Field(default="1 day", description="Ротация логов")
    # Настройки производительности
    max_workers: int = Field(default=4, ge=1, le=32, description="Максимум воркеров")
    timeout: float = Field(default=30.0, gt=0, description="Таймаут операций")
    retry_attempts: int = Field(default=3, ge=0, le=10, description="Попытки повтора")
    # Настройки безопасности
    api_key: Optional[str] = Field(None, description="API ключ")
    secret_key: Optional[str] = Field(None, description="Секретный ключ")
    encryption_enabled: bool = Field(default=True, description="Включить шифрование")
    # Настройки мониторинга
    monitoring_enabled: bool = Field(default=True, description="Включить мониторинг")
    metrics_interval: int = Field(default=30, gt=0, description="Интервал метрик")
    alert_threshold: float = Field(
        default=80.0, gt=0, le=100, description="Порог алертов"
    )

    @validator("environment")
    def validate_environment(cls, v):
        allowed = ["development", "staging", "production", "testing"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v

    @validator("logging_level")
    def validate_logging_level(cls, v):
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"Logging level must be one of {allowed}")
        return v.upper()


class TradingConfig(BaseModel):
    """Схема конфигурации торговли."""

    # Основные настройки торговли
    trading_enabled: bool = Field(default=True, description="Включить торговлю")
    max_position_size: float = Field(
        default=1000.0, gt=0, description="Максимальный размер позиции"
    )
    min_position_size: float = Field(
        default=10.0, gt=0, description="Минимальный размер позиции"
    )
    leverage: float = Field(default=1.0, gt=0, le=100, description="Плечо")
    # Настройки рисков
    max_drawdown: float = Field(
        default=20.0, gt=0, le=100, description="Максимальная просадка"
    )
    stop_loss_percent: float = Field(
        default=5.0, gt=0, le=50, description="Стоп-лосс в процентах"
    )
    take_profit_percent: float = Field(
        default=10.0, gt=0, le=100, description="Тейк-профит в процентах"
    )
    # Настройки стратегий
    strategy_timeout: int = Field(default=300, gt=0, description="Таймаут стратегии")
    max_concurrent_strategies: int = Field(
        default=5, ge=1, le=20, description="Максимум стратегий"
    )
    strategy_rotation: bool = Field(default=True, description="Ротация стратегий")
    # Настройки ордеров
    order_timeout: int = Field(default=60, gt=0, description="Таймаут ордера")
    max_retries: int = Field(default=3, ge=0, le=10, description="Максимум повторов")
    retry_delay: float = Field(default=1.0, gt=0, description="Задержка повтора")

    @validator("max_position_size")
    def validate_max_position_size(cls, v, values):
        if "min_position_size" in values and v < values["min_position_size"]:
            raise ValueError("Max position size must be greater than min position size")
        return v

    @validator("take_profit_percent")
    def validate_take_profit(cls, v, values):
        if "stop_loss_percent" in values and v <= values["stop_loss_percent"]:
            raise ValueError("Take profit must be greater than stop loss")
        return v


class DatabaseConfig(BaseModel):
    """Схема конфигурации базы данных."""

    # Основные настройки
    database_type: str = Field(..., description="Тип базы данных")
    host: str = Field(..., description="Хост базы данных")
    port: int = Field(default=5432, gt=0, le=65535, description="Порт базы данных")
    database_name: str = Field(..., description="Название базы данных")
    username: str = Field(..., description="Имя пользователя")
    password: Optional[str] = Field(None, description="Пароль")
    # Настройки подключения
    pool_size: int = Field(default=10, ge=1, le=100, description="Размер пула")
    max_overflow: int = Field(
        default=20, ge=0, le=100, description="Максимум переполнения"
    )
    timeout: int = Field(default=30, gt=0, description="Таймаут подключения")
    # Настройки SSL
    ssl_enabled: bool = Field(default=False, description="Включить SSL")
    ssl_cert: Optional[str] = Field(None, description="SSL сертификат")
    ssl_key: Optional[str] = Field(None, description="SSL ключ")

    @validator("database_type")
    def validate_database_type(cls, v):
        allowed = ["postgresql", "mysql", "sqlite", "mongodb"]
        if v.lower() not in allowed:
            raise ValueError(f"Database type must be one of {allowed}")
        return v.lower()


class ExchangeConfig(BaseModel):
    """Схема конфигурации биржи."""

    # Основные настройки
    exchange_name: str = Field(..., description="Название биржи")
    api_key: str = Field(..., description="API ключ")
    secret_key: str = Field(..., description="Секретный ключ")
    passphrase: Optional[str] = Field(None, description="Парольная фраза")
    # Настройки подключения
    sandbox: bool = Field(default=False, description="Песочница")
    timeout: int = Field(default=30, gt=0, description="Таймаут запросов")
    rate_limit: int = Field(default=100, gt=0, description="Лимит запросов")
    # Настройки торговли
    trading_pairs: List[str] = Field(default_factory=list, description="Торговые пары")
    min_order_size: Dict[str, float] = Field(
        default_factory=dict, description="Минимальные размеры ордеров"
    )
    fees: Dict[str, float] = Field(default_factory=dict, description="Комиссии")

    @validator("exchange_name")
    def validate_exchange_name(cls, v):
        allowed = ["binance", "bybit", "okx", "kucoin", "bitget"]
        if v.lower() not in allowed:
            raise ValueError(f"Exchange name must be one of {allowed}")
        return v.lower()


class ConfigValidator:
    """
    Валидатор конфигурации.
    Проверяет конфигурационные файлы на соответствие схемам,
    выявляет проблемы и предлагает исправления.
    """

    def __init__(self) -> None:
        self.validators: Dict[str, Type[BaseModel]] = {
            "application": ApplicationConfig,
            "trading": TradingConfig,
            "database": DatabaseConfig,
            "exchange": ExchangeConfig,
        }
        self.issues: List[ConfigIssue] = []
        self.auto_fix_enabled = True

    def validate_config_file(self, file_path: str) -> bool:
        """
        Валидация конфигурационного файла.
        Args:
            file_path: Путь к файлу конфигурации
        Returns:
            True если файл валиден
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.endswith(".yaml") or file_path.endswith(".yml"):
                    config = yaml.safe_load(f)
                elif file_path.endswith(".json"):
                    config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {file_path}")
            return self.validate_config(config, file_path)
        except Exception as e:
            self.add_issue(
                ConfigSeverity.CRITICAL,
                f"Failed to load config file: {e}",
                "file",
                file_path,
            )
            return False

    def validate_config(self, config: Dict[str, Any], source: str = "unknown") -> bool:
        """
        Валидация конфигурации.
        Args:
            config: Конфигурация для валидации
            source: Источник конфигурации
        Returns:
            True если конфигурация валидна
        """
        self.issues.clear()
        is_valid = True
        for section_name, section_config in config.items():
            if section_name in self.validators:
                try:
                    validator_class = self.validators[section_name]
                    validator_class(**section_config)
                except ValidationError as e:
                    is_valid = False
                    for error in e.errors():
                        line_number = error.get("line")
                        if line_number is not None:
                            try:
                                line_number = int(line_number) if isinstance(line_number, (int, str)) else None
                            except (ValueError, TypeError):
                                line_number = None
                        self.add_issue(
                            ConfigSeverity.ERROR,
                            error["msg"],
                            f"{section_name}.{'.'.join(str(x) for x in error['loc'])}",
                            error.get("input"),
                            error.get("ctx"),
                            line_number=line_number,
                        )
            else:
                self.add_issue(
                    ConfigSeverity.WARNING,
                    f"Unknown configuration section: {section_name}",
                    section_name,
                    section_config,
                )
        # Дополнительные проверки
        self._validate_cross_section_dependencies(config)
        self._validate_environment_specific_rules(config)
        return is_valid and not self.has_critical_issues()

    def _validate_cross_section_dependencies(self, config: Dict[str, Any]) -> None:
        """Проверка зависимостей между секциями."""
        # Проверка соответствия окружения и настроек
        if "application" in config and "trading" in config:
            app_config = config["application"]
            trading_config = config["trading"]
            if app_config.get("environment") == "production":
                if not trading_config.get("trading_enabled"):
                    self.add_issue(
                        ConfigSeverity.WARNING,
                        "Trading is disabled in production environment",
                        "trading.trading_enabled",
                        trading_config.get("trading_enabled"),
                        True,
                    )
                if trading_config.get("leverage", 1.0) > 10.0:
                    self.add_issue(
                        ConfigSeverity.ERROR,
                        "High leverage detected in production environment",
                        "trading.leverage",
                        trading_config.get("leverage"),
                        1.0,
                    )

    def _validate_environment_specific_rules(self, config: Dict[str, Any]) -> None:
        """Проверка правил для конкретного окружения."""
        if "application" in config:
            env = config["application"].get("environment", "development")
            if env == "production":
                # Проверки для продакшена
                if config["application"].get("debug", False):
                    self.add_issue(
                        ConfigSeverity.CRITICAL,
                        "Debug mode enabled in production",
                        "application.debug",
                        True,
                        False,
                    )
                if not config["application"].get("encryption_enabled", True):
                    self.add_issue(
                        ConfigSeverity.ERROR,
                        "Encryption disabled in production",
                        "application.encryption_enabled",
                        False,
                        True,
                    )
            elif env == "development":
                # Проверки для разработки
                if not config["application"].get("debug", False):
                    self.add_issue(
                        ConfigSeverity.INFO,
                        "Debug mode recommended for development",
                        "application.debug",
                        False,
                        True,
                    )

    def auto_fix_issues(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Автоматическое исправление проблем конфигурации.
        Args:
            config: Исходная конфигурация
        Returns:
            Исправленная конфигурация
        """
        if not self.auto_fix_enabled:
            return config
        fixed_config = config.copy()
        for issue in self.issues:
            if issue.severity in [ConfigSeverity.INFO, ConfigSeverity.WARNING]:
                if issue.expected_value is not None:
                    self._set_nested_value(
                        fixed_config, issue.field_path, issue.expected_value
                    )
        return fixed_config

    def _set_nested_value(
        self, config: Dict[str, Any], field_path: str, value: Any
    ) -> None:
        """Установить значение вложенного поля."""
        keys = field_path.split(".")
        current: Dict[str, Any] = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
            if not isinstance(current, dict):
                current = {}
        current[keys[-1]] = value

    def get_issues_summary(self) -> Dict[str, Any]:
        """Получить сводку проблем."""
        summary: Dict[str, Any] = {
            "total": len(self.issues),
            "by_severity": {},
            "by_section": {},
            "critical_count": 0,
            "error_count": 0,
            "warning_count": 0,
            "info_count": 0,
        }
        for issue in self.issues:
            severity = issue.severity.value
            section = issue.field_path.split(".")[0]
            summary["by_severity"][severity] = (
                summary["by_severity"].get(severity, 0) + 1
            )
            summary["by_section"][section] = summary["by_section"].get(section, 0) + 1
            if issue.severity == ConfigSeverity.CRITICAL:
                summary["critical_count"] += 1
            elif issue.severity == ConfigSeverity.ERROR:
                summary["error_count"] += 1
            elif issue.severity == ConfigSeverity.WARNING:
                summary["warning_count"] += 1
            elif issue.severity == ConfigSeverity.INFO:
                summary["info_count"] += 1
        return summary

    def add_issue(
        self,
        severity: ConfigSeverity,
        message: str,
        field_path: str,
        current_value: Any,
        expected_value: Optional[Any] = None,
        suggestion: Optional[str] = None,
        line_number: Optional[int] = None,
    ) -> None:
        """Добавить проблему."""
        issue = ConfigIssue(
            severity=severity,
            message=message,
            field_path=field_path,
            current_value=current_value,
            expected_value=expected_value,
            suggestion=suggestion,
            line_number=line_number,
        )
        self.issues.append(issue)

    def has_critical_issues(self) -> bool:
        """Проверить наличие критических проблем."""
        return any(issue.severity == ConfigSeverity.CRITICAL for issue in self.issues)

    def export_issues(self, format: str = "json") -> str:
        """
        Экспорт проблем конфигурации.
        Args:
            format: Формат экспорта
        Returns:
            Строка с экспортированными проблемами
        """
        if format == "json":
            data = {
                "issues": [issue.__dict__ for issue in self.issues],
                "summary": self.get_issues_summary(),
            }
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Глобальный экземпляр валидатора
config_validator = ConfigValidator()


def validate_config_file(file_path: str) -> bool:
    """
    Валидация конфигурационного файла.
    Args:
        file_path: Путь к файлу конфигурации
    Returns:
        True если файл валиден
    """
    return config_validator.validate_config_file(file_path)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Валидация конфигурации.
    Args:
        config: Конфигурация для валидации
    Returns:
        True если конфигурация валидна
    """
    return config_validator.validate_config(config)
