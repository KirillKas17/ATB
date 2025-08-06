"""
Конфигурация для торгового репозитория.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class CacheConfig:
    """Конфигурация кэширования."""

    enabled: bool = True
    max_size: int = 1000
    default_ttl_minutes: int = 5
    cleanup_interval_seconds: int = 60
    orders_cache_size: int = 1000
    positions_cache_size: int = 500
    trading_pairs_cache_size: int = 100
    accounts_cache_size: int = 50
    metrics_cache_size: int = 10
    patterns_cache_size: int = 200
    liquidity_cache_size: int = 100


@dataclass
class ValidationConfig:
    """Конфигурация валидации."""

    enabled: bool = True
    strict_mode: bool = False
    validate_business_rules: bool = True
    min_quantity: str = "0.00000001"
    max_quantity: str = "999999999.99999999"
    min_price: str = "0.00000001"
    max_price: str = "999999999.99999999"
    max_leverage: int = 125
    min_order_size: str = "0.001"
    max_order_size: str = "1000000"
    allowed_time_in_force: List[str] = field(
        default_factory=lambda: ["GTC", "IOC", "FOK", "GTX"]
    )


@dataclass
class EventConfig:
    """Конфигурация событий."""

    enabled: bool = True
    max_history_size: int = 10000
    publish_async: bool = True
    log_events: bool = True
    event_types_enabled: List[str] = field(
        default_factory=lambda: [
            "order_created",
            "order_updated",
            "order_deleted",
            "order_filled",
            "position_created",
            "position_updated",
            "position_deleted",
            "metrics_updated",
            "pattern_detected",
            "liquidity_analyzed",
        ]
    )


@dataclass
class DatabaseConfig:
    """Конфигурация базы данных."""

    type: str = "in_memory"  # "in_memory", "postgres", "mysql"
    connection_string: Optional[str] = None
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    auto_migrate: bool = True


@dataclass
class LoggingConfig:
    """Конфигурация логирования."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size_mb: int = 100
    backup_count: int = 5
    log_queries: bool = False
    log_events: bool = True
    log_performance: bool = True


@dataclass
class PerformanceConfig:
    """Конфигурация производительности."""

    enable_metrics: bool = True
    enable_profiling: bool = False
    slow_query_threshold_ms: int = 1000
    batch_size: int = 100
    max_concurrent_operations: int = 50
    connection_pool_size: int = 20
    query_timeout_seconds: int = 30


@dataclass
class SecurityConfig:
    """Конфигурация безопасности."""

    enable_audit_log: bool = True
    encrypt_sensitive_data: bool = False
    mask_personal_info: bool = True
    rate_limit_enabled: bool = True
    max_requests_per_minute: int = 1000
    allowed_ips: List[str] = field(default_factory=list)
    require_authentication: bool = True


@dataclass
class TradingRepositoryConfig:
    """Основная конфигурация торгового репозитория."""

    # Основные настройки
    name: str = "TradingRepository"
    version: str = "1.0.0"
    environment: str = "development"
    # Подконфигурации
    cache: CacheConfig = field(default_factory=CacheConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    events: EventConfig = field(default_factory=EventConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    # Дополнительные настройки
    default_currency: str = "USDT"
    supported_currencies: List[str] = field(
        default_factory=lambda: ["USDT", "USD", "BTC", "ETH"]
    )
    trading_hours: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "monday": ["00:00", "23:59"],
            "tuesday": ["00:00", "23:59"],
            "wednesday": ["00:00", "23:59"],
            "thursday": ["00:00", "23:59"],
            "friday": ["00:00", "23:59"],
            "saturday": ["00:00", "23:59"],
            "sunday": ["00:00", "23:59"],
        }
    )
    # Настройки по умолчанию для торговых пар
    default_trading_pair_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "base_asset_precision": 8,
            "quote_precision": 8,
            "min_quantity": "0.00000001",
            "max_quantity": "999999999.99999999",
            "min_price": "0.00000001",
            "max_price": "999999999.99999999",
            "min_notional": "10.0",
            "iceberg_allowed": True,
            "oco_allowed": True,
        }
    )


class TradingRepositoryConfigManager:
    """Менеджер конфигурации торгового репозитория."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.config = TradingRepositoryConfig()
        if config_path:
            self.load_from_file(config_path)

    def load_from_file(self, config_path: str) -> None:
        """Загрузка конфигурации из файла."""
        try:
            path = Path(config_path)
            if not path.exists():
                self.logger.warning(f"Config file not found: {config_path}")
                return
            with open(path, "r", encoding="utf-8") as file:
                config_data = yaml.safe_load(file)
            self._update_config_from_dict(config_data)
            self.logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            self.logger.error(f"Error loading configuration from {config_path}: {e}")

    def save_to_file(self, config_path: str) -> None:
        """Сохранение конфигурации в файл."""
        try:
            config_data = self._config_to_dict()
            with open(config_path, "w", encoding="utf-8") as file:
                yaml.dump(config_data, file, default_flow_style=False, indent=2)
            self.logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            self.logger.error(f"Error saving configuration to {config_path}: {e}")

    def load_from_dict(self, config_data: Dict[str, Any]) -> None:
        """Загрузка конфигурации из словаря."""
        try:
            self._update_config_from_dict(config_data)
            self.logger.info("Configuration loaded from dictionary")
        except Exception as e:
            self.logger.error(f"Error loading configuration from dictionary: {e}")

    def get_config(self) -> TradingRepositoryConfig:
        """Получение текущей конфигурации."""
        return self.config

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Обновление конфигурации."""
        try:
            self._update_config_from_dict(updates)
            self.logger.info("Configuration updated")
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")

    def validate_config(self) -> List[str]:
        """Валидация конфигурации."""
        errors = []
        # Проверка основных настроек
        if not self.config.name:
            errors.append("Repository name cannot be empty")
        if self.config.version not in ["1.0.0", "1.0.1", "1.1.0"]:
            errors.append("Unsupported version")
        if self.config.environment not in ["development", "staging", "production"]:
            errors.append("Invalid environment")
        # Проверка настроек кэша
        if self.config.cache.max_size <= 0:
            errors.append("Cache max_size must be positive")
        if self.config.cache.default_ttl_minutes <= 0:
            errors.append("Cache default_ttl_minutes must be positive")
        # Проверка настроек базы данных
        if self.config.database.type not in ["in_memory", "postgres", "mysql"]:
            errors.append("Unsupported database type")
        if (
            self.config.database.type != "in_memory"
            and not self.config.database.connection_string
        ):
            errors.append(
                "Database connection string is required for non-in-memory databases"
            )
        # Проверка настроек производительности
        if self.config.performance.batch_size <= 0:
            errors.append("Batch size must be positive")
        if self.config.performance.max_concurrent_operations <= 0:
            errors.append("Max concurrent operations must be positive")
        return errors

    def get_default_config(self) -> TradingRepositoryConfig:
        """Получение конфигурации по умолчанию."""
        return TradingRepositoryConfig()

    def _update_config_from_dict(self, config_data: Dict[str, Any]) -> None:
        """Обновление конфигурации из словаря."""
        if not config_data:
            return
        # Обновление основных настроек
        for key, value in config_data.items():
            if hasattr(self.config, key):
                if key == "cache" and isinstance(value, dict):
                    self._update_cache_config(value)
                elif key == "validation" and isinstance(value, dict):
                    self._update_validation_config(value)
                elif key == "events" and isinstance(value, dict):
                    self._update_event_config(value)
                elif key == "database" and isinstance(value, dict):
                    self._update_database_config(value)
                elif key == "logging" and isinstance(value, dict):
                    self._update_logging_config(value)
                elif key == "performance" and isinstance(value, dict):
                    self._update_performance_config(value)
                elif key == "security" and isinstance(value, dict):
                    self._update_security_config(value)
                else:
                    setattr(self.config, key, value)

    def _update_cache_config(self, cache_data: Dict[str, Any]) -> None:
        """Обновление конфигурации кэша."""
        for key, value in cache_data.items():
            if hasattr(self.config.cache, key):
                setattr(self.config.cache, key, value)

    def _update_validation_config(self, validation_data: Dict[str, Any]) -> None:
        """Обновление конфигурации валидации."""
        for key, value in validation_data.items():
            if hasattr(self.config.validation, key):
                setattr(self.config.validation, key, value)

    def _update_event_config(self, event_data: Dict[str, Any]) -> None:
        """Обновление конфигурации событий."""
        for key, value in event_data.items():
            if hasattr(self.config.events, key):
                setattr(self.config.events, key, value)

    def _update_database_config(self, database_data: Dict[str, Any]) -> None:
        """Обновление конфигурации базы данных."""
        for key, value in database_data.items():
            if hasattr(self.config.database, key):
                setattr(self.config.database, key, value)

    def _update_logging_config(self, logging_data: Dict[str, Any]) -> None:
        """Обновление конфигурации логирования."""
        for key, value in logging_data.items():
            if hasattr(self.config.logging, key):
                setattr(self.config.logging, key, value)

    def _update_performance_config(self, performance_data: Dict[str, Any]) -> None:
        """Обновление конфигурации производительности."""
        for key, value in performance_data.items():
            if hasattr(self.config.performance, key):
                setattr(self.config.performance, key, value)

    def _update_security_config(self, security_data: Dict[str, Any]) -> None:
        """Обновление конфигурации безопасности."""
        for key, value in security_data.items():
            if hasattr(self.config.security, key):
                setattr(self.config.security, key, value)

    def _config_to_dict(self) -> Dict[str, Any]:
        """Преобразование конфигурации в словарь."""
        return {
            "name": self.config.name,
            "version": self.config.version,
            "environment": self.config.environment,
            "cache": {
                "enabled": self.config.cache.enabled,
                "max_size": self.config.cache.max_size,
                "default_ttl_minutes": self.config.cache.default_ttl_minutes,
                "cleanup_interval_seconds": self.config.cache.cleanup_interval_seconds,
                "orders_cache_size": self.config.cache.orders_cache_size,
                "positions_cache_size": self.config.cache.positions_cache_size,
                "trading_pairs_cache_size": self.config.cache.trading_pairs_cache_size,
                "accounts_cache_size": self.config.cache.accounts_cache_size,
                "metrics_cache_size": self.config.cache.metrics_cache_size,
                "patterns_cache_size": self.config.cache.patterns_cache_size,
                "liquidity_cache_size": self.config.cache.liquidity_cache_size,
            },
            "validation": {
                "enabled": self.config.validation.enabled,
                "strict_mode": self.config.validation.strict_mode,
                "validate_business_rules": self.config.validation.validate_business_rules,
                "min_quantity": self.config.validation.min_quantity,
                "max_quantity": self.config.validation.max_quantity,
                "min_price": self.config.validation.min_price,
                "max_price": self.config.validation.max_price,
                "max_leverage": self.config.validation.max_leverage,
                "min_order_size": self.config.validation.min_order_size,
                "max_order_size": self.config.validation.max_order_size,
                "allowed_time_in_force": self.config.validation.allowed_time_in_force,
            },
            "events": {
                "enabled": self.config.events.enabled,
                "max_history_size": self.config.events.max_history_size,
                "publish_async": self.config.events.publish_async,
                "log_events": self.config.events.log_events,
                "event_types_enabled": self.config.events.event_types_enabled,
            },
            "database": {
                "type": self.config.database.type,
                "connection_string": self.config.database.connection_string,
                "pool_size": self.config.database.pool_size,
                "max_overflow": self.config.database.max_overflow,
                "pool_timeout": self.config.database.pool_timeout,
                "pool_recycle": self.config.database.pool_recycle,
                "echo": self.config.database.echo,
                "auto_migrate": self.config.database.auto_migrate,
            },
            "logging": {
                "level": self.config.logging.level,
                "format": self.config.logging.format,
                "file_path": self.config.logging.file_path,
                "max_file_size_mb": self.config.logging.max_file_size_mb,
                "backup_count": self.config.logging.backup_count,
                "log_queries": self.config.logging.log_queries,
                "log_events": self.config.logging.log_events,
                "log_performance": self.config.logging.log_performance,
            },
            "performance": {
                "enable_metrics": self.config.performance.enable_metrics,
                "enable_profiling": self.config.performance.enable_profiling,
                "slow_query_threshold_ms": self.config.performance.slow_query_threshold_ms,
                "batch_size": self.config.performance.batch_size,
                "max_concurrent_operations": self.config.performance.max_concurrent_operations,
                "connection_pool_size": self.config.performance.connection_pool_size,
                "query_timeout_seconds": self.config.performance.query_timeout_seconds,
            },
            "security": {
                "enable_audit_log": self.config.security.enable_audit_log,
                "encrypt_sensitive_data": self.config.security.encrypt_sensitive_data,
                "mask_personal_info": self.config.security.mask_personal_info,
                "rate_limit_enabled": self.config.security.rate_limit_enabled,
                "max_requests_per_minute": self.config.security.max_requests_per_minute,
                "allowed_ips": self.config.security.allowed_ips,
                "require_authentication": self.config.security.require_authentication,
            },
            "default_currency": self.config.default_currency,
            "supported_currencies": self.config.supported_currencies,
            "trading_hours": self.config.trading_hours,
            "default_trading_pair_config": self.config.default_trading_pair_config,
        }
