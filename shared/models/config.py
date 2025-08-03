"""
Типизированные модели конфигурации с безопасной обработкой секретов.
"""

import os
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union


class LogLevel(str, Enum):
    """Уровни логирования."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class CacheType(str, Enum):
    """Типы кэша."""

    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"


class DatabaseType(str, Enum):
    """Типы баз данных."""

    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"


@dataclass
class SecurityConfig:
    """Конфигурация безопасности."""

    # API ключи (загружаются из переменных окружения)
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    passphrase: Optional[str] = None
    # Настройки шифрования
    encryption_key: Optional[str] = None
    salt_rounds: int = 12
    # Настройки аутентификации
    jwt_secret: Optional[str] = None
    jwt_expiry_hours: int = 24
    # Настройки безопасности
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    password_min_length: int = 8
    require_special_chars: bool = True

    def __post_init__(self):
        """Загружаем секреты из переменных окружения."""
        self.api_key = os.getenv("API_KEY") or self.api_key
        self.api_secret = os.getenv("API_SECRET") or self.api_secret
        self.passphrase = os.getenv("API_PASSPHRASE") or self.passphrase
        self.encryption_key = os.getenv("ENCRYPTION_KEY") or self.encryption_key
        self.jwt_secret = os.getenv("JWT_SECRET") or self.jwt_secret


@dataclass
class DatabaseConfig:
    """Конфигурация базы данных."""

    type: DatabaseType
    host: str
    port: int
    database: str
    username: str
    password: Optional[str] = None
    pool_size: int = 10
    max_overflow: int = 20
    timeout: int = 30
    ssl_mode: str = "prefer"
    connection_string: Optional[str] = None

    def __post_init__(self):
        """Загружаем пароль из переменных окружения."""
        if not self.password:
            self.password = os.getenv("DB_PASSWORD")
        # Валидация
        if not self.host:
            raise ValueError("Database host cannot be empty")
        if self.port <= 0 or self.port > 65535:
            raise ValueError("Database port must be between 1 and 65535")
        if not self.database:
            raise ValueError("Database name cannot be empty")
        if not self.username:
            raise ValueError("Database username cannot be empty")


@dataclass
class CacheConfig:
    """Конфигурация кэша."""

    type: CacheType
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = None
    max_size: int = 1000
    ttl: int = 3600
    eviction_policy: str = "lru"

    def __post_init__(self):
        """Загружаем пароль из переменных окружения."""
        if not self.password:
            self.password = os.getenv("REDIS_PASSWORD")


@dataclass
class LoggingConfig:
    """Конфигурация логирования."""

    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_console: bool = True
    enable_file: bool = False
    # Безопасность логирования
    mask_sensitive_data: bool = True
    sensitive_fields: List[str] = field(
        default_factory=lambda: ["password", "api_key", "api_secret", "token", "secret"]
    )


@dataclass
class TradingConfig:
    """Конфигурация торговли."""

    trading_interval: int = 300  # 5 минут
    sentiment_analysis_interval: int = 600  # 10 минут
    portfolio_rebalance_interval: int = 3600  # 1 час
    evolution_cycle_interval: int = 1800  # 30 минут
    simulation_cycle_interval: int = 3600  # 1 час
    max_positions: int = 10
    risk_threshold: Decimal = Decimal("0.1")
    max_leverage: Decimal = Decimal("3.0")
    max_position_size: Decimal = Decimal("0.1")
    max_concentration: Decimal = Decimal("0.25")

    def __post_init__(self):
        """Валидация торговых параметров."""
        if self.risk_threshold <= 0:
            raise ValueError("risk_threshold must be positive")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be positive")
        if self.max_position_size <= 0:
            raise ValueError("max_position_size must be positive")
        if self.max_concentration <= 0:
            raise ValueError("max_concentration must be positive")


@dataclass
class RiskConfig:
    """Конфигурация управления рисками."""

    max_var: Decimal = Decimal("0.02")  # 2% максимальный VaR
    max_drawdown: Decimal = Decimal("0.15")  # 15% максимальная просадка
    max_position_size: Decimal = Decimal("0.1")  # 10% максимальный размер позиции
    max_leverage: Decimal = Decimal("3.0")  # 3x максимальное плечо
    max_concentration: Decimal = Decimal("0.25")  # 25% максимальная концентрация
    risk_cache_ttl: int = 300  # 5 минут

    def __post_init__(self):
        """Валидация параметров риска."""
        if self.max_var <= 0:
            raise ValueError("max_var must be positive")
        if self.max_drawdown <= 0:
            raise ValueError("max_drawdown must be positive")
        if self.max_position_size <= 0:
            raise ValueError("max_position_size must be positive")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be positive")
        if self.max_concentration <= 0:
            raise ValueError("max_concentration must be positive")


@dataclass
class MonitoringConfig:
    """Конфигурация мониторинга."""

    enabled: bool = True
    metrics_interval: int = 60  # 1 минута
    health_check_interval: int = 30  # 30 секунд
    alert_threshold: float = 0.8
    dashboard_enabled: bool = True
    dashboard_port: int = 8080

    def __post_init__(self):
        """Валидация параметров мониторинга."""
        if self.metrics_interval <= 0:
            raise ValueError("metrics_interval must be positive")
        if self.health_check_interval <= 0:
            raise ValueError("health_check_interval must be positive")
        if self.dashboard_port <= 0 or self.dashboard_port > 65535:
            raise ValueError("dashboard_port must be between 1 and 65535")


@dataclass
class MLConfig:
    """Конфигурация машинного обучения."""

    models_path: str = "models/"
    training_data_path: str = "data/training/"
    validation_split: float = 0.2
    batch_size: int = 32
    learning_rate: float = 0.001
    max_epochs: int = 100
    early_stopping_patience: int = 10
    model_checkpoint_interval: int = 5

    def __post_init__(self):
        """Валидация ML параметров."""
        if self.validation_split <= 0 or self.validation_split >= 1:
            raise ValueError("validation_split must be between 0 and 1")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")


@dataclass
class EvolutionConfig:
    """Конфигурация эволюционных алгоритмов."""

    enabled: bool = True
    population_size: int = 100
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 10
    tournament_size: int = 3
    fitness_threshold: float = 0.95

    def __post_init__(self):
        """Валидация параметров эволюции."""
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")
        if self.generations <= 0:
            raise ValueError("generations must be positive")
        if self.mutation_rate < 0 or self.mutation_rate > 1:
            raise ValueError("mutation_rate must be between 0 and 1")
        if self.crossover_rate < 0 or self.crossover_rate > 1:
            raise ValueError("crossover_rate must be between 0 and 1")
        if self.fitness_threshold < 0 or self.fitness_threshold > 1:
            raise ValueError("fitness_threshold must be between 0 and 1")


@dataclass
class ApplicationConfig:
    """Основная конфигурация приложения."""

    # Основные настройки
    app_name: str = "ATB Trading System"
    version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    # Компоненты конфигурации
    security: SecurityConfig = field(default_factory=SecurityConfig)
    database: DatabaseConfig = field(
        default_factory=lambda: DatabaseConfig(
            type=DatabaseType.POSTGRESQL,
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "atb_trading"),
            username=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD"),
        )
    )
    cache: CacheConfig = field(
        default_factory=lambda: CacheConfig(
            type=CacheType.REDIS,
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD"),
        )
    )
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    # Дополнительные параметры
    custom_parameters: Dict[str, Union[str, int, float, bool]] = field(
        default_factory=dict
    )

    def __post_init__(self):
        """Валидация конфигурации."""
        errors = self.validate()
        if errors:
            raise ValueError(f"Configuration validation failed: {', '.join(errors)}")

    def validate(self) -> List[str]:
        """Валидация конфигурации."""
        errors = []
        # Проверка основных параметров
        if not self.app_name:
            errors.append("app_name cannot be empty")
        # Проверка окружения
        if self.environment not in ["development", "staging", "production"]:
            errors.append(
                "environment must be one of: development, staging, production"
            )
        # Проверка безопасности
        if self.environment == "production":
            if not self.security.api_key:
                errors.append("API key is required in production")
            if not self.security.api_secret:
                errors.append("API secret is required in production")
            if not self.security.jwt_secret:
                errors.append("JWT secret is required in production")
        return errors


def create_default_config() -> ApplicationConfig:
    """Создание конфигурации по умолчанию."""
    return ApplicationConfig()


def create_production_config() -> ApplicationConfig:
    """Создание конфигурации для продакшена."""
    config = ApplicationConfig(
        environment="production",
        debug=False,
        logging=LoggingConfig(
            level=LogLevel.INFO, enable_file=True, mask_sensitive_data=True
        ),
    )
    return config


def validate_config(config: ApplicationConfig) -> List[str]:
    """Валидация конфигурации."""
    return config.validate()
