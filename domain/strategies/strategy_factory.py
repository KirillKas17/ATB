"""
Фабрика стратегий - промышленная реализация создания и управления стратегиями.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Type, Union
from uuid import uuid4

from domain.entities.strategy import StrategyType
from domain.strategies.arbitrage_strategy import ArbitrageStrategy
from domain.strategies.breakout_strategy import BreakoutStrategy
from domain.strategies.mean_reversion_strategy import MeanReversionStrategy
from domain.strategies.scalping_strategy import ScalpingStrategy
from domain.strategies.strategy_interface import StrategyInterface
from domain.strategies.trend_following_strategy import TrendFollowingStrategy
from domain.strategies.exceptions import (
    StrategyCreationError,
    StrategyFactoryError,
    StrategyNotFoundError,
    StrategyRegistrationError,
    StrategyValidationError,
)
from domain.strategies.validators import StrategyValidator
from domain.type_definitions import (
    ConfidenceLevel,
    RiskLevel,
    StrategyConfig,
    StrategyId,
)

logger = logging.getLogger(__name__)


@dataclass
class StrategyCreator:
    """Создатель стратегии с метаданными."""

    creator_func: Callable[..., StrategyInterface]
    strategy_type: StrategyType
    name: str
    description: str
    version: str
    author: str
    required_parameters: List[str] = field(default_factory=list)
    optional_parameters: List[str] = field(default_factory=list)
    supported_pairs: List[str] = field(default_factory=list)
    min_confidence: Decimal = Decimal("0.3")
    max_confidence: Decimal = Decimal("1.0")
    risk_levels: List[str] = field(default_factory=lambda: ["low", "medium", "high"])
    created_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    success_rate: Decimal = Decimal("0.0")
    avg_performance: Decimal = Decimal("0.0")

    def create(self, **kwargs: Any) -> StrategyInterface:
        """Создать экземпляр стратегии."""
        try:
            strategy = self.creator_func(**kwargs)
            self.usage_count += 1
            return strategy
        except Exception as e:
            logger.error(f"Error creating strategy {self.name}: {e}")
            raise StrategyCreationError(f"Failed to create strategy {self.name}: {e}")


@dataclass
class StrategyFactoryStats:
    """Статистика фабрики стратегий."""

    total_strategies: int = 0
    total_creations: int = 0
    successful_creations: int = 0
    failed_creations: int = 0
    avg_creation_time: float = 0.0
    most_popular_strategy: Optional[str] = None
    most_popular_type: Optional[StrategyType] = None
    last_creation_time: Optional[datetime] = None
    creation_history: List[Dict[str, Any]] = field(default_factory=list)


class StrategyFactory:
    """
    Промышленная фабрика стратегий с полным управлением жизненным циклом.
    Обеспечивает:
    - Регистрацию создателей стратегий
    - Валидацию конфигураций
    - Управление зависимостями
    - Статистику и мониторинг
    - Кэширование и оптимизацию
    """

    def __init__(self) -> None:
        """Инициализация фабрики."""
        self._creators: Dict[str, StrategyCreator] = {}
        self._type_registry: Dict[StrategyType, List[str]] = {}
        self._validator = StrategyValidator()
        self._stats = StrategyFactoryStats()
        self._cache: Dict[str, StrategyInterface] = {}
        self._cache_ttl = 3600  # 1 час
        self._cache_timestamps: Dict[str, datetime] = {}
        # Регистрируем встроенные стратегии
        self._register_builtin_strategies()

    def _register_builtin_strategies(self) -> None:
        """Регистрируем встроенные стратегии."""
        self.register_strategy(
            name="trend_following",
            creator_func=self._create_trend_following_strategy,
            strategy_type=StrategyType.TREND_FOLLOWING,
            description="Стратегия следования за трендом",
            version="1.0.0",
            author="System",
            required_parameters=["strategy_id", "name", "trading_pairs", "parameters"],
            optional_parameters=["risk_level", "confidence_threshold"],
            supported_pairs=["BTC/USDT", "ETH/USDT", "ADA/USDT"],
            min_confidence=Decimal("0.3"),
            max_confidence=Decimal("1.0"),
            risk_levels=["low", "medium", "high"],
        )
        self.register_strategy(
            name="mean_reversion",
            creator_func=self._create_mean_reversion_strategy,
            strategy_type=StrategyType.MEAN_REVERSION,
            description="Стратегия возврата к среднему",
            version="1.0.0",
            author="System",
            required_parameters=["strategy_id", "name", "trading_pairs", "parameters"],
            optional_parameters=["risk_level", "confidence_threshold"],
            supported_pairs=["BTC/USDT", "ETH/USDT", "ADA/USDT"],
            min_confidence=Decimal("0.3"),
            max_confidence=Decimal("1.0"),
            risk_levels=["low", "medium", "high"],
        )
        self.register_strategy(
            name="breakout",
            creator_func=self._create_breakout_strategy,
            strategy_type=StrategyType.BREAKOUT,
            description="Стратегия пробоя",
            version="1.0.0",
            author="System",
            required_parameters=["strategy_id", "name", "trading_pairs", "parameters"],
            optional_parameters=["risk_level", "confidence_threshold"],
            supported_pairs=["BTC/USDT", "ETH/USDT", "ADA/USDT"],
            min_confidence=Decimal("0.3"),
            max_confidence=Decimal("1.0"),
            risk_levels=["low", "medium", "high"],
        )
        self.register_strategy(
            name="scalping",
            creator_func=self._create_scalping_strategy,
            strategy_type=StrategyType.SCALPING,
            description="Скальпинг стратегия",
            version="1.0.0",
            author="System",
            required_parameters=["strategy_id", "name", "trading_pairs", "parameters"],
            optional_parameters=["risk_level", "confidence_threshold"],
            supported_pairs=["BTC/USDT", "ETH/USDT", "ADA/USDT"],
            min_confidence=Decimal("0.3"),
            max_confidence=Decimal("1.0"),
            risk_levels=["low", "medium", "high"],
        )
        self.register_strategy(
            name="arbitrage",
            creator_func=self._create_arbitrage_strategy,
            strategy_type=StrategyType.ARBITRAGE,
            description="Арбитражная стратегия",
            version="1.0.0",
            author="System",
            required_parameters=["strategy_id", "name", "trading_pairs", "parameters"],
            optional_parameters=["risk_level", "confidence_threshold"],
            supported_pairs=["BTC/USDT", "ETH/USDT", "ADA/USDT"],
            min_confidence=Decimal("0.3"),
            max_confidence=Decimal("1.0"),
            risk_levels=["low", "medium", "high"],
        )

    def _create_trend_following_strategy(self, **kwargs: Any) -> StrategyInterface:
        """Создать стратегию следования за трендом."""
        return TrendFollowingStrategy(**kwargs)

    def _create_mean_reversion_strategy(self, **kwargs: Any) -> StrategyInterface:
        """Создать стратегию возврата к среднему."""
        return MeanReversionStrategy(**kwargs)

    def _create_breakout_strategy(self, **kwargs: Any) -> StrategyInterface:
        """Создать стратегию пробоя."""
        return BreakoutStrategy(**kwargs)

    def _create_scalping_strategy(self, **kwargs: Any) -> StrategyInterface:
        """Создать скальпинг стратегию."""
        return ScalpingStrategy(**kwargs)

    def _create_arbitrage_strategy(self, **kwargs: Any) -> StrategyInterface:
        """Создать арбитражную стратегию."""
        return ArbitrageStrategy(**kwargs)

    def register_strategy(
        self,
        name: str,
        creator_func: Callable[..., StrategyInterface],
        strategy_type: StrategyType,
        description: str = "",
        version: str = "1.0.0",
        author: str = "Unknown",
        required_parameters: Optional[List[str]] = None,
        optional_parameters: Optional[List[str]] = None,
        supported_pairs: Optional[List[str]] = None,
        min_confidence: Decimal = Decimal("0.3"),
        max_confidence: Decimal = Decimal("1.0"),
        risk_levels: Optional[List[str]] = None,
    ) -> None:
        """
        Зарегистрировать создателя стратегии.
        Args:
            name: Уникальное имя стратегии
            creator_func: Функция создания стратегии
            strategy_type: Тип стратегии
            description: Описание стратегии
            version: Версия стратегии
            author: Автор стратегии
            required_parameters: Обязательные параметры
            optional_parameters: Опциональные параметры
            supported_pairs: Поддерживаемые торговые пары
            min_confidence: Минимальный уровень уверенности
            max_confidence: Максимальный уровень уверенности
            risk_levels: Поддерживаемые уровни риска
        Raises:
            StrategyRegistrationError: Если регистрация не удалась
        """
        try:
            if name in self._creators:
                raise StrategyRegistrationError(f"Strategy {name} already registered")
            creator = StrategyCreator(
                creator_func=creator_func,
                strategy_type=strategy_type,
                name=name,
                description=description,
                version=version,
                author=author,
                required_parameters=required_parameters or [],
                optional_parameters=optional_parameters or [],
                supported_pairs=supported_pairs or [],
                min_confidence=min_confidence,
                max_confidence=max_confidence,
                risk_levels=risk_levels or ["low", "medium", "high"],
            )
            self._creators[name] = creator
            # Регистрируем по типу
            if strategy_type not in self._type_registry:
                self._type_registry[strategy_type] = []
            self._type_registry[strategy_type].append(name)
            self._stats.total_strategies += 1
            logger.info(f"Registered strategy: {name} ({strategy_type.value})")
        except Exception as e:
            logger.error(f"Failed to register strategy {name}: {e}")
            raise StrategyRegistrationError(f"Registration failed: {e}")

    def create_strategy(
        self,
        name: str,
        strategy_id: Optional[StrategyId] = None,
        trading_pairs: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        risk_level: str = "medium",
        confidence_threshold: Optional[Decimal] = None,
        use_cache: bool = True,
    ) -> StrategyInterface:
        """
        Создать стратегию.
        Args:
            name: Имя стратегии
            strategy_id: ID стратегии (генерируется автоматически если не указан)
            trading_pairs: Торговые пары
            parameters: Параметры стратегии
            risk_level: Уровень риска
            confidence_threshold: Порог уверенности
            use_cache: Использовать кэш
        Returns:
            StrategyInterface: Созданная стратегия
        Raises:
            StrategyNotFoundError: Если стратегия не найдена
            StrategyCreationError: Если создание не удалось
            StrategyValidationError: Если валидация не прошла
        """
        start_time = datetime.now()
        try:
            # Проверяем кэш
            cache_key = self._generate_cache_key(name, parameters, risk_level)
            if use_cache and cache_key in self._cache:
                if self._is_cache_valid(cache_key):
                    logger.debug(f"Returning cached strategy: {name}")
                    return self._cache[cache_key]
            # Проверяем, что стратегия зарегистрирована
            if name not in self._creators:
                raise StrategyNotFoundError(f"Strategy {name} not found")
            creator = self._creators[name]
            # Подготавливаем параметры
            strategy_id = strategy_id or StrategyId(uuid4())
            trading_pairs = trading_pairs or ["BTC/USDT"]
            parameters = parameters or {}
            # Создаем стратегию
            strategy = creator.create(
                strategy_id=strategy_id,
                name=name,
                trading_pairs=trading_pairs,
                parameters=parameters,
                risk_level=RiskLevel(Decimal(risk_level)),
                confidence_threshold=ConfidenceLevel(
                    confidence_threshold or Decimal("0.6")
                ),
            )
            # Валидируем стратегию
            validation_errors = self._validator.validate_strategy_config(
                strategy.get_parameters()
            )
            if validation_errors:
                raise StrategyValidationError(
                    f"Strategy validation failed: {validation_errors}"
                )
            # Кэшируем стратегию
            if use_cache:
                self._cache[cache_key] = strategy
                self._cache_timestamps[cache_key] = datetime.now()
            # Обновляем статистику
            self._update_creation_stats(name, creator.strategy_type, start_time)
            logger.info(f"Created strategy: {name} (ID: {strategy_id})")
            return strategy
        except Exception as e:
            logger.error(f"Failed to create strategy {name}: {e}")
            raise StrategyCreationError(f"Creation failed: {e}")

    def get_available_strategies(
        self, strategy_type: Optional[StrategyType] = None
    ) -> List[Dict[str, Any]]:
        """
        Получить список доступных стратегий.
        Args:
            strategy_type: Фильтр по типу стратегии
        Returns:
            List[Dict[str, Any]]: Список доступных стратегий
        """
        strategies = []
        for name, creator in self._creators.items():
            if strategy_type and creator.strategy_type != strategy_type:
                continue
            strategies.append(
                {
                    "name": name,
                    "type": creator.strategy_type.value,
                    "description": creator.description,
                    "version": creator.version,
                    "author": creator.author,
                    "required_parameters": creator.required_parameters,
                    "optional_parameters": creator.optional_parameters,
                    "supported_pairs": creator.supported_pairs,
                    "min_confidence": str(creator.min_confidence),
                    "max_confidence": str(creator.max_confidence),
                    "risk_levels": creator.risk_levels,
                    "usage_count": creator.usage_count,
                    "success_rate": str(creator.success_rate),
                    "avg_performance": str(creator.avg_performance),
                }
            )
        return strategies

    def get_strategy_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Получить информацию о стратегии.
        Args:
            name: Имя стратегии
        Returns:
            Optional[Dict[str, Any]]: Информация о стратегии
        """
        if name not in self._creators:
            return None
        creator = self._creators[name]
        return {
            "name": name,
            "type": creator.strategy_type.value,
            "description": creator.description,
            "version": creator.version,
            "author": creator.author,
            "required_parameters": creator.required_parameters,
            "optional_parameters": creator.optional_parameters,
            "supported_pairs": creator.supported_pairs,
            "min_confidence": str(creator.min_confidence),
            "max_confidence": str(creator.max_confidence),
            "risk_levels": creator.risk_levels,
            "usage_count": creator.usage_count,
            "success_rate": str(creator.success_rate),
            "avg_performance": str(creator.avg_performance),
            "created_at": creator.created_at.isoformat(),
        }

    def unregister_strategy(self, name: str) -> bool:
        """
        Отменить регистрацию стратегии.
        Args:
            name: Имя стратегии
        Returns:
            bool: True если стратегия была отменена
        """
        if name not in self._creators:
            return False
        try:
            creator = self._creators[name]
            # Удаляем из реестра
            del self._creators[name]
            # Удаляем из индекса по типу
            if creator.strategy_type in self._type_registry:
                if name in self._type_registry[creator.strategy_type]:
                    self._type_registry[creator.strategy_type].remove(name)
            # Очищаем кэш
            self._clear_cache_for_strategy(name)
            self._stats.total_strategies -= 1
            logger.info(f"Unregistered strategy: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to unregister strategy {name}: {e}")
            return False

    def update_strategy_performance(
        self, name: str, success: bool, performance: Decimal
    ) -> None:
        """
        Обновить производительность стратегии.
        Args:
            name: Имя стратегии
            success: Успешность выполнения
            performance: Показатель производительности
        """
        if name in self._creators:
            creator = self._creators[name]
            # Обновляем статистику
            if success:
                creator.success_rate = (
                    creator.success_rate * creator.usage_count + Decimal("1")
                ) / (creator.usage_count + 1)
            creator.avg_performance = (
                creator.avg_performance * creator.usage_count + performance
            ) / (creator.usage_count + 1)

    def get_factory_stats(self) -> Dict[str, Any]:
        """
        Получить статистику фабрики.
        Returns:
            Dict[str, Any]: Статистика фабрики
        """
        return {
            "total_strategies": self._stats.total_strategies,
            "total_creations": self._stats.total_creations,
            "successful_creations": self._stats.successful_creations,
            "failed_creations": self._stats.failed_creations,
            "avg_creation_time": self._stats.avg_creation_time,
            "most_popular_strategy": self._stats.most_popular_strategy,
            "most_popular_type": (
                self._stats.most_popular_type.value
                if self._stats.most_popular_type
                else None
            ),
            "last_creation_time": (
                self._stats.last_creation_time.isoformat()
                if self._stats.last_creation_time
                else None
            ),
        }

    def clear_cache(self) -> None:
        """Очистить кэш стратегий."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Strategy cache cleared")

    def validate_strategy_config(self, name: str, config: StrategyConfig) -> List[str]:
        """
        Валидировать конфигурацию стратегии.
        Args:
            name: Имя стратегии
            config: Конфигурация стратегии
        Returns:
            List[str]: Список ошибок валидации
        """
        if name not in self._creators:
            return [f"Strategy {name} not found"]
        # Преобразуем конфигурацию в словарь для валидации
        config_dict = {
            "name": config.get("name", name),
            "strategy_type": config.get("strategy_type", ""),
            "trading_pairs": config.get("trading_pairs", []),
            "parameters": config.get("parameters", {}),
        }
        return self._validator.validate_strategy_config(config_dict)

    def _generate_cache_key(
        self, name: str, parameters: Optional[Dict[str, Any]], risk_level: str
    ) -> str:
        """Сгенерировать ключ кэша."""
        import hashlib
        import json

        key_data = {
            "name": name,
            "parameters": parameters or {},
            "risk_level": risk_level,
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Проверить валидность кэша."""
        if cache_key not in self._cache_timestamps:
            return False
        timestamp = self._cache_timestamps[cache_key]
        return (datetime.now() - timestamp).total_seconds() < self._cache_ttl

    def _update_creation_stats(
        self, name: str, strategy_type: StrategyType, start_time: datetime
    ) -> None:
        """Обновить статистику создания."""
        end_time = datetime.now()
        creation_time = (end_time - start_time).total_seconds()
        self._stats.total_creations += 1
        self._stats.successful_creations += 1
        self._stats.last_creation_time = end_time
        # Обновляем среднее время создания
        if self._stats.total_creations == 1:
            self._stats.avg_creation_time = creation_time
        else:
            self._stats.avg_creation_time = (
                self._stats.avg_creation_time * (self._stats.total_creations - 1)
                + creation_time
            ) / self._stats.total_creations
        # Обновляем самую популярную стратегию
        if name not in self._creators:
            return
        creator = self._creators[name]
        if (
            self._stats.most_popular_strategy is None
            or creator.usage_count
            > self._creators[self._stats.most_popular_strategy].usage_count
        ):
            self._stats.most_popular_strategy = name
        # Обновляем самый популярный тип
        type_count = len(self._type_registry.get(strategy_type, []))
        if self._stats.most_popular_type is None or type_count > len(
            self._type_registry.get(self._stats.most_popular_type, [])
        ):
            self._stats.most_popular_type = strategy_type

    def _clear_cache_for_strategy(self, name: str) -> None:
        """Очистить кэш для конкретной стратегии."""
        keys_to_remove = []
        for key in self._cache.keys():
            if name in key:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self._cache[key]
            if key in self._cache_timestamps:
                del self._cache_timestamps[key]


# Глобальный экземпляр фабрики
_factory_instance: Optional[StrategyFactory] = None


def get_strategy_factory() -> StrategyFactory:
    """
    Получить глобальный экземпляр фабрики стратегий.
    Returns:
        StrategyFactory: Глобальный экземпляр фабрики
    """
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = StrategyFactory()
    return _factory_instance


def register_strategy(
    name: str,
    strategy_type: StrategyType,
    description: str = "",
    version: str = "1.0.0",
    author: str = "Unknown",
    required_parameters: Optional[List[str]] = None,
    optional_parameters: Optional[List[str]] = None,
    supported_pairs: Optional[List[str]] = None,
    min_confidence: Decimal = Decimal("0.3"),
    max_confidence: Decimal = Decimal("1.0"),
    risk_levels: Optional[List[str]] = None,
) -> Callable[[Callable[..., StrategyInterface]], Callable[..., StrategyInterface]]:
    """
    Декоратор для регистрации стратегии.
    Args:
        name: Имя стратегии
        strategy_type: Тип стратегии
        description: Описание стратегии
        version: Версия стратегии
        author: Автор стратегии
        required_parameters: Обязательные параметры
        optional_parameters: Опциональные параметры
        supported_pairs: Поддерживаемые торговые пары
        min_confidence: Минимальный уровень уверенности
        max_confidence: Максимальный уровень уверенности
        risk_levels: Поддерживаемые уровни риска
    """

    def decorator(
        creator_func: Callable[..., StrategyInterface],
    ) -> Callable[..., StrategyInterface]:
        factory = get_strategy_factory()
        factory.register_strategy(
            name=name,
            creator_func=creator_func,
            strategy_type=strategy_type,
            description=description,
            version=version,
            author=author,
            required_parameters=required_parameters,
            optional_parameters=optional_parameters,
            supported_pairs=supported_pairs,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            risk_levels=risk_levels,
        )
        return creator_func

    return decorator
