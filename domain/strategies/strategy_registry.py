"""
Реестр стратегий - централизованное управление стратегиями.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

from domain.entities.strategy import StrategyStatus, StrategyType
from domain.strategies.exceptions import (
    StrategyDuplicateError,
    StrategyNotFoundError,
    StrategyRegistryError,
    StrategyStateError,
    StrategyValidationError,
)
from domain.strategies.strategy_interface import StrategyInterface
from domain.strategies.validators import StrategyValidator
from domain.type_definitions import MetadataDict, StrategyId, TradingPair
from domain.type_definitions.evolution_types import StrategyPerformance as EvolutionStrategyPerformance
from domain.entities.strategy_performance import StrategyPerformance as EntityStrategyPerformance


def _convert_performance_to_evolution_type(
    entity_performance: EntityStrategyPerformance
) -> EvolutionStrategyPerformance:
    """Конвертировать производительность из entity в evolution тип."""
    return EvolutionStrategyPerformance(
        total_trades=entity_performance.total_trades,
        winning_trades=entity_performance.winning_trades,
        losing_trades=entity_performance.losing_trades,
        win_rate=float(entity_performance.win_rate.value) if entity_performance.win_rate else 0.0,
        profit_factor=float(entity_performance.profit_factor),
        sharpe_ratio=float(entity_performance.sharpe_ratio),
        sortino_ratio=0.0,  # Не доступно в entity
        calmar_ratio=0.0,   # Не доступно в entity
        max_drawdown=float(entity_performance.max_drawdown.value) if entity_performance.max_drawdown else 0.0,
        total_pnl=float(entity_performance.total_pnl.value) if entity_performance.total_pnl else 0.0,
        net_pnl=float(entity_performance.total_pnl.value) if entity_performance.total_pnl else 0.0,
        average_trade=float(entity_performance.average_trade.value) if entity_performance.average_trade else 0.0,
        best_trade=float(entity_performance.best_trade.value) if entity_performance.best_trade else 0.0,
        worst_trade=float(entity_performance.worst_trade.value) if entity_performance.worst_trade else 0.0,
        average_win=0.0,  # Не доступно в entity
        average_loss=0.0, # Не доступно в entity
        largest_win=0.0,  # Не доступно в entity
        largest_loss=0.0, # Не доступно в entity
    )


logger = logging.getLogger(__name__)


@dataclass
class StrategyMetadata:
    """Метаданные стратегии в реестре."""

    strategy_id: StrategyId
    name: str
    strategy_type: StrategyType
    status: StrategyStatus
    trading_pairs: List[TradingPair]
    parameters: Dict[str, Any]
    performance: EvolutionStrategyPerformance
    created_at: datetime
    updated_at: datetime
    last_execution: Optional[datetime] = None
    execution_count: int = 0
    success_count: int = 0
    total_pnl: Decimal = Decimal("0")
    avg_execution_time: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))
    version: str = "1.0.0"
    author: str = "Unknown"
    description: str = ""
    is_active: bool = True
    priority: int = 0
    risk_level: str = "medium"
    confidence_threshold: Decimal = Decimal("0.6")
    max_position_size: Decimal = Decimal("0.1")
    stop_loss: Decimal = Decimal("0.02")
    take_profit: Decimal = Decimal("0.04")


@dataclass
class RegistryStats:
    """Статистика реестра стратегий."""

    total_strategies: int = 0
    active_strategies: int = 0
    paused_strategies: int = 0
    stopped_strategies: int = 0
    error_strategies: int = 0
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_pnl: Decimal = Decimal("0")
    avg_execution_time: float = 0.0
    most_profitable_strategy: Optional[str] = None
    most_active_strategy: Optional[str] = None
    strategy_type_distribution: Dict[str, int] = field(default_factory=dict)
    risk_level_distribution: Dict[str, int] = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.now)


class StrategyRegistry:
    """
    Централизованный реестр стратегий с полным управлением жизненным циклом.
    Обеспечивает:
    - Регистрацию и управление стратегиями
    - Индексацию и поиск
    - Мониторинг производительности
    - Управление состоянием
    - Статистику и аналитику
    """

    def __init__(self) -> None:
        """Инициализация реестра."""
        self._strategies: Dict[StrategyId, StrategyInterface] = {}
        self._metadata: Dict[StrategyId, StrategyMetadata] = {}
        self._name_index: Dict[str, StrategyId] = {}
        self._type_index: Dict[StrategyType, Set[StrategyId]] = {}
        self._status_index: Dict[StrategyStatus, Set[StrategyId]] = {}
        self._pair_index: Dict[str, Set[StrategyId]] = {}
        self._tag_index: Dict[str, Set[StrategyId]] = {}
        self._validator = StrategyValidator()
        self._stats = RegistryStats()
        self._lock = False  # Простая блокировка для атомарных операций

    def register_strategy(
        self,
        strategy: StrategyInterface,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        priority: int = 0,
        version: str = "1.0.0",
        author: str = "Unknown",
        description: str = "",
    ) -> StrategyId:
        """
        Зарегистрировать стратегию в реестре.
        Args:
            strategy: Стратегия для регистрации
            name: Имя стратегии (если отличается от встроенного)
            tags: Теги для категоризации
            priority: Приоритет стратегии
            version: Версия стратегии
            author: Автор стратегии
            description: Описание стратегии
        Returns:
            StrategyId: ID зарегистрированной стратегии
        Raises:
            StrategyDuplicateError: Если стратегия уже зарегистрирована
            StrategyValidationError: Если валидация не прошла
        """
        if self._lock:
            raise StrategyRegistryError("Registry is locked")
        try:
            strategy_id = strategy.get_strategy_id()
            # Проверяем, что стратегия не зарегистрирована
            if strategy_id in self._strategies:
                raise StrategyDuplicateError(
                    f"Strategy {strategy_id} already registered"
                )
            # Валидируем стратегию
            validation_errors = self._validator.validate_strategy_config(
                strategy.get_parameters()
            )
            if validation_errors:
                raise StrategyValidationError(
                    f"Strategy validation failed: {validation_errors}"
                )
            # Создаем метаданные
            metadata = StrategyMetadata(
                strategy_id=strategy_id,
                name=name or f"Strategy_{strategy_id}",
                strategy_type=strategy.get_strategy_type(),
                status=StrategyStatus.INACTIVE,
                trading_pairs=strategy.get_trading_pairs(),
                parameters=strategy.get_parameters(),
                performance=_convert_performance_to_evolution_type(
                    strategy.get_performance()
                ),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                tags=set(tags or []),
                version=version,
                author=author,
                description=description,
                priority=priority,
            )
            # Регистрируем стратегию
            self._strategies[strategy_id] = strategy
            self._metadata[strategy_id] = metadata
            # Обновляем индексы
            self._update_indexes(strategy_id, metadata, is_new=True)
            # Обновляем статистику
            self._update_stats_on_register(metadata)
            logger.info(f"Registered strategy: {metadata.name} (ID: {strategy_id})")
            return strategy_id
        except Exception as e:
            logger.error(f"Failed to register strategy: {e}")
            raise StrategyRegistryError(f"Registration failed: {e}")

    def unregister_strategy(self, strategy_id: StrategyId) -> bool:
        """
        Отменить регистрацию стратегии.
        Args:
            strategy_id: ID стратегии
        Returns:
            bool: True если стратегия была отменена
        """
        if self._lock:
            raise StrategyRegistryError("Registry is locked")
        if strategy_id not in self._strategies:
            return False
        try:
            metadata = self._metadata[strategy_id]
            # Удаляем из индексов
            self._remove_from_indexes(strategy_id, metadata)
            # Удаляем стратегию и метаданные
            del self._strategies[strategy_id]
            del self._metadata[strategy_id]
            # Обновляем статистику
            self._update_stats_on_unregister(metadata)
            logger.info(f"Unregistered strategy: {metadata.name} (ID: {strategy_id})")
            return True
        except Exception as e:
            logger.error(f"Failed to unregister strategy: {e}")
            return False

    def get_strategy(self, strategy_id: StrategyId) -> Optional[StrategyInterface]:
        """
        Получить стратегию по ID.
        Args:
            strategy_id: ID стратегии
        Returns:
            Optional[StrategyInterface]: Стратегия или None
        """
        return self._strategies.get(strategy_id)

    def get_strategy_by_name(self, name: str) -> Optional[StrategyInterface]:
        """
        Получить стратегию по имени.
        Args:
            name: Имя стратегии
        Returns:
            Optional[StrategyInterface]: Стратегия или None
        """
        strategy_id = self._name_index.get(name)
        if strategy_id:
            return self._strategies.get(strategy_id)
        return None

    def get_strategies_by_type(
        self, strategy_type: StrategyType
    ) -> List[StrategyInterface]:
        """
        Получить стратегии по типу.
        Args:
            strategy_type: Тип стратегии
        Returns:
            List[StrategyInterface]: Список стратегий
        """
        strategy_ids = self._type_index.get(strategy_type, set())
        return [
            self._strategies[sid] for sid in strategy_ids if sid in self._strategies
        ]

    def get_strategies_by_status(
        self, status: StrategyStatus
    ) -> List[StrategyInterface]:
        """
        Получить стратегии по статусу.
        Args:
            status: Статус стратегии
        Returns:
            List[StrategyInterface]: Список стратегий
        """
        strategy_ids = self._status_index.get(status, set())
        return [
            self._strategies[sid] for sid in strategy_ids if sid in self._strategies
        ]

    def get_strategies_by_pair(self, trading_pair: str) -> List[StrategyInterface]:
        """
        Получить стратегии по торговой паре.
        Args:
            trading_pair: Торговая пара
        Returns:
            List[StrategyInterface]: Список стратегий
        """
        strategy_ids = self._pair_index.get(trading_pair, set())
        return [
            self._strategies[sid] for sid in strategy_ids if sid in self._strategies
        ]

    def get_strategies_by_tag(self, tag: str) -> List[StrategyInterface]:
        """
        Получить стратегии по тегу.
        Args:
            tag: Тег
        Returns:
            List[StrategyInterface]: Список стратегий
        """
        strategy_ids = self._tag_index.get(tag, set())
        return [
            self._strategies[sid] for sid in strategy_ids if sid in self._strategies
        ]

    def search_strategies(
        self,
        name_pattern: Optional[str] = None,
        strategy_type: Optional[StrategyType] = None,
        status: Optional[StrategyStatus] = None,
        trading_pair: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_priority: Optional[int] = None,
        max_priority: Optional[int] = None,
        active_only: bool = False,
    ) -> List[StrategyInterface]:
        """
        Поиск стратегий по критериям.
        Args:
            name_pattern: Паттерн имени
            strategy_type: Тип стратегии
            status: Статус стратегии
            trading_pair: Торговая пара
            tags: Список тегов
            min_priority: Минимальный приоритет
            max_priority: Максимальный приоритет
            active_only: Только активные стратегии
        Returns:
            List[StrategyInterface]: Список найденных стратегий
        """
        results = []
        for strategy_id, strategy in self._strategies.items():
            metadata = self._metadata[strategy_id]
            # Фильтр по имени
            if name_pattern and name_pattern.lower() not in metadata.name.lower():
                continue
            # Фильтр по типу
            if strategy_type and strategy.get_strategy_type() != strategy_type:
                continue
            # Фильтр по статусу
            if status and metadata.status != status:
                continue
            # Фильтр по торговой паре
            if trading_pair and trading_pair not in [
                str(pair) for pair in metadata.trading_pairs
            ]:
                continue
            # Фильтр по тегам
            if tags and not any(tag in metadata.tags for tag in tags):
                continue
            # Фильтр по приоритету
            if min_priority is not None and metadata.priority < min_priority:
                continue
            if max_priority is not None and metadata.priority > max_priority:
                continue
            # Фильтр по активности
            if active_only and not metadata.is_active:
                continue
            results.append(strategy)
        return results

    def update_strategy_status(
        self, strategy_id: StrategyId, status: StrategyStatus
    ) -> bool:
        """
        Обновить статус стратегии.
        Args:
            strategy_id: ID стратегии
            status: Новый статус
        Returns:
            bool: True если статус обновлен
        """
        if strategy_id not in self._strategies:
            return False
        try:
            metadata = self._metadata[strategy_id]
            old_status = metadata.status
            # Обновляем статус
            metadata.status = status
            metadata.updated_at = datetime.now()
            # Обновляем индексы
            if old_status in self._status_index:
                self._status_index[old_status].discard(strategy_id)
            if status not in self._status_index:
                self._status_index[status] = set()
            self._status_index[status].add(strategy_id)
            # Обновляем статистику
            self._update_stats_on_status_change(old_status, status)
            logger.info(
                f"Updated strategy {strategy_id} status: {old_status.value} -> {status.value}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update strategy status: {e}")
            return False

    def update_strategy_performance(
        self,
        strategy_id: StrategyId,
        execution_time: float,
        success: bool,
        pnl: Decimal,
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Обновить производительность стратегии.
        Args:
            strategy_id: ID стратегии
            execution_time: Время выполнения
            success: Успешность выполнения
            pnl: Прибыль/убыток
            error_message: Сообщение об ошибке
        Returns:
            bool: True если производительность обновлена
        """
        if strategy_id not in self._strategies:
            return False
        try:
            metadata = self._metadata[strategy_id]
            # Обновляем метаданные
            metadata.execution_count += 1
            metadata.last_execution = datetime.now()
            metadata.updated_at = datetime.now()
            if success:
                metadata.success_count += 1
                metadata.total_pnl += pnl
            else:
                metadata.error_count += 1
                metadata.last_error = error_message
            # Обновляем среднее время выполнения
            if metadata.execution_count == 1:
                metadata.avg_execution_time = execution_time
            else:
                metadata.avg_execution_time = (
                    metadata.avg_execution_time * (metadata.execution_count - 1)
                    + execution_time
                ) / metadata.execution_count
            # Обновляем статистику реестра
            self._update_stats_on_execution(success, pnl, execution_time)
            return True
        except Exception as e:
            logger.error(f"Failed to update strategy performance: {e}")
            return False

    def get_strategy_metadata(
        self, strategy_id: StrategyId
    ) -> Optional[StrategyMetadata]:
        """
        Получить метаданные стратегии.
        Args:
            strategy_id: ID стратегии
        Returns:
            Optional[StrategyMetadata]: Метаданные стратегии
        """
        return self._metadata.get(strategy_id)

    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Получить статистику реестра.
        Returns:
            Dict[str, Any]: Статистика реестра
        """
        return {
            "total_strategies": self._stats.total_strategies,
            "active_strategies": self._stats.active_strategies,
            "paused_strategies": self._stats.paused_strategies,
            "stopped_strategies": self._stats.stopped_strategies,
            "error_strategies": self._stats.error_strategies,
            "total_executions": self._stats.total_executions,
            "successful_executions": self._stats.successful_executions,
            "failed_executions": self._stats.failed_executions,
            "total_pnl": str(self._stats.total_pnl),
            "avg_execution_time": self._stats.avg_execution_time,
            "most_profitable_strategy": self._stats.most_profitable_strategy,
            "most_active_strategy": self._stats.most_active_strategy,
            "strategy_type_distribution": self._stats.strategy_type_distribution,
            "risk_level_distribution": self._stats.risk_level_distribution,
            "last_update": self._stats.last_update.isoformat(),
        }

    def get_top_performers(self, limit: int = 10) -> List[Tuple[str, Decimal]]:
        """
        Получить топ-производителей.
        Args:
            limit: Количество стратегий
        Returns:
            List[Tuple[str, Decimal]]: Список (имя, PnL)
        """
        performers = []
        for strategy_id, metadata in self._metadata.items():
            if metadata.total_pnl != 0:
                performers.append((metadata.name, metadata.total_pnl))
        performers.sort(key=lambda x: x[1], reverse=True)
        return performers[:limit]

    def get_most_active_strategies(self, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Получить самые активные стратегии.
        Args:
            limit: Количество стратегий
        Returns:
            List[Tuple[str, int]]: Список (имя, количество выполнений)
        """
        active = []
        for strategy_id, metadata in self._metadata.items():
            if metadata.execution_count > 0:
                active.append((metadata.name, metadata.execution_count))
        active.sort(key=lambda x: x[1], reverse=True)
        return active[:limit]

    def cleanup_inactive_strategies(self, days_threshold: int = 30) -> int:
        """
        Очистить неактивные стратегии.
        Args:
            days_threshold: Порог дней неактивности
        Returns:
            int: Количество удаленных стратегий
        """
        if self._lock:
            raise StrategyRegistryError("Registry is locked")
        threshold_date = datetime.now() - timedelta(days=days_threshold)
        removed_count = 0
        strategy_ids_to_remove = []
        for strategy_id, metadata in self._metadata.items():
            if (
                metadata.last_execution is None
                or metadata.last_execution < threshold_date
            ):
                strategy_ids_to_remove.append(strategy_id)
        for strategy_id in strategy_ids_to_remove:
            if self.unregister_strategy(strategy_id):
                removed_count += 1
        logger.info(f"Cleaned up {removed_count} inactive strategies")
        return removed_count

    def lock_registry(self) -> None:
        """Заблокировать реестр."""
        self._lock = True
        logger.info("Registry locked")

    def unlock_registry(self) -> None:
        """Разблокировать реестр."""
        self._lock = False
        logger.info("Registry unlocked")

    def _update_indexes(
        self, strategy_id: StrategyId, metadata: StrategyMetadata, is_new: bool = False
    ) -> None:
        """Обновить индексы."""
        # Индекс по имени
        self._name_index[metadata.name] = strategy_id
        # Индекс по типу
        if metadata.strategy_type not in self._type_index:
            self._type_index[metadata.strategy_type] = set()
        self._type_index[metadata.strategy_type].add(strategy_id)
        # Индекс по статусу
        if metadata.status not in self._status_index:
            self._status_index[metadata.status] = set()
        self._status_index[metadata.status].add(strategy_id)
        # Индекс по торговым парам
        for pair in metadata.trading_pairs:
            pair_str = str(pair)
            if pair_str not in self._pair_index:
                self._pair_index[pair_str] = set()
            self._pair_index[pair_str].add(strategy_id)
        # Индекс по тегам
        for tag in metadata.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(strategy_id)

    def _remove_from_indexes(
        self, strategy_id: StrategyId, metadata: StrategyMetadata
    ) -> None:
        """Удалить из индексов."""
        # Удаляем из индекса по имени
        if metadata.name in self._name_index:
            del self._name_index[metadata.name]
        # Удаляем из индекса по типу
        if metadata.strategy_type in self._type_index:
            self._type_index[metadata.strategy_type].discard(strategy_id)
        # Удаляем из индекса по статусу
        if metadata.status in self._status_index:
            self._status_index[metadata.status].discard(strategy_id)
        # Удаляем из индекса по торговым парам
        for pair in metadata.trading_pairs:
            pair_str = str(pair)
            if pair_str in self._pair_index:
                self._pair_index[pair_str].discard(strategy_id)
        # Удаляем из индекса по тегам
        for tag in metadata.tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(strategy_id)

    def _update_stats_on_register(self, metadata: StrategyMetadata) -> None:
        """Обновить статистику при регистрации."""
        self._stats.total_strategies += 1
        # Обновляем распределение по типам
        type_name = metadata.strategy_type.value
        self._stats.strategy_type_distribution[type_name] = (
            self._stats.strategy_type_distribution.get(type_name, 0) + 1
        )
        # Обновляем распределение по уровням риска
        self._stats.risk_level_distribution[metadata.risk_level] = (
            self._stats.risk_level_distribution.get(metadata.risk_level, 0) + 1
        )
        self._stats.last_update = datetime.now()

    def _update_stats_on_unregister(self, metadata: StrategyMetadata) -> None:
        """Обновить статистику при отмене регистрации."""
        self._stats.total_strategies -= 1
        # Обновляем распределение по типам
        type_name = metadata.strategy_type.value
        if type_name in self._stats.strategy_type_distribution:
            self._stats.strategy_type_distribution[type_name] -= 1
            if self._stats.strategy_type_distribution[type_name] <= 0:
                del self._stats.strategy_type_distribution[type_name]
        # Обновляем распределение по уровням риска
        if metadata.risk_level in self._stats.risk_level_distribution:
            self._stats.risk_level_distribution[metadata.risk_level] -= 1
            if self._stats.risk_level_distribution[metadata.risk_level] <= 0:
                del self._stats.risk_level_distribution[metadata.risk_level]
        self._stats.last_update = datetime.now()

    def _update_stats_on_status_change(
        self, old_status: StrategyStatus, new_status: StrategyStatus
    ) -> None:
        """Обновить статистику при изменении статуса."""
        # Уменьшаем счетчик старого статуса
        if old_status == StrategyStatus.ACTIVE:
            self._stats.active_strategies -= 1
        elif old_status == StrategyStatus.PAUSED:
            self._stats.paused_strategies -= 1
        elif old_status == StrategyStatus.STOPPED:
            self._stats.stopped_strategies -= 1
        elif old_status == StrategyStatus.ERROR:
            self._stats.error_strategies -= 1
        # Увеличиваем счетчик нового статуса
        if new_status == StrategyStatus.ACTIVE:
            self._stats.active_strategies += 1
        elif new_status == StrategyStatus.PAUSED:
            self._stats.paused_strategies += 1
        elif new_status == StrategyStatus.STOPPED:
            self._stats.stopped_strategies += 1
        elif new_status == StrategyStatus.ERROR:
            self._stats.error_strategies += 1
        self._stats.last_update = datetime.now()

    def _update_stats_on_execution(
        self, success: bool, pnl: Decimal, execution_time: float
    ) -> None:
        """Обновить статистику при выполнении."""
        self._stats.total_executions += 1
        if success:
            self._stats.successful_executions += 1
            self._stats.total_pnl += pnl
        else:
            self._stats.failed_executions += 1
        # Обновляем среднее время выполнения
        if self._stats.total_executions == 1:
            self._stats.avg_execution_time = execution_time
        else:
            self._stats.avg_execution_time = (
                self._stats.avg_execution_time * (self._stats.total_executions - 1)
                + execution_time
            ) / self._stats.total_executions
        self._stats.last_update = datetime.now()


# Глобальный экземпляр реестра
_registry_instance: Optional[StrategyRegistry] = None


def get_strategy_registry() -> StrategyRegistry:
    """
    Получить глобальный экземпляр реестра стратегий.
    Returns:
        StrategyRegistry: Глобальный экземпляр реестра
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = StrategyRegistry()
    return _registry_instance
