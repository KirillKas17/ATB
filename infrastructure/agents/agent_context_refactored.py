"""
Рефакторированный контекст агента с полной реализацией всех методов.
Создан для обеспечения эффективного управления состоянием агентов и их модификаторов.
"""
import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Union, Type
import logging
from collections import defaultdict
import weakref
import threading
import json
import pickle
import copy

logger = logging.getLogger(__name__)

class ModifierType(Enum):
    """Типы модификаторов контекста."""
    BEHAVIOR = "behavior"
    STRATEGY = "strategy"
    RISK = "risk"
    PERFORMANCE = "performance"
    MEMORY = "memory"
    LEARNING = "learning"
    ADAPTATION = "adaptation"
    FILTER = "filter"

class ContextState(Enum):
    """Состояния контекста агента."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEGRADED = "degraded"
    TERMINATED = "terminated"
    ERROR = "error"

class Priority(Enum):
    """Приоритеты модификаторов."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5

@dataclass
class ContextMetadata:
    """Метаданные контекста."""
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    version: int = 1
    tags: Set[str] = field(default_factory=set)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def update(self) -> None:
        """Обновление времени модификации и версии."""
        self.last_modified = datetime.now()
        self.version += 1

@dataclass
class ModifierConfig:
    """Конфигурация модификатора."""
    enabled: bool = True
    priority: Priority = Priority.MEDIUM
    timeout_seconds: Optional[float] = None
    retry_count: int = 0
    conditions: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)

class ContextObserver(ABC):
    """Интерфейс наблюдателя за изменениями контекста."""
    
    @abstractmethod
    async def on_context_changed(self, context: "AgentContext", change_type: str, data: Any) -> None:
        """Вызывается при изменении контекста."""
        pass
    
    @abstractmethod
    async def on_modifier_applied(self, context: "AgentContext", modifier: "ContextModifier") -> None:
        """Вызывается при применении модификатора."""
        pass

class ContextModifier(ABC):
    """Базовый класс для модификаторов контекста агента."""
    
    def __init__(
        self, 
        modifier_id: Optional[str] = None,
        modifier_type: ModifierType = ModifierType.BEHAVIOR,
        config: Optional[ModifierConfig] = None
    ):
        self.modifier_id = modifier_id or str(uuid.uuid4())
        self.modifier_type = modifier_type
        self.config = config or ModifierConfig()
        self.created_at = datetime.now()
        self.application_count = 0
        self.last_applied = None
        self.errors: List[Exception] = []
        self._metadata = ContextMetadata()
    
    @abstractmethod
    def apply_modifier(self, context: "AgentContext") -> None:
        """Применение модификатора к контексту."""
        pass
    
    def is_applicable(self, context: "AgentContext") -> bool:
        """Проверка применимости модификатора."""
        if not self.config.enabled:
            return False
        
        # Проверка условий применения
        for condition_key, condition_value in self.config.conditions.items():
            context_value = context.get_property(condition_key)
            if not self._check_condition(context_value, condition_value):
                return False
        
        return True
    
    def _check_condition(self, context_value: Any, condition_value: Any) -> bool:
        """Проверка конкретного условия."""
        if isinstance(condition_value, dict):
            # Поддержка операторов сравнения
            if 'gt' in condition_value:
                return context_value > condition_value['gt']
            elif 'lt' in condition_value:
                return context_value < condition_value['lt']
            elif 'eq' in condition_value:
                return context_value == condition_value['eq']
            elif 'in' in condition_value:
                return context_value in condition_value['in']
        
        return context_value == condition_value
    
    async def apply_with_retry(self, context: "AgentContext") -> bool:
        """Применение модификатора с повторными попытками."""
        for attempt in range(self.config.retry_count + 1):
            try:
                if self.config.timeout_seconds:
                    await asyncio.wait_for(
                        self._apply_async(context),
                        timeout=self.config.timeout_seconds
                    )
                else:
                    await self._apply_async(context)
                
                self.application_count += 1
                self.last_applied = datetime.now()
                return True
                
            except Exception as e:
                self.errors.append(e)
                logger.warning(f"Попытка {attempt + 1} применения модификатора {self.modifier_id} неудачна: {e}")
                
                if attempt < self.config.retry_count:
                    await asyncio.sleep(0.1 * (2 ** attempt))  # Экспоненциальная задержка
        
        return False
    
    async def _apply_async(self, context: "AgentContext") -> None:
        """Асинхронное применение модификатора."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.apply_modifier, context)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики модификатора."""
        return {
            'modifier_id': self.modifier_id,
            'modifier_type': self.modifier_type.value,
            'application_count': self.application_count,
            'last_applied': self.last_applied.isoformat() if self.last_applied else None,
            'error_count': len(self.errors),
            'enabled': self.config.enabled,
            'priority': self.config.priority.value,
            'created_at': self.created_at.isoformat()
        }

class BehaviorModifier(ContextModifier):
    """Модификатор поведения агента."""
    
    def __init__(self, behavior_changes: Dict[str, Any], **kwargs):
        super().__init__(modifier_type=ModifierType.BEHAVIOR, **kwargs)
        self.behavior_changes = behavior_changes
    
    def apply_modifier(self, context: "AgentContext") -> None:
        """Применение изменений поведения."""
        try:
            for key, value in self.behavior_changes.items():
                if isinstance(value, dict) and key in context.data:
                    # Глубокое обновление словарей
                    if isinstance(context.data[key], dict):
                        context.data[key].update(value)
                    else:
                        context.data[key] = value
                else:
                    context.set_property(key, value)
            
            context._metadata.update()
            logger.debug(f"Применён модификатор поведения {self.modifier_id}")
            
        except Exception as e:
            logger.error(f"Ошибка применения модификатора поведения {self.modifier_id}: {e}")
            raise

class StrategyModifier(ContextModifier):
    """Модификатор стратегии."""
    
    def __init__(self, strategy_adjustments: Dict[str, Any], **kwargs):
        super().__init__(modifier_type=ModifierType.STRATEGY, **kwargs)
        self.strategy_adjustments = strategy_adjustments
    
    def apply_modifier(self, context: "AgentContext") -> None:
        """Применение стратегических модификаций."""
        try:
            strategy_section = context.data.setdefault('strategy', {})
            
            for key, value in self.strategy_adjustments.items():
                if key == 'parameters' and isinstance(value, dict):
                    strategy_params = strategy_section.setdefault('parameters', {})
                    strategy_params.update(value)
                elif key == 'risk_limits' and isinstance(value, dict):
                    risk_limits = strategy_section.setdefault('risk_limits', {})
                    risk_limits.update(value)
                else:
                    strategy_section[key] = value
            
            context._metadata.update()
            logger.debug(f"Применён модификатор стратегии {self.modifier_id}")
            
        except Exception as e:
            logger.error(f"Ошибка применения модификатора стратегии {self.modifier_id}: {e}")
            raise

class PerformanceModifier(ContextModifier):
    """Модификатор производительности."""
    
    def __init__(self, performance_settings: Dict[str, Any], **kwargs):
        super().__init__(modifier_type=ModifierType.PERFORMANCE, **kwargs)
        self.performance_settings = performance_settings
    
    def apply_modifier(self, context: "AgentContext") -> None:
        """Применение настроек производительности."""
        try:
            perf_section = context.data.setdefault('performance', {})
            
            for key, value in self.performance_settings.items():
                if key == 'optimization_level':
                    perf_section['optimization_level'] = max(0, min(10, value))
                elif key == 'memory_limit_mb':
                    perf_section['memory_limit_mb'] = max(64, value)
                elif key == 'cpu_allocation':
                    perf_section['cpu_allocation'] = max(0.1, min(1.0, value))
                else:
                    perf_section[key] = value
            
            context._metadata.update()
            logger.debug(f"Применён модификатор производительности {self.modifier_id}")
            
        except Exception as e:
            logger.error(f"Ошибка применения модификатора производительности {self.modifier_id}: {e}")
            raise

class AdaptiveModifier(ContextModifier):
    """Адаптивный модификатор с обучением."""
    
    def __init__(self, adaptation_rules: Dict[str, Any], **kwargs):
        super().__init__(modifier_type=ModifierType.ADAPTATION, **kwargs)
        self.adaptation_rules = adaptation_rules
        self.adaptation_history: List[Dict[str, Any]] = []
        self.learning_rate = 0.1
    
    def apply_modifier(self, context: "AgentContext") -> None:
        """Применение адаптивных изменений."""
        try:
            # Анализ текущего состояния
            current_state = self._analyze_context_state(context)
            
            # Применение правил адаптации
            adaptations = self._calculate_adaptations(current_state)
            
            # Применение изменений
            for key, value in adaptations.items():
                context.set_property(key, value)
            
            # Сохранение истории для обучения
            self.adaptation_history.append({
                'timestamp': datetime.now(),
                'state': current_state,
                'adaptations': adaptations
            })
            
            # Ограничение истории
            if len(self.adaptation_history) > 1000:
                self.adaptation_history = self.adaptation_history[-500:]
            
            context._metadata.update()
            logger.debug(f"Применён адаптивный модификатор {self.modifier_id}")
            
        except Exception as e:
            logger.error(f"Ошибка применения адаптивного модификатора {self.modifier_id}: {e}")
            raise
    
    def _analyze_context_state(self, context: "AgentContext") -> Dict[str, Any]:
        """Анализ текущего состояния контекста."""
        return {
            'performance_score': context.get_property('performance.score', 0.5),
            'error_rate': context.get_property('errors.rate', 0.0),
            'resource_usage': context.get_property('resources.usage', 0.5),
            'adaptation_count': len(self.adaptation_history)
        }
    
    def _calculate_adaptations(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Расчёт необходимых адаптаций."""
        adaptations = {}
        
        # Адаптация на основе производительности
        if state['performance_score'] < 0.3:
            adaptations['performance.optimization_level'] = min(10, 
                adaptations.get('performance.optimization_level', 5) + 1)
        elif state['performance_score'] > 0.8:
            adaptations['performance.optimization_level'] = max(1,
                adaptations.get('performance.optimization_level', 5) - 1)
        
        # Адаптация на основе ошибок
        if state['error_rate'] > 0.1:
            adaptations['strategy.conservative_mode'] = True
            adaptations['risk.tolerance'] = max(0.1,
                adaptations.get('risk.tolerance', 0.5) - 0.1)
        
        return adaptations

class AgentContext:
    """Продвинутый контекст агента с полной функциональностью."""
    
    def __init__(self, agent_id: str, initial_data: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id
        self.context_id = str(uuid.uuid4())
        self.data = initial_data or {}
        self.state = ContextState.INITIALIZING
        self._metadata = ContextMetadata()
        
        # Управление модификаторами
        self._modifiers: Dict[str, ContextModifier] = {}
        self._modifier_order: List[str] = []
        self._modifier_lock = asyncio.Lock()
        
        # Система наблюдателей
        self._observers: Set[ContextObserver] = set()
        
        # Кэширование и оптимизация
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, datetime] = {}
        
        # История изменений
        self._change_history: List[Dict[str, Any]] = []
        self._max_history_size = 1000
        
        # Валидация и безопасность
        self._validators: List[Callable[[str, Any], bool]] = []
        self._read_only_keys: Set[str] = set()
        
        # Метрики
        self._metrics = {
            'get_count': 0,
            'set_count': 0,
            'modifier_applications': 0,
            'validation_failures': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        self.state = ContextState.ACTIVE
        logger.info(f"Инициализирован контекст агента {agent_id} с ID {self.context_id}")
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """Получение свойства с поддержкой вложенных ключей и кэширования."""
        self._metrics['get_count'] += 1
        
        # Проверка кэша
        if key in self._cache:
            if key in self._cache_ttl and datetime.now() < self._cache_ttl[key]:
                self._metrics['cache_hits'] += 1
                return self._cache[key]
            else:
                # Истёкший кэш
                self._cache.pop(key, None)
                self._cache_ttl.pop(key, None)
        
        self._metrics['cache_misses'] += 1
        
        # Поддержка вложенных ключей (например, "strategy.parameters.risk_level")
        keys = key.split('.')
        value = self.data
        
        try:
            for k in keys:
                if isinstance(value, dict):
                    value = value[k]
                else:
                    return default
            
            # Кэширование результата на 60 секунд
            self._cache[key] = value
            self._cache_ttl[key] = datetime.now() + timedelta(seconds=60)
            
            return value
        except (KeyError, TypeError):
            return default
    
    def set_property(self, key: str, value: Any, validate: bool = True) -> bool:
        """Установка свойства с валидацией и уведомлениями."""
        self._metrics['set_count'] += 1
        
        # Проверка блокировки
        if key in self._read_only_keys:
            logger.warning(f"Попытка изменения read-only свойства {key}")
            return False
        
        # Валидация
        if validate:
            for validator in self._validators:
                if not validator(key, value):
                    self._metrics['validation_failures'] += 1
                    logger.warning(f"Валидация не пройдена для {key} = {value}")
                    return False
        
        # Сохранение старого значения для истории
        old_value = self.get_property(key)
        
        # Установка значения с поддержкой вложенных ключей
        keys = key.split('.')
        target = self.data
        
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        target[keys[-1]] = value
        
        # Очистка кэша
        self._invalidate_cache_for_key(key)
        
        # Сохранение в истории изменений
        self._add_to_history('property_changed', {
            'key': key,
            'old_value': old_value,
            'new_value': value,
            'timestamp': datetime.now()
        })
        
        # Уведомление наблюдателей
        asyncio.create_task(self._notify_observers('property_changed', {
            'key': key,
            'old_value': old_value,
            'new_value': value
        }))
        
        self._metadata.update()
        return True
    
    def _invalidate_cache_for_key(self, key: str) -> None:
        """Инвалидация кэша для ключа и связанных ключей."""
        keys_to_remove = []
        key_prefix = key.split('.')[0]
        
        for cached_key in self._cache.keys():
            if cached_key.startswith(key_prefix):
                keys_to_remove.append(cached_key)
        
        for k in keys_to_remove:
            self._cache.pop(k, None)
            self._cache_ttl.pop(k, None)
    
    async def add_modifier(self, modifier: ContextModifier) -> bool:
        """Добавление модификатора с автоматической сортировкой по приоритету."""
        async with self._modifier_lock:
            try:
                if modifier.modifier_id in self._modifiers:
                    logger.warning(f"Модификатор {modifier.modifier_id} уже существует")
                    return False
                
                self._modifiers[modifier.modifier_id] = modifier
                
                # Вставка в правильную позицию по приоритету
                inserted = False
                for i, existing_id in enumerate(self._modifier_order):
                    existing_modifier = self._modifiers[existing_id]
                    if modifier.config.priority.value < existing_modifier.config.priority.value:
                        self._modifier_order.insert(i, modifier.modifier_id)
                        inserted = True
                        break
                
                if not inserted:
                    self._modifier_order.append(modifier.modifier_id)
                
                logger.info(f"Добавлен модификатор {modifier.modifier_id} типа {modifier.modifier_type.value}")
                return True
                
            except Exception as e:
                logger.error(f"Ошибка добавления модификатора: {e}")
                return False
    
    async def remove_modifier(self, modifier_id: str) -> bool:
        """Удаление модификатора."""
        async with self._modifier_lock:
            if modifier_id in self._modifiers:
                del self._modifiers[modifier_id]
                if modifier_id in self._modifier_order:
                    self._modifier_order.remove(modifier_id)
                logger.info(f"Удалён модификатор {modifier_id}")
                return True
            return False
    
    async def apply_modifiers(self, modifier_type: Optional[ModifierType] = None) -> Dict[str, bool]:
        """Применение всех или определённого типа модификаторов."""
        results = {}
        applied_count = 0
        
        async with self._modifier_lock:
            modifiers_to_apply = []
            
            for modifier_id in self._modifier_order:
                modifier = self._modifiers[modifier_id]
                
                # Фильтрация по типу если указан
                if modifier_type is None or modifier.modifier_type == modifier_type:
                    if modifier.is_applicable(self):
                        modifiers_to_apply.append(modifier)
            
            # Применение модификаторов
            for modifier in modifiers_to_apply:
                try:
                    success = await modifier.apply_with_retry(self)
                    results[modifier.modifier_id] = success
                    
                    if success:
                        applied_count += 1
                        self._metrics['modifier_applications'] += 1
                        
                        # Уведомление наблюдателей
                        await self._notify_observers('modifier_applied', {
                            'modifier_id': modifier.modifier_id,
                            'modifier_type': modifier.modifier_type.value
                        })
                    
                except Exception as e:
                    logger.error(f"Ошибка применения модификатора {modifier.modifier_id}: {e}")
                    results[modifier.modifier_id] = False
        
        logger.debug(f"Применено {applied_count} из {len(modifiers_to_apply)} модификаторов")
        return results
    
    def add_observer(self, observer: ContextObserver) -> None:
        """Добавление наблюдателя."""
        self._observers.add(observer)
        logger.debug(f"Добавлен наблюдатель {type(observer).__name__}")
    
    def remove_observer(self, observer: ContextObserver) -> None:
        """Удаление наблюдателя."""
        self._observers.discard(observer)
        logger.debug(f"Удалён наблюдатель {type(observer).__name__}")
    
    async def _notify_observers(self, change_type: str, data: Any) -> None:
        """Уведомление всех наблюдателей."""
        for observer in self._observers.copy():  # Копия для безопасности
            try:
                await observer.on_context_changed(self, change_type, data)
            except Exception as e:
                logger.error(f"Ошибка в наблюдателе {type(observer).__name__}: {e}")
    
    def add_validator(self, validator: Callable[[str, Any], bool]) -> None:
        """Добавление валидатора."""
        self._validators.append(validator)
    
    def set_read_only(self, key: str) -> None:
        """Установка ключа как read-only."""
        self._read_only_keys.add(key)
    
    def _add_to_history(self, action_type: str, data: Dict[str, Any]) -> None:
        """Добавление записи в историю изменений."""
        self._change_history.append({
            'action_type': action_type,
            'timestamp': datetime.now(),
            'data': data
        })
        
        # Ограничение размера истории
        if len(self._change_history) > self._max_history_size:
            self._change_history = self._change_history[-self._max_history_size//2:]
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Получение истории изменений."""
        if limit:
            return self._change_history[-limit:]
        return self._change_history.copy()
    
    def get_modifiers_info(self) -> List[Dict[str, Any]]:
        """Получение информации о всех модификаторах."""
        return [modifier.get_statistics() for modifier in self._modifiers.values()]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Получение метрик контекста."""
        return {
            **self._metrics,
            'cache_size': len(self._cache),
            'modifiers_count': len(self._modifiers),
            'observers_count': len(self._observers),
            'history_size': len(self._change_history),
            'state': self.state.value,
            'uptime_seconds': (datetime.now() - self._metadata.created_at).total_seconds()
        }
    
    def create_snapshot(self) -> Dict[str, Any]:
        """Создание снимка состояния контекста."""
        return {
            'agent_id': self.agent_id,
            'context_id': self.context_id,
            'data': copy.deepcopy(self.data),
            'state': self.state.value,
            'metadata': {
                'created_at': self._metadata.created_at.isoformat(),
                'last_modified': self._metadata.last_modified.isoformat(),
                'version': self._metadata.version
            },
            'modifiers': [modifier.get_statistics() for modifier in self._modifiers.values()],
            'metrics': self.get_metrics()
        }
    
    async def restore_from_snapshot(self, snapshot: Dict[str, Any]) -> bool:
        """Восстановление состояния из снимка."""
        try:
            self.data = snapshot['data']
            self.state = ContextState(snapshot['state'])
            
            # Очистка кэша после восстановления
            self._cache.clear()
            self._cache_ttl.clear()
            
            self._metadata.update()
            
            logger.info(f"Восстановлен контекст из снимка версии {snapshot.get('metadata', {}).get('version', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка восстановления из снимка: {e}")
            self.state = ContextState.ERROR
            return False
    
    async def cleanup(self) -> None:
        """Очистка ресурсов контекста."""
        try:
            # Остановка всех модификаторов
            async with self._modifier_lock:
                self._modifiers.clear()
                self._modifier_order.clear()
            
            # Очистка наблюдателей
            self._observers.clear()
            
            # Очистка кэша
            self._cache.clear()
            self._cache_ttl.clear()
            
            # Очистка истории
            self._change_history.clear()
            
            self.state = ContextState.TERMINATED
            logger.info(f"Контекст {self.context_id} успешно очищен")
            
        except Exception as e:
            logger.error(f"Ошибка при очистке контекста: {e}")
            self.state = ContextState.ERROR

# Фабрика для создания стандартных модификаторов
class ModifierFactory:
    """Фабрика для создания стандартных модификаторов."""
    
    @staticmethod
    def create_performance_booster(optimization_level: int = 7) -> PerformanceModifier:
        """Создание модификатора повышения производительности."""
        return PerformanceModifier(
            performance_settings={
                'optimization_level': optimization_level,
                'memory_limit_mb': 512,
                'cpu_allocation': 0.8,
                'cache_enabled': True
            },
            config=ModifierConfig(priority=Priority.HIGH)
        )
    
    @staticmethod
    def create_risk_reducer(risk_tolerance: float = 0.3) -> BehaviorModifier:
        """Создание модификатора снижения рисков."""
        return BehaviorModifier(
            behavior_changes={
                'risk.tolerance': risk_tolerance,
                'strategy.conservative_mode': True,
                'validation.strict_mode': True
            },
            config=ModifierConfig(priority=Priority.CRITICAL)
        )
    
    @staticmethod
    def create_adaptive_learner(learning_rate: float = 0.1) -> AdaptiveModifier:
        """Создание адаптивного модификатора с обучением."""
        return AdaptiveModifier(
            adaptation_rules={
                'performance_threshold': 0.7,
                'error_threshold': 0.05,
                'adaptation_frequency': 100
            },
            config=ModifierConfig(priority=Priority.MEDIUM)
        )

# Менеджер контекстов для управления множественными агентами
class ContextManager:
    """Менеджер для управления контекстами множественных агентов."""
    
    def __init__(self):
        self._contexts: Dict[str, AgentContext] = {}
        self._context_lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
    
    async def create_context(self, agent_id: str, initial_data: Optional[Dict[str, Any]] = None) -> AgentContext:
        """Создание нового контекста агента."""
        async with self._context_lock:
            if agent_id in self._contexts:
                logger.warning(f"Контекст для агента {agent_id} уже существует")
                return self._contexts[agent_id]
            
            context = AgentContext(agent_id, initial_data)
            self._contexts[agent_id] = context
            logger.info(f"Создан контекст для агента {agent_id}")
            return context
    
    async def get_context(self, agent_id: str) -> Optional[AgentContext]:
        """Получение контекста агента."""
        async with self._context_lock:
            return self._contexts.get(agent_id)
    
    async def remove_context(self, agent_id: str) -> bool:
        """Удаление контекста агента."""
        async with self._context_lock:
            if agent_id in self._contexts:
                context = self._contexts[agent_id]
                await context.cleanup()
                del self._contexts[agent_id]
                logger.info(f"Удалён контекст агента {agent_id}")
                return True
            return False
    
    async def get_all_contexts(self) -> Dict[str, AgentContext]:
        """Получение всех контекстов."""
        async with self._context_lock:
            return self._contexts.copy()
    
    def _start_cleanup_task(self) -> None:
        """Запуск задачи очистки неактивных контекстов."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(300)  # Проверка каждые 5 минут
                    await self._cleanup_inactive_contexts()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Ошибка в процессе очистки контекстов: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def _cleanup_inactive_contexts(self) -> None:
        """Очистка неактивных контекстов."""
        current_time = datetime.now()
        inactive_threshold = timedelta(hours=24)  # 24 часа неактивности
        
        async with self._context_lock:
            contexts_to_remove = []
            
            for agent_id, context in self._contexts.items():
                if (current_time - context._metadata.last_modified) > inactive_threshold:
                    if context.state in [ContextState.SUSPENDED, ContextState.ERROR]:
                        contexts_to_remove.append(agent_id)
            
            for agent_id in contexts_to_remove:
                await self.remove_context(agent_id)
            
            if contexts_to_remove:
                logger.info(f"Очищено {len(contexts_to_remove)} неактивных контекстов")

# Глобальный менеджер контекстов
context_manager = ContextManager()
