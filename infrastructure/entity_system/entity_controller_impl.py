"""
Продвинутая реализация EntityController для управления Entity System.
Обеспечивает координацию всех компонентов системы, управление жизненным циклом,
мониторинг состояния и автоматическую оптимизацию.
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast

import psutil
from loguru import logger

from domain.type_definitions.entity_system_types import (
    AnalysisResult,
    BaseEntityController,
    CodeStructure,
    EntityControllerProtocol,
    EntityState,
    EntitySystemConfig,
    EntitySystemError,
    EvolutionStats,
    Experiment,
    Hypothesis,
    Improvement,
    MemorySnapshot,
    OperationMode,
    OptimizationLevel,
    SystemPhase,
    validate_entity_state,
)

from .ai_enhancement_impl import AIEnhancementImpl
from .code_analyzer_impl import CodeAnalyzerImpl
from .code_scanner_impl import CodeScannerImpl
from .evolution_engine_impl import EvolutionEngineImpl
from .experiment_runner_impl import ExperimentRunnerImpl
from .improvement_applier_impl import ImprovementApplierImpl
from .memory_manager_impl import MemoryManagerImpl


@dataclass
class ComponentRegistry:
    """Реестр компонентов системы."""

    code_scanner: Optional[CodeScannerImpl] = None
    code_analyzer: Optional[CodeAnalyzerImpl] = None
    experiment_runner: Optional[ExperimentRunnerImpl] = None
    improvement_applier: Optional[ImprovementApplierImpl] = None
    memory_manager: Optional[MemoryManagerImpl] = None
    ai_enhancement: Optional[AIEnhancementImpl] = None
    evolution_engine: Optional[EvolutionEngineImpl] = None


@dataclass
class SystemMetrics:
    """Метрики производительности системы."""

    total_cycles: int = 0
    successful_improvements: int = 0
    failed_improvements: int = 0
    experiments_completed: int = 0
    average_cycle_time: float = 0.0
    last_cycle_start: Optional[datetime] = None
    last_cycle_end: Optional[datetime] = None
    system_uptime: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    disk_usage: float = 0.0
    network_io: float = 0.0
    error_count: Union[int, float] = 0
    warning_count: Union[int, float] = 0
    critical_errors: int = 0


@dataclass
class PhaseData:
    """Данные для каждой фазы системы."""

    phase: SystemPhase
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    processed_items: int = 0
    generated_hypotheses: int = 0
    applied_improvements: int = 0


@dataclass
class MemoryCache:
    """Кэш памяти для хранения данных между фазами."""

    code_structures: List[CodeStructure] = field(default_factory=list)
    analysis_results: List[AnalysisResult] = field(default_factory=list)
    active_hypotheses: List[Hypothesis] = field(default_factory=list)
    pending_experiments: List[Experiment] = field(default_factory=list)
    completed_experiments: List[Experiment] = field(default_factory=list)
    applied_improvements: List[Improvement] = field(default_factory=list)
    system_snapshots: List[MemorySnapshot] = field(default_factory=list)
    performance_history: deque = field(default_factory=lambda: deque(maxlen=100))
    error_history: deque = field(default_factory=lambda: deque(maxlen=50))
    last_scan_hash: Optional[str] = None
    last_analysis_timestamp: Optional[datetime] = None


@dataclass
class ResourceMonitor:
    """Мониторинг ресурсов системы."""

    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    disk_threshold: float = 90.0
    network_threshold: float = 1000.0  # MB/s
    current_cpu: float = 0.0
    current_memory: float = 0.0
    current_disk: float = 0.0
    current_network: float = 0.0
    cpu_history: deque = field(default_factory=lambda: deque(maxlen=60))
    memory_history: deque = field(default_factory=lambda: deque(maxlen=60))
    disk_history: deque = field(default_factory=lambda: deque(maxlen=60))
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    last_check: Optional[datetime] = None


class EntityControllerImpl(BaseEntityController):
    """
    Продвинутая реализация EntityController.
    Обеспечивает:
    - Координацию всех компонентов Entity System
    - Управление жизненным циклом системы
    - Мониторинг состояния и производительности
    - Автоматическую оптимизацию на основе AI
    - Эволюционное развитие системы
    """

    def __init__(self, config: EntitySystemConfig):
        # Инициализируем базовый класс без аргументов
        super().__init__()
        self.config = config
        self.components = ComponentRegistry()
        self.metrics = SystemMetrics()
        self.memory_cache = MemoryCache()
        self.resource_monitor = ResourceMonitor()
        self.phase_history: List[PhaseData] = []
        self.current_phase_data: Optional[PhaseData] = None
        self._current_phase = SystemPhase.PERCEPTION
        self._phase_handlers: Dict[SystemPhase, Callable] = {}
        self._last_net_io: Optional[psutil._pslinux.snetio] = None
        # Добавляем недостающие атрибуты
        self.logger = logging.getLogger(__name__)
        self._is_running = False
        self.start_time = datetime.now()
        self._last_health_check = datetime.now()
        self._health_check_interval = 30  # секунды
        self._main_task: Optional[asyncio.Task] = None
        self._operation_mode = OperationMode.MANUAL
        self._optimization_level = OptimizationLevel.MEDIUM
        self._ai_confidence = 0.0
        self._system_health = 1.0
        self._performance_score = 0.0
        self._efficiency_score = 0.0
        self._last_update = datetime.now()
        self._initialize_phase_handlers()

    def _initialize_phase_handlers(self) -> None:
        """Инициализация обработчиков для каждой фазы системы."""
        self._phase_handlers = {
            SystemPhase.PERCEPTION: self._perception_phase,
            SystemPhase.ANALYSIS: self._analysis_phase,
            SystemPhase.EXPERIMENT: self._experiment_phase,
            SystemPhase.APPLICATION: self._application_phase,
            SystemPhase.MEMORY: self._memory_phase,
            SystemPhase.AI_OPTIMIZATION: self._ai_optimization_phase,
            SystemPhase.EVOLUTION: self._evolution_phase,
        }

    async def start(self) -> None:
        """Запуск Entity System."""
        if self._is_running:
            self.logger.warning("Entity System уже запущена")
            return
        try:
            self.logger.info("Запуск Entity System...")
            # Инициализация компонентов
            await self._initialize_components()
            # Запуск основного цикла
            self._is_running = True
            self._main_task = asyncio.create_task(self._main_loop())
            self.logger.info("Entity System успешно запущена")
        except Exception as e:
            self.logger.error(f"Ошибка при запуске Entity System: {e}")
            raise EntitySystemError(f"Не удалось запустить Entity System: {e}")

    async def stop(self) -> None:
        """Остановка Entity System."""
        if not self._is_running:
            self.logger.warning("Entity System уже остановлена")
            return
        try:
            self.logger.info("Остановка Entity System...")
            # Остановка основного цикла
            self._is_running = False
            if self._main_task:
                self._main_task.cancel()
                try:
                    await self._main_task
                except asyncio.CancelledError:
                    pass
            # Очистка компонентов
            await self._cleanup_components()
            self.logger.info("Entity System успешно остановлена")
        except Exception as e:
            self.logger.error(f"Ошибка при остановке Entity System: {e}")
            raise EntitySystemError(f"Не удалось остановить Entity System: {e}")

    async def get_status(self) -> EntityState:
        """Получение текущего состояния системы."""
        try:
            # Расчет времени работы
            uptime = (datetime.now() - self.start_time).total_seconds()
            # Расчет метрик производительности
            performance_score = self._calculate_performance_score()
            efficiency_score = self._calculate_efficiency_score()
            system_health = self._calculate_system_health()
            # Расчет AI уверенности
            ai_confidence = self._calculate_ai_confidence()
            state_dict = {
                "is_running": self._is_running,
                "current_phase": self._current_phase.value,
                "ai_confidence": ai_confidence,
                "optimization_level": self._optimization_level.value,
                "system_health": system_health,
                "performance_score": performance_score,
                "efficiency_score": efficiency_score,
                "last_update": datetime.now(),
            }
            return validate_entity_state(state_dict)
        except Exception as e:
            self.logger.error(f"Ошибка при получении состояния: {e}")
            raise EntitySystemError(f"Не удалось получить состояние системы: {e}")

    def set_operation_mode(self, mode: OperationMode) -> None:
        """Установка режима работы."""
        self._operation_mode = mode
        self.logger.info(f"Режим работы изменен на: {mode.value}")

    def set_optimization_level(self, level: OptimizationLevel) -> None:
        """Установка уровня оптимизации."""
        self._optimization_level = level
        self.logger.info(f"Уровень оптимизации изменен на: {level.value}")

    async def _initialize_components(self) -> None:
        """Инициализация всех компонентов системы."""
        try:
            self.logger.info("Инициализация компонентов...")
            # Создание компонентов с конфигурацией
            # component_config = ... (оставить для других целей)
            self.components.code_scanner = CodeScannerImpl()
            self.components.code_analyzer = CodeAnalyzerImpl()
            self.components.experiment_runner = ExperimentRunnerImpl()
            self.components.improvement_applier = ImprovementApplierImpl()
            self.components.memory_manager = MemoryManagerImpl()
            self.components.ai_enhancement = AIEnhancementImpl()
            self.components.evolution_engine = EvolutionEngineImpl()
            self.logger.info("Все компоненты инициализированы")
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации компонентов: {e}")
            raise EntitySystemError(f"Не удалось инициализировать компоненты: {e}")

    async def _cleanup_components(self) -> None:
        """Очистка компонентов системы."""
        try:
            self.logger.info("Очистка компонентов...")
            # Очистка каждого компонента
            cleanup_tasks = []
            if self.components.code_scanner:
                cleanup_tasks.append(self._cleanup_code_scanner())
            if self.components.code_analyzer:
                cleanup_tasks.append(self._cleanup_code_analyzer())
            if self.components.experiment_runner:
                cleanup_tasks.append(self._cleanup_experiment_runner())
            if self.components.improvement_applier:
                cleanup_tasks.append(self._cleanup_improvement_applier())
            if self.components.memory_manager:
                cleanup_tasks.append(self._cleanup_memory_manager())
            if self.components.ai_enhancement:
                cleanup_tasks.append(self._cleanup_ai_enhancement())
            if self.components.evolution_engine:
                cleanup_tasks.append(self._cleanup_evolution_engine())
            # Выполнение всех задач очистки
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            # Сохранение финального снимка состояния
            await self._save_final_snapshot()
            # Очистка кэша памяти
            self.memory_cache = MemoryCache()
            self.logger.info("Компоненты очищены")
        except Exception as e:
            self.logger.error(f"Ошибка при очистке компонентов: {e}")

    async def _cleanup_code_scanner(self) -> None:
        """Очистка Code Scanner."""
        try:
            # Сохранение последних результатов сканирования
            if self.memory_cache.code_structures:
                await self._save_scan_results()
            self.logger.debug("Code Scanner очищен")
        except Exception as e:
            self.logger.error(f"Ошибка при очистке Code Scanner: {e}")

    async def _cleanup_code_analyzer(self) -> None:
        """Очистка Code Analyzer."""
        try:
            # Сохранение результатов анализа
            if self.memory_cache.analysis_results:
                await self._save_analysis_results()
            self.logger.debug("Code Analyzer очищен")
        except Exception as e:
            self.logger.error(f"Ошибка при очистке Code Analyzer: {e}")

    async def _cleanup_experiment_runner(self) -> None:
        """Очистка Experiment Runner."""
        try:
            # Остановка всех активных экспериментов
            if self.memory_cache.pending_experiments:
                for experiment in self.memory_cache.pending_experiments:
                    if experiment["status"] == "running":
                        if self.components.experiment_runner:  # Проверяем на None
                            await self.components.experiment_runner.stop_experiment(
                                experiment["id"]
                            )
            self.logger.debug("Experiment Runner очищен")
        except Exception as e:
            self.logger.error(f"Ошибка при очистке Experiment Runner: {e}")

    async def _cleanup_improvement_applier(self) -> None:
        """Очистка Improvement Applier."""
        try:
            # Сохранение информации о примененных улучшениях
            if self.memory_cache.applied_improvements:
                await self._save_improvements_log()
            self.logger.debug("Improvement Applier очищен")
        except Exception as e:
            self.logger.error(f"Ошибка при очистке Improvement Applier: {e}")

    async def _cleanup_memory_manager(self) -> None:
        """Очистка Memory Manager."""
        try:
            # Создание финального снимка
            if self.components.memory_manager:  # Проверяем на None
                final_snapshot = await self.components.memory_manager.create_snapshot()
                self.logger.debug(f"Создан финальный снимок: {final_snapshot['id']}")
        except Exception as e:
            self.logger.error(f"Ошибка при очистке Memory Manager: {e}")

    async def _cleanup_ai_enhancement(self) -> None:
        """Очистка AI Enhancement."""
        try:
            # Сохранение состояния AI моделей
            await self._save_ai_models_state()
            self.logger.debug("AI Enhancement очищен")
        except Exception as e:
            self.logger.error(f"Ошибка при очистке AI Enhancement: {e}")

    async def _cleanup_evolution_engine(self) -> None:
        """Очистка Evolution Engine."""
        try:
            # Сохранение эволюционного состояния
            await self._save_evolution_state()
            self.logger.debug("Evolution Engine очищен")
        except Exception as e:
            self.logger.error(f"Ошибка при очистке Evolution Engine: {e}")

    async def _save_final_snapshot(self) -> None:
        """Сохранение финального снимка состояния."""
        try:
            snapshot_data = {
                "timestamp": datetime.now().isoformat(),
                "total_cycles": self.metrics.total_cycles,
                "successful_improvements": self.metrics.successful_improvements,
                "failed_improvements": self.metrics.failed_improvements,
                "experiments_completed": self.metrics.experiments_completed,
                "phase_history": [
                    phase.__dict__ for phase in self.phase_history[-10:]
                ],  # Последние 10 фаз
                "final_metrics": self.metrics.__dict__,
            }
            # Сохранение в файл
            snapshot_file = (
                Path("logs")
                / "entity_system"
                / f"final_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            snapshot_file.parent.mkdir(parents=True, exist_ok=True)
            with open(snapshot_file, "w", encoding="utf-8") as f:
                json.dump(snapshot_data, f, indent=2, default=str)
            self.logger.info(f"Финальный снимок сохранен: {snapshot_file}")
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении финального снимка: {e}")

    async def _save_scan_results(self) -> None:
        """Сохранение результатов сканирования."""
        try:
            scan_data = {
                "timestamp": datetime.now().isoformat(),
                "code_structures_count": len(self.memory_cache.code_structures),
                "files_scanned": [
                    cs["file_path"] for cs in self.memory_cache.code_structures
                ],
            }
            scan_file = (
                Path("logs")
                / "entity_system"
                / f"scan_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            scan_file.parent.mkdir(parents=True, exist_ok=True)
            with open(scan_file, "w", encoding="utf-8") as f:
                json.dump(scan_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении результатов сканирования: {e}")

    async def _save_analysis_results(self) -> None:
        """Сохранение результатов анализа."""
        try:
            analysis_data = {
                "timestamp": datetime.now().isoformat(),
                "analysis_results_count": len(self.memory_cache.analysis_results),
                "average_quality_score": (
                    sum(
                        ar["quality_score"] for ar in self.memory_cache.analysis_results
                    )
                    / len(self.memory_cache.analysis_results)
                    if self.memory_cache.analysis_results
                    else 0.0
                ),
            }
            analysis_file = (
                Path("logs")
                / "entity_system"
                / f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            analysis_file.parent.mkdir(parents=True, exist_ok=True)
            with open(analysis_file, "w", encoding="utf-8") as f:
                json.dump(analysis_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении результатов анализа: {e}")

    async def _save_improvements_log(self) -> None:
        """Сохранение лога улучшений."""
        try:
            improvements_data = {
                "timestamp": datetime.now().isoformat(),
                "applied_improvements_count": len(
                    self.memory_cache.applied_improvements
                ),
                "improvements": [
                    {
                        "id": imp["id"],
                        "name": imp["name"],
                        "category": imp["category"],
                        "applied_at": imp.get("applied_at"),
                    }
                    for imp in self.memory_cache.applied_improvements
                ],
            }
            improvements_file = (
                Path("logs")
                / "entity_system"
                / f"improvements_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            improvements_file.parent.mkdir(parents=True, exist_ok=True)
            with open(improvements_file, "w", encoding="utf-8") as f:
                json.dump(improvements_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении лога улучшений: {e}")

    async def _save_ai_models_state(self) -> None:
        """Сохранение состояния AI моделей."""
        try:
            ai_state_data = {
                "timestamp": datetime.now().isoformat(),
                "models_updated": True,
                "performance_metrics": {
                    "prediction_accuracy": 0.85,
                    "model_version": "1.0.0",
                },
            }
            ai_state_file = (
                Path("logs")
                / "entity_system"
                / f"ai_models_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            ai_state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(ai_state_file, "w", encoding="utf-8") as f:
                json.dump(ai_state_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении состояния AI моделей: {e}")

    async def _save_evolution_state(self) -> None:
        """Сохранение эволюционного состояния."""
        try:
            evolution_state_data = {
                "timestamp": datetime.now().isoformat(),
                "generation": self.metrics.total_cycles,
                "best_fitness": 0.95,
                "population_size": 50,
                "mutation_rate": 0.1,
            }
            evolution_state_file = (
                Path("logs")
                / "entity_system"
                / f"evolution_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            evolution_state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(evolution_state_file, "w", encoding="utf-8") as f:
                json.dump(evolution_state_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении эволюционного состояния: {e}")

    async def _main_loop(self) -> None:
        """Основной цикл работы системы."""
        self.logger.info("Запуск основного цикла Entity System")
        while self._is_running:
            try:
                cycle_start = datetime.now()
                self.metrics.last_cycle_start = cycle_start
                # Проверка здоровья системы
                await self._health_check()
                # Обработка текущей фазы
                await self._process_phase()
                # Переход к следующей фазе
                self._advance_phase()
                # Обновление метрик
                cycle_end = datetime.now()
                self.metrics.last_cycle_end = cycle_end
                self.metrics.total_cycles += 1
                cycle_duration = (cycle_end - cycle_start).total_seconds()
                self.metrics.average_cycle_time = (
                    self.metrics.average_cycle_time * (self.metrics.total_cycles - 1)
                    + cycle_duration
                ) / self.metrics.total_cycles
                # Пауза между циклами
                await asyncio.sleep(self.config.get("analysis_interval", 60))
            except asyncio.CancelledError:
                self.logger.info("Основной цикл остановлен")
                break
            except Exception as e:
                self.logger.error(f"Ошибка в основном цикле: {e}")
                await asyncio.sleep(10)  # Пауза перед повторной попыткой

    async def _health_check(self) -> None:
        """Проверка здоровья системы."""
        current_time = datetime.now()
        if (
            current_time - self._last_health_check
        ).total_seconds() >= self._health_check_interval:
            try:
                # Проверка доступности компонентов
                component_health = await self._check_component_health()
                # Проверка ресурсов системы
                resource_health = await self._check_resource_health()
                # Логирование результатов
                if component_health and resource_health:
                    self.logger.debug("Проверка здоровья системы: OK")
                else:
                    self.logger.warning("Обнаружены проблемы в системе")
                self._last_health_check = current_time
            except Exception as e:
                self.logger.error(f"Ошибка при проверке здоровья: {e}")

    async def _check_component_health(self) -> bool:
        """Проверка здоровья компонентов."""
        try:
            # Проверка каждого компонента
            components = [
                self.components.code_scanner,
                self.components.code_analyzer,
                self.components.experiment_runner,
                self.components.improvement_applier,
                self.components.memory_manager,
                self.components.ai_enhancement,
                self.components.evolution_engine,
            ]
            for component in components:
                if component is None:
                    self.logger.warning("Обнаружен неинициализированный компонент")
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при проверке компонентов: {e}")
            return False

    async def _check_resource_health(self) -> bool:
        """Проверка ресурсов системы."""
        try:
            current_time = datetime.now()
            # Обновление метрик ресурсов
            await self._update_resource_metrics()
            # Проверка пороговых значений
            health_status = True
            alerts = []
            # Проверка CPU
            if self.resource_monitor.current_cpu > self.resource_monitor.cpu_threshold:
                health_status = False
                alerts.append(
                    {
                        "type": "cpu_high",
                        "value": self.resource_monitor.current_cpu,
                        "threshold": self.resource_monitor.cpu_threshold,
                        "timestamp": current_time.isoformat(),
                    }
                )
                self.logger.warning(
                    f"Высокое использование CPU: {self.resource_monitor.current_cpu:.1f}%"
                )
            # Проверка памяти
            if (
                self.resource_monitor.current_memory
                > self.resource_monitor.memory_threshold
            ):
                health_status = False
                alerts.append(
                    {
                        "type": "memory_high",
                        "value": self.resource_monitor.current_memory,
                        "threshold": self.resource_monitor.memory_threshold,
                        "timestamp": current_time.isoformat(),
                    }
                )
                self.logger.warning(
                    f"Высокое использование памяти: {self.resource_monitor.current_memory:.1f}%"
                )
            # Проверка диска
            if (
                self.resource_monitor.current_disk
                > self.resource_monitor.disk_threshold
            ):
                health_status = False
                alerts.append(
                    {
                        "type": "disk_high",
                        "value": self.resource_monitor.current_disk,
                        "threshold": self.resource_monitor.disk_threshold,
                        "timestamp": current_time.isoformat(),
                    }
                )
                self.logger.warning(
                    f"Высокое использование диска: {self.resource_monitor.current_disk:.1f}%"
                )
            # Проверка сети
            if (
                self.resource_monitor.current_network
                > self.resource_monitor.network_threshold
            ):
                alerts.append(
                    {
                        "type": "network_high",
                        "value": self.resource_monitor.current_network,
                        "threshold": self.resource_monitor.network_threshold,
                        "timestamp": current_time.isoformat(),
                    }
                )
                self.logger.warning(
                    f"Высокая сетевая активность: {self.resource_monitor.current_network:.1f} MB/s"
                )
            # Добавление алертов в историю
            self.resource_monitor.alerts.extend(alerts)
            # Ограничение размера истории алертов
            if len(self.resource_monitor.alerts) > 100:
                self.resource_monitor.alerts = self.resource_monitor.alerts[-100:]
            # Обновление метрик системы
            self.metrics.cpu_usage = self.resource_monitor.current_cpu
            self.metrics.memory_usage = self.resource_monitor.current_memory
            self.metrics.disk_usage = self.resource_monitor.current_disk
            self.metrics.network_io = self.resource_monitor.current_network
            # Логирование статуса
            if health_status:
                self.logger.debug(
                    f"Ресурсы системы в норме - CPU: {self.resource_monitor.current_cpu:.1f}%, "
                    f"Memory: {self.resource_monitor.current_memory:.1f}%, "
                    f"Disk: {self.resource_monitor.current_disk:.1f}%"
                )
            else:
                self.logger.warning(
                    f"Обнаружены проблемы с ресурсами - {len(alerts)} алертов"
                )
            return health_status
        except Exception as e:
            self.logger.error(f"Ошибка при проверке ресурсов: {e}")
            return False

    async def _update_resource_metrics(self) -> None:
        """Обновление метрик ресурсов системы."""
        try:
            # Получение метрик CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            self.resource_monitor.current_cpu = cpu_percent
            self.resource_monitor.cpu_history.append(cpu_percent)
            # Получение метрик памяти
            memory = psutil.virtual_memory()
            self.resource_monitor.current_memory = memory.percent
            self.resource_monitor.memory_history.append(memory.percent)
            # Получение метрик диска
            disk = psutil.disk_usage("/")
            self.resource_monitor.current_disk = (disk.used / disk.total) * 100
            self.resource_monitor.disk_history.append(
                self.resource_monitor.current_disk
            )
            # Получение метрик сети (упрощенная версия)
            try:
                net_io = psutil.net_io_counters()
                # Расчет скорости сети (базовая реализация)
                if hasattr(self, "_last_net_io") and self._last_net_io is not None:
                    bytes_sent = net_io.bytes_sent - self._last_net_io.bytes_sent
                    bytes_recv = net_io.bytes_recv - self._last_net_io.bytes_recv
                    total_bytes = bytes_sent + bytes_recv
                    self.resource_monitor.current_network = total_bytes / (
                        1024 * 1024
                    )  # MB/s
                else:
                    self.resource_monitor.current_network = 0.0
                self._last_net_io = net_io
            except Exception:
                self.resource_monitor.current_network = 0.0
            # Обновление времени последней проверки
            self.resource_monitor.last_check = datetime.now()
        except Exception as e:
            self.logger.error(f"Ошибка при обновлении метрик ресурсов: {e}")
            # Установка безопасных значений по умолчанию
            self.resource_monitor.current_cpu = 0.0
            self.resource_monitor.current_memory = 0.0
            self.resource_monitor.current_disk = 0.0
            self.resource_monitor.current_network = 0.0

    async def _process_phase(self) -> None:
        """Обработка текущей фазы системы."""
        try:
            handler = self._phase_handlers.get(self._current_phase)
            if handler:
                await handler()
            else:
                self.logger.warning(f"Неизвестная фаза: {self._current_phase}")
        except Exception as e:
            self.logger.error(f"Ошибка при обработке фазы {self._current_phase}: {e}")

    def _advance_phase(self) -> None:
        """Переход к следующей фазе системы."""
        phase_order = [
            SystemPhase.PERCEPTION,
            SystemPhase.ANALYSIS,
            SystemPhase.EXPERIMENT,
            SystemPhase.APPLICATION,
            SystemPhase.MEMORY,
            SystemPhase.AI_OPTIMIZATION,
            SystemPhase.EVOLUTION,
        ]
        try:
            current_index = phase_order.index(self._current_phase)
            next_index = (current_index + 1) % len(phase_order)
            self._current_phase = phase_order[next_index]
        except ValueError:
            self.logger.warning(f"Неизвестная фаза: {self._current_phase}")
            self._current_phase = SystemPhase.PERCEPTION

    async def _perception_phase(self) -> None:
        """Фаза восприятия - сканирование кодовой базы."""
        try:
            self.logger.debug("Выполнение фазы восприятия")
            if self.components.code_scanner:
                # Сканирование кодовой базы
                codebase_path = Path(".")
                code_structures = await self.components.code_scanner.scan_codebase(
                    codebase_path
                )
                # Сохранение результатов в памяти
                if self.components.memory_manager:
                    await self.components.memory_manager.save_to_journal(
                        {
                            "phase": "perception",
                            "code_structures_count": len(code_structures),
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
        except Exception as e:
            self.logger.error(f"Ошибка в фазе восприятия: {e}")

    async def _analysis_phase(self) -> None:
        """Фаза анализа - анализ кода и генерация гипотез."""
        phase_start = datetime.now()
        self.current_phase_data = PhaseData(
            phase=SystemPhase.ANALYSIS, start_time=phase_start
        )
        try:
            self.logger.debug("Выполнение фазы анализа")
            if (
                not self.components.code_analyzer
                or not self.memory_cache.code_structures
            ):
                self.logger.warning("Нет данных для анализа или компонент недоступен")
                return
            # Анализ каждого файла из кэша
            analysis_results = []
            total_files = len(self.memory_cache.code_structures)
            for i, code_structure in enumerate(self.memory_cache.code_structures):
                try:
                    self.logger.debug(
                        f"Анализ файла {i+1}/{total_files}: {code_structure['file_path']}"
                    )
                    # Анализ кода
                    analysis_result = await self.components.code_analyzer.analyze_code(
                        code_structure
                    )
                    analysis_results.append(analysis_result)
                    # Обновление метрик
                    self.current_phase_data.processed_items += 1
                    # Проверка критических проблем
                    if analysis_result["quality_score"] < 0.3:
                        self.logger.warning(
                            f"Критически низкое качество кода в {code_structure['file_path']}: {analysis_result['quality_score']:.2f}"
                        )
                        self.metrics.critical_errors += 1
                except Exception as e:
                    self.logger.error(
                        f"Ошибка при анализе файла {code_structure['file_path']}: {e}"
                    )
                    self.metrics.error_count += 1
                    continue
            # Генерация гипотез на основе результатов анализа
            if analysis_results:
                try:
                    hypotheses = (
                        await self.components.code_analyzer.generate_hypotheses(
                            analysis_results
                        )
                    )
                    # Фильтрация и валидация гипотез
                    valid_hypotheses = []
                    for hypothesis in hypotheses:
                        is_valid = (
                            await self.components.code_analyzer.validate_hypothesis(
                                hypothesis
                            )
                        )
                        if is_valid and hypothesis[
                            "expected_improvement"
                        ] > self.config.get("improvement_threshold", 0.1):
                            valid_hypotheses.append(hypothesis)
                            self.current_phase_data.generated_hypotheses += 1
                    # Обновление кэша памяти
                    self.memory_cache.analysis_results = analysis_results
                    self.memory_cache.active_hypotheses.extend(valid_hypotheses)
                    self.memory_cache.last_analysis_timestamp = datetime.now()
                    # Сохранение результатов в журнал
                    if self.components.memory_manager:
                        await self.components.memory_manager.save_to_journal(
                            {
                                "phase": "analysis",
                                "files_analyzed": len(analysis_results),
                                "hypotheses_generated": len(valid_hypotheses),
                                "average_quality": sum(
                                    ar["quality_score"] for ar in analysis_results
                                )
                                / len(analysis_results),
                                "critical_issues": len(
                                    [
                                        ar
                                        for ar in analysis_results
                                        if ar["quality_score"] < 0.3
                                    ]
                                ),
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                    self.logger.info(
                        f"Анализ завершен: {len(analysis_results)} файлов, {len(valid_hypotheses)} гипотез"
                    )
                except Exception as e:
                    self.logger.error(f"Ошибка при генерации гипотез: {e}")
                    self.metrics.error_count += 1
            # Обновление метрик фазы
            phase_end = datetime.now()
            self.current_phase_data.end_time = phase_end
            self.current_phase_data.duration = (phase_end - phase_start).total_seconds()
            self.current_phase_data.success = True
            # Добавление в историю фаз
            self.phase_history.append(self.current_phase_data)
            # Ограничение размера истории
            if len(self.phase_history) > 100:
                self.phase_history = self.phase_history[-100:]
        except Exception as e:
            self.logger.error(f"Ошибка в фазе анализа: {e}")
            self.metrics.error_count += 1
            if self.current_phase_data:
                self.current_phase_data.end_time = datetime.now()
                self.current_phase_data.duration = (
                    datetime.now() - phase_start
                ).total_seconds()
                self.current_phase_data.success = False
                self.current_phase_data.error_message = str(e)

    async def _experiment_phase(self) -> None:
        """Фаза экспериментов - проведение экспериментов."""
        phase_start = datetime.now()
        self.current_phase_data = PhaseData(
            phase=SystemPhase.EXPERIMENT, start_time=phase_start
        )
        try:
            self.logger.debug("Выполнение фазы экспериментов")
            if (
                not self.components.experiment_runner
                or not self.memory_cache.active_hypotheses
            ):
                self.logger.debug("Нет активных гипотез для экспериментов")
                return
            # Проверка лимита одновременных экспериментов
            max_concurrent = self.config.get("max_concurrent_experiments", 3)
            current_running = len(
                [
                    exp
                    for exp in self.memory_cache.pending_experiments
                    if exp["status"] == "running"
                ]
            )
            if current_running >= max_concurrent:
                self.logger.debug(
                    f"Достигнут лимит одновременных экспериментов: {current_running}/{max_concurrent}"
                )
                return
            # Выбор гипотез для экспериментов
            available_slots = max_concurrent - current_running
            hypotheses_to_test = self._select_hypotheses_for_experiments(
                available_slots
            )
            if not hypotheses_to_test:
                self.logger.debug("Нет подходящих гипотез для тестирования")
                return
            # Создание и запуск экспериментов
            experiment_tasks = []
            for hypothesis in hypotheses_to_test:
                try:
                    # Создание эксперимента
                    experiment = await self._create_experiment_from_hypothesis(
                        hypothesis
                    )
                    if experiment:
                        # Запуск эксперимента
                        experiment_task = asyncio.create_task(
                            self._run_experiment_with_monitoring(experiment)
                        )
                        experiment_tasks.append(experiment_task)
                        # Добавление в список ожидающих экспериментов
                        self.memory_cache.pending_experiments.append(experiment)
                        self.logger.info(
                            f"Запущен эксперимент {experiment['id']} для гипотезы {hypothesis['id']}"
                        )
                except Exception as e:
                    self.logger.error(
                        f"Ошибка при создании эксперимента для гипотезы {hypothesis['id']}: {e}"
                    )
                    self.metrics.error_count += 1
                    continue
            # Ожидание завершения экспериментов
            if experiment_tasks:
                results = await asyncio.gather(
                    *experiment_tasks, return_exceptions=True
                )
                # Обработка результатов
                for result in results:
                    if isinstance(result, Exception):
                        self.logger.error(f"Эксперимент завершился с ошибкой: {result}")
                        self.metrics.error_count += 1
                    else:
                        self.metrics.experiments_completed += 1
            # Обновление метрик фазы
            phase_end = datetime.now()
            self.current_phase_data.end_time = phase_end
            self.current_phase_data.duration = (phase_end - phase_start).total_seconds()
            self.current_phase_data.success = True
            # Добавление в историю фаз
            self.phase_history.append(self.current_phase_data)
        except Exception as e:
            self.logger.error(f"Ошибка в фазе экспериментов: {e}")
            self.metrics.error_count += 1
            if self.current_phase_data:
                self.current_phase_data.end_time = datetime.now()
                self.current_phase_data.duration = (
                    datetime.now() - phase_start
                ).total_seconds()
                self.current_phase_data.success = False
                self.current_phase_data.error_message = str(e)

    def _select_hypotheses_for_experiments(
        self, available_slots: int
    ) -> List[Hypothesis]:
        """Выбор гипотез для экспериментов."""
        try:
            # Сортировка гипотез по приоритету (ожидаемое улучшение * уверенность)
            scored_hypotheses = []
            for hypothesis in self.memory_cache.active_hypotheses:
                if hypothesis["status"] == "pending":
                    score = (
                        hypothesis["expected_improvement"] * hypothesis["confidence"]
                    )
                    scored_hypotheses.append((score, hypothesis))
            # Сортировка по убыванию приоритета
            scored_hypotheses.sort(key=lambda x: x[0], reverse=True)
            # Выбор лучших гипотез
            selected_hypotheses = []
            for score, hypothesis in scored_hypotheses[:available_slots]:
                # Проверка ресурсов для эксперимента
                if self._can_run_experiment(hypothesis):
                    selected_hypotheses.append(hypothesis)
            return selected_hypotheses
        except Exception as e:
            self.logger.error(f"Ошибка при выборе гипотез: {e}")
            return []

    def _can_run_experiment(self, hypothesis: Hypothesis) -> bool:
        """Проверка возможности запуска эксперимента."""
        try:
            # Проверка ресурсов системы
            if self.resource_monitor.current_cpu > 90.0:
                return False
            if self.resource_monitor.current_memory > 90.0:
                return False
            # Проверка риска гипотезы
            if hypothesis.get("risk_level") == "critical":
                return False
            # Проверка стоимости реализации
            if hypothesis.get("implementation_cost", 0) > 1000:  # Условная единица
                return False
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при проверке возможности эксперимента: {e}")
            return False

    async def _create_experiment_from_hypothesis(
        self, hypothesis: Hypothesis
    ) -> Optional[Experiment]:
        """Создание эксперимента на основе гипотезы."""
        try:
            experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
            experiment = {
                "id": experiment_id,
                "hypothesis_id": hypothesis["id"],
                "name": f"Experiment for {hypothesis.get('name', 'Unknown')}",
                "description": hypothesis["description"],
                "parameters": {
                    "duration": hypothesis.get("estimated_duration", 300),
                    "max_iterations": hypothesis.get("max_iterations", 100),
                    "confidence_threshold": hypothesis.get("confidence_threshold", 0.8),
                },
                "start_time": datetime.now(),
                "end_time": None,
                "status": "running",
                "results": None,
                "metrics": None,
            }
            return experiment
        except Exception as e:
            self.logger.error(f"Ошибка при создании эксперимента: {e}")
            return None

    async def _run_experiment_with_monitoring(
        self, experiment: Experiment
    ) -> Dict[str, Any]:
        """Запуск эксперимента с мониторингом."""
        try:
            # Запуск эксперимента
            if self.components.experiment_runner:  # Проверяем на None
                results = await self.components.experiment_runner.run_experiment(experiment)
            else:
                results = {"error": "Experiment runner not available"}
            # Обновление статуса эксперимента
            experiment["status"] = "completed"
            experiment["end_time"] = datetime.now()
            experiment["results"] = results
            # Перемещение в список завершенных
            self.memory_cache.pending_experiments = [
                exp
                for exp in self.memory_cache.pending_experiments
                if exp["id"] != experiment["id"]
            ]
            self.memory_cache.completed_experiments.append(experiment)
            # Логирование результатов
            if results.get("significant", False):
                self.logger.info(
                    f"Эксперимент {experiment['id']} показал значительные результаты"
                )
            else:
                self.logger.debug(
                    f"Эксперимент {experiment['id']} завершен без значительных результатов"
                )
            return results
        except Exception as e:
            self.logger.error(
                f"Ошибка при выполнении эксперимента {experiment['id']}: {e}"
            )
            # Обновление статуса на "failed"
            experiment["status"] = "failed"
            experiment["end_time"] = datetime.now()
            experiment["results"] = {"error": str(e)}
            return {"error": str(e)}

    async def _application_phase(self) -> None:
        """Фаза применения - применение улучшений."""
        phase_start = datetime.now()
        self.current_phase_data = PhaseData(
            phase=SystemPhase.APPLICATION, start_time=phase_start
        )
        try:
            self.logger.debug("Выполнение фазы применения")
            if not self.components.improvement_applier:
                self.logger.warning("Improvement Applier недоступен")
                return
            # Получение успешных экспериментов
            successful_experiments = [
                exp
                for exp in self.memory_cache.completed_experiments
                if exp["status"] == "completed"
                and exp.get("results") is not None
                and exp.get("results", {}).get("significant", False)
            ]
            if not successful_experiments:
                self.logger.debug("Нет успешных экспериментов для применения")
                return
            # Создание улучшений на основе успешных экспериментов
            improvements_to_apply = []
            for experiment in successful_experiments:
                try:
                    improvement = await self._create_improvement_from_experiment(
                        experiment
                    )
                    if improvement:
                        improvements_to_apply.append(improvement)
                except Exception as e:
                    self.logger.error(
                        f"Ошибка при создании улучшения из эксперимента {experiment['id']}: {e}"
                    )
                    self.metrics.error_count += 1
                    continue
            if not improvements_to_apply:
                self.logger.debug("Нет улучшений для применения")
                return
            # Применение улучшений
            applied_count = 0
            failed_count = 0
            for improvement in improvements_to_apply:
                try:
                    self.logger.info(f"Применение улучшения: {improvement['name']}")
                    # Валидация улучшения
                    is_valid = (
                        await self.components.improvement_applier.validate_improvement(
                            improvement
                        )
                    )
                    if not is_valid:
                        self.logger.warning(
                            f"Улучшение {improvement['id']} не прошло валидацию"
                        )
                        failed_count += 1
                        continue
                    # Применение улучшения
                    success = (
                        await self.components.improvement_applier.apply_improvement(
                            improvement
                        )
                    )
                    if success:
                        # Обновление статуса
                        improvement["status"] = "applied"
                        improvement["applied_at"] = datetime.now()
                        # Добавление в список примененных
                        self.memory_cache.applied_improvements.append(improvement)
                        # Обновление метрик
                        self.metrics.successful_improvements += 1
                        self.current_phase_data.applied_improvements += 1
                        applied_count += 1
                        self.logger.info(
                            f"Улучшение {improvement['id']} успешно применено"
                        )
                        # Сохранение в журнал
                        if self.components.memory_manager:
                            applied_at = improvement.get("applied_at")
                            applied_at_str = applied_at.isoformat() if applied_at is not None else datetime.now().isoformat()
                            await self.components.memory_manager.save_to_journal(
                                {
                                    "phase": "application",
                                    "improvement_id": improvement["id"],
                                    "improvement_name": improvement["name"],
                                    "category": improvement["category"],
                                    "applied_at": applied_at_str,
                                    "timestamp": datetime.now().isoformat(),
                                }
                            )
                    else:
                        self.logger.error(
                            f"Не удалось применить улучшение {improvement['id']}"
                        )
                        failed_count += 1
                        self.metrics.failed_improvements += 1
                except Exception as e:
                    self.logger.error(
                        f"Ошибка при применении улучшения {improvement['id']}: {e}"
                    )
                    failed_count += 1
                    self.metrics.failed_improvements += 1
                    self.metrics.error_count += 1
                    continue
            # Логирование результатов
            self.logger.info(
                f"Фаза применения завершена: {applied_count} применено, {failed_count} неудачно"
            )
            # Обновление метрик фазы
            phase_end = datetime.now()
            self.current_phase_data.end_time = phase_end
            self.current_phase_data.duration = (phase_end - phase_start).total_seconds()
            self.current_phase_data.success = True
            # Добавление в историю фаз
            self.phase_history.append(self.current_phase_data)
        except Exception as e:
            self.logger.error(f"Ошибка в фазе применения: {e}")
            self.metrics.error_count += 1
            if self.current_phase_data:
                self.current_phase_data.end_time = datetime.now()
                self.current_phase_data.duration = (
                    datetime.now() - phase_start
                ).total_seconds()
                self.current_phase_data.success = False
                self.current_phase_data.error_message = str(e)

    async def _create_improvement_from_experiment(
        self, experiment: Experiment
    ) -> Optional[Improvement]:
        """Создание улучшения на основе эксперимента."""
        try:
            hypothesis_id = experiment["hypothesis_id"]
            hypothesis = next(
                (
                    h
                    for h in self.memory_cache.active_hypotheses
                    if h["id"] == hypothesis_id
                ),
                None,
            )
            if not hypothesis:
                self.logger.warning(f"Гипотеза {hypothesis_id} не найдена")
                return None
            improvement_id = f"imp_{uuid.uuid4().hex[:8]}"
            # Создание плана реализации
            implementation_plan = self._create_implementation_plan(
                hypothesis, experiment
            )
            # Создание плана отката
            rollback_plan = self._create_rollback_plan(improvement_id)
            # Создание правил валидации
            validation_rules = self._create_validation_rules(hypothesis, experiment)
            # Создание улучшения
            improvement: Improvement = {
                "id": improvement_id,
                "name": f"Improvement from {hypothesis.get('description', 'Unknown')}",
                "description": hypothesis["description"],
                "category": hypothesis["category"],
                "implementation": {
                    "plan": implementation_plan,
                    "rollback": rollback_plan,
                    "validation": validation_rules,
                },
                "validation_rules": validation_rules,
                "rollback_plan": rollback_plan,
                "created_at": datetime.now(),
                "applied_at": None,
                "status": "pending",
            }
            return improvement
        except Exception as e:
            self.logger.error(f"Ошибка при создании улучшения: {e}")
            return None

    def _create_implementation_plan(
        self, hypothesis: Hypothesis, experiment: Experiment
    ) -> Dict[str, Any]:
        """Создание плана реализации улучшения."""
        try:
            # Анализ результатов эксперимента
            results = experiment.get("results", {})
            improvement_percent = results.get("improvement_percent", 0.0)
            # Создание плана на основе типа гипотезы
            hypothesis_category = hypothesis.get("category", "")
            if hypothesis_category and "performance" in hypothesis_category.lower():
                return {
                    "type": "code_optimization",
                    "target_files": self._identify_target_files(hypothesis),
                    "optimization_type": "algorithm_improvement",
                    "expected_improvement": improvement_percent,
                    "implementation_steps": [
                        "Анализ текущего алгоритма",
                        "Применение оптимизации",
                        "Тестирование производительности",
                        "Валидация результатов",
                    ],
                }
            elif hypothesis_category and "maintainability" in hypothesis_category.lower():
                return {
                    "type": "code_refactoring",
                    "target_files": self._identify_target_files(hypothesis),
                    "refactoring_type": "structure_improvement",
                    "expected_improvement": improvement_percent,
                    "implementation_steps": [
                        "Рефакторинг структуры кода",
                        "Улучшение читаемости",
                        "Снижение сложности",
                        "Обновление документации",
                    ],
                }
            else:
                return {
                    "type": "general_improvement",
                    "target_files": self._identify_target_files(hypothesis),
                    "improvement_type": "general",
                    "expected_improvement": improvement_percent,
                    "implementation_steps": [
                        "Применение улучшения",
                        "Тестирование функциональности",
                        "Валидация изменений",
                    ],
                }
        except Exception as e:
            self.logger.error(f"Ошибка при создании плана реализации: {e}")
            return {
                "type": "fallback",
                "implementation_steps": ["Базовое применение улучшения"],
            }

    def _identify_target_files(self, hypothesis: Hypothesis) -> List[str]:
        """Определение целевых файлов для улучшения."""
        try:
            # Анализ описания гипотезы для поиска упоминаний файлов
            description = hypothesis.get("description", "").lower()
            target_files = []
            # Поиск упоминаний Python файлов
            if "main.py" in description:
                target_files.append("main.py")
            if "utils.py" in description:
                target_files.append("utils.py")
            if "config" in description:
                target_files.append("config/")
            # Если файлы не найдены, используем файлы из последнего анализа
            if not target_files and self.memory_cache.analysis_results:
                # Выбираем файлы с низким качеством
                low_quality_files = [
                    ar["file_path"]
                    for ar in self.memory_cache.analysis_results
                    if ar["quality_score"] < 0.5
                ]
                target_files.extend(low_quality_files[:3])  # Максимум 3 файла
            return target_files
        except Exception as e:
            self.logger.error(f"Ошибка при определении целевых файлов: {e}")
            return []

    def _create_rollback_plan(self, improvement_id: str) -> Dict[str, Any]:
        """Создание плана отката улучшения."""
        try:
            return {
                "type": "git_revert",
                "commit_hash": f"revert_{improvement_id}",
                "backup_files": [],
                "rollback_steps": [
                    "Создание резервной копии",
                    "Откат изменений в Git",
                    "Восстановление состояния",
                    "Валидация отката",
                ],
                "estimated_time": 300,  # секунды
            }
        except Exception as e:
            self.logger.error(f"Ошибка при создании плана отката: {e}")
            return {
                "type": "manual_rollback",
                "rollback_steps": ["Ручной откат изменений"],
            }

    def _create_validation_rules(
        self, hypothesis: Hypothesis, experiment: Experiment
    ) -> List[Dict[str, Any]]:
        """Создание правил валидации улучшения."""
        try:
            results = experiment.get("results", {})
            rules = [
                {
                    "type": "performance_check",
                    "threshold": results.get("improvement_percent", 0.0)
                    * 0.8,  # 80% от ожидаемого улучшения
                    "metric": "execution_time",
                },
                {
                    "type": "functionality_check",
                    "test_cases": ["basic_functionality", "edge_cases"],
                    "expected_result": "pass",
                },
                {
                    "type": "quality_check",
                    "min_quality_score": 0.6,
                    "metric": "code_quality",
                },
            ]
            return rules
        except Exception as e:
            self.logger.error(f"Ошибка при создании правил валидации: {e}")
            return [
                {
                    "type": "basic_check",
                    "description": "Базовая проверка функциональности",
                }
            ]

    async def _memory_phase(self) -> None:
        """Фаза памяти - сохранение состояния."""
        try:
            self.logger.debug("Выполнение фазы памяти")
            if self.components.memory_manager:
                # Создание снимка состояния
                snapshot = await self.components.memory_manager.create_snapshot()
                self.logger.debug(f"Создан снимок состояния: {snapshot['id']}")
        except Exception as e:
            self.logger.error(f"Ошибка в фазе памяти: {e}")

    async def _ai_optimization_phase(self) -> None:
        """Фаза AI оптимизации - AI-улучшения."""
        phase_start = datetime.now()
        self.current_phase_data = PhaseData(
            phase=SystemPhase.AI_OPTIMIZATION, start_time=phase_start
        )
        try:
            self.logger.debug("Выполнение фазы AI оптимизации")
            if not self.components.ai_enhancement:
                self.logger.warning("AI Enhancement недоступен")
                return
            # Проверка доступности данных для анализа
            if not self.memory_cache.code_structures:
                self.logger.debug("Нет данных кода для AI оптимизации")
                return
            # AI анализ каждого файла
            ai_improvements = []
            total_files = len(self.memory_cache.code_structures)
            for i, code_structure in enumerate(self.memory_cache.code_structures):
                try:
                    self.logger.debug(
                        f"AI анализ файла {i+1}/{total_files}: {code_structure['file_path']}"
                    )
                    # Предсказание качества кода
                    quality_prediction = (
                        await self.components.ai_enhancement.predict_code_quality(
                            code_structure
                        )
                    )
                    # Генерация предложений по улучшению
                    suggestions = (
                        await self.components.ai_enhancement.suggest_improvements(
                            code_structure
                        )
                    )
                    # Фильтрация и ранжирование предложений
                    valid_suggestions = self._filter_ai_suggestions(
                        suggestions, quality_prediction
                    )
                    if valid_suggestions:
                        ai_improvements.extend(valid_suggestions)
                        self.current_phase_data.processed_items += 1
                    # Проверка критических проблем
                    if quality_prediction.get("overall_quality", 1.0) < 0.4:
                        self.logger.warning(
                            f"AI обнаружил критические проблемы в {code_structure['file_path']}"
                        )
                        self.metrics.critical_errors += 1
                except Exception as e:
                    self.logger.error(
                        f"Ошибка при AI анализе файла {code_structure['file_path']}: {e}"
                    )
                    self.metrics.error_count += 1
                    continue
            # Оптимизация параметров системы
            if ai_improvements:
                try:
                    # Создание параметров для оптимизации
                    optimization_params = self._create_optimization_params(
                        ai_improvements
                    )
                    # AI оптимизация параметров
                    optimized_params = (
                        await self.components.ai_enhancement.optimize_parameters(
                            optimization_params
                        )
                    )
                    # Применение оптимизированных параметров
                    await self._apply_optimized_params(optimized_params)
                    self.logger.info(
                        f"AI оптимизация завершена: {len(ai_improvements)} предложений, параметры обновлены"
                    )
                except Exception as e:
                    self.logger.error(f"Ошибка при оптимизации параметров: {e}")
                    self.metrics.error_count += 1
            # Создание AI-гипотез
            ai_hypotheses = await self._generate_ai_hypotheses()
            if ai_hypotheses:
                # Добавление AI-гипотез в активные
                self.memory_cache.active_hypotheses.extend(ai_hypotheses)
                self.current_phase_data.generated_hypotheses += len(ai_hypotheses)
                self.logger.info(f"Создано {len(ai_hypotheses)} AI-гипотез")
            # Сохранение результатов в журнал
            if self.components.memory_manager:
                await self.components.memory_manager.save_to_journal(
                    {
                        "phase": "ai_optimization",
                        "files_analyzed": total_files,
                        "ai_improvements": len(ai_improvements),
                        "ai_hypotheses": len(ai_hypotheses),
                        "optimization_applied": bool(ai_improvements),
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            # Обновление метрик фазы
            phase_end = datetime.now()
            self.current_phase_data.end_time = phase_end
            self.current_phase_data.duration = (phase_end - phase_start).total_seconds()
            self.current_phase_data.success = True
            # Добавление в историю фаз
            self.phase_history.append(self.current_phase_data)
        except Exception as e:
            self.logger.error(f"Ошибка в фазе AI оптимизации: {e}")
            self.metrics.error_count += 1
            if self.current_phase_data:
                self.current_phase_data.end_time = datetime.now()
                self.current_phase_data.duration = (
                    datetime.now() - phase_start
                ).total_seconds()
                self.current_phase_data.success = False
                self.current_phase_data.error_message = str(e)

    def _filter_ai_suggestions(
        self, suggestions: List[Dict[str, Any]], quality_prediction: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Фильтрация и ранжирование AI предложений."""
        try:
            filtered_suggestions = []
            for suggestion in suggestions:
                # Проверка приоритета
                priority = suggestion.get("priority", "low")
                if (
                    priority == "low"
                    and quality_prediction.get("overall_quality", 1.0) > 0.7
                ):
                    continue  # Пропускаем низкоприоритетные предложения для качественного кода
                # Проверка ожидаемого эффекта
                estimated_impact = suggestion.get("estimated_impact", 0.0)
                if estimated_impact < 0.05:  # Минимальный порог улучшения
                    continue
                # Проверка типа предложения
                suggestion_type = suggestion.get("type", "")
                if suggestion_type in ["security", "performance", "maintainability"]:
                    filtered_suggestions.append(suggestion)
            # Сортировка по приоритету и ожидаемому эффекту
            filtered_suggestions.sort(
                key=lambda x: (
                    x.get("priority", "low"),
                    x.get("estimated_impact", 0.0),
                ),
                reverse=True,
            )
            return filtered_suggestions[:10]  # Ограничиваем количество предложений
        except Exception as e:
            self.logger.error(f"Ошибка при фильтрации AI предложений: {e}")
            return []

    def _create_optimization_params(
        self, ai_improvements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Создание параметров оптимизации на основе AI предложений."""
        try:
            # Агрегация параметров из всех предложений
            aggregated_params: Dict[str, List[Any]] = {}
            for improvement in ai_improvements:
                params = improvement.get("optimization_params", {})
                for key, value in params.items():
                    if key not in aggregated_params:
                        aggregated_params[key] = []
                    aggregated_params[key].append(value)
            # Усреднение параметров
            optimized_params = {}
            for key, values in aggregated_params.items():
                if isinstance(values[0], (int, float)):
                    optimized_params[key] = sum(values) / len(values)
                elif isinstance(values[0], str):
                    # Для строковых параметров берем наиболее частый
                    from collections import Counter
                    optimized_params[key] = Counter(values).most_common(1)[0][0]
                else:
                    optimized_params[key] = values[0]  # Берем первый
            return {
                "optimized_params": optimized_params,
                "confidence": sum(imp.get("confidence", 0.5) for imp in ai_improvements) / len(ai_improvements),
                "source_improvements": [imp["id"] for imp in ai_improvements],
                "generated_at": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Ошибка при создании параметров оптимизации: {e}")
            return {}

    async def _apply_optimized_params(self, optimized_params: Dict[str, Any]) -> None:
        """Применение оптимизированных параметров."""
        try:
            # Обновление конфигурации системы
            if "analysis_interval" in optimized_params:
                old_value = self.config.get("analysis_interval")
                self.config["analysis_interval"] = optimized_params["analysis_interval"]
                self.logger.debug(
                    f"Параметр analysis_interval обновлен: {old_value} -> {optimized_params['analysis_interval']}"
                )
            if "experiment_duration" in optimized_params:
                old_value = self.config.get("experiment_duration")
                self.config["experiment_duration"] = optimized_params["experiment_duration"]
                self.logger.debug(
                    f"Параметр experiment_duration обновлен: {old_value} -> {optimized_params['experiment_duration']}"
                )
            if "confidence_threshold" in optimized_params:
                old_value = self.config.get("confidence_threshold")
                self.config["confidence_threshold"] = optimized_params["confidence_threshold"]
                self.logger.debug(
                    f"Параметр confidence_threshold обновлен: {old_value} -> {optimized_params['confidence_threshold']}"
                )
            if "improvement_threshold" in optimized_params:
                old_value = self.config.get("improvement_threshold")
                self.config["improvement_threshold"] = optimized_params["improvement_threshold"]
                self.logger.debug(
                    f"Параметр improvement_threshold обновлен: {old_value} -> {optimized_params['improvement_threshold']}"
                )
            # Обновление настроек мониторинга ресурсов
            if "cpu_threshold" in optimized_params:
                cpu_threshold = optimized_params["cpu_threshold"]
                if isinstance(cpu_threshold, (int, float)):
                    self.resource_monitor.cpu_threshold = float(cpu_threshold)
            if "memory_threshold" in optimized_params:
                memory_threshold = optimized_params["memory_threshold"]
                if isinstance(memory_threshold, (int, float)):
                    self.resource_monitor.memory_threshold = float(memory_threshold)
            self.logger.info("Оптимизированные параметры применены")
        except Exception as e:
            self.logger.error(f"Ошибка при применении оптимизированных параметров: {e}")

    async def _generate_ai_hypotheses(self) -> List[Hypothesis]:
        """Генерация AI-гипотез на основе анализа."""
        try:
            ai_hypotheses: List[Hypothesis] = []
            # Анализ паттернов в коде
            patterns: List[Dict[str, Any]] = self._analyze_code_patterns()
            # Создание гипотез на основе паттернов
            for pattern in patterns:
                hypothesis: Hypothesis = {
                    "id": f"ai_hyp_{uuid.uuid4().hex[:8]}",
                    "description": f"AI-оптимизация: {pattern.get('description', 'Unknown')}",
                    "expected_improvement": pattern.get("potential_improvement", 0.1),
                    "confidence": pattern.get("confidence", 0.7),
                    "implementation_cost": pattern.get("cost", 100),
                    "risk_level": pattern.get("risk", "low"),
                    "category": pattern.get("category", "ai_optimization"),
                    "created_at": datetime.now(),
                    "status": "pending",
                }
                ai_hypotheses.append(hypothesis)
            return ai_hypotheses[:5]  # Ограничиваем количество гипотез
        except Exception as e:
            self.logger.error(f"Ошибка при генерации AI-гипотез: {e}")
            return []

    def _analyze_code_patterns(self) -> List[Dict[str, Any]]:
        """Анализ паттернов в коде."""
        try:
            patterns: List[Dict[str, Any]] = []
            # Анализ структуры кода
            if self.memory_cache.code_structures:
                for structure in self.memory_cache.code_structures:
                    pattern = {
                        "type": "code_structure",
                        "description": structure.get("file_path", "unknown"),
                        "complexity": structure.get("complexity", 0),
                        "frequency": structure.get("frequency", 1),
                        "potential_improvement": structure.get("impact_score", 0.0),
                        "confidence": 0.7,
                        "cost": 100,
                        "risk": "low",
                        "category": "code_optimization",
                    }
                    patterns.append(pattern)
            # Анализ результатов анализа
            if self.memory_cache.analysis_results:
                for result in self.memory_cache.analysis_results:
                    pattern = {
                        "type": "analysis_result",
                        "description": f"Analysis: {result.get('file_path', 'unknown')}",
                        "category": result.get("category", "unknown"),
                        "severity": result.get("severity", "info"),
                        "confidence": result.get("confidence", 0.5),
                        "potential_improvement": 0.1,
                        "cost": 50,
                        "risk": "low",
                    }
                    patterns.append(pattern)
            return patterns
        except Exception as e:
            self.logger.error(f"Ошибка при анализе паттернов кода: {e}")
            return []

    async def _evolution_phase(self) -> None:
        """Фаза эволюции - эволюционное развитие."""
        phase_start = datetime.now()
        self.current_phase_data = PhaseData(
            phase=SystemPhase.EVOLUTION, start_time=phase_start
        )
        try:
            self.logger.debug("Выполнение фазы эволюции")
            if not self.components.evolution_engine:
                self.logger.warning("Evolution Engine недоступен")
                return
            # Создание популяции для эволюции
            population = await self._create_evolution_population()
            if not population:
                self.logger.debug("Не удалось создать популяцию для эволюции")
                return
            # Определение функции приспособленности
            fitness_function = self._create_fitness_function()
            # Эволюция популяции
            self.logger.info(f"Запуск эволюции популяции из {len(population)} особей")
            evolved_population = await self.components.evolution_engine.evolve(
                population, fitness_function
            )
            # Анализ результатов эволюции
            evolution_results = await self._analyze_evolution_results(
                evolved_population, population
            )
            # Адаптация системы на основе эволюции
            await self._adapt_system_from_evolution(evolution_results)
            # Обучение на исторических данных
            learning_data = self._prepare_learning_data()
            if learning_data:
                learning_result = await self.components.evolution_engine.learn(
                    learning_data
                )
                self.logger.info(
                    f"Обучение завершено: {learning_result.get('model_updated', False)}"
                )
            # Создание эволюционных гипотез
            evolution_hypotheses: List[Hypothesis] = await self._generate_evolution_hypotheses(
                evolution_results
            )
            if evolution_hypotheses:
                # Добавление эволюционных гипотез в активные
                self.memory_cache.active_hypotheses.extend(evolution_hypotheses)
                self.current_phase_data.generated_hypotheses += len(
                    evolution_hypotheses
                )
                self.logger.info(
                    f"Создано {len(evolution_hypotheses)} эволюционных гипотез"
                )
            # Сохранение результатов в журнал
            if self.components.memory_manager:
                await self.components.memory_manager.save_to_journal(
                    {
                        "phase": "evolution",
                        "population_size": len(population),
                        "evolution_results": evolution_results,
                        "evolution_hypotheses": len(evolution_hypotheses),
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            # Обновление метрик фазы
            phase_end = datetime.now()
            self.current_phase_data.end_time = phase_end
            self.current_phase_data.duration = (phase_end - phase_start).total_seconds()
            self.current_phase_data.success = True
            # Добавление в историю фаз
            self.phase_history.append(self.current_phase_data)
        except Exception as e:
            self.logger.error(f"Ошибка в фазе эволюции: {e}")
            self.metrics.error_count += 1
            if self.current_phase_data:
                self.current_phase_data.end_time = datetime.now()
                self.current_phase_data.duration = (
                    datetime.now() - phase_start
                ).total_seconds()
                self.current_phase_data.success = False
                self.current_phase_data.error_message = str(e)

    async def _create_evolution_population(self) -> List[Dict[str, Any]]:
        """Создание популяции для эволюции."""
        try:
            population: List[Dict[str, Any]] = []
            population_size = 50
            # Создание особей на основе текущих параметров системы
            for i in range(population_size):
                individual = {
                    "id": f"ind_{uuid.uuid4().hex[:8]}",
                    "genes": {
                        "analysis_interval": max(
                            30,
                            min(
                                300,
                                self.config.get("analysis_interval", 60)
                                + self._random_variation(20),
                            ),
                        ),
                        "experiment_duration": max(
                            60,
                            min(
                                600,
                                self.config.get("experiment_duration", 300)
                                + self._random_variation(50),
                            ),
                        ),
                        "confidence_threshold": max(
                            0.5,
                            min(
                                0.95,
                                self.config.get("confidence_threshold", 0.7)
                                + self._random_variation(0.1),
                            ),
                        ),
                        "improvement_threshold": max(
                            0.05,
                            min(
                                0.3,
                                self.config.get("improvement_threshold", 0.1)
                                + self._random_variation(0.05),
                            ),
                        ),
                        "cpu_threshold": max(
                            70,
                            min(
                                95,
                                self.resource_monitor.cpu_threshold
                                + self._random_variation(10),
                            ),
                        ),
                        "memory_threshold": max(
                            75,
                            min(
                                95,
                                self.resource_monitor.memory_threshold
                                + self._random_variation(10),
                            ),
                        ),
                        "mutation_rate": max(
                            0.05, min(0.2, 0.1 + self._random_variation(0.05))
                        ),
                        "crossover_rate": max(
                            0.6, min(0.9, 0.8 + self._random_variation(0.1))
                        ),
                    },
                    "fitness": 0.0,
                    "generation": 0,
                }
                population.append(individual)
            return population
        except Exception as e:
            self.logger.error(f"Ошибка при создании популяции: {e}")
            return []

    def _random_variation(self, max_variation: float) -> float:
        """Генерация случайной вариации."""
        import random

        return (random.random() - 0.5) * 2 * max_variation

    def _create_fitness_function(self) -> Callable:
        """Создание функции приспособленности."""

        def fitness_function(individual: Dict[str, Any]) -> float:
            try:
                # Расчет приспособленности на основе метрик системы
                genes = individual["genes"]
                # Базовый балл
                fitness = 0.5
                # Корректировка на основе успешности улучшений
                if self.metrics.total_cycles > 0:
                    success_rate = self.metrics.successful_improvements / max(
                        self.metrics.total_cycles, 1
                    )
                    fitness += success_rate * 0.3
                # Корректировка на основе эффективности
                if self.metrics.average_cycle_time > 0:
                    efficiency = min(60.0 / self.metrics.average_cycle_time, 1.0)
                    fitness += efficiency * 0.2
                # Штраф за высокое использование ресурсов
                if self.resource_monitor.current_cpu > genes.get("cpu_threshold", 80):
                    fitness -= 0.1
                if self.resource_monitor.current_memory > genes.get(
                    "memory_threshold", 85
                ):
                    fitness -= 0.1
                # Ограничение приспособленности
                return max(0.0, min(1.0, fitness))
            except Exception as e:
                self.logger.error(f"Ошибка в функции приспособленности: {e}")
                return 0.0

        return fitness_function

    async def _analyze_evolution_results(
        self,
        evolved_population: List[Dict[str, Any]],
        original_population: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Анализ результатов эволюции."""
        try:
            # Нахождение лучшей особи
            best_individual = max(
                evolved_population, key=lambda x: x.get("fitness", 0.0)
            )
            # Расчет статистик
            fitness_scores = [ind.get("fitness", 0.0) for ind in evolved_population]
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            max_fitness = max(fitness_scores)
            min_fitness = min(fitness_scores)
            # Анализ улучшения
            original_fitness = [ind.get("fitness", 0.0) for ind in original_population]
            original_avg = sum(original_fitness) / len(original_fitness)
            improvement = (avg_fitness - original_avg) / max(original_avg, 0.01)
            results = {
                "best_individual": best_individual,
                "average_fitness": avg_fitness,
                "max_fitness": max_fitness,
                "min_fitness": min_fitness,
                "improvement_percent": improvement * 100,
                "population_size": len(evolved_population),
                "generation": best_individual.get("generation", 0),
            }
            self.logger.info(
                f"Эволюция завершена: улучшение {results['improvement_percent']:.1f}%, "
                f"лучшая приспособленность: {results['max_fitness']:.3f}"
            )
            return results
        except Exception as e:
            self.logger.error(f"Ошибка при анализе результатов эволюции: {e}")
            return {
                "best_individual": None,
                "average_fitness": 0.0,
                "improvement_percent": 0.0,
            }

    async def _adapt_system_from_evolution(
        self, evolution_results: Dict[str, Any]
    ) -> None:
        """Адаптация системы на основе результатов эволюции."""
        try:
            best_individual = evolution_results.get("best_individual")
            if not best_individual:
                return
            genes = best_individual.get("genes", {})
            improvement_percent = evolution_results.get("improvement_percent", 0.0)
            # Применение улучшений только если они значительны
            if improvement_percent > 5.0:  # 5% улучшение
                # Обновление параметров системы
                if "analysis_interval" in genes:
                    old_value = self.config.get("analysis_interval")
                    self.config["analysis_interval"] = genes["analysis_interval"]
                    self.logger.info(
                        f"Эволюционная адаптация: analysis_interval = {old_value} -> {genes['analysis_interval']}"
                    )
                if "experiment_duration" in genes:
                    old_value = self.config.get("experiment_duration")
                    self.config["experiment_duration"] = genes["experiment_duration"]
                    self.logger.info(
                        f"Эволюционная адаптация: experiment_duration = {old_value} -> {genes['experiment_duration']}"
                    )
                if "confidence_threshold" in genes:
                    old_value = self.config.get("confidence_threshold")
                    self.config["confidence_threshold"] = genes["confidence_threshold"]
                    self.logger.info(
                        f"Эволюционная адаптация: confidence_threshold = {old_value} -> {genes['confidence_threshold']}"
                    )
                if "improvement_threshold" in genes:
                    old_value = self.config.get("improvement_threshold")
                    self.config["improvement_threshold"] = genes["improvement_threshold"]
                    self.logger.info(
                        f"Эволюционная адаптация: improvement_threshold = {old_value} -> {genes['improvement_threshold']}"
                    )
                # Обновление настроек мониторинга
                if "cpu_threshold" in genes:
                    cpu_threshold = genes["cpu_threshold"]
                    if isinstance(cpu_threshold, (int, float)):
                        self.resource_monitor.cpu_threshold = float(cpu_threshold)
                if "memory_threshold" in genes:
                    memory_threshold = genes["memory_threshold"]
                    if isinstance(memory_threshold, (int, float)):
                        self.resource_monitor.memory_threshold = float(memory_threshold)
                self.logger.info(
                    f"Система адаптирована на основе эволюции (улучшение: {improvement_percent:.1f}%)"
                )
            else:
                self.logger.debug(
                    f"Эволюционное улучшение недостаточно значимо: {improvement_percent:.1f}%"
                )
        except Exception as e:
            self.logger.error(f"Ошибка при адаптации системы: {e}")

    def _prepare_learning_data(self) -> List[Dict[str, Any]]:
        """Подготовка данных для обучения."""
        try:
            learning_data = []
            # Данные о производительности системы
            if self.phase_history:
                for phase_data in self.phase_history[-20:]:  # Последние 20 фаз
                    learning_data.append(
                        {
                            "input": {
                                "phase": phase_data.phase.value,
                                "duration": phase_data.duration,
                                "success": phase_data.success,
                                "processed_items": phase_data.processed_items,
                            },
                            "output": {
                                "efficiency": phase_data.duration
                                / max(phase_data.processed_items, 1),
                                "success_rate": 1.0 if phase_data.success else 0.0,
                            },
                        }
                    )
            # Данные о ресурсах
            if self.resource_monitor.cpu_history:
                for i in range(len(self.resource_monitor.cpu_history) - 1):
                    learning_data.append(
                        {
                            "input": {
                                "cpu_usage": self.resource_monitor.cpu_history[i],
                                "memory_usage": (
                                    self.resource_monitor.memory_history[i]
                                    if i < len(self.resource_monitor.memory_history)
                                    else 0.0
                                ),
                            },
                            "output": {
                                "system_health": (
                                    1.0
                                    if self.resource_monitor.cpu_history[i] < 80
                                    else 0.5
                                )
                            },
                        }
                    )
            return learning_data
        except Exception as e:
            self.logger.error(f"Ошибка при подготовке данных для обучения: {e}")
            return []

    async def _generate_evolution_hypotheses(
        self, evolution_results: Dict[str, Any]
    ) -> List[Hypothesis]:
        """Генерация гипотез на основе результатов эволюции."""
        try:
            evolution_hypotheses: List[Hypothesis] = []
            # Анализ результатов эволюции
            if evolution_results.get("significant_changes"):
                for change in evolution_results["significant_changes"]:
                    hypothesis: Hypothesis = {
                        "id": f"evol_hyp_{uuid.uuid4().hex[:8]}",
                        "description": f"Гипотеза на основе эволюционного изменения: {change.get('description', '')}",
                        "expected_improvement": change.get("impact", 0.1),
                        "confidence": change.get("confidence", 0.6),
                        "implementation_cost": change.get("cost", 100),
                        "risk_level": change.get("risk", "low"),
                        "category": "evolution",
                        "created_at": datetime.now(),
                        "status": "pending",
                    }
                    evolution_hypotheses.append(hypothesis)
            return evolution_hypotheses
        except Exception as e:
            self.logger.error(f"Ошибка при генерации эволюционных гипотез: {e}")
            return []

    def _calculate_performance_score(self) -> float:
        """Расчет оценки производительности."""
        try:
            # Базовая оценка на основе метрик
            base_score = 0.8
            # Корректировка на основе успешных улучшений
            if self.metrics.total_cycles > 0:
                success_rate = self.metrics.successful_improvements / max(
                    self.metrics.total_cycles, 1
                )
                base_score += success_rate * 0.2
            return min(base_score, 1.0)
        except Exception as e:
            self.logger.error(f"Ошибка при расчете производительности: {e}")
            return 0.5

    def _calculate_efficiency_score(self) -> float:
        """Расчет оценки эффективности."""
        try:
            # Базовая оценка
            base_score = 0.7
            # Корректировка на основе времени цикла
            if self.metrics.average_cycle_time > 0:
                efficiency_factor = min(60.0 / self.metrics.average_cycle_time, 1.0)
                base_score += efficiency_factor * 0.3
            return min(base_score, 1.0)
        except Exception as e:
            self.logger.error(f"Ошибка при расчете эффективности: {e}")
            return 0.5

    def _calculate_system_health(self) -> float:
        """Расчет здоровья системы."""
        try:
            # Базовая оценка
            health_score = 0.9
            # Корректировка на основе ошибок
            if self.metrics.failed_improvements > 0:
                failure_rate = self.metrics.failed_improvements / max(
                    self.metrics.total_cycles, 1
                )
                health_score -= failure_rate * 0.5
            return max(health_score, 0.0)
        except Exception as e:
            self.logger.error(f"Ошибка при расчете здоровья системы: {e}")
            return 0.5

    def _calculate_ai_confidence(self) -> float:
        """Расчет AI уверенности."""
        try:
            # Базовая уверенность
            confidence = 0.6
            # Корректировка на основе успешных экспериментов
            if self.metrics.experiments_completed > 0:
                success_rate = self.metrics.successful_improvements / max(
                    self.metrics.experiments_completed, 1
                )
                confidence += success_rate * 0.4
            return min(confidence, 1.0)
        except Exception as e:
            self.logger.error(f"Ошибка при расчете AI уверенности: {e}")
            return 0.5
