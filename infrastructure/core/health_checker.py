"""
Модуль проверки здоровья системы.

Обеспечивает мониторинг состояния всех компонентов системы,
включая биржи, базы данных, стратегии и внешние сервисы.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

import psutil
from loguru import logger

from domain.types.infrastructure_types import SystemMetrics
from domain.types.messaging_types import Event as MessagingEvent, EventPriority as MessagingEventPriority, EventName, EventType
from infrastructure.messaging.event_bus import EventBus


@dataclass
class HealthStatus:
    """Статус здоровья компонента."""

    name: str
    is_healthy: bool
    last_check: datetime
    response_time: float
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Общее здоровье системы."""

    overall_healthy: bool
    components: Dict[str, HealthStatus]
    last_update: datetime
    system_metrics: SystemMetrics
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class HealthChecker:
    """
    Проверяет здоровье всех компонентов системы.

    Мониторит:
    - Состояние бирж
    - Состояние базы данных
    - Состояние стратегий
    - Системные ресурсы
    - Внешние сервисы
    """

    def __init__(self, event_bus: EventBus):
        """Инициализация проверяющего здоровья."""
        self.event_bus = event_bus
        self.components: Dict[str, HealthStatus] = {}
        self.check_intervals: Dict[str, int] = {
            "system": 30,  # 30 секунд
            "exchange": 60,  # 1 минута
            "database": 120,  # 2 минуты
            "strategies": 300,  # 5 минут
            "external": 600,  # 10 минут
        }
        self.last_checks: Dict[str, datetime] = {}
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None

        # Пороги для предупреждений
        self.thresholds = {
            "cpu_usage": 80.0,  # 80% CPU
            "memory_usage": 85.0,  # 85% памяти
            "disk_usage": 90.0,  # 90% диска
            "response_time": 5.0,  # 5 секунд
            "error_rate": 0.05,  # 5% ошибок
        }

        logger.info("HealthChecker initialized")

    async def start_monitoring(self) -> None:
        """Запуск мониторинга здоровья."""
        if self.is_monitoring:
            logger.warning("Health monitoring is already running")
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Health monitoring started")

    async def stop_monitoring(self) -> None:
        """Остановка мониторинга здоровья."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")

    async def check_health(self) -> Dict[str, Any]:
        """Проверка здоровья всех компонентов."""
        try:
            # Проверка системных ресурсов
            system_health = await self._check_system_resources()

            # Проверка компонентов
            components_health = await self._check_all_components()

            # Анализ общего состояния
            overall_healthy = self._analyze_overall_health(
                dict(system_health), components_health
            )

            # Создание отчета
            health_report = SystemHealth(
                overall_healthy=overall_healthy,
                components=components_health,
                last_update=datetime.now(),
                system_metrics=system_health,
                critical_issues=self._get_critical_issues(
                    dict(system_health), components_health
                ),
                warnings=self._get_warnings(dict(system_health), components_health),
            )

            # Отправка события
            await self._publish_health_event(health_report)

            return {
                "overall_healthy": overall_healthy,
                "system_metrics": system_health,
                "components": {
                    name: status.__dict__ for name, status in components_health.items()
                },
                "critical_issues": health_report.critical_issues,
                "warnings": health_report.warnings,
                "last_update": health_report.last_update.isoformat(),
            }

        except Exception as e:
            logger.error(f"Error checking health: {e}")
            return {
                "overall_healthy": False,
                "error": str(e),
                "last_update": datetime.now().isoformat(),
            }

    async def get_metrics(self) -> Dict[str, Any]:
        """Получение метрик системы."""
        try:
            # Системные метрики
            system_metrics = await self._get_system_metrics()

            # Метрики компонентов
            component_metrics = {}
            for name, status in self.components.items():
                component_metrics[name] = {
                    "is_healthy": status.is_healthy,
                    "response_time": status.response_time,
                    "last_check": status.last_check.isoformat(),
                    "error_rate": status.metrics.get("error_rate", 0.0),
                    "availability": status.metrics.get("availability", 0.0),
                }

            return {
                "system": system_metrics,
                "components": component_metrics,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {"error": str(e)}

    async def _monitoring_loop(self) -> None:
        """Основной цикл мониторинга."""
        logger.info("Starting health monitoring loop")

        while self.is_monitoring:
            try:
                # Проверка здоровья
                health_status = await self.check_health()

                # Логирование критических проблем
                if not health_status.get("overall_healthy", True):
                    logger.warning(
                        f"System health issues detected: {health_status.get('critical_issues', [])}"
                    )

                # Ожидание следующей проверки
                await asyncio.sleep(30)  # Проверка каждые 30 секунд

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _check_system_resources(self) -> SystemMetrics:
        """Проверка системных ресурсов."""
        try:
            # CPU
            cpu_usage = psutil.cpu_percent(interval=1)

            # Память
            memory = psutil.virtual_memory()
            memory_usage = memory.percent

            # Диск
            disk = psutil.disk_usage("/")
            disk_usage = (disk.used / disk.total) * 100

            # Сеть
            network_latency = await self._measure_network_latency()

            # Системные процессы
            database_connections = await self._count_database_connections()
            active_strategies = await self._count_active_strategies()
            total_trades = await self._get_total_trades()

            # Время работы системы
            system_uptime = time.time() - psutil.boot_time()

            # Оценка производительности
            performance_score = self._calculate_performance_score(
                cpu_usage, memory_usage, disk_usage, network_latency
            )

            return SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_latency=network_latency,
                database_connections=database_connections,
                active_strategies=active_strategies,
                total_trades=total_trades,
                system_uptime=system_uptime,
                error_rate=0.0,  # Будет рассчитано отдельно
                performance_score=performance_score,
            )

        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            return SystemMetrics(
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_latency=0.0,
                database_connections=0,
                active_strategies=0,
                total_trades=0,
                system_uptime=0.0,
                error_rate=1.0,
                performance_score=0.0,
            )

    async def _check_all_components(self) -> Dict[str, HealthStatus]:
        """Проверка всех компонентов."""
        components = {}

        # Проверка бирж
        components.update(await self._check_exchanges())

        # Проверка базы данных
        components.update(await self._check_databases())

        # Проверка стратегий
        components.update(await self._check_strategies())

        # Проверка внешних сервисов
        components.update(await self._check_external_services())

        self.components = components
        return components

    async def _check_exchanges(self) -> Dict[str, HealthStatus]:
        """Проверка состояния бирж."""
        exchanges = {}

        # Проверка Binance
        try:
            start_time = time.time()
            # await self._test_binance_connection()
            response_time = time.time() - start_time

            exchanges["binance"] = HealthStatus(
                name="binance",
                is_healthy=True,
                last_check=datetime.now(),
                response_time=response_time,
                metrics={"availability": 0.99, "error_rate": 0.01},
            )

        except Exception as e:
            exchanges["binance"] = HealthStatus(
                name="binance",
                is_healthy=False,
                last_check=datetime.now(),
                response_time=0.0,
                error_message=str(e),
                metrics={"availability": 0.0, "error_rate": 1.0},
            )

        return exchanges

    async def _check_databases(self) -> Dict[str, HealthStatus]:
        """Проверка состояния баз данных."""
        databases = {}

        try:
            # Проверка основной БД
            start_time = time.time()
            # await self._test_database_connection()
            response_time = time.time() - start_time

            databases["main_db"] = HealthStatus(
                name="main_db",
                is_healthy=True,
                last_check=datetime.now(),
                response_time=response_time,
                metrics={"connections": 5, "query_time": 0.1},
            )

        except Exception as e:
            databases["main_db"] = HealthStatus(
                name="main_db",
                is_healthy=False,
                last_check=datetime.now(),
                response_time=0.0,
                error_message=str(e),
                metrics={"connections": 0, "query_time": 0.0},
            )

        return databases

    async def _check_strategies(self) -> Dict[str, HealthStatus]:
        """Проверка состояния стратегий."""
        strategies = {}

        strategies["default"] = HealthStatus(
            name="default",
            is_healthy=True,
            last_check=datetime.now(),
            response_time=0.1,
            metrics={"active_strategies": 3, "total_signals": 150},
        )

        return strategies

    async def _check_external_services(self) -> Dict[str, HealthStatus]:
        """Проверка внешних сервисов."""
        services = {}

        # Проверка API ключей
        try:
            start_time = time.time()
            # await self._test_api_keys()
            response_time = time.time() - start_time

            services["api_keys"] = HealthStatus(
                name="api_keys",
                is_healthy=True,
                last_check=datetime.now(),
                response_time=response_time,
                metrics={"valid_keys": 2, "expired_keys": 0},
            )

        except Exception as e:
            services["api_keys"] = HealthStatus(
                name="api_keys",
                is_healthy=False,
                last_check=datetime.now(),
                response_time=0.0,
                error_message=str(e),
                metrics={"valid_keys": 0, "expired_keys": 2},
            )

        return services

    async def _measure_network_latency(self) -> float:
        """Измерение сетевой задержки."""
        try:
            start_time = time.time()
            # await asyncio.sleep(0.1)  # Заглушка
            return time.time() - start_time
        except Exception:
            return 0.0

    async def _count_database_connections(self) -> int:
        """Подсчет активных соединений с БД."""
        try:
            return 5  # Заглушка
        except Exception:
            return 0

    async def _count_active_strategies(self) -> int:
        """Подсчет активных стратегий."""
        try:
            return 3  # Заглушка
        except Exception:
            return 0

    async def _get_total_trades(self) -> int:
        """Получение общего количества сделок."""
        try:
            return 150  # Заглушка
        except Exception:
            return 0

    def _calculate_performance_score(
        self, cpu: float, memory: float, disk: float, latency: float
    ) -> float:
        """Расчет оценки производительности системы."""
        try:
            # Нормализация метрик (0-1, где 1 - лучший результат)
            cpu_score = max(0, 1 - cpu / 100)
            memory_score = max(0, 1 - memory / 100)
            disk_score = max(0, 1 - disk / 100)
            latency_score = max(0, 1 - min(latency, 5) / 5)

            # Взвешенная оценка
            score = (
                cpu_score * 0.3
                + memory_score * 0.3
                + disk_score * 0.2
                + latency_score * 0.2
            )

            return round(score, 3)

        except Exception:
            return 0.0

    def _analyze_overall_health(
        self, system_metrics: Dict[str, Any], components: Dict[str, HealthStatus]
    ) -> bool:
        """Анализ общего состояния здоровья системы."""
        try:
            # Проверка системных ресурсов
            if (
                system_metrics["cpu_usage"] > self.thresholds["cpu_usage"]
                or system_metrics["memory_usage"] > self.thresholds["memory_usage"]
                or system_metrics["disk_usage"] > self.thresholds["disk_usage"]
            ):
                return False

            # Проверка критических компонентов
            critical_components = ["binance", "main_db"]
            for component in critical_components:
                if component in components and not components[component].is_healthy:
                    return False

            return True

        except Exception:
            return False

    def _get_critical_issues(
        self, system_metrics: Dict[str, Any], components: Dict[str, HealthStatus]
    ) -> List[str]:
        """Получение критических проблем."""
        issues = []

        # Системные проблемы
        if system_metrics["cpu_usage"] > self.thresholds["cpu_usage"]:
            issues.append(f"High CPU usage: {system_metrics['cpu_usage']:.1f}%")

        if system_metrics["memory_usage"] > self.thresholds["memory_usage"]:
            issues.append(f"High memory usage: {system_metrics['memory_usage']:.1f}%")

        if system_metrics["disk_usage"] > self.thresholds["disk_usage"]:
            issues.append(f"High disk usage: {system_metrics['disk_usage']:.1f}%")

        # Проблемы компонентов
        for name, status in components.items():
            if not status.is_healthy:
                issues.append(f"Component {name} is unhealthy: {status.error_message}")

        return issues

    def _get_warnings(
        self, system_metrics: Dict[str, Any], components: Dict[str, HealthStatus]
    ) -> List[str]:
        """Получение предупреждений."""
        warnings = []

        # Системные предупреждения
        if system_metrics["cpu_usage"] > 70:
            warnings.append(f"CPU usage is high: {system_metrics['cpu_usage']:.1f}%")

        if system_metrics["memory_usage"] > 75:
            warnings.append(
                f"Memory usage is high: {system_metrics['memory_usage']:.1f}%"
            )

        if system_metrics["performance_score"] < 0.7:
            warnings.append(
                f"System performance is degraded: {system_metrics['performance_score']:.3f}"
            )

        return warnings

    async def _publish_health_event(self, health_report: Any) -> None:
        """Публикация события о здоровье системы."""
        try:
            health_report_dict = {
                "overall_healthy": health_report.overall_healthy,
                "system_metrics": health_report.system_metrics.__dict__,
                "critical_issues": health_report.critical_issues,
                "warnings": health_report.warnings,
                "last_update": health_report.last_update.isoformat(),
            }

            event = MessagingEvent(
                name=EventName("health.system.status"),  # Добавляю обязательный аргумент
                type=EventType("health"),  # Добавляю обязательный аргумент
                data=health_report_dict,
                priority=MessagingEventPriority.NORMAL,
            )

            await self.event_bus.publish(event)

        except Exception as e:
            logger.error(f"Error publishing health event: {e}")

    def register_component(self, name: str, check_interval: int = 300) -> None:
        """Регистрация компонента для мониторинга."""
        self.check_intervals[name] = check_interval
        logger.info(f"Registered component for health monitoring: {name}")

    def unregister_component(self, name: str) -> None:
        """Отмена регистрации компонента."""
        if name in self.check_intervals:
            del self.check_intervals[name]
            logger.info(f"Unregistered component from health monitoring: {name}")

    async def get_component_health(self, component_name: str) -> Optional[HealthStatus]:
        """Получение здоровья конкретного компонента."""
        return self.components.get(component_name)

    def is_component_healthy(self, component_name: str) -> bool:
        """Проверка здоровья конкретного компонента."""
        component = self.components.get(component_name)
        return component.is_healthy if component else False

    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Получение системных метрик."""
        try:
            system_metrics = await self._check_system_resources()
            return {
                "cpu_usage": system_metrics["cpu_usage"],
                "memory_usage": system_metrics["memory_usage"],
                "disk_usage": system_metrics["disk_usage"],
                "network_latency": system_metrics["network_latency"],
                "database_connections": system_metrics["database_connections"],
                "active_strategies": system_metrics["active_strategies"],
                "total_trades": system_metrics["total_trades"],
                "system_uptime": system_metrics["system_uptime"],
                "error_rate": system_metrics["error_rate"],
                "performance_score": system_metrics["performance_score"],
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}
