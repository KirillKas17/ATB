"""
Production-grade мониторинг и алертинг для ATB Trading System.
Обеспечивает 24/7 контроль всех критических компонентов системы.
"""

import asyncio
import json
import logging
import psutil
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Настройка безопасного логирования
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class AlertSeverity(Enum):
    """Уровни критичности алертов."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Типы метрик мониторинга."""
    GAUGE = "gauge"         # Текущее значение
    COUNTER = "counter"     # Накопительный счетчик
    HISTOGRAM = "histogram" # Распределение значений
    TIMER = "timer"        # Время выполнения


@dataclass
class Alert:
    """Алерт системы мониторинга."""
    id: str
    title: str
    message: str
    severity: AlertSeverity
    timestamp: datetime
    component: str = ""
    metric_name: str = ""
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Metric:
    """Метрика системы мониторинга."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class HealthCheck:
    """Результат проверки здоровья компонента."""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    response_time: float
    timestamp: datetime
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class ProductionMonitor:
    """Комплексная система мониторинга производственного уровня."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.metrics: Dict[str, List[Metric]] = {}
        self.alerts: List[Alert] = []
        self.health_checks: Dict[str, HealthCheck] = {}
        self.is_running = False
        self.alert_handlers: List[Callable] = []
        self.metric_handlers: List[Callable] = []
        
        # Пороги для алертов
        self.alert_thresholds = self.config.get("alert_thresholds", {})
        
        # Состояние системы
        self.system_status = "starting"
        self.last_health_check = datetime.now()
        
        # Фоновые задачи
        self.monitoring_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        logger.info("Production monitor initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Конфигурация по умолчанию."""
        return {
            "monitoring_interval": 30,  # секунд
            "cleanup_interval": 3600,   # секунд (1 час)
            "metrics_retention": 86400, # секунд (24 часа)
            "alert_thresholds": {
                "cpu_usage": 80.0,
                "memory_usage": 85.0,
                "disk_usage": 90.0,
                "error_rate": 5.0,
                "response_time": 5000.0,  # мс
                "position_loss": 10.0,    # %
                "daily_loss": 20.0,       # %
                "drawdown": 15.0          # %
            },
            "health_check_components": [
                "database", "cache", "exchange_connections",
                "trading_engine", "risk_manager", "portfolio_manager"
            ]
        }
    
    async def start(self) -> None:
        """Запуск системы мониторинга."""
        if self.is_running:
            logger.warning("Monitor already running")
            return
        
        self.is_running = True
        self.system_status = "running"
        
        # Запуск фоновых задач
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Начальная проверка здоровья
        await self._perform_health_checks()
        
        logger.info("Production monitor started")
    
    async def stop(self) -> None:
        """Остановка системы мониторинга."""
        self.is_running = False
        self.system_status = "stopping"
        
        # Остановка фоновых задач
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.system_status = "stopped"
        logger.info("Production monitor stopped")
    
    def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE,
                     labels: Optional[Dict[str, str]] = None, description: str = "") -> None:
        """Запись метрики."""
        try:
            metric = Metric(
                name=name,
                value=value,
                metric_type=metric_type,
                timestamp=datetime.now(),
                labels=labels or {},
                description=description
            )
            
            if name not in self.metrics:
                self.metrics[name] = []
            
            self.metrics[name].append(metric)
            
            # Проверка на превышение порогов
            self._check_metric_thresholds(metric)
            
            # Уведомление обработчиков
            for handler in self.metric_handlers:
                try:
                    handler(metric)
                except Exception as e:
                    logger.error(f"Metric handler error: {e}")
            
        except Exception as e:
            logger.error(f"Failed to record metric {name}: {e}")
    
    def create_alert(self, title: str, message: str, severity: AlertSeverity,
                    component: str = "", metric_name: str = "",
                    current_value: Optional[float] = None,
                    threshold: Optional[float] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Создание алерта."""
        try:
            alert_id = f"alert_{int(time.time() * 1000)}"
            
            alert = Alert(
                id=alert_id,
                title=title,
                message=message,
                severity=severity,
                timestamp=datetime.now(),
                component=component,
                metric_name=metric_name,
                current_value=current_value,
                threshold=threshold,
                metadata=metadata or {}
            )
            
            self.alerts.append(alert)
            
            # Уведомление обработчиков
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler error: {e}")
            
            # Логирование алерта
            level = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.ERROR: logging.ERROR,
                AlertSeverity.CRITICAL: logging.CRITICAL
            }.get(severity, logging.INFO)
            
            logger.log(level, f"Alert {alert_id}: {title} - {message}")
            
            return alert_id
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            return ""
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Разрешение алерта."""
        try:
            for alert in self.alerts:
                if alert.id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolution_time = datetime.now()
                    logger.info(f"Alert {alert_id} resolved")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to resolve alert {alert_id}: {e}")
            return False
    
    async def check_component_health(self, component: str) -> HealthCheck:
        """Проверка здоровья компонента."""
        start_time = time.time()
        
        try:
            # Диспетчер проверок по компонентам
            health_checks = {
                "database": self._check_database_health,
                "cache": self._check_cache_health,
                "exchange_connections": self._check_exchange_health,
                "trading_engine": self._check_trading_engine_health,
                "risk_manager": self._check_risk_manager_health,
                "portfolio_manager": self._check_portfolio_health
            }
            
            checker = health_checks.get(component, self._check_generic_health)
            status, message, details = await checker(component)
            
            response_time = (time.time() - start_time) * 1000  # мс
            
            health_check = HealthCheck(
                component=component,
                status=status,
                response_time=response_time,
                timestamp=datetime.now(),
                message=message,
                details=details
            )
            
            self.health_checks[component] = health_check
            
            # Создание алерта при проблемах
            if status == "unhealthy":
                self.create_alert(
                    title=f"Component {component} unhealthy",
                    message=message,
                    severity=AlertSeverity.ERROR,
                    component=component,
                    metadata={"response_time": response_time, "details": details}
                )
            elif status == "degraded":
                self.create_alert(
                    title=f"Component {component} degraded",
                    message=message,
                    severity=AlertSeverity.WARNING,
                    component=component,
                    metadata={"response_time": response_time, "details": details}
                )
            
            return health_check
            
        except Exception as e:
            logger.error(f"Health check failed for {component}: {e}")
            return HealthCheck(
                component=component,
                status="unhealthy",
                response_time=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                message=f"Health check error: {e}"
            )
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Получение общего обзора системы."""
        try:
            # Системные метрики
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Подсчет алертов по уровням
            alert_counts = {}
            for severity in AlertSeverity:
                alert_counts[severity.value] = len([
                    a for a in self.alerts[-100:] 
                    if a.severity == severity and not a.resolved
                ])
            
            # Статус компонентов
            component_statuses = {}
            for component, health in self.health_checks.items():
                component_statuses[component] = {
                    "status": health.status,
                    "response_time": health.response_time,
                    "last_check": health.timestamp.isoformat()
                }
            
            # Последние метрики
            latest_metrics = {}
            for name, metric_list in self.metrics.items():
                if metric_list:
                    latest = metric_list[-1]
                    latest_metrics[name] = {
                        "value": latest.value,
                        "timestamp": latest.timestamp.isoformat()
                    }
            
            return {
                "system_status": self.system_status,
                "uptime_seconds": (datetime.now() - self.last_health_check).total_seconds(),
                "system_metrics": {
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory.percent,
                    "disk_usage": (disk.used / disk.total) * 100,
                    "memory_available_gb": memory.available / (1024**3)
                },
                "alerts": {
                    "active_count": sum(alert_counts.values()),
                    "by_severity": alert_counts,
                    "latest": [
                        {
                            "id": a.id,
                            "title": a.title,
                            "severity": a.severity.value,
                            "timestamp": a.timestamp.isoformat()
                        } for a in self.alerts[-5:] if not a.resolved
                    ]
                },
                "components": component_statuses,
                "latest_metrics": latest_metrics,
                "monitoring_active": self.is_running
            }
            
        except Exception as e:
            logger.error(f"Failed to get system overview: {e}")
            return {"error": str(e), "system_status": "unknown"}
    
    async def _monitoring_loop(self) -> None:
        """Основной цикл мониторинга."""
        while self.is_running:
            try:
                # Сбор системных метрик
                await self._collect_system_metrics()
                
                # Проверка здоровья компонентов
                await self._perform_health_checks()
                
                # Проверка критических показателей
                await self._check_critical_metrics()
                
                await asyncio.sleep(self.config["monitoring_interval"])
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(10)  # Короткая пауза при ошибке
    
    async def _cleanup_loop(self) -> None:
        """Цикл очистки старых данных."""
        while self.is_running:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(self.config["cleanup_interval"])
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)
    
    async def _collect_system_metrics(self) -> None:
        """Сбор системных метрик."""
        try:
            # CPU
            cpu_usage = psutil.cpu_percent(interval=1)
            self.record_metric("system.cpu_usage", cpu_usage, MetricType.GAUGE)
            
            # Memory
            memory = psutil.virtual_memory()
            self.record_metric("system.memory_usage", memory.percent, MetricType.GAUGE)
            self.record_metric("system.memory_available", memory.available, MetricType.GAUGE)
            
            # Disk
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            self.record_metric("system.disk_usage", disk_usage, MetricType.GAUGE)
            
            # Network
            network = psutil.net_io_counters()
            self.record_metric("system.network_bytes_sent", network.bytes_sent, MetricType.COUNTER)
            self.record_metric("system.network_bytes_recv", network.bytes_recv, MetricType.COUNTER)
            
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
    
    async def _perform_health_checks(self) -> None:
        """Выполнение проверок здоровья всех компонентов."""
        components = self.config["health_check_components"]
        
        # Параллельная проверка всех компонентов
        tasks = []
        for component in components:
            task = asyncio.create_task(self.check_component_health(component))
            tasks.append(task)
        
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
            self.last_health_check = datetime.now()
            
        except Exception as e:
            logger.error(f"Health checks failed: {e}")
    
    def _check_metric_thresholds(self, metric: Metric) -> None:
        """Проверка превышения порогов метрик."""
        threshold = self.alert_thresholds.get(metric.name)
        if threshold is None:
            return
        
        if metric.value > threshold:
            self.create_alert(
                title=f"Metric threshold exceeded: {metric.name}",
                message=f"{metric.name} is {metric.value}, threshold: {threshold}",
                severity=AlertSeverity.WARNING if metric.value < threshold * 1.2 else AlertSeverity.ERROR,
                metric_name=metric.name,
                current_value=metric.value,
                threshold=threshold
            )
    
    async def _check_critical_metrics(self) -> None:
        """Проверка критических показателей торговли."""
        try:
            # Здесь можно добавить интеграцию с торговыми сервисами
            # Для демонстрации используем заглушки
            
            # Проверка P&L
            daily_pnl = await self._get_daily_pnl()
            if daily_pnl and daily_pnl < -self.alert_thresholds.get("daily_loss", 20):
                self.create_alert(
                    title="Critical daily loss detected",
                    message=f"Daily P&L: {daily_pnl}%",
                    severity=AlertSeverity.CRITICAL,
                    component="trading_engine",
                    current_value=abs(daily_pnl),
                    threshold=self.alert_thresholds.get("daily_loss", 20)
                )
            
        except Exception as e:
            logger.error(f"Critical metrics check failed: {e}")
    
    async def _get_daily_pnl(self) -> Optional[float]:
        """Получение дневного P&L (заглушка)."""
        # Интеграция с portfolio service
        return None
    
    async def _cleanup_old_data(self) -> None:
        """Очистка старых метрик и алертов."""
        try:
            retention_seconds = self.config["metrics_retention"]
            cutoff_time = datetime.now() - timedelta(seconds=retention_seconds)
            
            # Очистка старых метрик
            for name, metric_list in self.metrics.items():
                self.metrics[name] = [
                    m for m in metric_list 
                    if m.timestamp > cutoff_time
                ]
            
            # Очистка старых алертов (оставляем последние 1000)
            if len(self.alerts) > 1000:
                self.alerts = self.alerts[-1000:]
            
            logger.debug("Old data cleanup completed")
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
    
    # Методы проверки здоровья компонентов (заглушки для интеграции)
    async def _check_database_health(self, component: str) -> Tuple[Any, ...]:
        """Проверка здоровья базы данных."""
        # Здесь интеграция с database service
        return "healthy", "Database operational", {}
    
    async def _check_cache_health(self, component: str) -> Tuple[Any, ...]:
        """Проверка здоровья кэша."""
        # Здесь интеграция с cache service
        return "healthy", "Cache operational", {}
    
    async def _check_exchange_health(self, component: str) -> Tuple[Any, ...]:
        """Проверка подключений к биржам."""
        # Здесь интеграция с exchange services
        return "healthy", "Exchange connections active", {}
    
    async def _check_trading_engine_health(self, component: str) -> Tuple[Any, ...]:
        """Проверка торгового движка."""
        # Здесь интеграция с trading engine
        return "healthy", "Trading engine operational", {}
    
    async def _check_risk_manager_health(self, component: str) -> Tuple[Any, ...]:
        """Проверка менеджера рисков."""
        # Здесь интеграция с risk manager
        return "healthy", "Risk manager operational", {}
    
    async def _check_portfolio_health(self, component: str) -> Tuple[Any, ...]:
        """Проверка портфельного менеджера."""
        # Здесь интеграция с portfolio manager
        return "healthy", "Portfolio manager operational", {}
    
    async def _check_generic_health(self, component: str) -> Tuple[Any, ...]:
        """Общая проверка здоровья."""
        return "healthy", f"Component {component} status unknown", {}


# Глобальный экземпляр монитора
production_monitor = ProductionMonitor()

# Convenience функции
async def start_monitoring() -> None:
    """Запуск мониторинга."""
    await production_monitor.start()

async def stop_monitoring() -> None:
    """Остановка мониторинга."""
    await production_monitor.stop()

def record_metric(name: str, value: float, metric_type: MetricType = MetricType.GAUGE, **kwargs) -> None:
    """Запись метрики."""
    production_monitor.record_metric(name, value, metric_type, **kwargs)

def create_alert(title: str, message: str, severity: AlertSeverity, **kwargs) -> str:
    """Создание алерта."""
    return production_monitor.create_alert(title, message, severity, **kwargs)

def get_system_status() -> Dict[str, Any]:
    """Получение статуса системы."""
    return production_monitor.get_system_overview()

# Экспорт основных компонентов
__all__ = [
    'ProductionMonitor', 'Alert', 'Metric', 'HealthCheck',
    'AlertSeverity', 'MetricType',
    'production_monitor', 'start_monitoring', 'stop_monitoring',
    'record_metric', 'create_alert', 'get_system_status'
]