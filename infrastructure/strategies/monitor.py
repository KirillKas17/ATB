"""
Мониторинг стратегий.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
try:
    import psutil
except ImportError:
    psutil = None
from loguru import logger


@dataclass
class MonitorConfig:
    """Конфигурация мониторинга"""

    # Параметры мониторинга
    log_dir: str = "logs"  # Директория для логов
    interval: int = 60  # Интервал мониторинга (сек)
    metrics: List[str] = field(default_factory=lambda: [
        "cpu_percent",
        "memory_percent",
        "disk_usage",
        "network_io",
        "process_count",
    ])  # Метрики для мониторинга
    save_metrics: bool = True  # Сохранять метрики
    alert_threshold: Dict[str, float] = field(default_factory=lambda: {
        "cpu_percent": 80.0,
        "memory_percent": 80.0,
        "disk_usage": 80.0,
    })  # Пороги для алертов


class Monitor:
    """Мониторинг стратегий."""

    def __init__(self, config: Optional[Union[Dict[str, Any], MonitorConfig]] = None):
        """
        Инициализация монитора.
        Args:
            config: Конфигурация монитора или объект MonitorConfig
        """
        if isinstance(config, MonitorConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = MonitorConfig(**config)
        else:
            self.config = MonitorConfig()
        self._setup_logger()
        self._setup_monitor()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.active_strategies: List[str] = []
        self.strategy_metrics: Dict[str, float] = {}
        self.alerts: List[str] = []

    def _setup_logger(self) -> None:
        """Настройка логгера."""
        logger.add(
            f"{self.config.log_dir}/monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
        )

    def _setup_monitor(self) -> None:
        """Настройка монитора."""
        # Создаем директорию для логов
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)

    async def start(self) -> None:
        """Запуск монитора."""
        if self._running:
            logger.warning("Monitor is already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Monitor started")

    async def stop(self) -> None:
        """Остановка монитора."""
        if not self._running:
            logger.warning("Monitor is not running")
            return
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Monitor stopped")

    async def _monitor_loop(self) -> None:
        """Основной цикл мониторинга."""
        try:
            while self._running:
                # Собираем метрики
                metrics = await self._collect_metrics()
                # Проверяем пороги
                await self._check_thresholds(metrics)
                # Сохраняем метрики
                if self.config.save_metrics:
                    await self._save_metrics(metrics)
                # Ждем следующего интервала
                await asyncio.sleep(self.config.interval)
        except asyncio.CancelledError:
            logger.info("Monitor loop cancelled")
        except Exception as e:
            logger.error(f"Error in monitor loop: {str(e)}")

    async def _collect_metrics(self) -> Dict[str, Any]:
        """
        Сбор метрик.
        Returns:
            Dict с метриками
        """
        try:
            # Запускаем сбор метрик в отдельном потоке для избежания блокировки
            loop = asyncio.get_event_loop()
            metrics = await loop.run_in_executor(None, self._collect_metrics_sync)
            return metrics
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
            return {}

    def _collect_metrics_sync(self) -> Dict[str, Any]:
        """Синхронный сбор метрик."""
        metrics = {}
        # CPU
        if "cpu_percent" in self.config.metrics:
            metrics["cpu_percent"] = psutil.cpu_percent(interval=1)
        # Память
        if "memory_percent" in self.config.metrics:
            metrics["memory_percent"] = psutil.virtual_memory().percent
        # Диск
        if "disk_usage" in self.config.metrics:
            metrics["disk_usage"] = psutil.disk_usage("/").percent
        # Сеть
        if "network_io" in self.config.metrics:
            net_io = psutil.net_io_counters()
            metrics["network_io"] = {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
            }
        # Процессы
        if "process_count" in self.config.metrics:
            metrics["process_count"] = len(psutil.pids())
        # Добавляем временную метку
        metrics["timestamp"] = datetime.now().isoformat()
        return metrics

    async def _check_thresholds(self, metrics: Dict[str, Any]) -> None:
        """
        Проверка порогов для алертов.
        Args:
            metrics: Собранные метрики
        """
        for metric_name, threshold in self.config.alert_threshold.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                if isinstance(value, (int, float)) and value > threshold:
                    alert_msg = f"{metric_name} exceeded threshold: {value} > {threshold}"
                    self.add_alert(alert_msg)

    async def _save_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Сохранение метрик.
        Args:
            metrics: Метрики для сохранения
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_metrics_sync, metrics)
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")

    def _save_metrics_sync(self, metrics: Dict[str, Any]) -> None:
        """Синхронное сохранение метрик."""
        try:
            metrics_file = Path(self.config.log_dir) / "metrics.jsonl"
            with open(metrics_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")
        except Exception as e:
            logger.error(f"Error saving metrics to file: {str(e)}")

    async def get_metrics(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Получение метрик за период.
        Args:
            start_time: Начальное время
            end_time: Конечное время
        Returns:
            DataFrame с метриками
        """
        try:
            # Запускаем чтение в отдельном потоке
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._get_metrics_sync, start_time, end_time
            )
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return pd.DataFrame()

    def _get_metrics_sync(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Синхронное получение метрик."""
        try:
            metrics_file = Path(self.config.log_dir) / "metrics.jsonl"
            if not metrics_file.exists():
                return pd.DataFrame()
            # Читаем метрики из файла
            metrics_data = []
            with open(metrics_file, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        metrics_data.append(data)
                    except json.JSONDecodeError:
                        continue
            if not metrics_data:
                return pd.DataFrame()
            # Создаем DataFrame
            df = pd.DataFrame(metrics_data)
            # Фильтруем по времени
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                if start_time:
                    df = df[df["timestamp"] >= start_time]
                if end_time:
                    df = df[df["timestamp"] <= end_time]
            return df
        except Exception as e:
            logger.error(f"Error reading metrics: {str(e)}")
            return pd.DataFrame()

    async def get_current_metrics(self) -> Dict[str, Any]:
        """
        Получение текущих метрик.
        Returns:
            Dict с текущими метриками
        """
        try:
            return await self._collect_metrics()
        except Exception as e:
            logger.error(f"Error getting current metrics: {str(e)}")
            return {}

    async def get_process_metrics(self, pid: int) -> Dict[str, Any]:
        """
        Получение метрик процесса.
        Args:
            pid: ID процесса
        Returns:
            Dict с метриками процесса
        """
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._get_process_metrics_sync, pid)
        except Exception as e:
            logger.error(f"Error getting process metrics: {str(e)}")
            return {}

    def _get_process_metrics_sync(self, pid: int) -> Dict[str, Any]:
        """Синхронное получение метрик процесса."""
        process = psutil.Process(pid)
        return {
            "cpu_percent": process.cpu_percent(),
            "memory_percent": process.memory_percent(),
            "memory_info": process.memory_info()._asdict(),
            "io_counters": (
                process.io_counters()._asdict()
                if hasattr(process, "io_counters")
                else None
            ),
            "num_threads": process.num_threads(),
            "create_time": datetime.fromtimestamp(process.create_time()).isoformat(),
        }

    async def get_system_metrics(self) -> Dict[str, Any]:
        """
        Получение системных метрик.
        Returns:
            Dict с системными метриками
        """
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._get_system_metrics_sync)
        except Exception as e:
            logger.error(f"Error getting system metrics: {str(e)}")
            return {}

    def _get_system_metrics_sync(self) -> Dict[str, Any]:
        """Синхронное получение системных метрик."""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_total": psutil.disk_usage("/").total,
            "disk_free": psutil.disk_usage("/").free,
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
        }

    def add_strategy(self, strategy_name: str) -> None:
        """Добавить стратегию для мониторинга."""
        if strategy_name not in self.active_strategies:
            self.active_strategies.append(strategy_name)
            self.strategy_metrics[strategy_name] = 0.0
        logger.info(f"Added strategy for monitoring: {strategy_name}")

    def remove_strategy(self, strategy_name: str) -> None:
        """Удалить стратегию из мониторинга."""
        if strategy_name in self.active_strategies:
            self.active_strategies.remove(strategy_name)
            if strategy_name in self.strategy_metrics:
                del self.strategy_metrics[strategy_name]
        logger.info(f"Removed strategy from monitoring: {strategy_name}")

    def update_metrics(self, strategy_name: str, metric_value: float) -> None:
        """Обновить метрики стратегии."""
        if strategy_name in self.active_strategies:
            self.strategy_metrics[strategy_name] = metric_value
        logger.debug(f"Updated metrics for {strategy_name}: {metric_value}")

    def add_alert(self, alert_message: str) -> None:
        """Добавить алерт."""
        self.alerts.append(alert_message)
        logger.warning(f"Alert: {alert_message}")

    async def __aenter__(self) -> "Monitor":
        """Асинхронный контекстный менеджер - вход."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> None:
        """Асинхронный контекстный менеджер - выход."""
        await self.stop()

    def __enter__(self) -> "Monitor":
        """Синхронный контекстный менеджер - вход."""
        asyncio.create_task(self.start())
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> None:
        """Синхронный контекстный менеджер - выход."""
        asyncio.create_task(self.stop())
