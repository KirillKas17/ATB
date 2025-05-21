import json
import os
import queue
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import psutil
from loguru import logger


@dataclass
class MonitorConfig:
    """Конфигурация мониторинга"""

    # Параметры мониторинга
    log_dir: str = "logs"  # Директория для логов
    interval: int = 60  # Интервал мониторинга (сек)
    metrics: List[str] = None  # Метрики для мониторинга
    save_metrics: bool = True  # Сохранять метрики
    alert_threshold: Dict[str, float] = None  # Пороги для алертов

    def __post_init__(self):
        """Инициализация параметров по умолчанию"""
        if self.metrics is None:
            self.metrics = [
                "cpu_percent",
                "memory_percent",
                "disk_usage",
                "network_io",
                "process_count",
            ]

        if self.alert_threshold is None:
            self.alert_threshold = {"cpu_percent": 80.0, "memory_percent": 80.0, "disk_usage": 80.0}


class Monitor:
    """Монитор производительности"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Инициализация монитора.

        Args:
            config: Словарь с параметрами мониторинга
        """
        self.config = MonitorConfig(**(config or {}))
        self._setup_logger()
        self._setup_monitor()
        self.active_strategies: List[str] = []
        self.strategy_metrics: Dict[str, float] = {}
        self.alerts: List[str] = []

    def _setup_logger(self):
        """Настройка логгера"""
        logger.add(
            f"{self.config.log_dir}/monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
        )

    def _setup_monitor(self):
        """Настройка монитора"""
        try:
            # Создаем директорию для логов
            Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)

            # Инициализируем очередь для метрик
            self.metrics_queue = queue.Queue()

            # Запускаем поток мониторинга
            self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()

        except Exception as e:
            logger.error(f"Error setting up monitor: {str(e)}")

    def _monitor_loop(self):
        """Цикл мониторинга"""
        try:
            while True:
                # Собираем метрики
                metrics = self._collect_metrics()

                # Проверяем пороги
                self._check_thresholds(metrics)

                # Сохраняем метрики
                if self.config.save_metrics:
                    self._save_metrics(metrics)

                # Ждем следующего интервала
                time.sleep(self.config.interval)

        except Exception as e:
            logger.error(f"Error in monitor loop: {str(e)}")

    def _collect_metrics(self) -> Dict[str, Any]:
        """
        Сбор метрик.

        Returns:
            Dict с метриками
        """
        try:
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

        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
            return {}

    def _check_thresholds(self, metrics: Dict[str, Any]):
        """
        Проверка порогов.

        Args:
            metrics: Словарь с метриками
        """
        try:
            for metric, value in metrics.items():
                if metric in self.config.alert_threshold:
                    threshold = self.config.alert_threshold[metric]
                    if value > threshold:
                        logger.warning(f"Alert: {metric} = {value} > {threshold}")

        except Exception as e:
            logger.error(f"Error checking thresholds: {str(e)}")

    def _save_metrics(self, metrics: Dict[str, Any]):
        """
        Сохранение метрик.

        Args:
            metrics: Словарь с метриками
        """
        try:
            # Формируем имя файла
            filename = f"metrics_{datetime.now().strftime('%Y%m%d')}.json"
            filepath = Path(self.config.log_dir) / filename

            # Читаем существующие метрики
            if filepath.exists():
                with open(filepath, "r") as f:
                    existing_metrics = json.load(f)
            else:
                existing_metrics = []

            # Добавляем новые метрики
            existing_metrics.append(metrics)

            # Сохраняем метрики
            with open(filepath, "w") as f:
                json.dump(existing_metrics, f, indent=4)

        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")

    def get_metrics(
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
            # Получаем список файлов с метриками
            metric_files = list(Path(self.config.log_dir).glob("metrics_*.json"))

            # Читаем метрики
            metrics = []
            for file in metric_files:
                with open(file, "r") as f:
                    metrics.extend(json.load(f))

            # Преобразуем в DataFrame
            df = pd.DataFrame(metrics)

            # Фильтруем по времени
            if start_time:
                df = df[df["timestamp"] >= start_time.isoformat()]
            if end_time:
                df = df[df["timestamp"] <= end_time.isoformat()]

            return df

        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return pd.DataFrame()

    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Получение текущих метрик.

        Returns:
            Dict с текущими метриками
        """
        try:
            return self._collect_metrics()

        except Exception as e:
            logger.error(f"Error getting current metrics: {str(e)}")
            return {}

    def get_process_metrics(self, pid: int) -> Dict[str, Any]:
        """
        Получение метрик процесса.

        Args:
            pid: ID процесса

        Returns:
            Dict с метриками процесса
        """
        try:
            process = psutil.Process(pid)

            return {
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "memory_info": process.memory_info()._asdict(),
                "io_counters": (
                    process.io_counters()._asdict() if hasattr(process, "io_counters") else None
                ),
                "num_threads": process.num_threads(),
                "create_time": datetime.fromtimestamp(process.create_time()).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting process metrics: {str(e)}")
            return {}

    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Получение системных метрик.

        Returns:
            Dict с системными метриками
        """
        try:
            return {
                "cpu_count": psutil.cpu_count(),
                "cpu_freq": psutil.cpu_freq()._asdict(),
                "memory": psutil.virtual_memory()._asdict(),
                "swap": psutil.swap_memory()._asdict(),
                "disk": psutil.disk_usage("/")._asdict(),
                "network": psutil.net_io_counters()._asdict(),
            }

        except Exception as e:
            logger.error(f"Error getting system metrics: {str(e)}")
            return {}

    def add_strategy(self, strategy_name: str) -> None:
        """Добавление стратегии для мониторинга"""
        if strategy_name not in self.active_strategies:
            self.active_strategies.append(strategy_name)
            self.strategy_metrics[strategy_name] = 0.0

    def remove_strategy(self, strategy_name: str) -> None:
        """Удаление стратегии из мониторинга"""
        if strategy_name in self.active_strategies:
            self.active_strategies.remove(strategy_name)
            if strategy_name in self.strategy_metrics:
                del self.strategy_metrics[strategy_name]

    def update_metrics(self, strategy_name: str, metric_value: float) -> None:
        """Обновление метрик стратегии"""
        if strategy_name in self.active_strategies:
            self.strategy_metrics[strategy_name] = metric_value

    def add_alert(self, alert_message: str) -> None:
        """Добавление оповещения"""
        self.alerts.append(alert_message)
        logger.warning(f"Alert: {alert_message}")

    def __enter__(self):
        """Контекстный менеджер: вход"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер: выход"""
        pass
