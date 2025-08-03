import asyncio
import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger


@dataclass
class StateConfig:
    """Конфигурация менеджера состояний"""

    state_dir: str = "ml/state"
    cache_size: int = 1000
    save_interval: int = 60  # секунд
    max_history_size: int = 1000
    compression: bool = True
    backup_enabled: bool = True
    backup_interval: int = 3600  # секунд
    max_backups: int = 10


@dataclass
class StateMetrics:
    """Метрики состояния"""

    last_update: datetime
    update_count: int
    error_count: int
    cache_hits: int
    cache_misses: int
    save_count: int
    load_count: int
    backup_count: int
    total_size: int
    compression_ratio: float


class StateManager:
    """Управление состоянием системы"""

    def __init__(self, config: Optional[StateConfig] = None) -> None:
        """Инициализация менеджера состояний"""
        self.config = config or StateConfig()
        self.state_dir = Path(self.config.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        # Пути к файлам
        self.optimal_window_path = self.state_dir / "optimal_window.json"
        self.regime_history_path = self.state_dir / "regime_history.json"
        self.confidence_path = self.state_dir / "prediction_confidence.json"
        self.metrics_path = self.state_dir / "metrics.json"
        self.backup_dir = self.state_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        # Состояния
        self.optimal_window: Dict[str, int] = self._load_json(
            self.optimal_window_path, default={}
        )
        self.regime_history: Dict[str, List[Dict[str, str]]] = self._load_json(
            self.regime_history_path, default={}
        )
        self.confidence: Dict[str, float] = self._load_json(
            self.confidence_path, default={}
        )
        self.metrics: Dict[str, Union[str, int, float]] = self._load_json(
            self.metrics_path, default=self._create_default_metrics()
        )
        # Кэш
        self.cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        # Блокировки
        self.lock = threading.Lock()
        self.save_lock = asyncio.Lock()
        # Задачи
        self.save_task: Optional[asyncio.Task] = None
        self.backup_task: Optional[asyncio.Task] = None
        # Запуск задач
        self._start_tasks()

    def _create_default_metrics(self) -> Dict[str, Union[str, int, float]]:
        """Создание метрик по умолчанию"""
        return {
            "last_update": datetime.now().isoformat(),
            "update_count": 0,
            "error_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "save_count": 0,
            "load_count": 0,
            "backup_count": 0,
            "total_size": 0,
            "compression_ratio": 1.0,
        }

    def _start_tasks(self) -> None:
        """Запуск фоновых задач"""
        loop = asyncio.get_event_loop()
        self.save_task = loop.create_task(self._periodic_save())
        if self.config.backup_enabled:
            self.backup_task = loop.create_task(self._periodic_backup())

    async def _periodic_save(self) -> None:
        """Периодическое сохранение состояния"""
        while True:
            try:
                await asyncio.sleep(self.config.save_interval)
                await self.save_all()
            except Exception as e:
                logger.error(f"Ошибка периодического сохранения: {e}")
                self.metrics["error_count"] = int(self.metrics["error_count"]) + 1

    async def _periodic_backup(self) -> None:
        """Периодическое резервное копирование"""
        while True:
            try:
                await asyncio.sleep(self.config.backup_interval)
                await self._create_backup()
            except Exception as e:
                logger.error(f"Ошибка резервного копирования: {e}")
                self.metrics["error_count"] = int(self.metrics["error_count"]) + 1

    async def _create_backup(self) -> None:
        """Создание резервной копии"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"backup_{timestamp}"
            backup_path.mkdir(exist_ok=True)
            # Копирование файлов
            for file in self.state_dir.glob("*.json"):
                if file.name != "metrics.json":
                    await self._copy_file(file, backup_path / file.name)
            # Очистка старых бэкапов
            backups = sorted(self.backup_dir.glob("backup_*"))
            if len(backups) > self.config.max_backups:
                for old_backup in backups[: -self.config.max_backups]:
                    await self._remove_dir(old_backup)
            self.metrics["backup_count"] = int(self.metrics["backup_count"]) + 1
            logger.info(f"Создана резервная копия: {backup_path}")
        except Exception as e:
            logger.error(f"Ошибка создания резервной копии: {e}")
            self.metrics["error_count"] = int(self.metrics["error_count"]) + 1

    async def _copy_file(self, src: Path, dst: Path) -> None:
        """Асинхронное копирование файла"""
        try:
            with open(src, "rb") as fsrc:
                with open(dst, "wb") as fdst:
                    await asyncio.get_event_loop().run_in_executor(
                        None, lambda: fdst.write(fsrc.read())
                    )
        except Exception as e:
            logger.error(f"Ошибка копирования файла {src}: {e}")
            raise

    async def _remove_dir(self, path: Path) -> None:
        """Асинхронное удаление директории"""
        try:
            for file in path.glob("**/*"):
                if file.is_file():
                    file.unlink()
            path.rmdir()
        except Exception as e:
            logger.error(f"Ошибка удаления директории {path}: {e}")
            raise

    def _load_json(self, path: Path, default: Any) -> Any:
        """Загрузка JSON с кэшированием"""
        try:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.metrics["load_count"] = int(self.metrics["load_count"]) + 1
                self.metrics["cache_hits"] = int(self.metrics["cache_hits"]) + 1
                return data
            self.metrics["cache_misses"] = int(self.metrics["cache_misses"]) + 1
            return default
        except Exception as e:
            logger.error(f"Ошибка загрузки {path}: {e}")
            self.metrics["error_count"] = int(self.metrics["error_count"]) + 1
            return default

    async def _save_json(self, path: Path, data: Any) -> None:
        """Асинхронное сохранение JSON"""
        try:
            async with self.save_lock:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                self.metrics["save_count"] = int(self.metrics["save_count"]) + 1
                self.metrics["last_update"] = datetime.now().isoformat()
                self.metrics["total_size"] = path.stat().st_size
        except Exception as e:
            logger.error(f"Ошибка сохранения {path}: {e}")
            self.metrics["error_count"] = int(self.metrics["error_count"]) + 1
            raise

    async def save_all(self) -> None:
        """Сохранение всех состояний"""
        try:
            await asyncio.gather(
                self._save_json(self.optimal_window_path, self.optimal_window),
                self._save_json(self.regime_history_path, self.regime_history),
                self._save_json(self.confidence_path, self.confidence),
                self._save_json(self.metrics_path, self.metrics),
            )
        except Exception as e:
            logger.error(f"Ошибка сохранения состояний: {e}")
            self.metrics["error_count"] = int(self.metrics["error_count"]) + 1
            raise

    # --- Optimal window ---
    def get_optimal_window(self, pair: str) -> int:
        """Получение оптимального окна"""
        with self.lock:
            return self.optimal_window.get(pair, 300)

    async def set_optimal_window(self, pair: str, value: int) -> None:
        """Установка оптимального окна"""
        with self.lock:
            self.optimal_window[pair] = value
            await self._save_json(self.optimal_window_path, self.optimal_window)
            self.metrics["update_count"] = int(self.metrics["update_count"]) + 1

    # --- Regime history ---
    def get_regime_history(self, pair: str) -> List:
        """Получение истории режимов"""
        with self.lock:
            history = self.regime_history.get(pair, [])
            if len(history) > self.config.max_history_size:
                history = history[-self.config.max_history_size :]
            return history

    async def add_regime(self, pair: str, regime: str) -> None:
        """Добавление режима в историю"""
        with self.lock:
            if pair not in self.regime_history:
                self.regime_history[pair] = []
            self.regime_history[pair].append(
                {"regime": regime, "timestamp": datetime.now().isoformat()}
            )
            if len(self.regime_history[pair]) > self.config.max_history_size:
                self.regime_history[pair] = self.regime_history[pair][
                    -self.config.max_history_size :
                ]
            await self._save_json(self.regime_history_path, self.regime_history)
            self.metrics["update_count"] = int(self.metrics["update_count"]) + 1

    # --- Prediction confidence ---
    def get_confidence(self, pair: str) -> float:
        """Получение уверенности в предсказании"""
        with self.lock:
            return self.confidence.get(pair, 0.0)

    async def set_confidence(self, pair: str, value: float) -> None:
        """Установка уверенности в предсказании"""
        with self.lock:
            self.confidence[pair] = max(0.0, min(1.0, value))
            await self._save_json(self.confidence_path, self.confidence)
            self.metrics["update_count"] = int(self.metrics["update_count"]) + 1

    # --- Metrics ---
    def get_metrics(self) -> Dict:
        """Получение метрик"""
        with self.lock:
            return self.metrics.copy()

    async def reset_metrics(self) -> None:
        """Сброс метрик"""
        with self.lock:
            self.metrics = self._create_default_metrics()
            await self._save_json(self.metrics_path, self.metrics)

    async def cleanup(self) -> None:
        """Очистка ресурсов"""
        try:
            if self.save_task:
                self.save_task.cancel()
            if self.backup_task:
                self.backup_task.cancel()
            await self.save_all()
        except Exception as e:
            logger.error(f"Ошибка очистки: {e}")
            self.metrics["error_count"] = int(self.metrics["error_count"]) + 1
