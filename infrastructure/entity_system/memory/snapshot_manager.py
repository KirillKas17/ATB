"""Менеджер снимков памяти Entity System."""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class SnapshotManager:
    """Менеджер снимков памяти с промышленной реализацией."""

    def __init__(self, storage_path: Path = Path("./snapshots")):
        self.storage_path = storage_path
        self._snapshots: Dict[str, Dict[str, Any]] = {}
        self._snapshots_lock = threading.RLock()

        logger.info("SnapshotManager инициализирован")

    async def create_snapshot(self, data: Dict[str, Any]) -> str:
        """Создание снимка памяти."""
        with self._snapshots_lock:
            snapshot_id = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self._snapshots)}"
            timestamp = datetime.now()

            metadata = {
                "id": snapshot_id,
                "timestamp": timestamp.isoformat(),
                "size_bytes": 0,
                "format": "json",
            }

            snapshot_path = self.storage_path / f"{snapshot_id}.json"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)

            with open(snapshot_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

            metadata["size_bytes"] = snapshot_path.stat().st_size
            self._snapshots[snapshot_id] = metadata

            logger.info(f"Создан снимок {snapshot_id}")
            return snapshot_id

    async def load_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Загрузка снимка памяти."""
        with self._snapshots_lock:
            if snapshot_id not in self._snapshots:
                return None

            snapshot_path = self.storage_path / f"{snapshot_id}.json"

            if not snapshot_path.exists():
                return None

            try:
                with open(snapshot_path, "r", encoding="utf-8") as f:
                    data: Dict[str, Any] = json.load(f)
                return data
            except Exception as e:
                logger.error(f"Ошибка загрузки снимка {snapshot_id}: {e}")
                return None

    async def list_snapshots(self) -> List[Dict[str, Any]]:
        """Получение списка всех снимков."""
        with self._snapshots_lock:
            return list(self._snapshots.values())

    async def delete_snapshot(self, snapshot_id: str) -> bool:
        """Удаление снимка."""
        with self._snapshots_lock:
            if snapshot_id not in self._snapshots:
                return False

            try:
                snapshot_path = self.storage_path / f"{snapshot_id}.json"
                if snapshot_path.exists():
                    snapshot_path.unlink()

                del self._snapshots[snapshot_id]
                return True
            except Exception as e:
                logger.error(f"Ошибка удаления снимка {snapshot_id}: {e}")
                return False

    async def start(self) -> None:
        """Запуск менеджера снимков."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        logger.info("SnapshotManager запущен")

    async def stop(self) -> None:
        """Остановка менеджера снимков."""
        logger.info("SnapshotManager остановлен")
