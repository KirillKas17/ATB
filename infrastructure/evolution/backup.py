"""
Модуль резервного копирования для infrastructure/evolution слоя.
"""

import gzip
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union, Any, cast
from uuid import uuid4

from sqlalchemy.orm import Session

from domain.evolution.strategy_fitness import StrategyEvaluationResult
from domain.evolution.strategy_model import EvolutionContext, StrategyCandidate
from infrastructure.evolution.exceptions import BackupError
from infrastructure.evolution.serializers import (
    candidate_to_model,
    context_to_model,
    evaluation_to_model,
)
from infrastructure.evolution.storage import StrategyStorage
from infrastructure.evolution.types import (
    BackupFormat,
    BackupMetadata,
    BackupPath,
    EvolutionBackupProtocol,
)

logger = logging.getLogger(__name__)


class EvolutionBackup(EvolutionBackupProtocol):
    """
    Система резервного копирования для эволюционных стратегий.
    Поддерживает различные форматы экспорта с возможностью сжатия.
    """

    def __init__(
        self,
        storage_or_config: Optional[Union[StrategyStorage, dict, str]] = None,
        config: Optional[dict] = None,
    ) -> None:
        """Инициализация системы резервного копирования."""
        if isinstance(storage_or_config, StrategyStorage):
            self.storage: Optional[StrategyStorage] = storage_or_config
            config = config or {}
        elif isinstance(storage_or_config, dict):
            self.storage = None  # Для тестов может быть None
            config = storage_or_config
        elif isinstance(storage_or_config, str):
            self.storage = None  # Для тестов может быть None
            config = {"backup_path": storage_or_config}
        else:
            self.storage = None
            config = config or {}
        # Устанавливаем конфигурацию по умолчанию
        default_config = {
            "backup_path": "backups/evolution_backup",
            "backup_format": "json",
            "enable_compression": True,
            "compression_enabled": True,  # Алиас для совместимости
            "compression_level": 6,
            "max_backups": 10,
            "auto_backup": False,
            "backup_interval": 86400,  # 24 часа
            "encryption_enabled": False,
            "encryption_key": None,
        }
        # Объединяем с переданной конфигурацией
        self.config = {**default_config, **config}
        # Обрабатываем backup_dir
        backup_path = self.config["backup_path"]
        if isinstance(backup_path, (str, Path)):
            self.backup_dir = Path(backup_path)
        else:
            raise BackupError(
                f"Некорректный тип backup_path: {type(backup_path)}", "invalid_config"
            )
        # Создаем директорию, если она не существует
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise BackupError(
                f"Не удалось создать директорию бэкапа: {e}",
                "directory_creation_failed",
            )
        logger.info(
            f"Система резервного копирования инициализирована: {self.backup_dir}"
        )

    @property
    def max_backups(self) -> int:
        """Максимальное количество резервных копий."""
        value = self.config.get("max_backups", 10)
        if not isinstance(value, int):
            return 10
        return value

    @property
    def compression_enabled(self) -> bool:
        """Включено ли сжатие."""
        value = self.config.get(
            "compression_enabled", self.config.get("enable_compression", True)
        )
        if not isinstance(value, bool):
            return True
        return value

    @property
    def encryption_enabled(self) -> bool:
        """Включено ли шифрование."""
        value = self.config.get("encryption_enabled", False)
        if not isinstance(value, bool):
            return False
        return value

    def create_backup(
        self, backup_path: Optional[Union[str, Path]] = None
    ) -> BackupMetadata:
        """
        Создать резервную копию.
        Args:
            backup_path: Путь для сохранения резервной копии (опционально)
        Returns:
            Метаданные созданной резервной копии
        Raises:
            BackupError: При ошибке создания резервной копии
        """
        try:
            backup_id = str(uuid4())
            timestamp = datetime.now()
            if backup_path is None:
                backup_path = self._generate_backup_path(backup_id, timestamp)
            # Преобразуем в строку для совместимости
            backup_path_str = str(backup_path)
            # Экспортировать данные
            export_data = self._prepare_backup_data()
            # Сохранить в файл
            if self.config.get(
                "compression_enabled", self.config.get("enable_compression", True)
            ):
                backup_path_str = f"{backup_path_str}.gz"
                self._save_compressed_backup(export_data, backup_path_str)
            else:
                self._save_backup(export_data, backup_path_str)
            # Создать метаданные
            metadata = self._create_backup_metadata(
                backup_id, backup_path_str, timestamp, export_data
            )
            # Сохранить метаданные
            self._save_backup_metadata(backup_id, metadata)
            # Очистить старые резервные копии
            self._cleanup_old_backups()
            logger.info(f"Резервная копия создана: {backup_id}")
            return metadata
        except Exception as e:
            logger.error(f"Ошибка создания резервной копии: {e}")
            raise BackupError(
                f"Не удалось создать резервную копию: {e}", "backup_failed"
            )

    def restore_backup(self, backup_id: str) -> bool:
        """
        Восстановить резервную копию.
        Args:
            backup_id: ID резервной копии для восстановления
        Returns:
            True, если восстановление прошло успешно
        Raises:
            BackupError: При ошибке восстановления
        """
        try:
            # Загрузить метаданные
            metadata = self._load_backup_metadata(backup_id)
            if not metadata:
                raise BackupError(
                    f"Метаданные резервной копии не найдены: {backup_id}",
                    "backup_not_found",
                )
            # Загрузить данные резервной копии
            backup_data = self._load_backup_data(metadata["backup_path"])
            # Восстановить данные
            self._restore_backup_data(backup_data)
            logger.info(f"Резервная копия восстановлена: {backup_id}")
            return True
        except Exception as e:
            logger.error(f"Ошибка восстановления резервной копии: {e}")
            raise BackupError(
                f"Не удалось восстановить резервную копию: {e}", "restore_failed"
            )

    def list_backups(self) -> List[BackupMetadata]:
        """
        Получить список резервных копий.
        Returns:
            Список метаданных резервных копий
        """
        try:
            backups: List[BackupMetadata] = []
            metadata_dir = self.backup_dir / "metadata"
            if not metadata_dir.exists():
                return backups
            for metadata_file in metadata_dir.glob("*.json"):
                try:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                        backups.append(metadata)
                except Exception as e:
                    logger.warning(f"Ошибка загрузки метаданных {metadata_file}: {e}")
            # Сортировать по времени создания
            backups.sort(key=lambda x: x["backup_time"], reverse=True)
            logger.debug(f"Найдено резервных копий: {len(backups)}")
            return backups
        except Exception as e:
            logger.error(f"Ошибка получения списка резервных копий: {e}")
            return []

    def delete_backup(self, backup_id: str) -> bool:
        """
        Удалить резервную копию.
        Args:
            backup_id: ID резервной копии для удаления
        Returns:
            True, если удаление прошло успешно
        Raises:
            BackupError: При ошибке удаления
        """
        try:
            # Загрузить метаданные
            metadata = self._load_backup_metadata(backup_id)
            if not metadata:
                logger.warning(f"Резервная копия не найдена: {backup_id}")
                return False
            # Удалить файл резервной копии
            backup_file = Path(metadata["backup_path"])
            if backup_file.exists():
                backup_file.unlink()
            # Удалить метаданные
            metadata_file = self.backup_dir / "metadata" / f"{backup_id}.json"
            if metadata_file.exists():
                metadata_file.unlink()
            logger.info(f"Резервная копия удалена: {backup_id}")
            return True
        except Exception as e:
            logger.error(f"Ошибка удаления резервной копии: {e}")
            raise BackupError(
                f"Не удалось удалить резервную копию: {e}", "delete_failed"
            )

    def _generate_backup_path(self, backup_id: str, timestamp: datetime) -> str:
        """Сгенерировать путь для резервной копии."""
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        return str(self.backup_dir / f"evolution_backup_{timestamp_str}_{backup_id}")

    def _prepare_backup_data(self) -> dict[str, Any]:
        """Подготовка данных для резервного копирования."""
        return {
            "candidates": [],
            "evaluations": [],
            "contexts": [],
            "backup_time": datetime.now().isoformat(),
            "version": "1.0",
        }

    def _save_backup(self, data: dict, backup_path: str) -> None:
        """Сохранение резервной копии в файл."""
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_compressed_backup(self, data: dict, backup_path: str) -> None:
        """Сохранение сжатой резервной копии."""
        compression_level = self.config.get("compression_level", 6)
        if not isinstance(compression_level, int):
            compression_level = 6
        with gzip.open(
            backup_path, "wt", encoding="utf-8", compresslevel=compression_level
        ) as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _create_backup_metadata(
        self, backup_id: str, backup_path: str, timestamp: datetime, data: dict
    ) -> BackupMetadata:
        """Создание метаданных резервной копии."""
        backup_format = self.config.get("backup_format", "json")
        if not isinstance(backup_format, str) or backup_format not in [
            "json",
            "sql",
            "pickle",
            "yaml",
        ]:
            backup_format = "json"
        # Приводим к правильному типу BackupFormat
        backup_format_typed: BackupFormat = cast(BackupFormat, backup_format)
        return {
            "backup_id": backup_id,
            "backup_path": backup_path,
            "backup_time": timestamp.isoformat(),
            "backup_format": backup_format_typed,
            "backup_size_bytes": len(json.dumps(data)),
            "compression_ratio": 1.0,  # Будет рассчитано позже
            "checksum": str(hash(json.dumps(data, sort_keys=True))),
            "version": "1.0",
            "description": f"Backup created at {timestamp.isoformat()}",
        }

    def _save_backup_metadata(self, backup_id: str, metadata: BackupMetadata) -> None:
        """Сохранение метаданных резервной копии."""
        metadata_dir = self.backup_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        metadata_file = metadata_dir / f"{backup_id}.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _load_backup_metadata(self, backup_id: str) -> Optional[BackupMetadata]:
        """Загрузка метаданных резервной копии."""
        metadata_file = self.backup_dir / "metadata" / f"{backup_id}.json"
        if not metadata_file.exists():
            return None
        with open(metadata_file, "r", encoding="utf-8") as f:
            return cast(BackupMetadata, json.load(f))

    def _load_backup_data(self, backup_path: str) -> dict[str, Any]:
        """Загрузка данных резервной копии."""
        backup_file = Path(backup_path)
        if backup_path.endswith(".gz"):
            with gzip.open(backup_file, "rt", encoding="utf-8") as f:
                return cast(dict, json.load(f))
        else:
            with open(backup_file, "r", encoding="utf-8") as f:
                return cast(dict, json.load(f))

    def _restore_backup_data(self, backup_data: dict) -> None:
        """Восстановление данных из резервной копии."""
        try:
            # Валидация структуры
            required_keys = ["candidates", "evaluations", "contexts"]
            for key in required_keys:
                if key not in backup_data:
                    raise BackupError(
                        f"В резервной копии отсутствует раздел: {key}", "invalid_backup"
                    )
            # Атомарная операция: используем транзакцию
            if self.storage is not None:
                with Session(self.storage.engine) as session:
                    # Очищаем существующие данные
                    from sqlalchemy import text
                    session.execute(text("DELETE FROM strategy_candidates"))
                    session.execute(text("DELETE FROM strategy_evaluations"))
                    session.execute(text("DELETE FROM evolution_contexts"))
                    # Восстанавливаем кандидатов
                    for candidate in backup_data["candidates"]:
                        candidate_obj = StrategyCandidate.from_dict(candidate)
                        candidate_model: Any = candidate_to_model(candidate_obj)
                        session.add(candidate_model)
                    # Восстанавливаем оценки
                    for evaluation in backup_data["evaluations"]:
                        evaluation_obj = StrategyEvaluationResult.from_dict(evaluation)
                        evaluation_model: Any = evaluation_to_model(evaluation_obj)
                        session.add(evaluation_model)
                    # Восстанавливаем контексты
                    for context in backup_data["contexts"]:
                        context_obj = EvolutionContext.from_dict(context)
                        context_model: Any = context_to_model(context_obj)
                        session.add(context_model)
                    session.commit()
            logger.info("Данные успешно восстановлены из резервной копии")
        except Exception as e:
            logger.error(f"Ошибка восстановления данных из резервной копии: {e}")
            raise BackupError(f"Ошибка восстановления данных: {e}", "restore_failed")

    def _cleanup_old_backups(self) -> None:
        """Очистить старые резервные копии."""
        try:
            max_backups = self.config.get("max_backups", 10)
            if not isinstance(max_backups, int) or max_backups <= 0:
                max_backups = 10
            backups = self.list_backups()
            if len(backups) <= max_backups:
                return
            # Сортировать по времени создания (новые в начале)
            backups.sort(key=lambda x: x["backup_time"], reverse=True)
            # Удалить старые резервные копии
            for backup in backups[max_backups:]:
                try:
                    self.delete_backup(backup["backup_id"])
                except Exception as e:
                    logger.warning(
                        f"Не удалось удалить старую резервную копию {backup['backup_id']}: {e}"
                    )
        except Exception as e:
            logger.error(f"Ошибка очистки старых резервных копий: {e}")
