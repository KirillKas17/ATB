"""Менеджер резервных копий для системы улучшений."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class BackupManager:
    """Менеджер резервных копий."""

    def __init__(self) -> None:
        self.backup_dir = Path("entity/backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.max_backups: int = 10

    async def create_backup(self, improvement_id: str) -> Optional[str]:
        """Создание резервной копии проекта."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{improvement_id}_{timestamp}"
            backup_path = self.backup_dir / backup_name
            # Создание директории для резервной копии
            backup_path.mkdir(parents=True, exist_ok=True)
            # Создание резервной копии проекта
            await self._backup_project(backup_path)
            # Создание метаданных резервной копии
            metadata = {
                "improvement_id": improvement_id,
                "created_at": datetime.now().isoformat(),
                "backup_path": str(backup_path),
                "size": await self._calculate_backup_size(backup_path),
            }
            metadata_file = backup_path / "backup_metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Создана резервная копия: {backup_path}")
            return str(backup_path)
        except Exception as e:
            logger.error(f"Ошибка создания резервной копии: {e}")
            return None

    async def _backup_project(self, backup_path: Path) -> None:
        """Создание резервной копии проекта."""
        try:
            # Копирование основных директорий проекта
            project_dirs = [
                "domain",
                "application",
                "infrastructure",
                "interfaces",
                "shared",
                "config",
            ]
            for dir_name in project_dirs:
                src_dir = Path(dir_name)
                if src_dir.exists():
                    dst_dir = backup_path / dir_name
                    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            # Копирование важных файлов
            important_files = [
                "main.py",
                "requirements.txt",
                "setup.py",
                "README.md",
                "config.yaml",
            ]
            for file_name in important_files:
                src_file = Path(file_name)
                if src_file.exists():
                    dst_file = backup_path / file_name
                    shutil.copy2(src_file, dst_file)
            logger.info(f"Проект скопирован в {backup_path}")
        except Exception as e:
            logger.error(f"Ошибка копирования проекта: {e}")
            raise

    async def rollback(self, backup_path: str) -> bool:
        """Откат к резервной копии."""
        try:
            backup_path_obj = Path(backup_path)
            if not backup_path_obj.exists():
                logger.error(f"Резервная копия не найдена: {backup_path}")
                return False
            logger.info(f"Начало отката к резервной копии: {backup_path}")
            # Восстановление из резервной копии
            await self._restore_from_backup(backup_path_obj)
            logger.info(f"Откат завершен успешно: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка отката: {e}")
            return False

    async def _restore_from_backup(self, backup_path: Path) -> None:
        """Восстановление из резервной копии."""
        try:
            # Восстановление директорий проекта
            project_dirs = [
                "domain",
                "application",
                "infrastructure",
                "interfaces",
                "shared",
                "config",
            ]
            for dir_name in project_dirs:
                src_dir = backup_path / dir_name
                if src_dir.exists():
                    dst_dir = Path(dir_name)
                    if dst_dir.exists():
                        shutil.rmtree(dst_dir)
                    shutil.copytree(src_dir, dst_dir)
            # Восстановление важных файлов
            important_files = [
                "main.py",
                "requirements.txt",
                "setup.py",
                "README.md",
                "config.yaml",
            ]
            for file_name in important_files:
                src_file = backup_path / file_name
                if src_file.exists():
                    dst_file = Path(file_name)
                    shutil.copy2(src_file, dst_file)
            logger.info(f"Восстановление из {backup_path} завершено")
        except Exception as e:
            logger.error(f"Ошибка восстановления: {e}")
            raise

    async def _calculate_backup_size(self, backup_path: Path) -> int:
        """Расчет размера резервной копии."""
        try:
            total_size = 0
            for file_path in backup_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size
        except Exception as e:
            logger.error(f"Ошибка расчета размера резервной копии: {e}")
            return 0

    async def cleanup_old_backups(self) -> None:
        """Очистка старых резервных копий."""
        try:
            backups = []
            for backup_dir in self.backup_dir.iterdir():
                if backup_dir.is_dir():
                    metadata_file = backup_dir / "backup_metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                        backups.append((backup_dir, metadata))
            # Сортировка по времени создания
            backups.sort(key=lambda x: x[1]["created_at"], reverse=True)
            # Удаление старых резервных копий
            for backup_dir, metadata in backups[self.max_backups :]:
                shutil.rmtree(backup_dir)
                logger.info(f"Удалена старая резервная копия: {backup_dir}")
        except Exception as e:
            logger.error(f"Ошибка очистки старых резервных копий: {e}")

    def get_backup_info(self, backup_path: str) -> Optional[Dict[str, Any]]:
        """Получение информации о резервной копии."""
        try:
            metadata_file = Path(backup_path) / "backup_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata: Dict[str, Any] = json.load(f)
                    return metadata
            return None
        except Exception as e:
            logger.error(f"Ошибка получения информации о резервной копии: {e}")
            return None

    def list_backups(self) -> List[Dict[str, Any]]:
        """Получение списка резервных копий."""
        backups = []
        try:
            for backup_dir in self.backup_dir.iterdir():
                if backup_dir.is_dir():
                    metadata_file = backup_dir / "backup_metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                        backups.append(metadata)
            # Сортировка по времени создания
            backups.sort(key=lambda x: x["created_at"], reverse=True)
            return backups
        except Exception as e:
            logger.error(f"Ошибка получения списка резервных копий: {e}")
            return []
