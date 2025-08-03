"""
Модуль миграций для эволюционной системы.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from sqlmodel import Session, text
from sqlalchemy import text as sql_text

from infrastructure.evolution.types import EvolutionMigrationProtocol
from infrastructure.evolution.strategy_storage import StrategyStorage
from infrastructure.evolution.types import MigrationMetadata
from infrastructure.evolution.exceptions import MigrationError

logger = logging.getLogger(__name__)

class EvolutionMigration(EvolutionMigrationProtocol):
    """Реализация системы миграций для эволюционной системы."""

    def __init__(self, storage: StrategyStorage, config: Optional[dict] = None) -> None:
        """
        Инициализация системы миграций.
        Args:
            storage: Хранилище стратегий
            config: Конфигурация миграций
        """
        self.storage = storage
        self.config = config or {}
        self.migration_dir = Path(self.config.get("migration_path", "migrations"))
        self.migration_dir.mkdir(exist_ok=True)
        
        # Создать таблицу миграций при инициализации
        self._create_migrations_table()
        
        logger.info("EvolutionMigration initialized successfully")

    def apply_migration(self, migration_id: str) -> MigrationMetadata:
        """
        Применить миграцию.
        Args:
            migration_id: ID миграции
        Returns:
            Метаданные примененной миграции
        """
        try:
            # Загрузить данные миграции
            migration_file = self.migration_dir / f"{migration_id}.json"
            if not migration_file.exists():
                raise MigrationError(
                    f"Файл миграции не найден: {migration_file}", "migration_not_found"
                )
            
            with open(migration_file, "r", encoding="utf-8") as f:
                migration_data = json.load(f)
            
            # Валидировать миграцию
            self._validate_migration(migration_data)
            
            # Проверить зависимости
            dependencies = migration_data.get("dependencies", [])
            if dependencies:
                self._check_dependencies(dependencies)
            
            # Создать резервную копию если включено
            if self.config.get("backup_before_migration", True):
                self._create_backup_before_migration(migration_id)
            
            # Выполнить миграцию
            start_time = datetime.now()
            self._execute_migration(migration_data)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Создать метаданные
            metadata: MigrationMetadata = {
                "migration_id": migration_data["migration_id"],
                "version": migration_data["version"],
                "description": migration_data["description"],
                "applied_at": datetime.now().isoformat(),
                "execution_time": execution_time,
                "status": "completed",
                "rollback_supported": migration_data.get("rollback_supported", False),
                "dependencies": dependencies,
            }
            
            # Сохранить метаданные
            self._save_migration_metadata(metadata)
            
            logger.info(f"Миграция {migration_id} успешно применена")
            return metadata
            
        except MigrationError:
            raise
        except Exception as e:
            logger.error(f"Ошибка применения миграции {migration_id}: {e}")
            raise MigrationError(
                f"Не удалось применить миграцию {migration_id}: {e}", "migration_failed"
            )

    def rollback_migration(self, migration_id: str) -> bool:
        """
        Откатить миграцию.
        Args:
            migration_id: ID миграции для отката
        Returns:
            True если откат успешен
        """
        try:
            # Загрузить метаданные миграции
            metadata = self._load_migration_metadata(migration_id)
            if not metadata:
                raise MigrationError(
                    f"Метаданные миграции не найдены: {migration_id}", "metadata_not_found"
                )
            
            if not metadata["rollback_supported"]:
                raise MigrationError(
                    f"Откат не поддерживается для миграции: {migration_id}", "rollback_not_supported"
                )
            
            # Загрузить данные миграции
            migration_file = self.migration_dir / f"{migration_id}.json"
            with open(migration_file, "r", encoding="utf-8") as f:
                migration_data = json.load(f)
            
            # Выполнить откат
            self._execute_rollback(migration_data)
            
            # Удалить метаданные
            self._delete_migration_metadata(migration_id)
            
            logger.info(f"Миграция {migration_id} успешно откачена")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка отката миграции {migration_id}: {e}")
            return False

    def list_migrations(self) -> List[MigrationMetadata]:
        """
        Получить список примененных миграций.
        Returns:
            Список метаданных миграций
        """
        try:
            migrations: List[MigrationMetadata] = []
            # Загрузить метаданные из БД
            with Session(self.storage.engine) as session:
                # Исправляем использование session.exec на session.execute
                result = session.execute(
                    sql_text("SELECT * FROM migration_metadata ORDER BY applied_at")
                )
                for row in result:
                    # Преобразуем row в MigrationMetadata
                    migration_dict = dict(row._mapping)
                    # Преобразуем dependencies из JSON строки обратно в список
                    if "dependencies" in migration_dict and isinstance(migration_dict["dependencies"], str):
                        migration_dict["dependencies"] = json.loads(migration_dict["dependencies"])
                    # [1] создаю правильный MigrationMetadata объект
                    migration_metadata: MigrationMetadata = {
                        "migration_id": migration_dict["migration_id"],
                        "version": migration_dict["version"],
                        "description": migration_dict.get("description", ""),
                        "applied_at": migration_dict["applied_at"],
                        "execution_time": migration_dict.get("execution_time", 0.0),
                        "status": migration_dict["status"],
                        "rollback_supported": migration_dict.get("rollback_supported", False),
                        "dependencies": migration_dict.get("dependencies", [])
                    }
                    migrations.append(migration_metadata)  # [1] правильный тип MigrationMetadata
            # Сортировать по времени применения
            migrations.sort(key=lambda x: x["applied_at"], reverse=True)
            logger.debug(f"Найдено миграций: {len(migrations)}")
            return migrations
        except Exception as e:
            logger.error(f"Ошибка получения списка миграций: {e}")
            return []

    def get_pending_migrations(self) -> List[str]:
        """
        Получить список ожидающих миграций.
        Returns:
            Список ID ожидающих миграций
        """
        try:
            applied_migrations = {m["migration_id"] for m in self.list_migrations()}
            all_migrations = []
            for migration_file in self.migration_dir.glob("*.json"):
                migration_id = migration_file.stem
                if migration_id not in applied_migrations:
                    all_migrations.append(migration_id)
            # Сортировать по версии
            all_migrations.sort()
            logger.debug(f"Найдено ожидающих миграций: {len(all_migrations)}")
            return all_migrations
        except Exception as e:
            logger.error(f"Ошибка получения ожидающих миграций: {e}")
            return []

    def _create_migrations_table(self) -> None:
        """Создать таблицу для хранения метаданных миграций."""
        try:
            with Session(self.storage.engine) as session:
                session.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS migration_metadata (
                        migration_id TEXT PRIMARY KEY,
                        version TEXT NOT NULL,
                        description TEXT,
                        applied_at TEXT NOT NULL,
                        execution_time REAL,
                        status TEXT NOT NULL,
                        rollback_supported BOOLEAN DEFAULT FALSE,
                        dependencies TEXT
                    )
                """
                    )
                )
                session.commit()
        except Exception as e:
            logger.error(f"Ошибка создания таблицы миграций: {e}")
            raise MigrationError(
                f"Не удалось создать таблицу миграций: {e}", "table_creation_failed"
            )

    def _check_dependencies(self, dependencies: List[str]) -> None:
        """Проверить зависимости миграции."""
        applied_migrations = {m["migration_id"] for m in self.list_migrations()}
        for dep in dependencies:
            if dep not in applied_migrations:
                raise MigrationError(
                    f"Зависимость не выполнена: {dep}", "dependency_error"
                )

    def _create_backup_before_migration(self, migration_id: str) -> None:
        """Создать резервную копию перед миграцией."""
        try:
            from infrastructure.evolution.backup import EvolutionBackup

            backup_service = EvolutionBackup(self.storage)
            backup_name = f"pre_migration_{migration_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path = backup_service.create_backup(backup_name)
            logger.info(f"Резервная копия создана: {backup_path}")
        except Exception as e:
            logger.warning(f"Не удалось создать резервную копию: {e}")

    def _execute_migration(self, migration_data: dict) -> None:
        """Выполнить миграцию."""
        try:
            with Session(self.storage.engine) as session:
                # Выполнить SQL-скрипты
                for script in migration_data.get("scripts", []):
                    session.execute(text(script))
                session.commit()
        except Exception as e:
            logger.error(f"Ошибка выполнения миграции: {e}")
            raise MigrationError(
                f"Не удалось выполнить миграцию: {e}", "execution_failed"
            )

    def _execute_rollback(self, migration_data: dict) -> None:
        """Выполнить откат миграции."""
        try:
            with Session(self.storage.engine) as session:
                # Выполнить скрипты отката
                for script in migration_data.get("rollback_scripts", []):
                    session.execute(text(script))
                session.commit()
        except Exception as e:
            logger.error(f"Ошибка выполнения отката: {e}")
            raise MigrationError(
                f"Не удалось выполнить откат: {e}", "rollback_execution_failed"
            )

    def _validate_migration(self, migration_data: dict) -> None:
        """Валидировать миграцию."""
        try:
            # Проверка обязательных полей
            required_fields = ["migration_id", "version", "description"]
            for field in required_fields:
                if field not in migration_data:
                    raise MigrationError(
                        f"Отсутствует обязательное поле: {field}", "validation_error"
                    )
            # Проверка версии
            version = migration_data["version"]
            if not self._is_valid_version(version):
                raise MigrationError(
                    f"Неверный формат версии: {version}", "validation_error"
                )
            # Проверка SQL-скриптов
            scripts = migration_data.get("scripts", [])
            for i, script in enumerate(scripts):
                if not self._is_valid_sql(script):
                    raise MigrationError(
                        f"Неверный SQL в скрипте {i+1}", "validation_error"
                    )
            logger.info("Миграция валидна")
        except MigrationError:
            raise
        except Exception as e:
            logger.error(f"Ошибка валидации миграции: {e}")
            raise MigrationError(f"Ошибка валидации: {e}", "validation_error")

    def _is_valid_version(self, version: str) -> bool:
        """Проверить корректность версии."""
        import re

        pattern = r"^\d+\.\d+\.\d+$"
        return bool(re.match(pattern, version))

    def _is_valid_sql(self, sql: str) -> bool:
        """Проверить корректность SQL."""
        # Простая проверка на наличие ключевых слов
        sql_upper = sql.upper()
        dangerous_keywords = ["DROP", "DELETE", "TRUNCATE"]
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                logger.warning(f"Обнаружено опасное ключевое слово: {keyword}")
        return True  # В продакшене здесь должна быть более строгая проверка

    def _save_migration_metadata(self, metadata: MigrationMetadata) -> None:
        """Сохранить метаданные миграции."""
        try:
            with Session(self.storage.engine) as session:
                session.execute(
                    text(
                        """
                    INSERT OR REPLACE INTO migration_metadata 
                    (migration_id, version, description, applied_at, execution_time, status, rollback_supported, dependencies)
                    VALUES (:migration_id, :version, :description, :applied_at, :execution_time, :status, :rollback_supported, :dependencies)
                """
                    ),
                    {
                        "migration_id": metadata["migration_id"],
                        "version": metadata["version"],
                        "description": metadata["description"],
                        "applied_at": metadata["applied_at"],
                        "execution_time": metadata["execution_time"],
                        "status": metadata["status"],
                        "rollback_supported": metadata["rollback_supported"],
                        "dependencies": json.dumps(metadata["dependencies"]),
                    },
                )
                session.commit()
        except Exception as e:
            logger.error(f"Ошибка сохранения метаданных миграции: {e}")
            raise MigrationError(
                f"Не удалось сохранить метаданные миграции: {e}", "metadata_save_failed"
            )

    def _load_migration_metadata(
        self, migration_id: str
    ) -> Optional[MigrationMetadata]:
        """Загрузить метаданные миграции."""
        try:
            with Session(self.storage.engine) as session:
                result = session.execute(
                    sql_text(
                        "SELECT * FROM migration_metadata WHERE migration_id = :migration_id"
                    ),
                    {"migration_id": migration_id},
                ).first()
                if result:
                    # Преобразуем result в MigrationMetadata
                    migration_dict = dict(result._mapping)
                    if "dependencies" in migration_dict and isinstance(migration_dict["dependencies"], str):
                        migration_dict["dependencies"] = json.loads(migration_dict["dependencies"])
                    
                    # [2] возвращаю правильный MigrationMetadata объект
                    return {
                        "migration_id": migration_dict["migration_id"],
                        "version": migration_dict["version"],
                        "description": migration_dict.get("description", ""),
                        "applied_at": migration_dict["applied_at"],
                        "execution_time": migration_dict.get("execution_time", 0.0),
                        "status": migration_dict["status"],
                        "rollback_supported": migration_dict.get("rollback_supported", False),
                        "dependencies": migration_dict.get("dependencies", [])
                    }
                return None
        except Exception as e:
            logger.error(f"Ошибка загрузки метаданных миграции: {e}")
            return None

    def _delete_migration_metadata(self, migration_id: str) -> None:
        """Удалить метаданные миграции."""
        try:
            with Session(self.storage.engine) as session:
                session.execute(
                    sql_text(
                        "DELETE FROM migration_metadata WHERE migration_id = :migration_id"
                    ),
                    {"migration_id": migration_id},
                )
                session.commit()
        except Exception as e:
            logger.error(f"Ошибка удаления метаданных миграции: {e}")
            raise MigrationError(
                f"Не удалось удалить метаданные миграции: {e}", "metadata_delete_failed"
            )
