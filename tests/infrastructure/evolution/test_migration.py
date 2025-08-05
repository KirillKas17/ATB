"""
Юнит-тесты для EvolutionMigration.
"""
import json
from pathlib import Path
from typing import Dict, Any, List
import pytest
from infrastructure.evolution.migration import EvolutionMigration
from infrastructure.evolution.exceptions import MigrationError
class TestEvolutionMigration:
    """Тесты для EvolutionMigration."""
    def test_init_default_config(self, temp_migration_dir: Path) -> None:
        """Тест инициализации с конфигурацией по умолчанию."""
        migration = EvolutionMigration(str(temp_migration_dir))
        assert migration.migrations_dir == temp_migration_dir
        assert migration.auto_migrate is True
        assert migration.backup_before_migrate is True
        assert migration.rollback_supported is True
    def test_init_custom_config(self, temp_migration_dir: Path) -> None:
        """Тест инициализации с пользовательской конфигурацией."""
        config = {
            "migrations_dir": str(temp_migration_dir),
            "auto_migrate": False,
            "backup_before_migrate": False,
            "rollback_supported": False
        }
        migration = EvolutionMigration(config)
        assert migration.migrations_dir == temp_migration_dir
        assert migration.auto_migrate is False
        assert migration.backup_before_migrate is False
        assert migration.rollback_supported is False
    def test_init_invalid_config(self: "TestEvolutionMigration") -> None:
        """Тест инициализации с некорректной конфигурацией."""
        config = {
            "migrations_dir": "/invalid/path",
            "auto_migrate": "invalid"
        }
        with pytest.raises(MigrationError) as exc_info:
            EvolutionMigration(config)
        assert "Некорректная конфигурация миграций" in str(exc_info.value)
    def test_create_migration_success(self, temp_migration_dir: Path) -> None:
        """Тест успешного создания миграции."""
        migration = EvolutionMigration(str(temp_migration_dir))
        migration_data = {
            "version": "1.1",
            "description": "Add new table for strategy metadata",
            "scripts": [
                "CREATE TABLE IF NOT EXISTS strategy_metadata (id INTEGER PRIMARY KEY, strategy_id TEXT, metadata TEXT)",
                "CREATE INDEX IF NOT EXISTS idx_strategy_metadata_strategy_id ON strategy_metadata(strategy_id)"
            ],
            "rollback_scripts": [
                "DROP TABLE IF EXISTS strategy_metadata"
            ],
            "rollback_supported": True,
            "dependencies": ["1.0"]
        }
        migration_path = migration.create_migration(migration_data)
        assert migration_path.exists()
        assert migration_path.suffix == ".json"
        # Проверить содержимое миграции
        with open(migration_path, "r", encoding="utf-8") as f:
            created_data = json.load(f)
        assert created_data["version"] == "1.1"
        assert created_data["description"] == "Add new table for strategy metadata"
        assert len(created_data["scripts"]) == 2
        assert len(created_data["rollback_scripts"]) == 1
        assert created_data["rollback_supported"] is True
        assert created_data["dependencies"] == ["1.0"]
    def test_create_migration_validation_error(self, temp_migration_dir: Path) -> None:
        """Тест ошибки валидации при создании миграции."""
        migration = EvolutionMigration(str(temp_migration_dir))
        # Некорректные данные миграции
        invalid_data = {
            "version": "",  # Пустая версия
            "description": "Test migration",
            "scripts": []
        }
        with pytest.raises(MigrationError) as exc_info:
            migration.create_migration(invalid_data)
        assert "Версия миграции обязательна" in str(exc_info.value)
    def test_create_migration_file_error(self, temp_migration_dir: Path, monkeypatch) -> None:
        """Тест ошибки создания файла миграции."""
        def mock_open(*args, **kwargs) -> Any:
            raise PermissionError("Permission denied")
        monkeypatch.setattr("builtins.open", mock_open)
        migration = EvolutionMigration(str(temp_migration_dir))
        migration_data = {
            "version": "1.0",
            "description": "Test migration",
            "scripts": ["CREATE TABLE test (id INTEGER)"]
        }
        with pytest.raises(MigrationError) as exc_info:
            migration.create_migration(migration_data)
        assert "Не удалось создать файл миграции" in str(exc_info.value)
    def test_load_migration_success(self, temp_migration_dir: Path, sample_migration_data: Dict[str, Any]) -> None:
        """Тест успешной загрузки миграции."""
        migration = EvolutionMigration(str(temp_migration_dir))
        # Создать файл миграции
        migration_path = temp_migration_dir / "1.0.json"
        with open(migration_path, "w", encoding="utf-8") as f:
            json.dump(sample_migration_data, f)
        # Загрузить миграцию
        loaded_data = migration.load_migration(migration_path)
        assert loaded_data["version"] == sample_migration_data["version"]
        assert loaded_data["description"] == sample_migration_data["description"]
        assert loaded_data["scripts"] == sample_migration_data["scripts"]
    def test_load_migration_file_not_found(self, temp_migration_dir: Path) -> None:
        """Тест загрузки несуществующего файла миграции."""
        migration = EvolutionMigration(str(temp_migration_dir))
        non_existent_path = temp_migration_dir / "non_existent.json"
        with pytest.raises(MigrationError) as exc_info:
            migration.load_migration(non_existent_path)
        assert "Файл миграции не найден" in str(exc_info.value)
    def test_load_migration_invalid_format(self, temp_migration_dir: Path) -> None:
        """Тест загрузки миграции с некорректным форматом."""
        migration = EvolutionMigration(str(temp_migration_dir))
        # Создать файл с некорректным JSON
        invalid_path = temp_migration_dir / "invalid.json"
        with open(invalid_path, "w", encoding="utf-8") as f:
            f.write("invalid json content")
        with pytest.raises(MigrationError) as exc_info:
            migration.load_migration(invalid_path)
        assert "Некорректный формат файла миграции" in str(exc_info.value)
    def test_list_migrations(self, temp_migration_dir: Path) -> None:
        """Тест получения списка миграций."""
        migration = EvolutionMigration(str(temp_migration_dir))
        # Создать несколько миграций
        migrations_data = [
            {"version": "1.0", "description": "Initial migration", "scripts": []},
            {"version": "1.1", "description": "Add table", "scripts": ["CREATE TABLE test"]},
            {"version": "1.2", "description": "Add index", "scripts": ["CREATE INDEX idx_test"]}
        ]
        for i, data in enumerate(migrations_data):
            migration_path = temp_migration_dir / f"{data['version']}.json"
            with open(migration_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        migrations = migration.list_migrations()
        assert len(migrations) == 3
        assert all(Path(str(migration_path)).suffix == ".json" for migration_path in migrations)
        # Проверить, что миграции отсортированы по версии
        versions = [Path(str(migration_path)).stem for migration_path in migrations]
        assert versions == ["1.0", "1.1", "1.2"]
    def test_list_migrations_empty(self, temp_migration_dir: Path) -> None:
        """Тест получения списка миграций из пустой директории."""
        migration = EvolutionMigration(str(temp_migration_dir))
        migrations = migration.list_migrations()
        assert len(migrations) == 0
    def test_get_migration_info(self, temp_migration_dir: Path, sample_migration_data: Dict[str, Any]) -> None:
        """Тест получения информации о миграции."""
        migration = EvolutionMigration(str(temp_migration_dir))
        # Создать файл миграции
        migration_path = temp_migration_dir / "1.0.json"
        with open(migration_path, "w", encoding="utf-8") as f:
            json.dump(sample_migration_data, f)
        # Получить информацию о миграции
        info = migration.get_migration_info(str(migration_path))  # [1] передаю str вместо Path
        assert info["path"] == str(migration_path)
        assert info["version"] == "1.0"
        assert info["description"] == "Test migration"
        assert info["scripts_count"] == 1
        assert info["rollback_scripts_count"] == 1
        assert info["rollback_supported"] is True
        assert info["dependencies"] == []
        assert info["size"] > 0
        assert info["created_at"] is not None
    def test_get_migration_info_not_found(self, temp_migration_dir: Path) -> None:
        """Тест получения информации о несуществующей миграции."""
        migration = EvolutionMigration(str(temp_migration_dir))
        non_existent_path = temp_migration_dir / "non_existent.json"
        with pytest.raises(MigrationError) as exc_info:
            migration.get_migration_info(str(non_existent_path))
        assert "Файл миграции не найден" in str(exc_info.value)
    def test_validate_migration_success(self, temp_migration_dir: Path, sample_migration_data: Dict[str, Any]) -> None:
        """Тест успешной валидации миграции."""
        migration = EvolutionMigration(str(temp_migration_dir))
        # Валидация должна пройти успешно
        migration._validate_migration(sample_migration_data)
    def test_validate_migration_missing_version(self, temp_migration_dir: Path) -> None:
        """Тест валидации миграции без версии."""
        migration = EvolutionMigration(str(temp_migration_dir))
        invalid_data = {
            "description": "Test migration",
            "scripts": []
        }
        with pytest.raises(MigrationError) as exc_info:
            migration._validate_migration(invalid_data)
        assert "Версия миграции обязательна" in str(exc_info.value)
    def test_validate_migration_missing_scripts(self, temp_migration_dir: Path) -> None:
        """Тест валидации миграции без скриптов."""
        migration = EvolutionMigration(str(temp_migration_dir))
        invalid_data = {
            "version": "1.0",
            "description": "Test migration"
        }
        with pytest.raises(MigrationError) as exc_info:
            migration._validate_migration(invalid_data)
        assert "Скрипты миграции обязательны" in str(exc_info.value)
    def test_validate_migration_invalid_scripts(self, temp_migration_dir: Path) -> None:
        """Тест валидации миграции с некорректными скриптами."""
        migration = EvolutionMigration(str(temp_migration_dir))
        invalid_data = {
            "version": "1.0",
            "description": "Test migration",
            "scripts": "not_a_list"
        }
        with pytest.raises(MigrationError) as exc_info:
            migration._validate_migration(invalid_data)
        assert "Скрипты миграции должны быть списком" in str(exc_info.value)
    def test_check_dependencies_success(self, temp_migration_dir: Path) -> None:
        """Тест успешной проверки зависимостей."""
        migration = EvolutionMigration(str(temp_migration_dir))
        # Создать зависимые миграции
        dependencies = [
            {"version": "1.0", "description": "Base migration", "scripts": []},
            {"version": "1.1", "description": "Dependent migration", "scripts": [], "dependencies": ["1.0"]}
        ]
        for data in dependencies:
            migration_path = temp_migration_dir / f"{data['version']}.json"
            with open(migration_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        # Проверить зависимости
        migration._check_dependencies(["1.0"])  # [2] правильное количество аргументов
    def test_check_dependencies_missing(self, temp_migration_dir: Path) -> None:
        """Тест проверки отсутствующих зависимостей."""
        migration = EvolutionMigration(str(temp_migration_dir))
        # Создать миграцию без зависимости
        migration_data = {
            "version": "1.1",
            "description": "Dependent migration",
            "scripts": [],
            "dependencies": ["1.0"]
        }
        migration_path = temp_migration_dir / "1.1.json"
        with open(migration_path, "w", encoding="utf-8") as f:
            json.dump(migration_data, f)
        with pytest.raises(MigrationError) as exc_info:
            migration._check_dependencies(["1.0"])  # [2] правильное количество аргументов
        assert "Зависимость 1.0 не найдена" in str(exc_info.value)
    def test_check_dependencies_circular(self, temp_migration_dir: Path) -> None:
        """Тест проверки циклических зависимостей."""
        migration = EvolutionMigration(str(temp_migration_dir))
        # Создать миграции с циклическими зависимостями
        migrations_data = [
            {"version": "1.0", "description": "Migration 1", "scripts": [], "dependencies": ["1.1"]},
            {"version": "1.1", "description": "Migration 2", "scripts": [], "dependencies": ["1.0"]}
        ]
        for data in migrations_data:
            migration_path = temp_migration_dir / f"{data['version']}.json"
            with open(migration_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        with pytest.raises(MigrationError) as exc_info:
            migration._check_dependencies(["1.0"])  # [2] правильное количество аргументов
        assert "Обнаружена циклическая зависимость" in str(exc_info.value)
    def test_sort_migrations_by_dependencies(self, temp_migration_dir: Path) -> None:
        """Тест сортировки миграций по зависимостям."""
        migration = EvolutionMigration(str(temp_migration_dir))
        # Создать миграции с зависимостями
        migrations_data = [
            {"version": "1.2", "description": "Migration 3", "scripts": [], "dependencies": ["1.1"]},
            {"version": "1.0", "description": "Migration 1", "scripts": [], "dependencies": []},
            {"version": "1.1", "description": "Migration 2", "scripts": [], "dependencies": ["1.0"]}
        ]
        for data in migrations_data:
            migration_path = temp_migration_dir / f"{data['version']}.json"
            with open(migration_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        migrations = migration.list_migrations()
        sorted_migrations = migration._sort_migrations_by_dependencies(migrations)
        # Проверить правильный порядок
        versions = [migration_path.stem for migration_path in sorted_migrations]
        assert versions == ["1.0", "1.1", "1.2"]
    def test_execute_migration_success(self, temp_migration_dir: Path, sample_migration_data: Dict[str, Any]) -> None:
        """Тест успешного выполнения миграции."""
        migration = EvolutionMigration(str(temp_migration_dir))
        # Создать файл миграции
        migration_path = temp_migration_dir / "1.0.json"
        with open(migration_path, "w", encoding="utf-8") as f:
            json.dump(sample_migration_data, f)
        # Выполнить миграцию
        result = migration.execute_migration(str(migration_path))
        assert result["success"] is True
        assert result["version"] == "1.0"
        assert result["executed_scripts"] == 1
        assert result["execution_time"] > 0
    def test_execute_migration_script_error(self, temp_migration_dir: Path) -> None:
        """Тест ошибки выполнения скрипта миграции."""
        migration = EvolutionMigration(str(temp_migration_dir))
        # Создать миграцию с некорректным SQL
        migration_data = {
            "version": "1.0",
            "description": "Invalid SQL migration",
            "scripts": ["INVALID SQL STATEMENT"],
            "rollback_scripts": [],
            "rollback_supported": True,
            "dependencies": []
        }
        migration_path = temp_migration_dir / "1.0.json"
        with open(migration_path, "w", encoding="utf-8") as f:
            json.dump(migration_data, f)
        with pytest.raises(MigrationError) as exc_info:
            migration.execute_migration(str(migration_path))
        assert "Ошибка выполнения скрипта миграции" in str(exc_info.value)
    def test_rollback_migration_success(self, temp_migration_dir: Path, sample_migration_data: Dict[str, Any]) -> None:
        """Тест успешного отката миграции."""
        migration = EvolutionMigration(str(temp_migration_dir))
        # Создать файл миграции
        migration_path = temp_migration_dir / "1.0.json"
        with open(migration_path, "w", encoding="utf-8") as f:
            json.dump(sample_migration_data, f)
        # Выполнить откат миграции
        result = migration.rollback_migration(str(migration_path))  # [1] передаю str вместо Path
        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["version"] == "1.0"
        assert result["executed_scripts"] == 1
        assert result["execution_time"] > 0
    def test_rollback_migration_not_supported(self, temp_migration_dir: Path) -> None:
        """Тест отката миграции без поддержки отката."""
        migration = EvolutionMigration(str(temp_migration_dir))
        # Создать миграцию без поддержки отката
        migration_data = {
            "version": "1.0",
            "description": "No rollback migration",
            "scripts": ["CREATE TABLE test (id INTEGER)"],
            "rollback_scripts": [],
            "rollback_supported": False,
            "dependencies": []
        }
        migration_path = temp_migration_dir / "1.0.json"
        with open(migration_path, "w", encoding="utf-8") as f:
            json.dump(migration_data, f)
        with pytest.raises(MigrationError) as exc_info:
            migration.rollback_migration(str(migration_path))  # [1] передаю str вместо Path
        assert "Откат миграции не поддерживается" in str(exc_info.value)
    def test_get_migration_status(self, temp_migration_dir: Path) -> None:
        """Тест получения статуса миграций."""
        migration = EvolutionMigration(str(temp_migration_dir))
        # Создать несколько миграций
        migrations_data = [
            {"version": "1.0", "description": "Migration 1", "scripts": []},
            {"version": "1.1", "description": "Migration 2", "scripts": []},
            {"version": "1.2", "description": "Migration 3", "scripts": []}
        ]
        for data in migrations_data:
            migration_path = temp_migration_dir / f"{data['version']}.json"
            with open(migration_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        status = migration.get_migration_status()
        assert status["total_migrations"] == 3
        assert status["pending_migrations"] == 3
        assert status["executed_migrations"] == 0
        assert status["failed_migrations"] == 0
        assert len(status["migrations"]) == 3
    def test_validate_migration_file(self, temp_migration_dir: Path, sample_migration_data: Dict[str, Any]) -> None:
        """Тест валидации файла миграции."""
        migration = EvolutionMigration(str(temp_migration_dir))
        # Создать корректный файл миграции
        migration_path = temp_migration_dir / "1.0.json"
        with open(migration_path, "w", encoding="utf-8") as f:
            json.dump(sample_migration_data, f)
        # Валидация должна пройти успешно
        migration._validate_migration_file(migration_path)
        # Создать некорректный файл
        invalid_path = temp_migration_dir / "invalid.json"
        with open(invalid_path, "w", encoding="utf-8") as f:
            f.write("invalid content")
        with pytest.raises(MigrationError) as exc_info:
            migration._validate_migration_file(invalid_path)
        assert "Некорректный формат файла миграции" in str(exc_info.value) 
