"""
Unit тесты для DatabaseManager.
Тестирует управление базой данных, включая подключения,
запросы, транзакции и оптимизацию производительности.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import tempfile
import os
from datetime import datetime, timedelta
from infrastructure.core.database_manager import DatabaseManager
class TestDatabaseManager:
    """Тесты для DatabaseManager."""
    @pytest.fixture
    def database_manager(self) -> DatabaseManager:
        """Фикстура для DatabaseManager."""
        # Создание временной базы данных для тестов
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        db_manager = DatabaseManager()
        db_manager.initialize_database(temp_db.name)
        yield db_manager
        # Очистка после тестов
        db_manager.close_connection()
        os.unlink(temp_db.name)
    @pytest.fixture
    def sample_table_data(self) -> dict:
        """Фикстура с тестовыми данными таблицы."""
        return {
            "table_name": "test_trades",
            "columns": [
                {"name": "id", "type": "INTEGER", "primary_key": True},
                {"name": "symbol", "type": "TEXT", "not_null": True},
                {"name": "side", "type": "TEXT", "not_null": True},
                {"name": "quantity", "type": "REAL", "not_null": True},
                {"name": "price", "type": "REAL", "not_null": True},
                {"name": "timestamp", "type": "DATETIME", "not_null": True}
            ],
            "sample_data": [
                {
                    "symbol": "BTCUSDT",
                    "side": "buy",
                    "quantity": 0.1,
                    "price": 50000.0,
                    "timestamp": datetime.now()
                },
                {
                    "symbol": "ETHUSDT",
                    "side": "sell",
                    "quantity": 1.0,
                    "price": 3000.0,
                    "timestamp": datetime.now()
                }
            ]
        }
    def test_initialization(self, database_manager: DatabaseManager) -> None:
        """Тест инициализации менеджера базы данных."""
        assert database_manager is not None
        assert hasattr(database_manager, 'connection')
        assert hasattr(database_manager, 'query_executors')
        assert hasattr(database_manager, 'transaction_managers')
    def test_initialize_database(self, database_manager: DatabaseManager) -> None:
        """Тест инициализации базы данных."""
        # Проверка, что база данных инициализирована
        assert database_manager.connection is not None
        # Проверка, что можно выполнять запросы
        cursor = database_manager.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        assert isinstance(tables, list)
    def test_create_table(self, database_manager: DatabaseManager, sample_table_data: dict) -> None:
        """Тест создания таблицы."""
        # Создание таблицы
        create_result = database_manager.create_table(
            sample_table_data["table_name"],
            sample_table_data["columns"]
        )
        # Проверки
        assert create_result is not None
        assert "success" in create_result
        assert "table_name" in create_result
        assert "creation_time" in create_result
        # Проверка типов данных
        assert isinstance(create_result["success"], bool)
        assert isinstance(create_result["table_name"], str)
        assert isinstance(create_result["creation_time"], datetime)
        # Проверка, что таблица создана
        cursor = database_manager.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", 
                      (sample_table_data["table_name"],))
        table_exists = cursor.fetchone()
        assert table_exists is not None
    def test_insert_data(self, database_manager: DatabaseManager, sample_table_data: dict) -> None:
        """Тест вставки данных."""
        # Создание таблицы
        database_manager.create_table(
            sample_table_data["table_name"],
            sample_table_data["columns"]
        )
        # Вставка данных
        insert_result = database_manager.insert_data(
            sample_table_data["table_name"],
            sample_table_data["sample_data"]
        )
        # Проверки
        assert insert_result is not None
        assert "success" in insert_result
        assert "inserted_rows" in insert_result
        assert "insert_time" in insert_result
        # Проверка типов данных
        assert isinstance(insert_result["success"], bool)
        assert isinstance(insert_result["inserted_rows"], int)
        assert isinstance(insert_result["insert_time"], datetime)
        # Проверка, что данные вставлены
        cursor = database_manager.connection.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {sample_table_data['table_name']}")
        count = cursor.fetchone()[0]
        assert count == len(sample_table_data["sample_data"])
    def test_query_data(self, database_manager: DatabaseManager, sample_table_data: dict) -> None:
        """Тест запроса данных."""
        # Создание таблицы и вставка данных
        database_manager.create_table(
            sample_table_data["table_name"],
            sample_table_data["columns"]
        )
        database_manager.insert_data(
            sample_table_data["table_name"],
            sample_table_data["sample_data"]
        )
        # Запрос данных
        query_result = database_manager.query_data(
            f"SELECT * FROM {sample_table_data['table_name']}"
        )
        # Проверки
        assert query_result is not None
        assert "success" in query_result
        assert "data" in query_result
        assert "row_count" in query_result
        assert "query_time" in query_result
        # Проверка типов данных
        assert isinstance(query_result["success"], bool)
        assert isinstance(query_result["data"], list)
        assert isinstance(query_result["row_count"], int)
        assert isinstance(query_result["query_time"], datetime)
        # Проверка данных
        assert query_result["row_count"] == len(sample_table_data["sample_data"])
    def test_update_data(self, database_manager: DatabaseManager, sample_table_data: dict) -> None:
        """Тест обновления данных."""
        # Создание таблицы и вставка данных
        database_manager.create_table(
            sample_table_data["table_name"],
            sample_table_data["columns"]
        )
        database_manager.insert_data(
            sample_table_data["table_name"],
            sample_table_data["sample_data"]
        )
        # Обновление данных
        update_result = database_manager.update_data(
            sample_table_data["table_name"],
            {"price": 51000.0},
            {"symbol": "BTCUSDT"}
        )
        # Проверки
        assert update_result is not None
        assert "success" in update_result
        assert "updated_rows" in update_result
        assert "update_time" in update_result
        # Проверка типов данных
        assert isinstance(update_result["success"], bool)
        assert isinstance(update_result["updated_rows"], int)
        assert isinstance(update_result["update_time"], datetime)
    def test_delete_data(self, database_manager: DatabaseManager, sample_table_data: dict) -> None:
        """Тест удаления данных."""
        # Создание таблицы и вставка данных
        database_manager.create_table(
            sample_table_data["table_name"],
            sample_table_data["columns"]
        )
        database_manager.insert_data(
            sample_table_data["table_name"],
            sample_table_data["sample_data"]
        )
        # Удаление данных
        delete_result = database_manager.delete_data(
            sample_table_data["table_name"],
            {"symbol": "ETHUSDT"}
        )
        # Проверки
        assert delete_result is not None
        assert "success" in delete_result
        assert "deleted_rows" in delete_result
        assert "delete_time" in delete_result
        # Проверка типов данных
        assert isinstance(delete_result["success"], bool)
        assert isinstance(delete_result["deleted_rows"], int)
        assert isinstance(delete_result["delete_time"], datetime)
    def test_execute_transaction(self, database_manager: DatabaseManager, sample_table_data: dict) -> None:
        """Тест выполнения транзакции."""
        # Создание таблицы
        database_manager.create_table(
            sample_table_data["table_name"],
            sample_table_data["columns"]
        )
        # Выполнение транзакции
        transaction_operations = [
            {
                "type": "insert",
                "table": sample_table_data["table_name"],
                "data": sample_table_data["sample_data"]
            },
            {
                "type": "update",
                "table": sample_table_data["table_name"],
                "updates": {"price": 52000.0},
                "conditions": {"symbol": "BTCUSDT"}
            }
        ]
        transaction_result = database_manager.execute_transaction(transaction_operations)
        # Проверки
        assert transaction_result is not None
        assert "success" in transaction_result
        assert "transaction_id" in transaction_result
        assert "operations_count" in transaction_result
        assert "transaction_time" in transaction_result
        # Проверка типов данных
        assert isinstance(transaction_result["success"], bool)
        assert isinstance(transaction_result["transaction_id"], str)
        assert isinstance(transaction_result["operations_count"], int)
        assert isinstance(transaction_result["transaction_time"], datetime)
    def test_backup_database(self, database_manager: DatabaseManager) -> None:
        """Тест резервного копирования базы данных."""
        # Создание резервной копии
        backup_result = database_manager.backup_database("test_backup.db")
        # Проверки
        assert backup_result is not None
        assert "success" in backup_result
        assert "backup_path" in backup_result
        assert "backup_size" in backup_result
        assert "backup_time" in backup_result
        # Проверка типов данных
        assert isinstance(backup_result["success"], bool)
        assert isinstance(backup_result["backup_path"], str)
        assert isinstance(backup_result["backup_size"], int)
        assert isinstance(backup_result["backup_time"], datetime)
        # Очистка
        if os.path.exists("test_backup.db"):
            os.unlink("test_backup.db")
    def test_restore_database(self, database_manager: DatabaseManager) -> None:
        """Тест восстановления базы данных."""
        # Создание резервной копии
        backup_result = database_manager.backup_database("test_backup.db")
        # Восстановление базы данных
        restore_result = database_manager.restore_database("test_backup.db")
        # Проверки
        assert restore_result is not None
        assert "success" in restore_result
        assert "restore_time" in restore_result
        # Проверка типов данных
        assert isinstance(restore_result["success"], bool)
        assert isinstance(restore_result["restore_time"], datetime)
        # Очистка
        if os.path.exists("test_backup.db"):
            os.unlink("test_backup.db")
    def test_optimize_database(self, database_manager: DatabaseManager) -> None:
        """Тест оптимизации базы данных."""
        # Оптимизация базы данных
        optimization_result = database_manager.optimize_database()
        # Проверки
        assert optimization_result is not None
        assert "success" in optimization_result
        assert "optimization_score" in optimization_result
        assert "space_saved" in optimization_result
        assert "optimization_time" in optimization_result
        # Проверка типов данных
        assert isinstance(optimization_result["success"], bool)
        assert isinstance(optimization_result["optimization_score"], float)
        assert isinstance(optimization_result["space_saved"], int)
        assert isinstance(optimization_result["optimization_time"], float)
        # Проверка диапазона
        assert 0.0 <= optimization_result["optimization_score"] <= 1.0
    def test_get_database_statistics(self, database_manager: DatabaseManager) -> None:
        """Тест получения статистики базы данных."""
        # Получение статистики
        statistics = database_manager.get_database_statistics()
        # Проверки
        assert statistics is not None
        assert "total_tables" in statistics
        assert "total_size" in statistics
        assert "table_sizes" in statistics
        assert "index_count" in statistics
        assert "last_optimization" in statistics
        # Проверка типов данных
        assert isinstance(statistics["total_tables"], int)
        assert isinstance(statistics["total_size"], int)
        assert isinstance(statistics["table_sizes"], dict)
        assert isinstance(statistics["index_count"], int)
        assert isinstance(statistics["last_optimization"], datetime)
    def test_create_index(self, database_manager: DatabaseManager, sample_table_data: dict) -> None:
        """Тест создания индекса."""
        # Создание таблицы
        database_manager.create_table(
            sample_table_data["table_name"],
            sample_table_data["columns"]
        )
        # Создание индекса
        index_result = database_manager.create_index(
            sample_table_data["table_name"],
            "idx_symbol",
            ["symbol"]
        )
        # Проверки
        assert index_result is not None
        assert "success" in index_result
        assert "index_name" in index_result
        assert "creation_time" in index_result
        # Проверка типов данных
        assert isinstance(index_result["success"], bool)
        assert isinstance(index_result["index_name"], str)
        assert isinstance(index_result["creation_time"], datetime)
    def test_analyze_query_performance(self, database_manager: DatabaseManager, sample_table_data: dict) -> None:
        """Тест анализа производительности запросов."""
        # Создание таблицы и вставка данных
        database_manager.create_table(
            sample_table_data["table_name"],
            sample_table_data["columns"]
        )
        database_manager.insert_data(
            sample_table_data["table_name"],
            sample_table_data["sample_data"]
        )
        # Анализ производительности
        performance_result = database_manager.analyze_query_performance(
            f"SELECT * FROM {sample_table_data['table_name']} WHERE symbol = 'BTCUSDT'"
        )
        # Проверки
        assert performance_result is not None
        assert "execution_time" in performance_result
        assert "query_plan" in performance_result
        assert "performance_score" in performance_result
        assert "optimization_suggestions" in performance_result
        # Проверка типов данных
        assert isinstance(performance_result["execution_time"], float)
        assert isinstance(performance_result["query_plan"], str)
        assert isinstance(performance_result["performance_score"], float)
        assert isinstance(performance_result["optimization_suggestions"], list)
        # Проверка диапазона
        assert 0.0 <= performance_result["performance_score"] <= 1.0
    def test_error_handling(self, database_manager: DatabaseManager) -> None:
        """Тест обработки ошибок."""
        # Тест с некорректными данными
        with pytest.raises(ValueError):
            database_manager.create_table(None, None)
        with pytest.raises(ValueError):
            database_manager.query_data(None)
    def test_edge_cases(self, database_manager: DatabaseManager) -> None:
        """Тест граничных случаев."""
        # Тест с очень большими данными
        large_data = [{"large_field": "x" * 10000} for _ in range(100)]
        table_data = {
            "table_name": "large_table",
            "columns": [{"name": "large_field", "type": "TEXT"}],
            "sample_data": large_data
        }
        create_result = database_manager.create_table(
            table_data["table_name"], table_data["columns"]
        )
        assert create_result["success"] is True
        # Тест с пустыми данными
        empty_data: list = []
        insert_result = database_manager.insert_data("large_table", empty_data)
        assert insert_result["success"] is True
    def test_cleanup(self, database_manager: DatabaseManager) -> None:
        """Тест очистки ресурсов."""
        # Проверка корректной очистки
        database_manager.cleanup()
        # Проверка, что ресурсы освобождены
        assert database_manager.connection is None
        assert database_manager.query_executors == {}
        assert database_manager.transaction_managers == {} 
