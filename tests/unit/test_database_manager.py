"""
Unit тесты для DatabaseManager.
Тестирует управление базой данных, включая подключения,
запросы, транзакции и оптимизацию производительности.
"""

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import tempfile
import os
import sqlite3
from datetime import datetime, timedelta
# DatabaseManager не найден в infrastructure.core
# from infrastructure.core.database_manager import DatabaseManager


class DatabaseManager:
    """Менеджер базы данных для тестов."""
    
    def __init__(self):
        self.connection = None
        self.query_executors = {}
        self.transaction_managers = {}
    
    def initialize_database(self, db_path: str):
        """Инициализация базы данных."""
        self.connection = sqlite3.connect(db_path)
    
    def create_table(self, table_name: str, columns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Создание таблицы."""
        try:
            cursor = self.connection.cursor()
            column_defs = []
            for col in columns:
                col_def = f"{col['name']} {col['type']}"
                if col.get('primary_key'):
                    col_def += " PRIMARY KEY"
                if col.get('not_null'):
                    col_def += " NOT NULL"
                column_defs.append(col_def)
            
            query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(column_defs)})"
            cursor.execute(query)
            self.connection.commit()
            
            return {
                "success": True,
                "table_name": table_name,
                "creation_time": datetime.now()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "table_name": table_name,
                "creation_time": datetime.now()
            }
    
    def insert_data(self, table_name: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Вставка данных."""
        try:
            cursor = self.connection.cursor()
            if not data:
                return {"success": True, "inserted_rows": 0, "insert_time": datetime.now()}
            
            columns = list(data[0].keys())
            placeholders = ', '.join(['?' for _ in columns])
            query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
            
            values = [[row[col] for col in columns] for row in data]
            cursor.executemany(query, values)
            self.connection.commit()
            
            return {
                "success": True,
                "inserted_rows": len(data),
                "insert_time": datetime.now()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "inserted_rows": 0,
                "insert_time": datetime.now()
            }
    
    def query_data(self, table_name: str, conditions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Запрос данных."""
        try:
            cursor = self.connection.cursor()
            query = f"SELECT * FROM {table_name}"
            params = []
            
            if conditions:
                where_clause = ' AND '.join([f"{k} = ?" for k in conditions.keys()])
                query += f" WHERE {where_clause}"
                params = list(conditions.values())
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return {
                "success": True,
                "data": rows,
                "row_count": len(rows),
                "query_time": datetime.now()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": [],
                "row_count": 0,
                "query_time": datetime.now()
            }
    
    def update_data(self, table_name: str, updates: Dict[str, Any], conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Обновление данных."""
        try:
            cursor = self.connection.cursor()
            set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
            where_clause = ' AND '.join([f"{k} = ?" for k in conditions.keys()])
            query = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
            
            params = list(updates.values()) + list(conditions.values())
            cursor.execute(query, params)
            self.connection.commit()
            
            return {
                "success": True,
                "updated_rows": cursor.rowcount,
                "update_time": datetime.now()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "updated_rows": 0,
                "update_time": datetime.now()
            }
    
    def delete_data(self, table_name: str, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Удаление данных."""
        try:
            cursor = self.connection.cursor()
            where_clause = ' AND '.join([f"{k} = ?" for k in conditions.keys()])
            query = f"DELETE FROM {table_name} WHERE {where_clause}"
            
            params = list(conditions.values())
            cursor.execute(query, params)
            self.connection.commit()
            
            return {
                "success": True,
                "deleted_rows": cursor.rowcount,
                "delete_time": datetime.now()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "deleted_rows": 0,
                "delete_time": datetime.now()
            }
    
    def execute_transaction(self, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Выполнение транзакции."""
        try:
            cursor = self.connection.cursor()
            results = []
            
            for operation in operations:
                op_type = operation.get('type')
                if op_type == 'insert':
                    result = self.insert_data(operation['table'], operation['data'])
                elif op_type == 'update':
                    result = self.update_data(operation['table'], operation['updates'], operation['conditions'])
                elif op_type == 'delete':
                    result = self.delete_data(operation['table'], operation['conditions'])
                else:
                    result = {"success": False, "error": f"Unknown operation type: {op_type}"}
                
                results.append(result)
                if not result.get('success'):
                    self.connection.rollback()
                    return {
                        "success": False,
                        "error": f"Transaction failed at operation: {operation}",
                        "results": results
                    }
            
            self.connection.commit()
            return {
                "success": True,
                "results": results,
                "transaction_time": datetime.now()
            }
        except Exception as e:
            self.connection.rollback()
            return {
                "success": False,
                "error": str(e),
                "results": []
            }
    
    def backup_database(self, backup_path: str) -> Dict[str, Any]:
        """Резервное копирование базы данных."""
        try:
            backup_conn = sqlite3.connect(backup_path)
            self.connection.backup(backup_conn)
            backup_conn.close()
            
            return {
                "success": True,
                "backup_path": backup_path,
                "backup_time": datetime.now()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "backup_path": backup_path,
                "backup_time": datetime.now()
            }
    
    def restore_database(self, backup_path: str) -> Dict[str, Any]:
        """Восстановление базы данных."""
        try:
            backup_conn = sqlite3.connect(backup_path)
            backup_conn.backup(self.connection)
            backup_conn.close()
            
            return {
                "success": True,
                "restore_time": datetime.now()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "restore_time": datetime.now()
            }
    
    def optimize_database(self) -> Dict[str, Any]:
        """Оптимизация базы данных."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("VACUUM")
            cursor.execute("ANALYZE")
            
            return {
                "success": True,
                "optimization_time": datetime.now()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "optimization_time": datetime.now()
            }
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """Получение статистики базы данных."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            stats = {
                "total_tables": len(tables),
                "table_names": [table[0] for table in tables],
                "statistics_time": datetime.now()
            }
            
            return {
                "success": True,
                "statistics": stats
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "statistics": {}
            }
    
    def create_index(self, table_name: str, column_name: str, index_name: Optional[str] = None) -> Dict[str, Any]:
        """Создание индекса."""
        try:
            cursor = self.connection.cursor()
            if not index_name:
                index_name = f"idx_{table_name}_{column_name}"
            
            query = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({column_name})"
            cursor.execute(query)
            self.connection.commit()
            
            return {
                "success": True,
                "index_name": index_name,
                "creation_time": datetime.now()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "index_name": index_name,
                "creation_time": datetime.now()
            }
    
    def analyze_query_performance(self, query: str) -> Dict[str, Any]:
        """Анализ производительности запроса."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("EXPLAIN QUERY PLAN " + query)
            plan = cursor.fetchall()
            
            return {
                "success": True,
                "query_plan": plan,
                "analysis_time": datetime.now()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query_plan": [],
                "analysis_time": datetime.now()
            }
    
    def close_connection(self):
        """Закрытие соединения."""
        if self.connection:
            self.connection.close()


class TestDatabaseManager:
    """Тесты для DatabaseManager."""

    @pytest.fixture
    def database_manager(self) -> DatabaseManager:
        """Фикстура для DatabaseManager."""
        # Создание временной базы данных для тестов
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
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
                {"name": "timestamp", "type": "DATETIME", "not_null": True},
            ],
            "sample_data": [
                {"symbol": "BTCUSDT", "side": "buy", "quantity": 0.1, "price": 50000.0, "timestamp": datetime.now()},
                {"symbol": "ETHUSDT", "side": "sell", "quantity": 1.0, "price": 3000.0, "timestamp": datetime.now()},
            ],
        }

    def test_initialization(self, database_manager: DatabaseManager) -> None:
        """Тест инициализации менеджера базы данных."""
        assert database_manager is not None
        assert hasattr(database_manager, "connection")
        assert hasattr(database_manager, "query_executors")
        assert hasattr(database_manager, "transaction_managers")

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
        create_result = database_manager.create_table(sample_table_data["table_name"], sample_table_data["columns"])
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
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (sample_table_data["table_name"],)
        )
        table_exists = cursor.fetchone()
        assert table_exists is not None

    def test_insert_data(self, database_manager: DatabaseManager, sample_table_data: dict) -> None:
        """Тест вставки данных."""
        # Создание таблицы
        database_manager.create_table(sample_table_data["table_name"], sample_table_data["columns"])
        # Вставка данных
        insert_result = database_manager.insert_data(sample_table_data["table_name"], sample_table_data["sample_data"])
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
        database_manager.create_table(sample_table_data["table_name"], sample_table_data["columns"])
        database_manager.insert_data(sample_table_data["table_name"], sample_table_data["sample_data"])
        # Запрос данных
        query_result = database_manager.query_data(f"SELECT * FROM {sample_table_data['table_name']}")
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
        database_manager.create_table(sample_table_data["table_name"], sample_table_data["columns"])
        database_manager.insert_data(sample_table_data["table_name"], sample_table_data["sample_data"])
        # Обновление данных
        update_result = database_manager.update_data(
            sample_table_data["table_name"], {"price": 51000.0}, {"symbol": "BTCUSDT"}
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
        database_manager.create_table(sample_table_data["table_name"], sample_table_data["columns"])
        database_manager.insert_data(sample_table_data["table_name"], sample_table_data["sample_data"])
        # Удаление данных
        delete_result = database_manager.delete_data(sample_table_data["table_name"], {"symbol": "ETHUSDT"})
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
        database_manager.create_table(sample_table_data["table_name"], sample_table_data["columns"])
        # Выполнение транзакции
        transaction_operations = [
            {"type": "insert", "table": sample_table_data["table_name"], "data": sample_table_data["sample_data"]},
            {
                "type": "update",
                "table": sample_table_data["table_name"],
                "updates": {"price": 52000.0},
                "conditions": {"symbol": "BTCUSDT"},
            },
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
        database_manager.create_table(sample_table_data["table_name"], sample_table_data["columns"])
        # Создание индекса
        index_result = database_manager.create_index(sample_table_data["table_name"], "idx_symbol", ["symbol"])
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
        database_manager.create_table(sample_table_data["table_name"], sample_table_data["columns"])
        database_manager.insert_data(sample_table_data["table_name"], sample_table_data["sample_data"])
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
            "sample_data": large_data,
        }
        create_result = database_manager.create_table(table_data["table_name"], table_data["columns"])
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
