"""
Юнит-тесты для исключений.
"""

from infrastructure.evolution.exceptions import (
    StorageError,
    CacheError,
    BackupError,
    MigrationError,
    SerializationError,
    ValidationError,
    ConnectionError,
    QueryError,
)


class TestStorageError:
    """Тесты для StorageError."""

    def test_init_with_message(self: "TestStorageError") -> None:
        """Тест инициализации с сообщением."""
        error = StorageError("Test storage error", "test_error")
        assert str(error) == "Test storage error"
        assert error.message == "Test storage error"
        assert error.error_type == "storage_test_error"
        assert error.details == {}

    def test_init_with_message_and_details(self: "TestStorageError") -> None:
        """Тест инициализации с сообщением и деталями."""
        details = {"operation": "save", "entity": "strategy"}
        error = StorageError("Test storage error", "test_error", details)
        assert str(error) == "Test storage error"
        assert error.message == "Test storage error"
        assert error.error_type == "storage_test_error"
        assert error.details == details

    def test_repr(self: "TestStorageError") -> None:
        """Тест строкового представления."""
        error = StorageError("Test error", "test_error", {"key": "value"})
        repr_str = repr(error)
        assert "StorageError" in repr_str
        assert "Test error" in repr_str


class TestCacheError:
    """Тесты для CacheError."""

    def test_init_with_message(self: "TestCacheError") -> None:
        """Тест инициализации с сообщением."""
        error = CacheError("Test cache error", "test_error")
        assert str(error) == "Test cache error"
        assert error.message == "Test cache error"
        assert error.error_type == "cache_test_error"
        assert error.details == {}

    def test_init_with_message_and_details(self: "TestCacheError") -> None:
        """Тест инициализации с сообщением и деталями."""
        details = {"operation": "get", "key": "test_key"}
        error = CacheError("Test cache error", "test_error", details)
        assert str(error) == "Test cache error"
        assert error.message == "Test cache error"
        assert error.error_type == "cache_test_error"
        assert error.details == details

    def test_repr(self: "TestCacheError") -> None:
        """Тест строкового представления."""
        error = CacheError("Test error", "test_error", {"key": "value"})
        repr_str = repr(error)
        assert "CacheError" in repr_str
        assert "Test error" in repr_str


class TestBackupError:
    """Тесты для BackupError."""

    def test_init_with_message(self: "TestBackupError") -> None:
        """Тест инициализации с сообщением."""
        error = BackupError("Test backup error", "test_error")
        assert str(error) == "Test backup error"
        assert error.message == "Test backup error"
        assert error.error_type == "backup_test_error"
        assert error.details == {}

    def test_init_with_message_and_details(self: "TestBackupError") -> None:
        """Тест инициализации с сообщением и деталями."""
        details = {"operation": "create", "path": "/backup/path"}
        error = BackupError("Test backup error", "test_error", details)
        assert str(error) == "Test backup error"
        assert error.message == "Test backup error"
        assert error.error_type == "backup_test_error"
        assert error.details == details

    def test_repr(self: "TestBackupError") -> None:
        """Тест строкового представления."""
        error = BackupError("Test error", "test_error", {"key": "value"})
        repr_str = repr(error)
        assert "BackupError" in repr_str
        assert "Test error" in repr_str


class TestMigrationError:
    """Тесты для MigrationError."""

    def test_init_with_message(self: "TestMigrationError") -> None:
        """Тест инициализации с сообщением."""
        error = MigrationError("Test migration error", "test_error")
        assert str(error) == "Test migration error"
        assert error.message == "Test migration error"
        assert error.error_type == "migration_test_error"
        assert error.details == {}

    def test_init_with_message_and_details(self: "TestMigrationError") -> None:
        """Тест инициализации с сообщением и деталями."""
        details = {"operation": "execute", "version": "1.0"}
        error = MigrationError("Test migration error", "test_error", details)
        assert str(error) == "Test migration error"
        assert error.message == "Test migration error"
        assert error.error_type == "migration_test_error"
        assert error.details == details

    def test_repr(self: "TestMigrationError") -> None:
        """Тест строкового представления."""
        error = MigrationError("Test error", "test_error", {"key": "value"})
        repr_str = repr(error)
        assert "MigrationError" in repr_str
        assert "Test error" in repr_str


class TestSerializationError:
    """Тесты для SerializationError."""

    def test_init_with_message(self: "TestSerializationError") -> None:
        """Тест инициализации с сообщением."""
        error = SerializationError("Test serialization error", "test_error")
        assert str(error) == "Test serialization error"
        assert error.message == "Test serialization error"
        assert error.error_type == "serialization_test_error"
        assert error.details == {}

    def test_init_with_message_and_details(self: "TestSerializationError") -> None:
        """Тест инициализации с сообщением и деталями."""
        details = {"operation": "serialize", "object_type": "StrategyCandidate"}
        error = SerializationError("Test serialization error", "test_error", details)
        assert str(error) == "Test serialization error"
        assert error.message == "Test serialization error"
        assert error.error_type == "serialization_test_error"
        assert error.details == details

    def test_repr(self: "TestSerializationError") -> None:
        """Тест строкового представления."""
        error = SerializationError("Test error", "test_error", {"key": "value"})
        repr_str = repr(error)
        assert "SerializationError" in repr_str
        assert "Test error" in repr_str


class TestValidationError:
    """Тесты для ValidationError."""

    def test_init_with_message(self: "TestValidationError") -> None:
        """Тест инициализации с сообщением."""
        error = ValidationError("Test validation error", "name", "")
        assert str(error) == "Test validation error"
        assert error.message == "Test validation error"
        assert error.error_type == "validation_error"
        assert error.details == {"field": "name", "value": ""}

    def test_init_with_message_and_details(self: "TestValidationError") -> None:
        """Тест инициализации с сообщением и деталями."""
        error = ValidationError("Test validation error", "name", "")
        assert str(error) == "Test validation error"
        assert error.message == "Test validation error"
        assert error.error_type == "validation_error"
        assert error.details["field"] == "name"
        assert error.details["value"] == ""

    def test_repr(self: "TestValidationError") -> None:
        """Тест строкового представления."""
        error = ValidationError("Test error", "field", "value")
        repr_str = repr(error)
        assert "ValidationError" in repr_str
        assert "Test error" in repr_str


class TestConnectionError:
    """Тесты для ConnectionError."""

    def test_init_with_message(self: "TestConnectionError") -> None:
        """Тест инициализации с сообщением."""
        error = ConnectionError("Test connection error", "/path/to/db")
        assert str(error) == "Test connection error"
        assert error.message == "Test connection error"
        assert error.error_type == "storage_connection_error"
        assert error.details == {"db_path": "/path/to/db"}

    def test_init_with_message_and_details(self: "TestConnectionError") -> None:
        """Тест инициализации с сообщением и деталями."""
        error = ConnectionError("Test connection error", "/path/to/db")
        assert str(error) == "Test connection error"
        assert error.message == "Test connection error"
        assert error.error_type == "storage_connection_error"
        assert error.details["db_path"] == "/path/to/db"

    def test_repr(self: "TestConnectionError") -> None:
        """Тест строкового представления."""
        error = ConnectionError("Test error", "/path/to/db")
        repr_str = repr(error)
        assert "ConnectionError" in repr_str
        assert "Test error" in repr_str


class TestQueryError:
    """Тесты для QueryError."""

    def test_init_with_message(self: "TestQueryError") -> None:
        """Тест инициализации с сообщением."""
        error = QueryError("Test query error", "SELECT * FROM table")
        assert str(error) == "Test query error"
        assert error.message == "Test query error"
        assert error.error_type == "storage_query_error"
        assert error.details == {"query": "SELECT * FROM table", "params": None}

    def test_init_with_message_and_details(self: "TestQueryError") -> None:
        """Тест инициализации с сообщением и деталями."""
        params = {"id": 1, "name": "test"}
        error = QueryError("Test query error", "SELECT * FROM table", params)
        assert str(error) == "Test query error"
        assert error.message == "Test query error"
        assert error.error_type == "storage_query_error"
        assert error.details["query"] == "SELECT * FROM table"
        assert error.details["params"] == params

    def test_repr(self: "TestQueryError") -> None:
        """Тест строкового представления."""
        error = QueryError("Test error", "SELECT * FROM table")
        repr_str = repr(error)
        assert "QueryError" in repr_str
        assert "Test error" in repr_str


class TestExceptionInheritance:
    """Тесты наследования исключений."""

    def test_all_exceptions_inherit_from_base(self: "TestExceptionInheritance") -> None:
        """Тест, что все исключения наследуются от базового класса."""
        exceptions = [
            StorageError("test", "test_error"),
            CacheError("test", "test_error"),
            BackupError("test", "test_error"),
            MigrationError("test", "test_error"),
            SerializationError("test", "test_error"),
            ValidationError("test", "field", "value"),
            ConnectionError("test", "/path/to/db"),
            QueryError("test", "SELECT * FROM table"),
        ]
        from infrastructure.evolution.exceptions import EvolutionInfrastructureError

        for exception in exceptions:
            assert isinstance(exception, EvolutionInfrastructureError)

    def test_exception_types(self: "TestExceptionInheritance") -> None:
        """Тест типов исключений."""
        assert issubclass(StorageError, Exception)
        assert issubclass(CacheError, Exception)
        assert issubclass(BackupError, Exception)
        assert issubclass(MigrationError, Exception)
        assert issubclass(SerializationError, Exception)
        assert issubclass(ValidationError, Exception)
        assert issubclass(ConnectionError, Exception)
        assert issubclass(QueryError, Exception)

    def test_exception_attributes(self: "TestExceptionInheritance") -> None:
        """Тест атрибутов исключений."""
        details = {"key": "value"}
        error = StorageError("test message", "test_error", details)
        assert hasattr(error, "message")
        assert hasattr(error, "error_type")
        assert hasattr(error, "details")
        assert error.message == "test message"
        assert error.error_type == "storage_test_error"
        assert error.details == details


class TestExceptionContext:
    """Тесты контекста исключений."""

    def test_exception_context_manager(self: "TestExceptionContext") -> None:
        """Тест использования исключений в контекстном менеджере."""
        try:
            raise StorageError("Test error", "test_error", {"operation": "test"})
        except StorageError as e:
            assert e.message == "Test error"
            assert e.details["operation"] == "test"

    def test_exception_chaining(self: "TestExceptionContext") -> None:
        """Тест цепочки исключений."""
        original_error = ValueError("Original value error")
        try:
            raise StorageError("Storage failed", "test_error") from original_error
        except StorageError as e:
            assert e.message == "Storage failed"
            assert e.__cause__ == original_error

    def test_exception_details_access(self: "TestExceptionContext") -> None:
        """Тест доступа к деталям исключения."""
        details = {"operation": "save", "entity": "strategy", "id": "123"}
        error = StorageError("Save failed", "test_error", details)
        assert error.details["operation"] == "save"
        assert error.details["entity"] == "strategy"
        assert error.details["id"] == "123"

    def test_exception_string_representation(self: "TestExceptionContext") -> None:
        """Тест строкового представления исключений."""
        error = StorageError("Database connection failed", "test_error", {"host": "localhost"})
        str_repr = str(error)
        assert "Database connection failed" in str_repr
        repr_repr = repr(error)
        assert "StorageError" in repr_repr
        assert "Database connection failed" in repr_repr
