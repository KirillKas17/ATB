"""
Конфигурация для хранения паттернов маркет-мейкера.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from domain.type_definitions.market_maker_types import Accuracy, Confidence, SimilarityScore


@dataclass(frozen=True)
class StorageConfig:
    """
    Конфигурация для хранения паттернов маркет-мейкера.
    Attributes:
        base_path: Базовый путь для хранения данных
        max_patterns_per_symbol: Максимальное количество паттернов на символ
        cleanup_days: Количество дней для очистки старых данных
        min_accuracy_for_cleanup: Минимальная точность для сохранения паттерна
        backup_enabled: Включение резервного копирования
        compression_enabled: Включение сжатия данных
        cache_size: Размер кэша в памяти
        max_file_size_mb: Максимальный размер файла в МБ
        compression_level: Уровень сжатия (1-9)
        encryption_enabled: Включение шифрования
        encryption_key: Ключ шифрования
        async_io_enabled: Включение асинхронного ввода-вывода
        max_workers: Максимальное количество рабочих потоков
        retry_attempts: Количество попыток повтора операций
        retry_delay_seconds: Задержка между попытками в секундах
        validation_enabled: Включение валидации данных
        integrity_check_enabled: Включение проверки целостности
    """

    # Пути и структура
    base_path: Path = field(default_factory=lambda: Path("market_profiles"))
    max_patterns_per_symbol: int = 1000
    # Очистка и управление данными
    cleanup_days: int = 30
    min_accuracy_for_cleanup: Accuracy = Accuracy(0.5)
    # Производительность и кэширование
    backup_enabled: bool = True
    compression_enabled: bool = True
    cache_size: int = 1000
    max_file_size_mb: int = 100
    # Сжатие и шифрование
    compression_level: int = 6
    encryption_enabled: bool = False
    encryption_key: Optional[str] = None
    # Асинхронность и многопоточность
    async_io_enabled: bool = True
    max_workers: int = 4
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    # Валидация и безопасность
    validation_enabled: bool = True
    integrity_check_enabled: bool = True

    def __post_init__(self) -> None:
        """Валидация конфигурации после инициализации."""
        if self.max_patterns_per_symbol <= 0:
            raise ValueError("max_patterns_per_symbol must be positive")
        if self.cleanup_days <= 0:
            raise ValueError("cleanup_days must be positive")
        if not (0.0 <= float(self.min_accuracy_for_cleanup) <= 1.0):
            raise ValueError("min_accuracy_for_cleanup must be between 0.0 and 1.0")
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if self.max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be positive")
        if not (1 <= self.compression_level <= 9):
            raise ValueError("compression_level must be between 1 and 9")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if self.retry_attempts <= 0:
            raise ValueError("retry_attempts must be positive")
        if self.retry_delay_seconds <= 0:
            raise ValueError("retry_delay_seconds must be positive")

    @property
    def patterns_directory(self) -> Path:
        """Директория для хранения паттернов."""
        return self.base_path / "patterns"

    @property
    def metadata_directory(self) -> Path:
        """Директория для метаданных."""
        return self.base_path / "metadata"

    @property
    def behavior_directory(self) -> Path:
        """Директория для истории поведения."""
        return self.base_path / "behavior"

    @property
    def backup_directory(self) -> Path:
        """Директория для резервных копий."""
        return self.base_path / "backups"

    def get_symbol_directory(self, symbol: str) -> Path:
        """Получение директории для конкретного символа."""
        return self.patterns_directory / symbol

    def get_metadata_file_path(self, symbol: str) -> Path:
        """Получение пути к файлу метаданных символа."""
        return self.metadata_directory / f"{symbol}_metadata.json"

    def get_behavior_file_path(self, symbol: str) -> Path:
        """Получение пути к файлу истории поведения."""
        return self.behavior_directory / f"{symbol}_behavior.db"

    def get_backup_file_path(self, symbol: str, timestamp: str) -> Path:
        """Получение пути к файлу резервной копии."""
        return self.backup_directory / f"{symbol}_{timestamp}.backup"
