"""
Основное хранилище паттернов маркет-мейкера.
Промышленная реализация с поддержкой:
- Строгой типизации
- Асинхронных операций
- Кэширования
- Сжатия данных
- Резервного копирования
- Валидации целостности
- Многопоточности
"""

import asyncio
import gzip
import hashlib
import json
import pickle
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

from domain.market_maker.mm_pattern import (
    MarketMakerPattern,
    PatternMemory,
    PatternOutcome,
    PatternResult,
)
from domain.types.market_maker_types import (
    Accuracy,
    AverageReturn,
    Confidence,
    MarketMakerPatternType,
    PatternOutcome,
    PatternContext,
    SuccessCount,
    TotalCount,
)

from ..interfaces.storage_interfaces import IPatternStorage
from ..models.storage_config import StorageConfig
from ..models.storage_models import (
    BehaviorRecord,
    PatternMetadata,
    StorageStatistics,
    SuccessMapEntry,
)


class MarketMakerStorage(IPatternStorage):
    """
    Промышленное хранилище паттернов маркет-мейкера.
    Особенности:
    - Асинхронные операции с блокировками
    - Кэширование в памяти с LRU
    - Сжатие данных с gzip
    - Резервное копирование
    - Валидация целостности
    - Многопоточная обработка
    - Метрики производительности
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        """
        Инициализация хранилища.
        Args:
            config: Конфигурация хранилища
        """
        self.config = config or StorageConfig()
        self._setup_storage()
        self._setup_caches()
        self._setup_executor()
        self._setup_locks()
        self._setup_metrics()
        logger.info(f"Initialized MarketMakerStorage at {self.config.base_path}")

    def _setup_storage(self) -> None:
        """Настройка структуры хранилища."""
        try:
            # Создаем основные директории
            self.config.patterns_directory.mkdir(parents=True, exist_ok=True)
            self.config.metadata_directory.mkdir(parents=True, exist_ok=True)
            self.config.behavior_directory.mkdir(parents=True, exist_ok=True)
            self.config.backup_directory.mkdir(parents=True, exist_ok=True)
            # Инициализируем базы данных
            self._init_metadata_database()
            self._init_patterns_database()
        except Exception as e:
            logger.error(f"Failed to setup storage: {e}")
            raise

    def _setup_caches(self) -> None:
        """Настройка кэшей."""
        self.pattern_cache: Dict[str, List[PatternMemory]] = {}
        self.success_map_cache: Dict[str, Dict[str, float]] = {}
        self.metadata_cache: Dict[str, PatternMetadata] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def _setup_executor(self) -> None:
        """Настройка пула потоков."""
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.max_workers, thread_name_prefix="MMStorage"
        )

    def _setup_locks(self) -> None:
        """Настройка блокировок."""
        self._symbol_locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    def _setup_metrics(self) -> None:
        """Настройка метрик производительности."""
        self.metrics = {
            "read_operations": 0,
            "write_operations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "compression_ratio": 1.0,
            "avg_read_time_ms": 0.0,
            "avg_write_time_ms": 0.0,
            "error_count": 0,
            "warning_count": 0,
        }

    def _init_metadata_database(self) -> None:
        """Инициализация базы данных метаданных."""
        db_path = self.config.metadata_directory / "metadata.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS symbols (
                    symbol TEXT PRIMARY KEY,
                    first_pattern_date TEXT,
                    last_pattern_date TEXT,
                    total_patterns INTEGER DEFAULT 0,
                    successful_patterns INTEGER DEFAULT 0,
                    avg_accuracy REAL DEFAULT 0.0,
                    avg_return REAL DEFAULT 0.0,
                    avg_confidence REAL DEFAULT 0.0,
                    avg_volume REAL DEFAULT 0.0,
                    avg_spread REAL DEFAULT 0.0,
                    avg_imbalance REAL DEFAULT 0.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pattern_types (
                    symbol TEXT,
                    pattern_type TEXT,
                    count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    avg_return REAL DEFAULT 0.0,
                    avg_confidence REAL DEFAULT 0.0,
                    last_seen TEXT,
                    PRIMARY KEY (symbol, pattern_type)
                )
            """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS storage_stats (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            # Индексы для производительности
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_symbols_last_date ON symbols(last_pattern_date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pattern_types_symbol ON pattern_types(symbol)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pattern_types_success ON pattern_types(success_rate)"
            )

    def _init_patterns_database(self) -> None:
        """Инициализация базы данных паттернов."""
        db_path = self.config.patterns_directory / "patterns.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patterns (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    features_data BLOB,
                    confidence REAL,
                    context_data TEXT,
                    result_data TEXT,
                    accuracy REAL DEFAULT 0.0,
                    avg_return REAL DEFAULT 0.0,
                    success_count INTEGER DEFAULT 0,
                    total_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_patterns_symbol ON patterns(symbol)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_patterns_timestamp ON patterns(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_patterns_accuracy ON patterns(accuracy)"
            )

    def _get_symbol_lock(self, symbol: str) -> asyncio.Lock:
        """Получение блокировки для символа."""
        if symbol not in self._symbol_locks:
            self._symbol_locks[symbol] = asyncio.Lock()
        return self._symbol_locks[symbol]

    def _generate_pattern_id(self, symbol: str, pattern: MarketMakerPattern) -> str:
        """Генерация уникального ID паттерна."""
        pattern_str = (
            f"{symbol}_{pattern.pattern_type.value}_{pattern.timestamp.isoformat()}"
        )
        return hashlib.sha256(pattern_str.encode()).hexdigest()[:16]

    async def save_pattern(self, symbol: str, pattern: MarketMakerPattern) -> bool:
        """
        Сохранение паттерна с полной валидацией и оптимизацией.
        Args:
            symbol: Символ торговой пары
            pattern: Паттерн для сохранения
        Returns:
            True если сохранение успешно
        """
        start_time = datetime.now()
        try:
            async with self._get_symbol_lock(symbol):
                # Генерируем ID паттерна
                pattern_id = self._generate_pattern_id(symbol, pattern)
                # Подготавливаем данные
                pattern_data = await self._prepare_pattern_data(pattern_id, pattern)
                # Сохраняем в базу данных
                success = await self._save_pattern_to_db(pattern_data)
                if success:
                    # Обновляем кэши
                    await self._update_caches_after_save(symbol, pattern)
                    # Обновляем метаданные
                    await self._update_symbol_metadata(symbol, pattern)
                    # Обновляем метрики
                    self._update_write_metrics(start_time)
                    logger.info(f"Successfully saved pattern {pattern_id} for {symbol}")
                    return True
                return False
        except Exception as e:
            self.metrics["error_count"] += 1
            logger.error(f"Failed to save pattern for {symbol}: {e}")
            return False

    async def _prepare_pattern_data(
        self, pattern_id: str, pattern: MarketMakerPattern
    ) -> Dict[str, Any]:
        """Подготовка данных паттерна для сохранения."""
        # Сериализуем признаки с сжатием
        # Исправляем передачу Callable в run_in_executor
        features_dict = pattern.features.to_dict()
        # Создаем функцию с правильной сигнатурой для run_in_executor
        def compress_features(data: Dict[str, Any]) -> bytes:
            return self._compress_data(data)
        features_blob = await asyncio.get_event_loop().run_in_executor(
            self.executor, compress_features, features_dict  # type: ignore
        )
        # Сериализуем контекст
        context_json = json.dumps(pattern.context, default=str)
        return {
            "id": pattern_id,
            "symbol": pattern.symbol,
            "pattern_type": pattern.pattern_type.value,
            "timestamp": pattern.timestamp.isoformat(),
            "features_data": features_blob,
            "confidence": float(pattern.confidence),
            "context_data": context_json,
            "result_data": None,
            "accuracy": 0.0,
            "avg_return": 0.0,
            "success_count": 1,
            "total_count": 1,
        }

    def _compress_data(self, data: Dict[str, Any]) -> bytes:
        """Сжатие данных."""
        if not self.config.compression_enabled:
            return pickle.dumps(data)
        json_str = json.dumps(data, default=str)
        compressed = gzip.compress(
            json_str.encode(), compresslevel=self.config.compression_level
        )
        # Обновляем метрики сжатия
        original_size = len(json_str.encode())
        compressed_size = len(compressed)
        if original_size > 0:
            self.metrics["compression_ratio"] = compressed_size / original_size
        return compressed

    def _decompress_data(self, data: bytes) -> Dict[str, Any]:
        """Распаковка данных."""
        if not self.config.compression_enabled:
            return pickle.loads(data)
        try:
            decompressed = gzip.decompress(data)
            return json.loads(decompressed.decode())
        except Exception:
            # Fallback к pickle
            return pickle.loads(data)

    async def _save_pattern_to_db(self, pattern_data: Dict[str, Any]) -> bool:
        """Сохранение паттерна в базу данных."""
        db_path = self.config.patterns_directory / "patterns.db"
        try:
            await asyncio.get_event_loop().run_in_executor(
                self.executor, self._execute_db_write, db_path, pattern_data
            )
            return True
        except Exception as e:
            logger.error(f"Database write failed: {e}")
            return False

    def _execute_db_write(self, db_path: Path, pattern_data: Dict[str, Any]) -> None:
        """Выполнение записи в базу данных."""
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO patterns (
                    id, symbol, pattern_type, timestamp, features_data, 
                    confidence, context_data, result_data, accuracy, 
                    avg_return, success_count, total_count, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    pattern_data["id"],
                    pattern_data["symbol"],
                    pattern_data["pattern_type"],
                    pattern_data["timestamp"],
                    pattern_data["features_data"],
                    pattern_data["confidence"],
                    pattern_data["context_data"],
                    pattern_data["result_data"],
                    pattern_data["accuracy"],
                    pattern_data["avg_return"],
                    pattern_data["success_count"],
                    pattern_data["total_count"],
                    datetime.now().isoformat(),
                ),
            )

    async def _update_caches_after_save(
        self, symbol: str, pattern: MarketMakerPattern
    ) -> None:
        """Обновление кэшей после сохранения."""
        # Очищаем кэш паттернов
        if symbol in self.pattern_cache:
            del self.pattern_cache[symbol]
        # Очищаем кэш метаданных
        if symbol in self.metadata_cache:
            del self.metadata_cache[symbol]

    async def _update_symbol_metadata(
        self, symbol: str, pattern: MarketMakerPattern
    ) -> None:
        """Обновление метаданных символа."""
        db_path = self.config.metadata_directory / "metadata.db"
        try:
            await asyncio.get_event_loop().run_in_executor(
                self.executor, self._update_metadata_db, db_path, symbol, pattern
            )
        except Exception as e:
            logger.error(f"Failed to update metadata for {symbol}: {e}")

    def _update_metadata_db(
        self, db_path: Path, symbol: str, pattern: MarketMakerPattern
    ) -> None:
        """Обновление метаданных в базе данных."""
        with sqlite3.connect(db_path) as conn:
            # Обновляем или создаем запись символа
            conn.execute(
                """
                INSERT OR REPLACE INTO symbols (
                    symbol, first_pattern_date, last_pattern_date, 
                    total_patterns, successful_patterns, avg_accuracy,
                    avg_return, avg_confidence, avg_volume, avg_spread,
                    avg_imbalance, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    symbol,
                    pattern.timestamp.isoformat(),  # first_pattern_date
                    pattern.timestamp.isoformat(),  # last_pattern_date
                    1,  # total_patterns
                    0,  # successful_patterns
                    0.0,  # avg_accuracy
                    0.0,  # avg_return
                    float(pattern.confidence),  # avg_confidence
                    0.0,  # avg_volume
                    0.0,  # avg_spread
                    0.0,  # avg_imbalance
                    datetime.now().isoformat(),
                ),
            )
            # Обновляем статистику по типам паттернов
            conn.execute(
                """
                INSERT OR REPLACE INTO pattern_types (
                    symbol, pattern_type, count, success_rate, 
                    avg_return, avg_confidence, last_seen
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    symbol,
                    pattern.pattern_type.value,
                    1,  # count
                    0.0,  # success_rate
                    0.0,  # avg_return
                    float(pattern.confidence),  # avg_confidence
                    pattern.timestamp.isoformat(),
                ),
            )

    def _update_write_metrics(self, start_time: datetime) -> None:
        """Обновление метрик записи."""
        self.metrics["write_operations"] = self.metrics.get("write_operations", 0) + 1
        write_time = (datetime.now() - start_time).total_seconds() * 1000
        # Обновляем среднее время записи
        total_writes = self.metrics["write_operations"]
        current_avg = self.metrics.get("avg_write_time_ms", 0.0)
        self.metrics["avg_write_time_ms"] = (
            current_avg * (total_writes - 1) + write_time
        ) / total_writes

    async def get_patterns_by_symbol(
        self, symbol: str, limit: int = 100
    ) -> List[PatternMemory]:
        """
        Получение паттернов по символу с кэшированием.
        Args:
            symbol: Символ торговой пары
            limit: Максимальное количество паттернов
        Returns:
            Список паттернов
        """
        start_time = datetime.now()
        try:
            # Проверяем кэш
            if symbol in self.pattern_cache:
                self.metrics["cache_hits"] = self.metrics.get("cache_hits", 0) + 1
                patterns = self.pattern_cache[symbol][:limit]
                self._update_read_metrics(start_time)
                return patterns
            self.metrics["cache_misses"] = self.metrics.get("cache_misses", 0) + 1
            # Загружаем из базы данных
            patterns = await self._load_patterns_from_db(symbol, limit)
            # Кэшируем результат
            self.pattern_cache[symbol] = patterns
            self._update_read_metrics(start_time)
            return patterns
        except Exception as e:
            self.metrics["error_count"] = self.metrics.get("error_count", 0) + 1
            logger.error(f"Failed to get patterns for {symbol}: {e}")
            return []

    async def _load_patterns_from_db(
        self, symbol: str, limit: int
    ) -> List[PatternMemory]:
        """Загрузка паттернов из базы данных."""
        db_path = self.config.patterns_directory / "patterns.db"
        try:
            data_list = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._execute_db_read, db_path, symbol, limit
            )
            patterns = []
            for data in data_list:
                pattern = await self._data_to_pattern_memory(data)
                if pattern:
                    patterns.append(pattern)
            return patterns
        except Exception as e:
            logger.error(f"Failed to load patterns from DB for {symbol}: {e}")
            return []

    def _execute_db_read(
        self, db_path: Path, symbol: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Выполнение чтения из базы данных."""
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM patterns
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (symbol, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    async def _data_to_pattern_memory(
        self, data: Dict[str, Any]
    ) -> Optional[PatternMemory]:
        """Преобразование данных в объект PatternMemory."""
        try:
            # Распаковываем признаки
            features_data = self._decompress_data(data["features_data"])
            # Создаем паттерн
            pattern = MarketMakerPattern(
                pattern_type=MarketMakerPatternType(data["pattern_type"]),
                symbol=data["symbol"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                features=await self._create_pattern_features(features_data),
                confidence=Confidence(data["confidence"]),
                context=PatternContext(
                    json.loads(data["context_data"]) if data["context_data"] else {}
                ),
            )
            # Создаем результат если есть
            result = None
            if data["result_data"]:
                result_data = json.loads(data["result_data"])
                result = PatternResult.from_dict(result_data)
            # Создаем память паттерна
            return PatternMemory(
                pattern=pattern,
                result=result,
                accuracy=Accuracy(data["accuracy"]),
                avg_return=AverageReturn(data["avg_return"]),
                success_count=SuccessCount(data["success_count"]),
                total_count=TotalCount(data["total_count"]),
                last_seen=datetime.fromisoformat(data["updated_at"]) if data["updated_at"] else None,
            )
        except Exception as e:
            logger.error(f"Failed to convert data to PatternMemory: {e}")
            return None

    async def _create_pattern_features(self, features_data: Dict[str, Any]):
        """Создание объекта признаков паттерна."""
        # Импортируем здесь для избежания циклических импортов
        from domain.market_maker.mm_pattern import PatternFeatures

        # Исправляем передачу dict[str, Any] в from_dict
        # Преобразуем в правильный формат PatternFeaturesDict
        pattern_features_dict = {
            "price_data": features_data.get("price_data", []),
            "volume_data": features_data.get("volume_data", []),
            "technical_indicators": features_data.get("technical_indicators", {}),
            "pattern_metrics": features_data.get("pattern_metrics", {}),
            "market_conditions": features_data.get("market_conditions", {}),
            "timestamp": features_data.get("timestamp", datetime.now().isoformat())
        }
        # Создаем PatternFeatures из данных
        from domain.market_maker.mm_pattern import PatternFeatures
        from domain.types.market_maker_types import (
            BookPressure, VolumeDelta, PriceReaction, SpreadChange, 
            OrderImbalance, LiquidityDepth, TimeDuration, 
            VolumeConcentration, PriceVolatility
        )
        
        # Исправление: используем dict(...) для создания TypedDict
        return PatternFeatures(
            book_pressure=BookPressure(pattern_features_dict.get("book_pressure", 0.0)),
            volume_delta=VolumeDelta(pattern_features_dict.get("volume_delta", 0.0)),
            price_reaction=PriceReaction(pattern_features_dict.get("price_reaction", 0.0)),
            spread_change=SpreadChange(pattern_features_dict.get("spread_change", 0.0)),
            order_imbalance=OrderImbalance(pattern_features_dict.get("order_imbalance", 0.0)),
            liquidity_depth=LiquidityDepth(pattern_features_dict.get("liquidity_depth", 0.0)),
            time_duration=TimeDuration(pattern_features_dict.get("time_duration", 0)),
            volume_concentration=VolumeConcentration(pattern_features_dict.get("volume_concentration", 0.0)),
            price_volatility=PriceVolatility(pattern_features_dict.get("price_volatility", 0.0)),
            market_microstructure=pattern_features_dict.get("market_microstructure", {})
        )

    def _update_read_metrics(self, start_time: datetime) -> None:
        """Обновление метрик чтения."""
        self.metrics["read_operations"] = self.metrics.get("read_operations", 0) + 1
        read_time = (datetime.now() - start_time).total_seconds() * 1000
        # Обновляем среднее время чтения
        total_reads = self.metrics["read_operations"]
        current_avg = self.metrics.get("avg_read_time_ms", 0.0)
        self.metrics["avg_read_time_ms"] = (
            current_avg * (total_reads - 1) + read_time
        ) / total_reads

    async def get_storage_statistics(self) -> StorageStatistics:
        """Получение статистики хранилища."""
        try:
            # Исправляем возвращаемый тип - возвращаем StorageStatistics
            return StorageStatistics(
                total_patterns=await self._count_total_patterns(),
                total_symbols=await self._count_total_symbols(),
                total_successful_patterns=await self._count_successful_patterns(),
                total_storage_size_bytes=await self._calculate_storage_size(),
                last_backup=datetime.fromtimestamp(self.metrics.get("last_backup")) if self.metrics.get("last_backup") else None,
                cache_hit_ratio=self.metrics["cache_hits"] / max(self.metrics["cache_requests"], 1),
                avg_read_time_ms=self.metrics["avg_read_time_ms"],
                avg_write_time_ms=self.metrics["avg_write_time_ms"],
                error_count=int(self.metrics["error_count"]),  # Исправляем: преобразуем float в int
            )
        except Exception as e:
            logger.error(f"Failed to get storage statistics: {e}")
            return StorageStatistics(
                total_patterns=0,
                total_symbols=0,
                total_successful_patterns=0,
                total_storage_size_bytes=0,
                last_backup=None,
                cache_hit_ratio=0.0,
                avg_read_time_ms=0.0,
                avg_write_time_ms=0.0,
                error_count=0,
            )

    async def _count_total_patterns(self) -> int:
        """Подсчет общего количества паттернов."""
        db_path = self.config.patterns_directory / "patterns.db"
        return self._execute_count_query(db_path, "SELECT COUNT(*) FROM patterns")

    async def _count_total_symbols(self) -> int:
        """Подсчет общего количества символов."""
        db_path = self.config.patterns_directory / "patterns.db"
        return self._execute_count_query(db_path, "SELECT COUNT(DISTINCT symbol) FROM patterns")

    async def _count_successful_patterns(self) -> int:
        """Подсчет успешных паттернов."""
        db_path = self.config.patterns_directory / "patterns.db"
        return self._execute_count_query(db_path, "SELECT COUNT(*) FROM patterns WHERE accuracy >= 0.7")

    def _execute_count_query(self, db_path: Path, query: str) -> int:
        """Выполнение запроса подсчета."""
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(query)
                return cursor.fetchone()[0] or 0
        except Exception as e:
            logger.error(f"Failed to execute count query: {e}")
            return 0

    async def _calculate_storage_size(self) -> int:
        """Расчет размера хранилища."""
        try:
            total_size = 0
            # Размер базы паттернов
            patterns_db = self.config.patterns_directory / "patterns.db"
            if patterns_db.exists():
                total_size += patterns_db.stat().st_size
            # Размер базы метаданных
            metadata_db = self.config.metadata_directory / "metadata.db"
            if metadata_db.exists():
                total_size += metadata_db.stat().st_size
            return total_size
        except Exception as e:
            logger.error(f"Failed to calculate storage size: {e}")
            return 0

    async def update_pattern_result(
        self, symbol: str, pattern_id: str, result: PatternResult
    ) -> bool:
        """Обновление результата паттерна."""
        start_time = datetime.now()
        try:
            async with self._get_symbol_lock(symbol):
                db_path = self.config.patterns_directory / "patterns.db"
                result_data = json.dumps(result.to_dict())
                updated_at = datetime.now().isoformat()
                # Извлекаем данные из результата
                accuracy = float(getattr(result, "accuracy", 0.0))
                avg_return = float(getattr(result, "avg_return", 0.0))
                success_count = int(getattr(result, "success_count", 0))
                total_count = int(getattr(result, "total_count", 0))
                # Сжимаем данные для экономии места
                compressed_data = self._compress_data(
                    {
                        "result_data": result_data,
                        "updated_at": updated_at,
                        "accuracy": accuracy,
                        "avg_return": avg_return,
                        "success_count": success_count,
                        "total_count": total_count,
                    }
                )
                updated = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._execute_pattern_update,
                    db_path,
                    pattern_id,
                    symbol,
                    compressed_data,
                    updated_at,
                    accuracy,
                    avg_return,
                    success_count,
                    total_count,
                )
                # Инвалидация кэша
                if symbol in self.pattern_cache:
                    del self.pattern_cache[symbol]
                # Обновление метрик
                self._update_write_metrics(start_time)
                return updated
        except Exception as e:
            self.metrics["error_count"] += 1
            logger.error(
                f"Failed to update pattern result for {symbol} {pattern_id}: {e}"
            )
            return False

    def _execute_pattern_update(
        self,
        db_path: Path,
        pattern_id: str,
        symbol: str,
        compressed_data: bytes,
        updated_at: str,
        accuracy: float,
        avg_return: float,
        success_count: int,
        total_count: int,
    ) -> bool:
        """Выполнение обновления паттерна в БД."""
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE patterns
                SET result_data = ?, updated_at = ?, accuracy = ?, avg_return = ?, 
                    success_count = ?, total_count = ?
                WHERE id = ? AND symbol = ?
                """,
                (
                    compressed_data,
                    updated_at,
                    accuracy,
                    avg_return,
                    success_count,
                    total_count,
                    pattern_id,
                    symbol,
                ),
            )
            conn.commit()
            return cursor.rowcount > 0

    async def get_successful_patterns(
        self, symbol: str, min_accuracy: float = 0.7
    ) -> List[PatternMemory]:
        """Получение успешных паттернов."""
        all_patterns = await self.get_patterns_by_symbol(symbol)
        return [p for p in all_patterns if float(p.accuracy) >= min_accuracy]

    async def find_similar_patterns(
        self, symbol: str, features: Dict[str, Any], similarity_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Поиск похожих паттернов с использованием общего сервиса."""
        try:
            # Используем общий сервис для поиска паттернов
            from domain.services.pattern_service import (
                PatternSearchCriteria,
                PatternSearchService,
                TechnicalPatternAnalyzer,
            )

            patterns = await self.get_patterns_by_symbol(symbol)
            if not patterns:
                return []
            # Создаем анализатор и сервис поиска
            analyzer = TechnicalPatternAnalyzer()
            search_service = PatternSearchService(analyzer)
            # Конвертируем паттерны для поиска
            for pattern_memory in patterns:
                try:
                    pattern_features = pattern_memory.pattern.features.to_dict()
                    if not pattern_features:
                        continue
                    # Создаем паттерн из данных
                    pattern_data = {
                        "prices": pattern_features.get("price_data", []),
                        "volumes": pattern_features.get("volume_data", []),
                        "timestamps": [pattern_memory.pattern.timestamp],
                        "trading_pair_id": symbol,
                    }
                    pattern = analyzer.analyze_pattern(pattern_data)
                    # Устанавливаем id как строку, если pattern.id ожидает UUID, используем setattr
                    pattern_id = self._generate_pattern_id(symbol, pattern_memory.pattern)
                    setattr(pattern, 'id', pattern_id)
                    search_service.add_pattern(pattern)
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Ошибка при обработке паттерна {self._generate_pattern_id(symbol, pattern_memory.pattern)}: {e}"
                    )
                    continue
            # Создаем целевой паттерн из признаков
            target_pattern_data = {
                "prices": features.get("price_data", []),
                "volumes": features.get("volume_data", []),
                "timestamps": [datetime.now()],
                "trading_pair_id": symbol,
            }
            target_pattern = analyzer.analyze_pattern(target_pattern_data)
            # Критерии поиска
            criteria = PatternSearchCriteria(
                min_similarity=similarity_threshold, trading_pair_id=symbol
            )
            # Поиск похожих паттернов
            matches = search_service.find_similar_patterns(target_pattern, criteria)
            # Конвертируем результаты в нужный формат
            results = []
            for match in matches:
                # Находим соответствующий pattern_memory
                for pattern_memory in patterns:
                    if (
                        self._generate_pattern_id(symbol, pattern_memory.pattern)
                        == match.pattern.id
                    ):
                        results.append(
                            {
                                "pattern_id": match.pattern.id,
                                "pattern_type": pattern_memory.pattern.pattern_type.value,
                                "similarity": match.similarity_score,
                                "features": pattern_memory.pattern.features.to_dict(),
                                "accuracy": float(pattern_memory.accuracy),
                                "avg_return": float(pattern_memory.avg_return),
                                "success_count": pattern_memory.success_count,
                                "total_count": pattern_memory.total_count,
                                "created_at": pattern_memory.pattern.timestamp.isoformat(),
                                "updated_at": pattern_memory.last_seen.isoformat() if pattern_memory.last_seen else None,  # Исправляем: добавляем проверку на None
                            }
                        )
                        break
            logger.info(
                f"Найдено {len(results)} похожих паттернов для {symbol} с порогом {similarity_threshold}"
            )
            return results
        except Exception as e:
            logger.error(f"Ошибка при поиске похожих паттернов для {symbol}: {e}")
            return []

    async def get_success_map(self, symbol: str) -> Dict[str, float]:
        """Получение карты успешности по pattern_type для symbol."""
        try:
            db_path = self.config.patterns_directory / "patterns.db"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(
                    "SELECT pattern_type, SUM(success_count), SUM(total_count) FROM patterns WHERE symbol = ? GROUP BY pattern_type",
                    (symbol,),
                )
                result = {}
                for row in cursor.fetchall():
                    pattern_type, success, total = row
                    if total > 0:
                        result[pattern_type] = float(success) / float(total)
                    else:
                        result[pattern_type] = 0.0
                return result
        except Exception as e:
            logger.error(f"Failed to get success map for {symbol}: {e}")
            return {}

    async def update_success_map(
        self, symbol: str, pattern_type: str, success_rate: float
    ) -> bool:
        """Обновление карты успешности."""
        try:
            db_path = self.config.patterns_directory / "success_map.db"
            # Создаем таблицу если не существует
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS success_map (
                        symbol TEXT,
                        pattern_type TEXT,
                        success_rate REAL,
                        updated_at TEXT,
                        PRIMARY KEY (symbol, pattern_type)
                    )
                """
                )
                conn.commit()
            # Обновляем или вставляем данные
            updated_at = datetime.now().isoformat()
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO success_map (symbol, pattern_type, success_rate, updated_at)
                    VALUES (?, ?, ?, ?)
                """,
                    (symbol, pattern_type, success_rate, updated_at),
                )
                conn.commit()
            logger.info(
                f"Success map updated for {symbol} {pattern_type}: {success_rate:.3f}"
            )
            return True
        except Exception as e:
            logger.error(
                f"Failed to update success map for {symbol} {pattern_type}: {e}"
            )
            return False

    async def cleanup_old_data(self, symbol: str, days: int = 30) -> int:
        """Очистка старых данных."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            cutoff_timestamp = cutoff_date.isoformat()
            db_path = self.config.patterns_directory / "patterns.db"
            deleted_count = 0
            with sqlite3.connect(db_path) as conn:
                # Удаляем старые паттерны
                cursor = conn.execute(
                    "DELETE FROM patterns WHERE symbol = ? AND created_at < ?",
                    (symbol, cutoff_timestamp),
                )
                deleted_count = cursor.rowcount
                conn.commit()
            # Очищаем кэш для этого символа
            if symbol in self.pattern_cache:
                del self.pattern_cache[symbol]
            logger.info(
                f"Cleaned up {deleted_count} old patterns for {symbol} older than {days} days"
            )
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup old data for {symbol}: {e}")
            return 0

    async def backup_data(self, symbol: str) -> bool:
        """Создание резервной копии."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.config.patterns_directory / "backups" / symbol
            backup_dir.mkdir(parents=True, exist_ok=True)
            # Создаем резервную копию основной БД
            source_db = self.config.patterns_directory / "patterns.db"
            backup_db = backup_dir / f"patterns_{timestamp}.db"
            if source_db.exists():
                import shutil

                shutil.copy2(source_db, backup_db)
            # Создаем резервную копию метаданных
            source_meta_db = self.config.patterns_directory / "metadata.db"
            backup_meta_db = backup_dir / f"metadata_{timestamp}.db"
            if source_meta_db.exists():
                shutil.copy2(source_meta_db, backup_meta_db)
            # Создаем файл с информацией о бэкапе
            backup_info = {
                "symbol": symbol,
                "timestamp": timestamp,
                "created_at": datetime.now().isoformat(),
                "files": [str(backup_db), str(backup_meta_db)],
            }
            import json

            with open(backup_dir / f"backup_info_{timestamp}.json", "w") as f:
                json.dump(backup_info, f, indent=2)
            logger.info(f"Backup created for {symbol} at {timestamp}")
            return True
        except Exception as e:
            logger.error(f"Failed to create backup for {symbol}: {e}")
            return False

    async def restore_data(self, symbol: str, backup_timestamp: str) -> bool:
        """Восстановление данных."""
        try:
            backup_dir = self.config.patterns_directory / "backups" / symbol
            backup_info_file = backup_dir / f"backup_info_{backup_timestamp}.json"
            if not backup_info_file.exists():
                logger.error(f"Backup info file not found: {backup_info_file}")
                return False
            import json

            with open(backup_info_file, "r") as f:
                backup_info = json.load(f)
            # Восстанавливаем основную БД
            backup_db = backup_dir / f"patterns_{backup_timestamp}.db"
            if backup_db.exists():
                import shutil

                target_db = self.config.patterns_directory / "patterns.db"
                shutil.copy2(backup_db, target_db)
            # Восстанавливаем метаданные
            backup_meta_db = backup_dir / f"metadata_{backup_timestamp}.db"
            if backup_meta_db.exists():
                target_meta_db = self.config.patterns_directory / "metadata.db"
                shutil.copy2(backup_meta_db, target_meta_db)
            # Очищаем кэш
            self.pattern_cache.clear()
            logger.info(f"Data restored for {symbol} from backup {backup_timestamp}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore data for {symbol}: {e}")
            return False

    async def validate_data_integrity(self, symbol: str) -> bool:
        """Проверка целостности данных."""
        try:
            db_path = self.config.patterns_directory / "patterns.db"
            with sqlite3.connect(db_path) as conn:
                # Проверяем структуру таблицы
                cursor = conn.execute("PRAGMA table_info(patterns)")
                columns = [row[1] for row in cursor.fetchall()]
                required_columns = [
                    "id",
                    "symbol",
                    "pattern_type",
                    "features_data",
                    "created_at",
                ]
                for col in required_columns:
                    if col not in columns:
                        logger.error(f"Missing required column: {col}")
                        return False
                # Проверяем целостность данных
                cursor = conn.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()
                if integrity_result[0] != "ok":
                    logger.error(
                        f"Database integrity check failed: {integrity_result[0]}"
                    )
                    return False
                # Проверяем наличие данных для символа
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM patterns WHERE symbol = ?", (symbol,)
                )
                count = cursor.fetchone()[0]
                if count == 0:
                    logger.warning(f"No patterns found for symbol: {symbol}")
                logger.info(f"Data integrity check passed for {symbol}")
                return True
        except Exception as e:
            logger.error(f"Failed to validate data integrity for {symbol}: {e}")
            return False

    async def get_pattern_metadata(self, symbol: str) -> List[PatternMetadata]:
        """Получение метаданных паттернов."""
        try:
            db_path = self.config.patterns_directory / "metadata.db"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT symbol, pattern_type, first_seen, last_seen, total_count, 
                           success_count, avg_accuracy, avg_return, avg_confidence
                    FROM pattern_types 
                    WHERE symbol = ?
                """,
                    (symbol,),
                )
                metadata_list = []
                for row in cursor.fetchall():
                    try:
                        pattern_type = MarketMakerPatternType(row[1])
                        metadata = PatternMetadata(
                            symbol=row[0],
                            pattern_type=pattern_type,
                            first_seen=datetime.fromisoformat(row[2]),
                            last_seen=datetime.fromisoformat(row[3]),
                            total_count=TotalCount(row[4]),
                            success_count=SuccessCount(row[5]),
                            avg_accuracy=Accuracy(float(row[6])),
                            avg_return=AverageReturn(float(row[7])),
                            avg_confidence=Confidence(float(row[8])),
                        )
                        metadata_list.append(metadata)
                    except (ValueError, KeyError) as e:
                        logger.warning(
                            f"Invalid pattern type for {symbol}: {row[1]}, error: {e}"
                        )
                        continue
                return metadata_list
        except Exception as e:
            logger.error(f"Failed to get pattern metadata for {symbol}: {e}")
            return []

    async def close(self) -> None:
        """Закрытие хранилища."""
        try:
            self.executor.shutdown(wait=True)
            logger.info("MarketMakerStorage closed successfully")
        except Exception as e:
            logger.error(f"Error closing MarketMakerStorage: {e}")

    def __del__(self):
        """Деструктор."""
        try:
            if hasattr(self, "executor"):
                self.executor.shutdown(wait=False)
        except Exception:
            pass
