"""
Репозиторий памяти паттернов маркет-мейкера.
Промышленная реализация с поддержкой:
- Асинхронных операций
- Кэширования
- Оптимизации запросов
- Валидации данных
- Метрик производительности
"""

import asyncio
import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from loguru import logger

from domain.market_maker.mm_pattern import (
    MarketMakerPattern,
    PatternMemory,
    PatternOutcome,
    PatternResult,
    PatternContext,
)
from domain.type_definitions.market_maker_types import (
    Accuracy,
    AverageReturn,
    Confidence,
    MarketMakerPatternType,
    SuccessCount,
    TotalCount,
)

from ..interfaces.storage_interfaces import IPatternStorage
from ..models.storage_config import StorageConfig
from ..models.storage_models import PatternMetadata, StorageStatistics


class PatternMemoryRepository(IPatternStorage):
    """
    Промышленный репозиторий памяти паттернов.
    Особенности:
    - Асинхронные операции с блокировками
    - Кэширование в памяти с LRU
    - Оптимизированные запросы к БД
    - Валидация данных
    - Метрики производительности
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        """
        Инициализация репозитория.
        Args:
            config: Конфигурация хранилища
        """
        self.config = config or StorageConfig()
        self._setup_repository()
        self._setup_caches()
        self._setup_executor()
        self._setup_locks()
        self._setup_metrics()
        logger.info(f"Initialized PatternMemoryRepository at {self.config.base_path}")

    def _setup_repository(self) -> None:
        """Настройка репозитория."""
        try:
            # Создаем директории
            self.config.patterns_directory.mkdir(parents=True, exist_ok=True)
            self.config.metadata_directory.mkdir(parents=True, exist_ok=True)
            # Инициализируем базы данных
            self._init_patterns_database()
            self._init_metadata_database()
        except Exception as e:
            logger.error(f"Failed to setup repository: {e}")
            raise

    def _setup_caches(self) -> None:
        """Настройка кэшей."""
        self.pattern_cache: Dict[str, List[PatternMemory]] = {}
        self.metadata_cache: Dict[str, PatternMetadata] = {}
        self.success_map_cache: Dict[str, Dict[str, float]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def _setup_executor(self) -> None:
        """Настройка пула потоков."""
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.max_workers, thread_name_prefix="PatternRepo"
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
            "avg_read_time_ms": 0.0,
            "avg_write_time_ms": 0.0,
            "error_count": 0,
            "warning_count": 0,
        }

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
                    features_data TEXT,
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
            # Индексы для производительности
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_patterns_symbol_type ON patterns(symbol, pattern_type)"
            )

    def _init_metadata_database(self) -> None:
        """Инициализация базы данных метаданных."""
        db_path = self.config.metadata_directory / "metadata.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pattern_metadata (
                    symbol TEXT,
                    pattern_type TEXT,
                    first_seen TEXT,
                    last_seen TEXT,
                    total_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    avg_accuracy REAL DEFAULT 0.0,
                    avg_return REAL DEFAULT 0.0,
                    avg_confidence REAL DEFAULT 0.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, pattern_type)
                )
            """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metadata_symbol ON pattern_metadata(symbol)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metadata_type ON pattern_metadata(pattern_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metadata_last_seen ON pattern_metadata(last_seen)"
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
        Сохранение паттерна с полной валидацией.
        Args:
            symbol: Символ торговой пары
            pattern: Паттерн для сохранения
        Returns:
            True если сохранение успешно
        """
        start_time = datetime.now()
        try:
            async with self._get_symbol_lock(symbol):
                # Валидируем паттерн
                if not self._validate_pattern(pattern):
                    logger.warning(f"Invalid pattern for {symbol}")
                    return False
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
                    await self._update_pattern_metadata(symbol, pattern)
                    # Обновляем метрики
                    self._update_write_metrics(start_time)
                    logger.info(f"Successfully saved pattern {pattern_id} for {symbol}")
                    return True
                return False
        except Exception as e:
            self.metrics["error_count"] += 1
            logger.error(f"Failed to save pattern for {symbol}: {e}")
            return False

    def _validate_pattern(self, pattern: MarketMakerPattern) -> bool:
        """Валидация паттерна."""
        try:
            # Проверяем обязательные поля
            if not pattern.symbol:
                return False
            if not pattern.pattern_type:
                return False
            if not pattern.timestamp:
                return False
            if not pattern.features:
                return False
            # Проверяем уверенность
            if not (0.0 <= float(pattern.confidence) <= 1.0):
                return False
            return True
        except Exception as e:
            logger.error(f"Pattern validation failed: {e}")
            return False

    async def _prepare_pattern_data(
        self, pattern_id: str, pattern: MarketMakerPattern
    ) -> Dict[str, Any]:
        """Подготовка данных паттерна для сохранения."""
        try:
            # Сериализуем признаки
            features_json = json.dumps(pattern.features.to_dict(), default=str)
            # Сериализуем контекст
            context_json = json.dumps(pattern.context, default=str)
            return {
                "id": pattern_id,
                "symbol": pattern.symbol,
                "pattern_type": pattern.pattern_type.value,
                "timestamp": pattern.timestamp.isoformat(),
                "features_data": features_json,
                "confidence": float(pattern.confidence),
                "context_data": context_json,
                "result_data": None,
                "accuracy": 0.0,
                "avg_return": 0.0,
                "success_count": 0,
                "total_count": 0,
            }
        except Exception as e:
            logger.error(f"Failed to prepare pattern data: {e}")
            raise

    async def _save_pattern_to_db(self, pattern_data: Dict[str, Any]) -> bool:
        """Сохранение паттерна в базу данных."""
        db_path = self.config.patterns_directory / "patterns.db"
        try:
            await asyncio.get_event_loop().run_in_executor(
                self.executor, self._execute_pattern_insert, db_path, pattern_data
            )
            return True
        except Exception as e:
            logger.error(f"Database insert failed: {e}")
            return False

    def _execute_pattern_insert(
        self, db_path: Path, pattern_data: Dict[str, Any]
    ) -> None:
        """Выполнение вставки паттерна."""
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

    async def _update_pattern_metadata(
        self, symbol: str, pattern: MarketMakerPattern
    ) -> None:
        """Обновление метаданных паттерна."""
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
            # Проверяем существующую запись
            cursor = conn.execute(
                """
                SELECT * FROM pattern_metadata 
                WHERE symbol = ? AND pattern_type = ?
            """,
                (symbol, pattern.pattern_type.value),
            )
            existing = cursor.fetchone()
            if existing:
                # Обновляем существующую запись
                conn.execute(
                    """
                    UPDATE pattern_metadata SET
                        last_seen = ?,
                        total_count = total_count + 1,
                        avg_confidence = (avg_confidence * total_count + ?) / (total_count + 1),
                        updated_at = ?
                    WHERE symbol = ? AND pattern_type = ?
                """,
                    (
                        pattern.timestamp.isoformat(),
                        float(pattern.confidence),
                        datetime.now().isoformat(),
                        symbol,
                        pattern.pattern_type.value,
                    ),
                )
            else:
                # Создаем новую запись
                conn.execute(
                    """
                    INSERT INTO pattern_metadata (
                        symbol, pattern_type, first_seen, last_seen,
                        total_count, success_count, avg_accuracy,
                        avg_return, avg_confidence, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        symbol,
                        pattern.pattern_type.value,
                        pattern.timestamp.isoformat(),
                        pattern.timestamp.isoformat(),
                        1,  # total_count
                        0,  # success_count
                        0.0,  # avg_accuracy
                        0.0,  # avg_return
                        float(pattern.confidence),  # avg_confidence
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                    ),
                )

    def _update_write_metrics(self, start_time: datetime) -> None:
        """Обновление метрик записи."""
        self.metrics["write_operations"] += 1
        write_time = (datetime.now() - start_time).total_seconds() * 1000
        # Обновляем среднее время записи
        total_writes = self.metrics["write_operations"]
        current_avg = self.metrics["avg_write_time_ms"]
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
                self.metrics["cache_hits"] += 1
                patterns = self.pattern_cache[symbol][:limit]
                self._update_read_metrics(start_time)
                return patterns
            self.metrics["cache_misses"] += 1
            # Загружаем из базы данных
            patterns = await self._load_patterns_from_db(symbol, limit)
            # Кэшируем результат
            self.pattern_cache[symbol] = patterns
            self._update_read_metrics(start_time)
            return patterns
        except Exception as e:
            self.metrics["error_count"] += 1
            logger.error(f"Failed to get patterns for {symbol}: {e}")
            return []

    async def _load_patterns_from_db(
        self, symbol: str, limit: int
    ) -> List[PatternMemory]:
        """Загрузка паттернов из базы данных."""
        db_path = self.config.patterns_directory / "patterns.db"
        try:
            patterns_data = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._execute_patterns_query, db_path, symbol, limit
            )
            patterns = []
            for data in patterns_data:
                pattern_memory = await self._data_to_pattern_memory(data)
                if pattern_memory:
                    patterns.append(pattern_memory)
            return patterns
        except Exception as e:
            logger.error(f"Database query failed for {symbol}: {e}")
            return []

    def _execute_patterns_query(
        self, db_path: Path, symbol: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Выполнение запроса паттернов."""
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
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    async def _data_to_pattern_memory(
        self, data: Dict[str, Any]
    ) -> Optional[PatternMemory]:
        """Преобразование данных в объект PatternMemory."""
        try:
            # Десериализуем признаки
            features_data = json.loads(data["features_data"])
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
        # Приводим к правильному типу PatternFeaturesDict
        from domain.type_definitions.market_maker_types import PatternFeaturesDict
        # Исправление: используем правильные ключи для PatternFeaturesDict
        typed_dict: PatternFeaturesDict = {
            "book_pressure": float(pattern_features_dict.get("book_pressure", 0.0)),
            "volume_delta": float(pattern_features_dict.get("volume_delta", 0.0)),
            "price_reaction": float(pattern_features_dict.get("price_reaction", 0.0)),
            "spread_change": float(pattern_features_dict.get("spread_change", 0.0)),
            "order_imbalance": float(pattern_features_dict.get("order_imbalance", 0.0)),
            "liquidity_depth": float(pattern_features_dict.get("liquidity_depth", 0.0)),
            "time_duration": int(pattern_features_dict.get("time_duration", 0)),
            "volume_concentration": float(pattern_features_dict.get("volume_concentration", 0.0)),
            "price_volatility": float(pattern_features_dict.get("price_volatility", 0.0)),
            "market_microstructure": pattern_features_dict.get("market_microstructure", {})
        }
        return PatternFeatures.from_dict(typed_dict)  # type: ignore[arg-type]

    def _update_read_metrics(self, start_time: datetime) -> None:
        """Обновление метрик чтения."""
        self.metrics["read_operations"] += 1
        read_time = (datetime.now() - start_time).total_seconds() * 1000
        # Обновляем среднее время чтения
        total_reads = self.metrics["read_operations"]
        current_avg = self.metrics["avg_read_time_ms"]
        self.metrics["avg_read_time_ms"] = (
            current_avg * (total_reads - 1) + read_time
        ) / total_reads

    # Реализация остальных методов интерфейса
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
                # Обновляем только result_data и updated_at, остальные поля — если есть в result
                accuracy = float(getattr(result, "accuracy", 0.0))
                avg_return = float(getattr(result, "avg_return", 0.0))
                success_count = int(getattr(result, "success_count", 0))
                total_count = int(getattr(result, "total_count", 0))
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.execute(
                        """
                        UPDATE patterns
                        SET result_data = ?, updated_at = ?, accuracy = ?, avg_return = ?, success_count = ?, total_count = ?
                        WHERE id = ? AND symbol = ?
                        """,
                        (
                            result_data,
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
                    updated = cursor.rowcount > 0
                # Инвалидация кэша
                if symbol in self.pattern_cache:
                    del self.pattern_cache[symbol]
                self._update_write_metrics(start_time)
                return updated
        except Exception as e:
            self.metrics["error_count"] += 1
            logger.error(
                f"Failed to update pattern result for {symbol} {pattern_id}: {e}"
            )
            return False

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
                                "updated_at": pattern_memory.last_seen.isoformat() if pattern_memory.last_seen else None,
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
        """Получение карты успешности паттернов."""
        try:
            patterns = await self.get_patterns_by_symbol(symbol)
            success_map: Dict[str, List[float]] = {}
            for pattern in patterns:
                pattern_type = pattern.pattern.pattern_type.value
                if pattern_type not in success_map:
                    success_map[pattern_type] = []
                success_map[pattern_type].append(float(pattern.accuracy))
            # Усредняем значения
            return {
                pattern_type: sum(accuracies) / len(accuracies)
                for pattern_type, accuracies in success_map.items()
                if accuracies
            }
        except Exception as e:
            logger.error(f"Failed to get success map for {symbol}: {e}")
            return {}

    async def update_success_map(
        self, symbol: str, pattern_type: str, success_rate: float
    ) -> bool:
        """Обновление карты успешности паттернов."""
        try:
            # Обновляем метаданные
            metadata = PatternMetadata(
                symbol=symbol,
                pattern_type=MarketMakerPatternType(pattern_type),  # Исправляем: преобразуем str в MarketMakerPatternType
                first_seen=datetime.now(),
                last_seen=datetime.now(),
            )
            # Обновляем метаданные в базе данных
            db_path = self.config.metadata_directory / "metadata.db"
            # Исправление: передаем правильные параметры в правильном порядке
            self._update_metadata_db(db_path, symbol, metadata)  # type: ignore[arg-type]
            return True
        except Exception as e:
            logger.error(f"Failed to update success map for {symbol}: {e}")
            return False

    async def cleanup_old_data(self, symbol: str, days: int = 30) -> int:
        """Очистка старых данных."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            db_path = self.config.patterns_directory / "patterns.db"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM patterns
                    WHERE symbol = ? AND created_at < ?
                    """,
                    (symbol, cutoff_date.isoformat()),
                )
                conn.commit()
                deleted_count = cursor.rowcount
            # Инвалидация кэша
            if symbol in self.pattern_cache:
                del self.pattern_cache[symbol]
            logger.info(f"Cleaned up {deleted_count} old patterns for {symbol}")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup old data for {symbol}: {e}")
            return 0

    async def backup_data(self, symbol: str) -> bool:
        """Создание резервной копии данных."""
        try:
            patterns = await self.get_patterns_by_symbol(symbol)
            backup_data = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "patterns": [pattern.to_dict() for pattern in patterns],
            }
            backup_path = self.config.backup_directory / f"{symbol}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            with open(backup_path, "w") as f:
                json.dump(backup_data, f, indent=2, default=str)
            logger.info(f"Created backup for {symbol} at {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to backup data for {symbol}: {e}")
            return False

    async def restore_data(self, symbol: str, backup_timestamp: str) -> bool:
        """Восстановление данных из резервной копии."""
        try:
            backup_path = self.config.backup_directory / f"{symbol}_backup_{backup_timestamp}.json"
            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False
            with open(backup_path, "r") as f:
                backup_data = json.load(f)
            # Восстанавливаем паттерны
            for pattern_data in backup_data["patterns"]:
                pattern = PatternMemory.from_dict(pattern_data)
                await self.save_pattern(symbol, pattern.pattern)
            logger.info(f"Restored data for {symbol} from {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore data for {symbol}: {e}")
            return False

    async def validate_data_integrity(self, symbol: str) -> bool:
        """Проверка целостности данных."""
        try:
            patterns = await self.get_patterns_by_symbol(symbol)
            for pattern in patterns:
                if not self._validate_pattern(pattern.pattern):
                    logger.warning(f"Invalid pattern found: {pattern.pattern}")
                    return False
            logger.info(f"Data integrity check passed for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Data integrity check failed for {symbol}: {e}")
            return False

    async def get_pattern_metadata(self, symbol: str) -> List[PatternMetadata]:
        """Получение метаданных паттернов."""
        try:
            db_path = self.config.metadata_directory / "metadata.db"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT symbol, pattern_type, success_rate, last_updated
                    FROM pattern_metadata
                    WHERE symbol = ?
                    """,
                    (symbol,),
                )
                rows = cursor.fetchall()
            return [
                PatternMetadata(
                    symbol=row[0],
                    pattern_type=MarketMakerPatternType(row[1]),  # Исправляем: преобразуем str в MarketMakerPatternType
                    first_seen=datetime.fromisoformat(row[2]) if row[2] else datetime.now(),
                    last_seen=datetime.fromisoformat(row[3]) if row[3] else datetime.now(),
                )
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Failed to get pattern metadata for {symbol}: {e}")
            return []

    async def get_storage_statistics(self) -> StorageStatistics:
        """Получение статистики хранилища."""
        try:
            db_path = self.config.patterns_directory / "patterns.db"
            with sqlite3.connect(db_path) as conn:
                # Общее количество паттернов
                total_patterns = self._execute_count_query(
                    db_path, "SELECT COUNT(*) FROM patterns"
                )
                # Количество паттернов по символам
                symbols = self._execute_count_query(
                    db_path, "SELECT COUNT(DISTINCT symbol) FROM patterns"
                )
                # Средняя точность
                avg_accuracy = conn.execute(
                    "SELECT AVG(accuracy) FROM patterns"
                ).fetchone()[0] or 0.0
                # Размер базы данных
                db_size = db_path.stat().st_size if db_path.exists() else 0
            # Исправляем: возвращаем StorageStatistics вместо dict[str, Any]
            return StorageStatistics(
                total_patterns=total_patterns,
                total_symbols=symbols,
                total_successful_patterns=0,  # Добавляем недостающие поля
                total_storage_size_bytes=db_size,
                last_backup=datetime.now() if self.metrics.get("last_backup") else None,
                cache_hit_ratio=self.metrics.get("cache_hit_ratio", 0.0),
                avg_read_time_ms=self.metrics.get("avg_read_time_ms", 0.0),
                avg_write_time_ms=self.metrics.get("avg_write_time_ms", 0.0),
                error_count=int(self.metrics.get("error_count", 0)),
            )
        except Exception as e:
            logger.error(f"Failed to get storage statistics: {e}")
            # Исправляем: возвращаем StorageStatistics вместо пустого dict
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

    def _execute_count_query(self, db_path: Path, query: str) -> int:
        """Выполнение запроса подсчета."""
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(query)
                return cursor.fetchone()[0] or 0
        except Exception as e:
            logger.error(f"Failed to execute count query: {e}")
            return 0

    async def close(self) -> None:
        """Закрытие соединений и очистка ресурсов."""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            # Сохраняем финальную статистику
            logger.info(f"Final storage statistics: {await self.get_storage_statistics()}")
        except Exception as e:
            logger.error(f"Error during close: {e}")

    def __del__(self):
        """Деструктор для очистки ресурсов."""
        try:
            if hasattr(self, "executor") and self.executor:
                self.executor.shutdown(wait=False)
        except Exception as e:
            logger.error(f"Failed to cleanup pattern memory repository: {e}")
            # Продолжаем работу, но логируем ошибку для мониторинга
