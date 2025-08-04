"""
Репозиторий истории поведения маркет-мейкера.
Промышленная реализация с поддержкой:
- Асинхронных операций
- Кэширования
- Анализа поведения
- Статистики
- Оптимизации запросов
"""

import asyncio
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from loguru import logger

from domain.types.market_maker_types import MarketMakerPatternType

from ..interfaces.storage_interfaces import IBehaviorHistoryStorage
from ..models.storage_config import StorageConfig
from ..models.storage_models import BehaviorRecord


class BehaviorHistoryRepository(IBehaviorHistoryStorage):
    """
    Промышленный репозиторий истории поведения маркет-мейкера.
    Особенности:
    - Асинхронные операции с блокировками
    - Кэширование в памяти
    - Анализ поведения
    - Статистика
    - Оптимизированные запросы
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
        logger.info(f"Initialized BehaviorHistoryRepository at {self.config.base_path}")

    def _setup_repository(self) -> None:
        """Настройка репозитория."""
        try:
            # Создаем директории
            self.config.behavior_directory.mkdir(parents=True, exist_ok=True)
            # Инициализируем базу данных
            self._init_behavior_database()
        except Exception as e:
            logger.error(f"Failed to setup repository: {e}")
            raise

    def _setup_caches(self) -> None:
        """Настройка кэшей."""
        self.behavior_cache: Dict[str, List[BehaviorRecord]] = {}
        self.statistics_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def _setup_executor(self) -> None:
        """Настройка пула потоков."""
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.max_workers, thread_name_prefix="BehaviorRepo"
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

    def _init_behavior_database(self) -> None:
        """Инициализация базы данных поведения."""
        db_path = self.config.behavior_directory / "behavior.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS behavior_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    market_phase TEXT NOT NULL,
                    volatility_regime TEXT NOT NULL,
                    liquidity_regime TEXT NOT NULL,
                    volume_profile TEXT,
                    price_action TEXT,
                    order_flow TEXT,
                    spread_behavior TEXT,
                    imbalance_behavior TEXT,
                    pressure_behavior TEXT,
                    reaction_time REAL,
                    persistence REAL,
                    effectiveness REAL,
                    risk_level TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            # Индексы для производительности
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_behavior_symbol ON behavior_records(symbol)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_behavior_timestamp ON behavior_records(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_behavior_pattern_type ON behavior_records(pattern_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_behavior_market_phase ON behavior_records(market_phase)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_behavior_symbol_timestamp ON behavior_records(symbol, timestamp)"
            )

    def _get_symbol_lock(self, symbol: str) -> asyncio.Lock:
        """Получение блокировки для символа."""
        if symbol not in self._symbol_locks:
            self._symbol_locks[symbol] = asyncio.Lock()
        return self._symbol_locks[symbol]

    async def save_behavior_record(self, record: BehaviorRecord) -> bool:
        """
        Сохранение записи поведения с валидацией.
        Args:
            record: Запись поведения
        Returns:
            True если сохранение успешно
        """
        start_time = datetime.now()
        try:
            async with self._get_symbol_lock(record.symbol):
                # Валидируем запись
                if not self._validate_behavior_record(record):
                    logger.warning(f"Invalid behavior record for {record.symbol}")
                    return False
                # Подготавливаем данные
                record_data = self._prepare_behavior_data(record)
                # Сохраняем в базу данных
                success = await self._save_behavior_to_db(record_data)
                if success:
                    # Обновляем кэши
                    await self._update_caches_after_save(record)
                    # Обновляем метрики
                    self._update_write_metrics(start_time)
                    logger.info(
                        f"Successfully saved behavior record for {record.symbol}"
                    )
                    return True
                return False
        except Exception as e:
            self.metrics["error_count"] += 1
            logger.error(f"Failed to save behavior record for {record.symbol}: {e}")
            return False

    def _validate_behavior_record(self, record: BehaviorRecord) -> bool:
        """Валидация записи поведения."""
        try:
            # Проверяем обязательные поля
            if not record.symbol:
                return False
            if not record.pattern_type:
                return False
            if not record.timestamp:
                return False
            if not record.market_phase:
                return False
            if not record.volatility_regime:
                return False
            if not record.liquidity_regime:
                return False
            if not record.risk_level:
                return False
            # Проверяем числовые значения
            if record.reaction_time < 0:
                return False
            if not (0.0 <= record.persistence <= 1.0):
                return False
            if not (0.0 <= record.effectiveness <= 1.0):
                return False
            return True
        except Exception as e:
            logger.error(f"Behavior record validation failed: {e}")
            return False

    def _prepare_behavior_data(self, record: BehaviorRecord) -> Dict[str, Any]:
        """Подготовка данных записи поведения для сохранения."""
        try:
            return {
                "symbol": record.symbol,
                "timestamp": record.timestamp.isoformat(),
                "pattern_type": record.pattern_type.value,
                "market_phase": record.market_phase,
                "volatility_regime": record.volatility_regime,
                "liquidity_regime": record.liquidity_regime,
                "volume_profile": json.dumps(record.volume_profile),
                "price_action": json.dumps(record.price_action),
                "order_flow": json.dumps(record.order_flow),
                "spread_behavior": json.dumps(record.spread_behavior),
                "imbalance_behavior": json.dumps(record.imbalance_behavior),
                "pressure_behavior": json.dumps(record.pressure_behavior),
                "reaction_time": record.reaction_time,
                "persistence": record.persistence,
                "effectiveness": record.effectiveness,
                "risk_level": record.risk_level,
                "metadata": json.dumps(record.metadata),
            }
        except Exception as e:
            logger.error(f"Failed to prepare behavior data: {e}")
            raise

    async def _save_behavior_to_db(self, record_data: Dict[str, Any]) -> bool:
        """Сохранение записи поведения в базу данных."""
        db_path = self.config.behavior_directory / "behavior.db"
        try:
            await asyncio.get_event_loop().run_in_executor(
                self.executor, self._execute_behavior_insert, db_path, record_data
            )
            return True
        except Exception as e:
            logger.error(f"Database insert failed: {e}")
            return False

    def _execute_behavior_insert(
        self, db_path: Path, record_data: Dict[str, Any]
    ) -> None:
        """Выполнение вставки записи поведения."""
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO behavior_records (
                    symbol, timestamp, pattern_type, market_phase, volatility_regime,
                    liquidity_regime, volume_profile, price_action, order_flow,
                    spread_behavior, imbalance_behavior, pressure_behavior,
                    reaction_time, persistence, effectiveness, risk_level, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    record_data["symbol"],
                    record_data["timestamp"],
                    record_data["pattern_type"],
                    record_data["market_phase"],
                    record_data["volatility_regime"],
                    record_data["liquidity_regime"],
                    record_data["volume_profile"],
                    record_data["price_action"],
                    record_data["order_flow"],
                    record_data["spread_behavior"],
                    record_data["imbalance_behavior"],
                    record_data["pressure_behavior"],
                    record_data["reaction_time"],
                    record_data["persistence"],
                    record_data["effectiveness"],
                    record_data["risk_level"],
                    record_data["metadata"],
                ),
            )

    async def _update_caches_after_save(self, record: BehaviorRecord) -> None:
        """Обновление кэшей после сохранения."""
        # Очищаем кэш поведения
        if record.symbol in self.behavior_cache:
            del self.behavior_cache[record.symbol]
        # Очищаем кэш статистики
        if record.symbol in self.statistics_cache:
            del self.statistics_cache[record.symbol]

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

    async def get_behavior_history(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        pattern_type: Optional[MarketMakerPatternType] = None,
    ) -> List[BehaviorRecord]:
        """
        Получение истории поведения с фильтрацией.
        Args:
            symbol: Символ торговой пары
            start_date: Начальная дата
            end_date: Конечная дата
            pattern_type: Тип паттерна для фильтрации
        Returns:
            Список записей поведения
        """
        start_time = datetime.now()
        try:
            # Проверяем кэш
            cache_key = f"{symbol}_{start_date}_{end_date}_{pattern_type}"
            if cache_key in self.behavior_cache:
                self.metrics["cache_hits"] += 1
                records = self.behavior_cache[cache_key]
                self._update_read_metrics(start_time)
                return records
            self.metrics["cache_misses"] += 1
            # Загружаем из базы данных
            records = await self._load_behavior_from_db(
                symbol, start_date, end_date, pattern_type
            )
            # Кэшируем результат
            self.behavior_cache[cache_key] = records
            self._update_read_metrics(start_time)
            return records
        except Exception as e:
            self.metrics["error_count"] += 1
            logger.error(f"Failed to get behavior history for {symbol}: {e}")
            return []

    async def _load_behavior_from_db(
        self,
        symbol: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        pattern_type: Optional[MarketMakerPatternType],
    ) -> List[BehaviorRecord]:
        """Загрузка истории поведения из базы данных."""
        db_path = self.config.behavior_directory / "behavior.db"
        try:
            records_data = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._execute_behavior_query,
                db_path,
                symbol,
                start_date,
                end_date,
                pattern_type,
            )
            records = []
            for data in records_data:
                behavior_record = self._data_to_behavior_record(data)
                if behavior_record:
                    records.append(behavior_record)
            return records
        except Exception as e:
            logger.error(f"Database query failed for {symbol}: {e}")
            return []

    def _execute_behavior_query(
        self,
        db_path: Path,
        symbol: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        pattern_type: Optional[MarketMakerPatternType],
    ) -> List[Dict[str, Any]]:
        """Выполнение запроса истории поведения."""
        with sqlite3.connect(db_path) as conn:
            query = "SELECT * FROM behavior_records WHERE symbol = ?"
            params = [symbol]
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            if pattern_type:
                query += " AND pattern_type = ?"
                params.append(pattern_type.value)
            query += " ORDER BY timestamp DESC"
            cursor = conn.execute(query, params)
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def _data_to_behavior_record(
        self, data: Dict[str, Any]
    ) -> Optional[BehaviorRecord]:
        """Преобразование данных в объект BehaviorRecord."""
        try:
            return BehaviorRecord(
                symbol=data["symbol"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                pattern_type=MarketMakerPatternType(data["pattern_type"]),
                market_phase=data["market_phase"],
                volatility_regime=data["volatility_regime"],
                liquidity_regime=data["liquidity_regime"],
                volume_profile=(
                    json.loads(data["volume_profile"]) if data["volume_profile"] else {}
                ),
                price_action=(
                    json.loads(data["price_action"]) if data["price_action"] else {}
                ),
                order_flow=json.loads(data["order_flow"]) if data["order_flow"] else {},
                spread_behavior=(
                    json.loads(data["spread_behavior"])
                    if data["spread_behavior"]
                    else {}
                ),
                imbalance_behavior=(
                    json.loads(data["imbalance_behavior"])
                    if data["imbalance_behavior"]
                    else {}
                ),
                pressure_behavior=(
                    json.loads(data["pressure_behavior"])
                    if data["pressure_behavior"]
                    else {}
                ),
                reaction_time=data["reaction_time"],
                persistence=data["persistence"],
                effectiveness=data["effectiveness"],
                risk_level=data["risk_level"],
                metadata=json.loads(data["metadata"]) if data["metadata"] else {},
            )
        except Exception as e:
            logger.error(f"Failed to convert data to BehaviorRecord: {e}")
            return None

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

    async def get_behavior_statistics(
        self, symbol: str, days: int = 30
    ) -> Dict[str, Any]:
        """
        Получение статистики поведения.
        Args:
            symbol: Символ торговой пары
            days: Количество дней для анализа
        Returns:
            Статистика поведения
        """
        try:
            # Проверяем кэш
            cache_key = f"{symbol}_stats_{days}"
            if cache_key in self.statistics_cache:
                return self.statistics_cache[cache_key]
            # Получаем историю поведения
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            records = await self.get_behavior_history(symbol, start_date, end_date)
            if not records:
                return self._create_default_statistics()
            # Рассчитываем статистику
            statistics = self._calculate_behavior_statistics(records)
            # Кэшируем результат
            self.statistics_cache[cache_key] = statistics
            return statistics
        except Exception as e:
            logger.error(f"Failed to get behavior statistics for {symbol}: {e}")
            return self._create_default_statistics()

    def _calculate_behavior_statistics(
        self, records: List[BehaviorRecord]
    ) -> Dict[str, Any]:
        """Расчет статистики поведения."""
        try:
            if not records:
                return self._create_default_statistics()
            # Базовая статистика
            total_records = len(records)
            # Статистика по типам паттернов
            pattern_type_stats = {}
            for record in records:
                pattern_type = record.pattern_type.value
                if pattern_type not in pattern_type_stats:
                    pattern_type_stats[pattern_type] = {
                        "count": 0,
                        "avg_effectiveness": 0.0,
                        "avg_persistence": 0.0,
                        "avg_reaction_time": 0.0,
                    }
                stats = pattern_type_stats[pattern_type]
                stats["count"] += 1
                stats["avg_effectiveness"] += record.effectiveness
                stats["avg_persistence"] += record.persistence
                stats["avg_reaction_time"] += record.reaction_time
            # Нормализуем средние значения
            for pattern_type, stats in pattern_type_stats.items():
                count = stats["count"]
                if count > 0:
                    stats["avg_effectiveness"] /= count
                    stats["avg_persistence"] /= count
                    stats["avg_reaction_time"] /= count
            # Статистика по рыночным фазам
            market_phase_stats = {}
            for record in records:
                phase = record.market_phase
                if phase not in market_phase_stats:
                    market_phase_stats[phase] = {"count": 0, "avg_effectiveness": 0.0}
                stats = market_phase_stats[phase]
                stats["count"] += 1
                stats["avg_effectiveness"] += record.effectiveness
            # Нормализуем средние значения
            for phase, stats in market_phase_stats.items():
                count = stats["count"]
                if count > 0:
                    stats["avg_effectiveness"] /= count
            # Статистика по уровням риска
            risk_level_stats = {}
            for record in records:
                risk_level = record.risk_level
                if risk_level not in risk_level_stats:
                    risk_level_stats[risk_level] = {
                        "count": 0,
                        "avg_effectiveness": 0.0,
                    }
                stats = risk_level_stats[risk_level]
                stats["count"] += 1
                stats["avg_effectiveness"] += record.effectiveness
            # Нормализуем средние значения
            for risk_level, stats in risk_level_stats.items():
                count = stats["count"]
                if count > 0:
                    stats["avg_effectiveness"] /= count
            # Общая статистика
            avg_effectiveness = sum(r.effectiveness for r in records) / total_records
            avg_persistence = sum(r.persistence for r in records) / total_records
            avg_reaction_time = sum(r.reaction_time for r in records) / total_records
            return {
                "total_records": total_records,
                "avg_effectiveness": avg_effectiveness,
                "avg_persistence": avg_persistence,
                "avg_reaction_time": avg_reaction_time,
                "pattern_type_stats": pattern_type_stats,
                "market_phase_stats": market_phase_stats,
                "risk_level_stats": risk_level_stats,
                "analysis_timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Failed to calculate behavior statistics: {e}")
            return self._create_default_statistics()

    def _create_default_statistics(self) -> Dict[str, Any]:
        """Создание статистики по умолчанию."""
        return {
            "total_records": 0,
            "avg_effectiveness": 0.0,
            "avg_persistence": 0.0,
            "avg_reaction_time": 0.0,
            "pattern_type_stats": {},
            "market_phase_stats": {},
            "risk_level_stats": {},
            "analysis_timestamp": datetime.now().isoformat(),
        }

    async def cleanup_old_behavior_data(self, symbol: str, days: int = 90) -> int:
        """
        Очистка старых данных поведения.
        Args:
            symbol: Символ торговой пары
            days: Количество дней для сохранения
        Returns:
            Количество удаленных записей
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            db_path = self.config.behavior_directory / "behavior.db"
            deleted_count = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._execute_cleanup, db_path, symbol, cutoff_date
            )
            # Очищаем кэши
            if symbol in self.behavior_cache:
                del self.behavior_cache[symbol]
            if symbol in self.statistics_cache:
                del self.statistics_cache[symbol]
            logger.info(f"Cleaned up {deleted_count} old behavior records for {symbol}")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup old behavior data for {symbol}: {e}")
            return 0

    def _execute_cleanup(
        self, db_path: Path, symbol: str, cutoff_date: datetime
    ) -> int:
        """Выполнение очистки старых данных."""
        with sqlite3.connect(db_path) as conn:
            # Получаем количество записей для удаления
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM behavior_records 
                WHERE symbol = ? AND timestamp < ?
            """,
                (symbol, cutoff_date.isoformat()),
            )
            result = cursor.fetchone()
            deleted_count = result[0] if result else 0
            # Удаляем старые записи
            conn.execute(
                """
                DELETE FROM behavior_records 
                WHERE symbol = ? AND timestamp < ?
            """,
                (symbol, cutoff_date.isoformat()),
            )
            return deleted_count

    async def close(self) -> None:
        """Закрытие репозитория."""
        try:
            self.executor.shutdown(wait=True)
            logger.info("BehaviorHistoryRepository closed successfully")
        except Exception as e:
            logger.error(f"Error closing BehaviorHistoryRepository: {e}")

    def __del__(self) -> None:
        """Деструктор."""
        try:
            if hasattr(self, "executor"):
                self.executor.shutdown(wait=False)
        except Exception as e:
            logger.error(f"Failed to cleanup expired behavior records: {e}")
            # Возвращаем 0 при ошибке, но логируем проблему
            return 0
