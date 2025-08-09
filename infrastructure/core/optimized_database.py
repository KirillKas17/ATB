# -*- coding: utf-8 -*-
"""
Оптимизированный модуль для работы с базой данных торгового бота.
Включает:
- Пулы соединений для эффективного управления подключениями
- Кеширование часто используемых запросов
- Мониторинг производительности
- Индексы для оптимизации запросов
- Асинхронную обработку
"""
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, cast
from decimal import Decimal
from uuid import uuid4

import pandas as pd
from loguru import logger
try:
    from psycopg2.pool import SimpleConnectionPool
except ImportError:
    SimpleConnectionPool = None
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.pool import QueuePool

from domain.type_definitions import SignalTypeType, TradeId, Symbol, TimestampValue
from domain.entities.trade import Trade
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency


class OptimizedDatabase:
    """
    Оптимизированный класс для работы с базой данных торгового бота.
    Включает пулы соединений, кеширование, мониторинг производительности
    и асинхронную обработку.
    """

    def __init__(
        self, connection_string: str, pool_size: int = 10, max_overflow: int = 20
    ) -> None:
        """
        Инициализация оптимизированного подключения к базе данных.
        Args:
            connection_string: Строка подключения к базе данных
            pool_size: Размер пула соединений
            max_overflow: Максимальное количество дополнительных соединений
        """
        self.connection_string = connection_string
        self.engine: Optional[Engine] = None
        self.pool: Optional[SimpleConnectionPool] = None
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.executor = ThreadPoolExecutor(max_workers=pool_size)
        # Метрики производительности
        self.performance_metrics = {
            "query_count": 0,
            "total_query_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "slow_queries": [],
            "connection_pool_usage": 0,
            "avg_response_time": 0.0,
        }
        # Кеш для часто используемых данных
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 300  # 5 минут
        self._initialize()

    def _initialize(self) -> None:
        """Инициализация подключения и создание таблиц."""
        try:
            # Создаем engine с пулом соединений
            self.engine = create_engine(
                self.connection_string,
                poolclass=QueuePool,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600,  # Пересоздаем соединения каждый час
                echo=False,  # Отключаем логирование SQL
                pool_timeout=30,  # Таймаут ожидания соединения
                pool_reset_on_return="commit",  # Автоматический коммит при возврате соединения
            )
            # Создаем пул соединений для psycopg2
            self.pool = SimpleConnectionPool(
                minconn=1, maxconn=self.pool_size, dsn=self.connection_string
            )
            self._create_tables()
            self._create_indexes()
            self._create_partitions()
            logger.info("Optimized database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing optimized database: {e}")
            raise

    def _create_tables(self) -> None:
        """Создание оптимизированных таблиц."""
        try:
            if self.engine is None:
                raise ValueError("Engine is not initialized")
            with self.engine.connect() as connection:
                cursor = connection.connection.cursor()
                # Таблица для сделок с партиционированием
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS trades (
                        id SERIAL,
                        pair VARCHAR(20) NOT NULL,
                        side VARCHAR(10) NOT NULL,
                        size DECIMAL NOT NULL,
                        price DECIMAL NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        fee DECIMAL DEFAULT 0,
                        pnl DECIMAL DEFAULT NULL,
                        PRIMARY KEY (id, timestamp)
                    ) PARTITION BY RANGE (timestamp)
                """
                )
                # Таблица для сигналов
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS signals (
                        id SERIAL PRIMARY KEY,
                        pair VARCHAR(20) NOT NULL,
                        source VARCHAR(50) NOT NULL,
                        signal_type VARCHAR(20) NOT NULL,
                        strength DECIMAL NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        metadata JSONB
                    )
                """
                )
                # Таблица для рыночных данных с партиционированием
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS market_data (
                        id SERIAL,
                        pair VARCHAR(20) NOT NULL,
                        timeframe VARCHAR(10) NOT NULL,
                        open DECIMAL NOT NULL,
                        high DECIMAL NOT NULL,
                        low DECIMAL NOT NULL,
                        close DECIMAL NOT NULL,
                        volume DECIMAL NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        indicators JSONB,
                        PRIMARY KEY (id, timestamp)
                    ) PARTITION BY RANGE (timestamp)
                """
                )
                # Таблица для производительности
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS performance (
                        id SERIAL PRIMARY KEY,
                        metric VARCHAR(50) NOT NULL,
                        value DECIMAL NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        metadata JSONB
                    )
                """
                )
                connection.commit()
                cursor.close()
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise

    def _create_indexes(self) -> None:
        """Создание оптимизированных индексов."""
        try:
            if self.engine is None:
                raise ValueError("Engine is not initialized")
            with self.engine.connect() as connection:
                cursor = connection.connection.cursor()
                # Составные индексы для trades
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trades_pair_timestamp ON trades(pair, timestamp DESC)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trades_side_timestamp ON trades(side, timestamp DESC)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trades_timestamp_btree ON trades USING btree(timestamp)"
                )
                # Индексы для signals
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_signals_pair_timestamp ON signals(pair, timestamp DESC)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_signals_source_type ON signals(source, signal_type)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_signals_strength ON signals(strength DESC)"
                )
                # Составные индексы для market_data
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_market_data_pair_timeframe_timestamp ON market_data(pair, timeframe, timestamp DESC)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_market_data_timestamp_btree ON market_data USING btree(timestamp)"
                )
                # Индексы для performance
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_performance_metric_timestamp ON performance(metric, timestamp DESC)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_performance_value ON performance(value DESC)"
                )
                # GIN индексы для JSONB полей
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_signals_metadata_gin ON signals USING gin(metadata)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_market_data_indicators_gin ON market_data USING gin(indicators)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_performance_metadata_gin ON performance USING gin(metadata)"
                )
                connection.commit()
                cursor.close()
                logger.info("Database indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            raise

    def _create_partitions(self) -> None:
        """Создание партиций для больших таблиц."""
        try:
            if self.engine is None:
                raise ValueError("Engine is not initialized")
            with self.engine.connect() as connection:
                cursor = connection.connection.cursor()
                # Создаем партиции для trades по месяцам
                current_date = datetime.now()
                for i in range(12):  # Создаем партиции на год вперед
                    partition_date = current_date.replace(
                        month=((current_date.month + i - 1) % 12) + 1
                    )
                    partition_name = f"trades_{partition_date.strftime('%Y_%m')}"
                    start_date = partition_date.replace(day=1)
                    if partition_date.month == 12:
                        end_date = partition_date.replace(
                            year=partition_date.year + 1, month=1, day=1
                        )
                    else:
                        end_date = partition_date.replace(
                            month=partition_date.month + 1, day=1
                        )
                    cursor.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS {partition_name} 
                        PARTITION OF trades 
                        FOR VALUES FROM ('{start_date.isoformat()}') TO ('{end_date.isoformat()}')
                    """
                    )
                # Создаем партиции для market_data по месяцам
                for i in range(12):
                    partition_date = current_date.replace(
                        month=((current_date.month + i - 1) % 12) + 1
                    )
                    partition_name = f"market_data_{partition_date.strftime('%Y_%m')}"
                    start_date = partition_date.replace(day=1)
                    if partition_date.month == 12:
                        end_date = partition_date.replace(
                            year=partition_date.year + 1, month=1, day=1
                        )
                    else:
                        end_date = partition_date.replace(
                            month=partition_date.month + 1, day=1
                        )
                    cursor.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS {partition_name} 
                        PARTITION OF market_data 
                        FOR VALUES FROM ('{start_date.isoformat()}') TO ('{end_date.isoformat()}')
                    """
                    )
                connection.commit()
                cursor.close()
                logger.info("Database partitions created successfully")
        except Exception as e:
            logger.error(f"Error creating partitions: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """Контекстный менеджер для получения соединения из пула."""
        conn = None
        start_time = time.time()
        try:
            conn = self.pool.getconn()
            self.performance_metrics["connection_pool_usage"] += 1
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)
                self.performance_metrics["connection_pool_usage"] -= 1
                # Обновляем среднее время ответа
                response_time = time.time() - start_time
                self.performance_metrics["avg_response_time"] = (
                    self.performance_metrics["avg_response_time"] + response_time
                ) / 2

    def _log_query_performance(self, query: str, execution_time: float) -> None:
        """Логирование производительности запросов."""
        query_count = self.performance_metrics.get("query_count", 0)
        total_query_time = self.performance_metrics.get("total_query_time", 0.0)

        # Безопасное преобразование в int
        if isinstance(query_count, (int, float)):
            self.performance_metrics["query_count"] = int(query_count) + 1
        else:
            self.performance_metrics["query_count"] = 1

        # Безопасное преобразование в float
        if isinstance(total_query_time, (int, float)):
            self.performance_metrics["total_query_time"] = float(total_query_time) + execution_time
        else:
            self.performance_metrics["total_query_time"] = execution_time
            
        # Логируем медленные запросы (>100ms)
        if execution_time > 0.1:
            slow_queries = self.performance_metrics.get("slow_queries", [])
            if not isinstance(slow_queries, list):
                slow_queries = []
            slow_queries.append(
                {
                    "query": query[:100] + "..." if len(query) > 100 else query,
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            self.performance_metrics["slow_queries"] = slow_queries
            # Ограничиваем количество записей о медленных запросах
            if len(slow_queries) > 100:
                self.performance_metrics["slow_queries"] = slow_queries[-50:]

    def _get_cache_key(self, method: str, **kwargs) -> str:
        """Генерация ключа кеша."""
        return f"{method}:{hash(frozenset(kwargs.items()))}"

    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Проверка валидности кеша."""
        return time.time() - cache_entry["timestamp"] < self._cache_ttl

    @lru_cache(maxsize=1000)
    def _get_cached_trades(self, pair: str, limit: int = 100) -> List[Trade]:
        """Кешированное получение сделок."""
        cache_hits = self.performance_metrics.get("cache_hits", 0)
        if isinstance(cache_hits, (int, float)):
            self.performance_metrics["cache_hits"] = int(cache_hits) + 1
        else:
            self.performance_metrics["cache_hits"] = 1
        return self._get_trades_uncached(pair, limit)

    def _get_trades_uncached(self, pair: str, limit: int = 100) -> List[Trade]:
        """Некешированное получение сделок."""
        cache_misses = self.performance_metrics.get("cache_misses", 0)
        if isinstance(cache_misses, (int, float)):
            self.performance_metrics["cache_misses"] = int(cache_misses) + 1
        else:
            self.performance_metrics["cache_misses"] = 1
        start_time = time.time()
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                # Используем подготовленные запросы для лучшей производительности
                query = """
                    SELECT id, pair, side, size, price, timestamp, fee, pnl 
                    FROM trades 
                    WHERE pair = %s 
                    ORDER BY timestamp DESC 
                    LIMIT %s
                """
                cursor.execute(query, (pair, limit))
                rows = cursor.fetchall()
                result = []
                for row in rows:
                    try:
                        # Корректно определяем side
                        side = str(row[2]).lower()
                        if side not in ["buy", "sell"]:
                            side = "buy"
                        side = SignalTypeType(side)
                        # Корректно создаём executed_at
                        ts = row[5]
                        if hasattr(ts, 'to_pydatetime'):
                            ts = ts.to_pydatetime()
                        elif hasattr(ts, '__getitem__') and not isinstance(ts, str):
                            ts = ts[-1] if len(ts) else datetime.now()
                        # Корректно создаём Trade
                        trade = Trade(
                            id=TradeId(uuid4()),
                            symbol=Symbol(str(row[1])),
                            side=side,
                            price=Price(Decimal(str(row[4])), Currency.USDT),
                            volume=Volume(Decimal(str(row[3]))),
                            executed_at=TimestampValue(ts),
                            fee=Money(Decimal(str(row[6])), Currency.USDT),
                            realized_pnl=Money(Decimal(str(row[7])), Currency.USDT) if row[7] is not None else None,
                        )
                        result.append(trade)
                    except Exception as e:
                        logger.warning(f"Error creating Trade object from row {row}: {e}")
                        continue
                execution_time = time.time() - start_time
                self._log_query_performance(query, execution_time)
                return result
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []

    async def save_trade_async(self, trade: Trade) -> int:
        """Асинхронное сохранение сделки."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.save_trade, trade)

    def save_trade(self, trade: Trade) -> int:
        """Сохранение сделки с оптимизацией."""
        start_time = time.time()
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                # Используем подготовленные запросы
                cursor.execute(
                    """
                    INSERT INTO trades (pair, side, size, price, timestamp, fee, pnl)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """,
                    (
                        str(trade.symbol),
                        trade.side,
                        float(trade.volume.value),
                        float(trade.price.value),
                        trade.executed_at,
                        float(trade.fee.value),
                        float(trade.realized_pnl.value) if trade.realized_pnl else None,
                    ),
                )
                result = cursor.fetchone()
                if result is None:
                    raise ValueError("Failed to get trade ID")
                trade_id = result[0]
                conn.commit()
                execution_time = time.time() - start_time
                self._log_query_performance("INSERT INTO trades", execution_time)
                # Инвалидируем кеш
                self._get_cached_trades.cache_clear()
                return int(trade_id)
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
            raise

    def get_trades(
        self, pair: Optional[str] = None, use_cache: bool = True
    ) -> List[Trade]:
        """Получение сделок с кешированием."""
        if pair and use_cache:
            return self._get_cached_trades(pair)
        return self._get_trades_uncached(pair) if pair else []

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Получение метрик производительности."""
        metrics = self.performance_metrics.copy()
        query_count = metrics.get("query_count", 0)
        if query_count > 0:
            total_query_time = metrics.get("total_query_time", 0.0)
            cache_hits = metrics.get("cache_hits", 0)
            cache_misses = metrics.get("cache_misses", 0)
            metrics["avg_query_time"] = total_query_time / query_count
            # Явное приведение типов для арифметических операций
            cache_hits_int = int(cache_hits) if isinstance(cache_hits, (int, float, str)) else 0
            cache_misses_int = int(cache_misses) if isinstance(cache_misses, (int, float, str)) else 0
            total_cache = cache_hits_int + cache_misses_int
            metrics["cache_hit_rate"] = cache_hits_int / total_cache if total_cache > 0 else 0.0
        else:
            metrics["avg_query_time"] = 0.0
            metrics["cache_hit_rate"] = 0.0
        # Добавляем информацию о пуле соединений
        if self.pool:
            metrics["pool_size"] = self.pool_size
            try:
                conn = self.pool.getconn()
                # Исправляем операции с объектами
                if hasattr(conn, 'closed') and callable(conn.closed):
                    is_closed = conn.closed()
                else:
                    is_closed = False
                pool_size_int = int(self.pool_size) if isinstance(self.pool_size, (int, float, str)) else 0
                active_connections = pool_size_int - (1 if is_closed else 0)  # type: ignore[operator]
                metrics["active_connections"] = active_connections
                self.pool.putconn(conn)
            except Exception:
                metrics["active_connections"] = 0
        return metrics

    def optimize_database(self) -> Dict[str, Any]:
        """Оптимизация базы данных."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                # Анализируем таблицы
                cursor.execute("ANALYZE trades")
                cursor.execute("ANALYZE signals")
                cursor.execute("ANALYZE market_data")
                cursor.execute("ANALYZE performance")
                # Очищаем старые данные (старше 1 года)
                cursor.execute(
                    "DELETE FROM trades WHERE timestamp < NOW() - INTERVAL '1 year'"
                )
                cursor.execute(
                    "DELETE FROM market_data WHERE timestamp < NOW() - INTERVAL '1 year'"
                )
                # Перестраиваем индексы
                cursor.execute("REINDEX INDEX idx_trades_pair_timestamp")
                cursor.execute("REINDEX INDEX idx_market_data_pair_timeframe_timestamp")
                conn.commit()
                return {
                    "status": "success",
                    "message": "Database optimization completed",
                    "timestamp": datetime.now().isoformat(),
                }
        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def close(self) -> None:
        """Закрытие соединений."""
        try:
            if self.pool:
                self.pool.closeall()
                self.pool = None
            if self.engine is not None:
                self.engine.dispose()
                self.engine = None
            if self.executor:
                self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
            raise

    def __enter__(self):
        """Контекстный менеджер: вход."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер: выход."""
        self.close()
