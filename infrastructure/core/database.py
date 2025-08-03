# -*- coding: utf-8 -*-
"""
Модуль для работы с базой данных торгового бота.
Предоставляет класс Database для управления подключением к базе данных,
создания таблиц, сохранения и получения торговых данных, сигналов,
рыночных данных и метрик производительности.
"""
import json
import time
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, cast
from decimal import Decimal
from uuid import uuid4

from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.pool import QueuePool

import pandas as pd
try:
    import psycopg2
    from psycopg2.pool import SimpleConnectionPool
    from psycopg2.extensions import connection as PGConnection
except ImportError:
    psycopg2 = None
    SimpleConnectionPool = None
    PGConnection = None

from domain.entities.market import MarketData
from domain.entities.trading import Trade, OrderSide
from domain.value_objects.price import Price
from domain.value_objects.money import Money
from domain.types import TradeId, TimestampValue, Symbol, TradingPair
from domain.value_objects.currency import Currency
from domain.value_objects.volume import Volume
# from domain.value_objects.metadata import MetadataDict


class Database:
    """
    Класс для работы с базой данных торгового бота.
    Предоставляет методы для сохранения и получения торговых данных,
    сигналов, рыночных данных и метрик производительности.
    """

    def __init__(
        self, connection_string: str, pool_size: int = 10, max_overflow: int = 20
    ):
        """
        Инициализация подключения к базе данных.
        Args:
            connection_string: Строка подключения к базе данных
            pool_size: Размер пула соединений
            max_overflow: Максимальное количество дополнительных соединений
        """
        self.connection_string = connection_string
        self.engine: Optional[Engine] = None
        self.conn: Optional[PGConnection] = None
        self.pool: Optional[SimpleConnectionPool] = None
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.performance_metrics: Dict[str, Any] = {
            "query_count": 0,
            "total_query_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "slow_queries": [],
        }
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
            )
            # Создаем пул соединений для psycopg2
            self.pool = SimpleConnectionPool(
                minconn=1, maxconn=self.pool_size, dsn=self.connection_string
            )
            self._create_tables()
            self._create_indexes()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def _create_tables(self) -> None:
        """Создание таблиц в базе данных."""
        try:
            if self.engine is None:
                raise ValueError("Database engine is not initialized")
            with self.engine.connect() as connection:
                cursor = connection.connection.cursor()
                if cursor is None:
                    raise ValueError("Failed to create cursor")
                # Таблица для сделок
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS trades (
                        id SERIAL PRIMARY KEY,
                        pair VARCHAR(20) NOT NULL,
                        side VARCHAR(10) NOT NULL,
                        size DECIMAL NOT NULL,
                        price DECIMAL NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        fee DECIMAL DEFAULT 0,
                        pnl DECIMAL DEFAULT NULL
                    )
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
                # Таблица для рыночных данных
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS market_data (
                        id SERIAL PRIMARY KEY,
                        pair VARCHAR(20) NOT NULL,
                        timeframe VARCHAR(10) NOT NULL,
                        open DECIMAL NOT NULL,
                        high DECIMAL NOT NULL,
                        low DECIMAL NOT NULL,
                        close DECIMAL NOT NULL,
                        volume DECIMAL NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        indicators JSONB
                    )
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
        """Создание индексов для оптимизации запросов."""
        try:
            if self.engine is None:
                raise ValueError("Database engine is not initialized")
            with self.engine.connect() as connection:
                connection = cast(Connection, connection)
                cursor = connection.connection.cursor()
                # Индексы для таблицы trades
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trades_pair_timestamp ON trades(pair, timestamp)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trades_side ON trades(side)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)"
                )
                # Индексы для таблицы signals
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_signals_pair_timestamp ON signals(pair, timestamp)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_signals_source ON signals(source)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_signals_signal_type ON signals(signal_type)"
                )
                # Индексы для таблицы market_data
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_market_data_pair_timeframe ON market_data(pair, timeframe)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp)"
                )
                # Индексы для таблицы performance
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_performance_metric_timestamp ON performance(metric, timestamp)"
                )
                connection.commit()
                cursor.close()
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """Контекстный менеджер для получения соединения с базой данных."""
        if self.pool is None:
            raise ValueError("Database pool is not initialized")
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
                self.pool.putconn(conn)

    def _log_query_performance(self, query: str, execution_time: float) -> None:
        """Логирование производительности запросов."""
        self.performance_metrics["query_count"] += 1
        self.performance_metrics["total_query_time"] += execution_time
        if execution_time > 1.0:  # Запросы дольше 1 секунды считаются медленными
            self.performance_metrics["slow_queries"].append({
                "query": query,
                    "execution_time": execution_time,
                "timestamp": datetime.now(),
            })

    @lru_cache(maxsize=1000)
    def _get_cached_trades(self, pair: str, limit: int = 100) -> List[Trade]:
        """Получение сделок из кэша."""
        self.performance_metrics["cache_hits"] += 1
        return self._get_trades_uncached(pair, limit)

    def _get_trades_uncached(self, pair: str, limit: int = 100) -> List[Trade]:
        """Получение сделок без кэширования."""
        self.performance_metrics["cache_misses"] += 1
        start_time = time.time()
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
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
                    # Преобразуем side в OrderSide enum
                    side = OrderSide.BUY if row[2] == "buy" else OrderSide.SELL
                    
                    # Обработка timestamp
                    ts = row[5]
                    if hasattr(ts, 'to_pydatetime'):
                        ts = ts.to_pydatetime()
                    elif hasattr(ts, '__getitem__') and not isinstance(ts, str):
                        ts = ts[-1] if len(ts) else datetime.now()
                    
                    trade = Trade(
                        id=TradeId(uuid4()),  # Генерируем новый UUID
                        trading_pair=TradingPair(str(row[1])),
                        side=side,  # Используем OrderSide enum
                        quantity=Volume(Decimal(str(row[3]))),
                        price=Price(Decimal(str(row[4])), Currency.USDT),
                        commission=Money(Decimal(str(row[6])), Currency.USDT),
                        timestamp=TimestampValue(ts),
                    )
                    result.append(trade)
                execution_time = time.time() - start_time
                self._log_query_performance(query, execution_time)
                return result
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []

    def save_trade(self, trade: Trade) -> int:
        """
        Сохранение сделки в базу данных.
        Args:
            trade: Объект сделки для сохранения
        Returns:
            ID сохраненной сделки
        Raises:
            ValueError: При ошибке инициализации подключения или создания курсора
        """
        start_time = time.time()
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO trades (pair, side, size, price, timestamp, fee, pnl)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """,
                    (
                        str(trade.trading_pair),
                        trade.side.value,  # Используем value для enum
                        float(trade.quantity.value),
                        float(trade.price.value),
                        trade.timestamp.isoformat() if hasattr(trade.timestamp, 'isoformat') else str(trade.timestamp),
                        float(trade.commission.value),
                        None,  # pnl не доступен в Trade
                    ),
                )
                result = cursor.fetchone()
                if result is None:
                    raise ValueError("Failed to get trade ID")
                trade_id = result[0]
                conn.commit()
                execution_time = time.time() - start_time
                self._log_query_performance("INSERT INTO trades", execution_time)
                return int(trade_id)
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
            raise

    def save_signal(self, signal_data: Dict[str, Any]) -> int:
        """
        Сохранение сигнала в базу данных.
        Args:
            signal_data: Данные сигнала для сохранения
        Returns:
            ID сохраненного сигнала
        Raises:
            ValueError: При ошибке инициализации подключения или создания курсора
        """
        start_time = time.time()
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO signals (pair, source, signal_type, strength, timestamp, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """,
                    (
                        signal_data["pair"],
                        signal_data["source"],
                        signal_data["signal_type"],
                        float(signal_data["strength"]),
                        signal_data["timestamp"],
                        json.dumps(signal_data.get("metadata", {})),
                    ),
                )
                result = cursor.fetchone()
                if result is None:
                    raise ValueError("Failed to get signal ID")
                signal_id = result[0]
                conn.commit()
                execution_time = time.time() - start_time
                self._log_query_performance("INSERT INTO signals", execution_time)
                return int(signal_id)
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
            raise

    def save_market_data(self, market_data: MarketData) -> int:
        """
        Сохранение рыночных данных.
        Args:
            market_data: Объект рыночных данных для сохранения
        Returns:
            ID сохраненных рыночных данных
        Raises:
            ValueError: При ошибке инициализации подключения или создания курсора
        """
        start_time = time.time()
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO market_data (pair, timeframe, open, high, low, close, volume, timestamp, indicators)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """,
                    (
                        str(market_data.symbol),
                        str(market_data.timeframe),
                        float(market_data.open.value),
                        float(market_data.high.value),
                        float(market_data.low.value),
                        float(market_data.close.value),
                        float(market_data.volume.value),
                        market_data.timestamp.isoformat() if hasattr(market_data.timestamp, 'isoformat') else str(market_data.timestamp),
                        json.dumps(market_data.metadata),
                    ),
                )
                result = cursor.fetchone()
                if result is None:
                    raise ValueError("Failed to get market data ID")
                data_id = result[0]
                conn.commit()
                execution_time = time.time() - start_time
                self._log_query_performance("INSERT INTO market_data", execution_time)
                return int(data_id)
        except Exception as e:
            logger.error(f"Error saving market data: {e}")
            raise

    def save_performance(self, performance_data: Dict[str, Any]) -> int:
        """
        Сохранение информации о производительности.
        Args:
            performance_data: Данные производительности для сохранения
        Returns:
            ID сохраненной записи производительности
        Raises:
            ValueError: При ошибке инициализации подключения или создания курсора
        """
        start_time = time.time()
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO performance (metric, value, timestamp, metadata)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """,
                    (
                        str(performance_data["metric"]),
                        float(performance_data["value"]),
                        datetime.now(),
                        json.dumps(performance_data.get("metadata", {})),
                    ),
                )
                result = cursor.fetchone()
                if result is None:
                    raise ValueError("Failed to get performance ID")
                perf_id = result[0]
                conn.commit()
                execution_time = time.time() - start_time
                self._log_query_performance("INSERT INTO performance", execution_time)
                return int(perf_id)
        except Exception as e:
            logger.error(f"Error saving performance: {e}")
            raise

    def get_trades(
        self, pair: Optional[str] = None, use_cache: bool = True
    ) -> List[Trade]:
        """
        Получение сделок.
        Args:
            pair: Фильтр по торговой паре (опционально)
            use_cache: Использовать кеширование
        Returns:
            Список объектов сделок
        """
        if pair and use_cache:
            return self._get_cached_trades(pair)
        return self._get_trades_uncached(pair) if pair else []

    def get_signals(
        self,
        pair: Optional[str] = None,
        source: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Получение сигналов.
        Args:
            pair: Фильтр по торговой паре (опционально)
            source: Фильтр по источнику (опционально)
            start_time: Начальное время (опционально)
            end_time: Конечное время (опционально)
        Returns:
            Список словарей с данными сигналов
        Raises:
            ValueError: При ошибке инициализации подключения или создания курсора
        """
        start_time_query = time.time()
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                query = "SELECT * FROM signals WHERE 1=1"
                params = []
                if pair:
                    query += " AND pair = %s"
                    params.append(pair)
                if source:
                    query += " AND source = %s"
                    params.append(source)
                if start_time:
                    query += " AND timestamp >= %s"
                    params.append(str(start_time))
                if end_time:
                    query += " AND timestamp <= %s"
                    params.append(str(end_time))
                cursor.execute(query, params)
                result = [dict(row) for row in cursor.fetchall()]
                execution_time = time.time() - start_time_query
                self._log_query_performance(query, execution_time)
                return result
        except Exception as e:
            logger.error(f"Error getting signals: {e}")
            raise

    def get_market_data(
        self, pair: str, start_time: datetime, end_time: datetime
    ) -> List[MarketData]:
        """
        Получение рыночных данных.
        Args:
            pair: Торговая пара
            start_time: Начальное время
            end_time: Конечное время
        Returns:
            Список объектов рыночных данных
        """
        start_time_query = time.time()
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM market_data WHERE pair = %s AND timestamp BETWEEN %s AND %s ORDER BY timestamp",
                    (pair, start_time.isoformat(), end_time.isoformat()),
                )
                rows = cursor.fetchall()
                result = []
                for row in rows:
                    # Создаем MarketData объект с правильной структурой
                    market_data = MarketData(
                        symbol=Symbol(row[1]),  # pair column
                        timeframe=row[2],  # timeframe column
                        timestamp=TimestampValue(pd.to_datetime(row[8]).to_pydatetime()),  # timestamp column
                        open=Price(Decimal(str(row[3])), Currency.USDT),
                        high=Price(Decimal(str(row[4])), Currency.USDT),
                        low=Price(Decimal(str(row[5])), Currency.USDT),
                        close=Price(Decimal(str(row[6])), Currency.USDT),
                        volume=Volume(Decimal(str(row[7]))),
                        metadata=MetadataDict(
                            json.loads(row[9]) if row[9] else {}
                        ),  # indicators column
                    )
                    result.append(market_data)
                execution_time = time.time() - start_time_query
                self._log_query_performance("SELECT FROM market_data", execution_time)
                return result
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return []

    def get_performance(
        self,
        metric: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Получение информации о производительности.
        Args:
            metric: Фильтр по метрике (опционально)
            start_time: Начальное время (опционально)
            end_time: Конечное время (опционально)
        Returns:
            Список словарей с данными производительности
        Raises:
            ValueError: При ошибке инициализации подключения или создания курсора
        """
        start_time_query = time.time()
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                query = "SELECT * FROM performance WHERE 1=1"
                params = []
                if metric:
                    query += " AND metric = %s"
                    params.append(metric)
                if start_time:
                    query += " AND timestamp >= %s"
                    params.append(str(start_time))
                if end_time:
                    query += " AND timestamp <= %s"
                    params.append(str(end_time))
                cursor.execute(query, params)
                result = [dict(row) for row in cursor.fetchall()]
                execution_time = time.time() - start_time_query
                self._log_query_performance(query, execution_time)
                return result
        except Exception as e:
            logger.error(f"Error getting performance: {e}")
            raise

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Получение метрик производительности базы данных."""
        metrics = self.performance_metrics.copy()
        if metrics["query_count"] > 0:
            metrics["avg_query_time"] = (
                metrics["total_query_time"] / metrics["query_count"]
            )
            metrics["cache_hit_rate"] = metrics["cache_hits"] / (
                metrics["cache_hits"] + metrics["cache_misses"]
            )
        else:
            metrics["avg_query_time"] = 0.0
            metrics["cache_hit_rate"] = 0.0
        return metrics

    def close(self) -> None:
        """
        Закрытие соединения с базой данных.
        Raises:
            Exception: При ошибке закрытия соединения
        """
        try:
            if self.pool:
                self.pool.closeall()
                self.pool = None
            if self.engine is not None:
                self.engine.dispose()
                self.engine = None
            if self.conn is not None:
                self.conn.close()
                self.conn = None
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
            raise

    def __enter__(self):
        """
        Контекстный менеджер: вход.
        Returns:
            Экземпляр класса Database
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Контекстный менеджер: выход.
        Args:
            exc_type: Тип исключения
            exc_val: Значение исключения
            exc_tb: Трейсбек исключения
        """
        self.close()
