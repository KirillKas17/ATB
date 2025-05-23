import json
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import pandas as pd
import psycopg2
import psycopg2.extensions
import psycopg2.extras
from loguru import logger
from psycopg2.extensions import connection as PGConnection
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session

from core.models import MarketData, Trade


class Database:
    """Класс для работы с базой данных."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engine: Optional[Engine] = None
        self.conn: Optional[PGConnection] = None
        self.session: Optional[Session] = None
        self._initialize()

    def _initialize(self) -> None:
        try:
            connection_string = self.config["database"]["connection_string"]
            self.engine = create_engine(connection_string)
            self.conn = psycopg2.connect(connection_string)
            self.session = Session(self.engine)
            self._create_tables()
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            self.engine = None
            self.conn = None
            self.session = None

    def _create_tables(self) -> None:
        """Создание необходимых таблиц."""
        if self.engine is None or self.conn is None:
            return

        try:
            with self.engine.connect() as connection:
                connection = cast(Connection, connection)
                cursor = connection.connection.cursor()
                if cursor is None:
                    return

                # Таблица для сделок
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS trades (
                        id SERIAL PRIMARY KEY,
                        pair VARCHAR(20) NOT NULL,
                        type VARCHAR(10) NOT NULL,
                        price DECIMAL NOT NULL,
                        size DECIMAL NOT NULL,
                        pnl DECIMAL,
                        timestamp TIMESTAMP NOT NULL,
                        metadata JSONB
                    )
                """
                )

                # Таблица для сигналов
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS signals (
                        id SERIAL PRIMARY KEY,
                        pair VARCHAR(20) NOT NULL,
                        type VARCHAR(10) NOT NULL,
                        confidence DECIMAL NOT NULL,
                        source VARCHAR(50) NOT NULL,
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

                if self.session:
                    self.session.commit()
                cursor.close()

        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            if self.session:
                self.session.rollback()
            raise

    def save_trade(self, trade: Trade) -> bool:
        """Сохранение сделки."""
        if self.engine is None or self.conn is None or self.session is None:
            return False

        try:
            with self.engine.connect() as connection:
                connection = cast(Connection, connection)
                cursor = connection.connection.cursor()
                if cursor is None:
                    return False

                cursor.execute(
                    "INSERT INTO trades (id, pair, side, size, price, timestamp, fee, pnl) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        trade.id,
                        trade.pair,
                        trade.side,
                        trade.size,
                        trade.price,
                        trade.timestamp.isoformat(),
                        trade.fee,
                        trade.pnl,
                    ),
                )
                if self.session:
                    self.session.commit()
                cursor.close()
                return True
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
            if self.session:
                self.session.rollback()
            return False

    def save_signal(self, signal_data: Dict[str, Any]) -> int:
        """Сохранение информации о сигнале."""
        try:
            if self.engine is None or self.conn is None:
                raise ValueError("Database connection is not initialized")

            with self.engine.connect() as connection:
                connection = cast(Connection, connection)
                cursor = connection.connection.cursor()
                if cursor is None:
                    raise ValueError("Failed to create cursor")

                cursor.execute(
                    """
                    INSERT INTO signals (pair, type, confidence, source, timestamp, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """,
                    (
                        str(signal_data["pair"]),
                        str(signal_data["type"]),
                        float(signal_data["confidence"]),
                        str(signal_data["source"]),
                        datetime.now(),
                        json.dumps(signal_data.get("metadata", {})),
                    ),
                )
                result = cursor.fetchone()
                if result is None:
                    raise ValueError("Failed to get signal ID")
                signal_id = result[0]
                connection.commit()
                cursor.close()
                return int(signal_id)

        except Exception as e:
            if self.engine is not None:
                with self.engine.connect() as connection:
                    connection = cast(Connection, connection)
                    connection.rollback()
            logger.error(f"Error saving signal: {e}")
            raise

    def save_market_data(self, data: List[MarketData]) -> bool:
        """Сохранение рыночных данных."""
        if self.engine is None or self.conn is None:
            return False

        try:
            with self.engine.connect() as connection:
                connection = cast(Connection, connection)
                cursor = connection.connection.cursor()
                if cursor is None:
                    return False

                for item in data:
                    cursor.execute(
                        "INSERT INTO market_data (timestamp, pair, open, high, low, close, volume) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        (
                            item.timestamp.isoformat(),
                            item.pair,
                            item.open,
                            item.high,
                            item.low,
                            item.close,
                            item.volume,
                        ),
                    )
                connection.commit()
                cursor.close()
                return True
        except Exception as e:
            logger.error(f"Error saving market data: {e}")
            return False

    def save_performance(self, performance_data: Dict[str, Any]) -> int:
        """Сохранение информации о производительности."""
        try:
            if self.engine is None or self.conn is None:
                raise ValueError("Database connection is not initialized")

            with self.engine.connect() as connection:
                connection = cast(Connection, connection)
                cursor = connection.connection.cursor()
                if cursor is None:
                    raise ValueError("Failed to create cursor")

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
                connection.commit()
                cursor.close()
                return int(perf_id)

        except Exception as e:
            if self.engine is not None:
                with self.engine.connect() as connection:
                    connection = cast(Connection, connection)
                    connection.rollback()
            logger.error(f"Error saving performance: {e}")
            raise

    def get_trades(self, pair: Optional[str] = None) -> List[Trade]:
        """Получение сделок."""
        if self.engine is None or self.conn is None:
            return []

        try:
            with self.engine.connect() as connection:
                connection = cast(Connection, connection)
                cursor = connection.connection.cursor()
                if cursor is None:
                    return []

                if pair:
                    cursor.execute("SELECT * FROM trades WHERE pair = %s", (pair,))
                else:
                    cursor.execute("SELECT * FROM trades")
                rows = cursor.fetchall()
                if rows is None:
                    return []
                result = [
                    Trade(
                        id=row[0],
                        pair=row[1],
                        side=row[2],
                        size=float(row[3]),
                        price=float(row[4]),
                        timestamp=pd.to_datetime(row[5]),
                        fee=float(row[6]),
                        pnl=float(row[7]) if row[7] is not None else None,
                    )
                    for row in rows
                ]
                cursor.close()
                return result
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []

    def get_signals(
        self,
        pair: Optional[str] = None,
        source: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Получение информации о сигналах."""
        try:
            if self.engine is None or self.conn is None:
                raise ValueError("Database connection is not initialized")

            with self.engine.connect() as connection:
                connection = cast(Connection, connection)
                cursor = connection.connection.cursor()
                if cursor is None:
                    raise ValueError("Failed to create cursor")

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
                    params.append(start_time)

                if end_time:
                    query += " AND timestamp <= %s"
                    params.append(end_time)

                cursor.execute(query, params)
                result = [dict(row) for row in cursor.fetchall()]
                cursor.close()
                return result

        except Exception as e:
            logger.error(f"Error getting signals: {e}")
            raise

    def get_market_data(
        self, pair: str, start_time: datetime, end_time: datetime
    ) -> List[MarketData]:
        """Получение рыночных данных."""
        if self.engine is None or self.conn is None:
            return []

        try:
            with self.engine.connect() as connection:
                connection = cast(Connection, connection)
                cursor = connection.connection.cursor()
                if cursor is None:
                    return []

                cursor.execute(
                    "SELECT * FROM market_data WHERE pair = %s AND timestamp BETWEEN %s AND %s",
                    (pair, start_time.isoformat(), end_time.isoformat()),
                )
                rows = cursor.fetchall()
                if rows is None:
                    return []
                result = [
                    MarketData(
                        timestamp=pd.to_datetime(row[0]),
                        pair=row[1],
                        open=float(row[2]),
                        high=float(row[3]),
                        low=float(row[4]),
                        close=float(row[5]),
                        volume=float(row[6]),
                    )
                    for row in rows
                ]
                cursor.close()
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
        """Получение информации о производительности."""
        try:
            if self.engine is None or self.conn is None:
                raise ValueError("Database connection is not initialized")

            with self.engine.connect() as connection:
                connection = cast(Connection, connection)
                cursor = connection.connection.cursor()
                if cursor is None:
                    raise ValueError("Failed to create cursor")

                query = "SELECT * FROM performance WHERE 1=1"
                params = []

                if metric:
                    query += " AND metric = %s"
                    params.append(metric)

                if start_time:
                    query += " AND timestamp >= %s"
                    params.append(start_time)

                if end_time:
                    query += " AND timestamp <= %s"
                    params.append(end_time)

                cursor.execute(query, params)
                result = [dict(row) for row in cursor.fetchall()]
                cursor.close()
                return result

        except Exception as e:
            logger.error(f"Error getting performance: {e}")
            raise

    def close(self) -> None:
        """Закрытие соединения с базой данных."""
        try:
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
        """Контекстный менеджер: вход."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер: выход."""
        self.close()
