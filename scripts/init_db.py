#!/usr/bin/env python3
"""
Скрипт инициализации базы данных
"""

import sys
from pathlib import Path

import psycopg2
import yaml
from loguru import logger
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Добавление корневой директории в путь
sys.path.append(str(Path(__file__).parent.parent))


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Загрузка конфигурации"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        sys.exit(1)


def create_database(config: dict):
    """Создание базы данных"""
    db_config = config.get("database", {})

    # Параметры подключения к PostgreSQL
    host = db_config.get("host", "localhost")
    port = db_config.get("port", 5432)
    user = db_config.get("user", "postgres")
    password = db_config.get("password", "")
    database = db_config.get("database", "trading_bot")

    try:
        # Подключение к PostgreSQL для создания БД
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database="postgres",  # Подключаемся к системной БД
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Проверка существования БД
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,))
        exists = cursor.fetchone()

        if not exists:
            # Создание БД
            cursor.execute(f'CREATE DATABASE "{database}"')
            logger.info(f"Database '{database}' created successfully")
        else:
            logger.info(f"Database '{database}' already exists")

        cursor.close()
        conn.close()

    except Exception as e:
        logger.error(f"Error creating database: {e}")
        sys.exit(1)


def create_tables(config: dict):
    """Создание таблиц"""
    db_config = config.get("database", {})

    # Параметры подключения
    host = db_config.get("host", "localhost")
    port = db_config.get("port", 5432)
    user = db_config.get("user", "postgres")
    password = db_config.get("password", "")
    database = db_config.get("database", "trading_bot")

    try:
        # Подключение к созданной БД
        conn = psycopg2.connect(
            host=host, port=port, user=user, password=password, database=database
        )
        cursor = conn.cursor()

        # SQL скрипты для создания таблиц
        tables_sql = [
            # Таблица торговых операций
            """
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                trade_id VARCHAR(100) UNIQUE NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                side VARCHAR(10) NOT NULL,
                size DECIMAL(20, 8) NOT NULL,
                price DECIMAL(20, 8) NOT NULL,
                strategy VARCHAR(50) NOT NULL,
                pnl DECIMAL(20, 8),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR(20) DEFAULT 'open'
            )
            """,
            # Таблица позиций
            """
            CREATE TABLE IF NOT EXISTS positions (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                size DECIMAL(20, 8) NOT NULL,
                avg_price DECIMAL(20, 8) NOT NULL,
                unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            # Таблица ордеров
            """
            CREATE TABLE IF NOT EXISTS orders (
                id SERIAL PRIMARY KEY,
                order_id VARCHAR(100) UNIQUE NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                side VARCHAR(10) NOT NULL,
                order_type VARCHAR(20) NOT NULL,
                size DECIMAL(20, 8) NOT NULL,
                price DECIMAL(20, 8),
                status VARCHAR(20) DEFAULT 'pending',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            # Таблица метрик
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id SERIAL PRIMARY KEY,
                metric_name VARCHAR(100) NOT NULL,
                metric_value DECIMAL(20, 8) NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            # Таблица событий
            """
            CREATE TABLE IF NOT EXISTS events (
                id SERIAL PRIMARY KEY,
                event_type VARCHAR(100) NOT NULL,
                event_data JSONB,
                priority VARCHAR(20) DEFAULT 'normal',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            # Таблица стратегий
            """
            CREATE TABLE IF NOT EXISTS strategies (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) UNIQUE NOT NULL,
                status VARCHAR(20) DEFAULT 'disabled',
                parameters JSONB,
                performance JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            # Таблица режимов рынка
            """
            CREATE TABLE IF NOT EXISTS market_regimes (
                id SERIAL PRIMARY KEY,
                regime_name VARCHAR(100) NOT NULL,
                regime_type VARCHAR(50) NOT NULL,
                characteristics JSONB,
                confidence DECIMAL(5, 4),
                is_active BOOLEAN DEFAULT false,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            # Таблица решений ИИ
            """
            CREATE TABLE IF NOT EXISTS ai_decisions (
                id SERIAL PRIMARY KEY,
                action VARCHAR(100) NOT NULL,
                confidence DECIMAL(5, 4) NOT NULL,
                reasoning TEXT,
                parameters JSONB,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            # Таблица рисков
            """
            CREATE TABLE IF NOT EXISTS risk_metrics (
                id SERIAL PRIMARY KEY,
                daily_loss DECIMAL(10, 4) DEFAULT 0,
                weekly_loss DECIMAL(10, 4) DEFAULT 0,
                portfolio_risk DECIMAL(10, 4) DEFAULT 0,
                max_drawdown DECIMAL(10, 4) DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            # Таблица здоровья системы
            """
            CREATE TABLE IF NOT EXISTS system_health (
                id SERIAL PRIMARY KEY,
                component VARCHAR(50) NOT NULL,
                status VARCHAR(20) NOT NULL,
                details JSONB,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
        ]

        # Создание таблиц
        for sql in tables_sql:
            cursor.execute(sql)
            logger.info("Table created/verified")

        # Создание индексов
        indexes_sql = [
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy)",
            "CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)",
            "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_ai_decisions_action ON ai_decisions(action)",
            "CREATE INDEX IF NOT EXISTS idx_ai_decisions_timestamp ON ai_decisions(timestamp)",
        ]

        for sql in indexes_sql:
            cursor.execute(sql)
            logger.info("Index created/verified")

        conn.commit()
        cursor.close()
        conn.close()

        logger.info("All tables and indexes created successfully")

    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        sys.exit(1)


def insert_initial_data(config: dict):
    """Вставка начальных данных"""
    db_config = config.get("database", {})

    # Параметры подключения
    host = db_config.get("host", "localhost")
    port = db_config.get("port", 5432)
    user = db_config.get("user", "postgres")
    password = db_config.get("password", "")
    database = db_config.get("database", "trading_bot")

    try:
        conn = psycopg2.connect(
            host=host, port=port, user=user, password=password, database=database
        )
        cursor = conn.cursor()

        # Вставка начальных стратегий
        strategies = [
            (
                "trend_strategy",
                "disabled",
                '{"fast_period": 10, "slow_period": 20}',
                "{}",
            ),
            (
                "mean_reversion_strategy",
                "disabled",
                '{"rsi_period": 14, "oversold": 30, "overbought": 70}',
                "{}",
            ),
            (
                "scalping_strategy",
                "disabled",
                '{"min_profit": 0.001, "max_loss": 0.002}',
                "{}",
            ),
            ("arbitrage_strategy", "disabled", '{"min_spread": 0.005}', "{}"),
            (
                "volatility_strategy",
                "disabled",
                '{"atr_period": 14, "volatility_threshold": 0.02}',
                "{}",
            ),
        ]

        for strategy in strategies:
            cursor.execute(
                """
                INSERT INTO strategies (name, status, parameters, performance)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (name) DO NOTHING
            """,
                strategy,
            )

        # Вставка начальных режимов рынка
        regimes = [
            ("trending", "TRENDING", '{"volatility": 0.02, "momentum": 0.01}', 0.8),
            ("sideways", "SIDEWAYS", '{"volatility": 0.01, "momentum": 0.0}', 0.7),
            (
                "volatile",
                "HIGH_VOLATILITY",
                '{"volatility": 0.05, "momentum": 0.02}',
                0.6,
            ),
        ]

        for regime in regimes:
            cursor.execute(
                """
                INSERT INTO market_regimes (regime_name, regime_type, characteristics, confidence)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """,
                regime,
            )

        # Вставка начальных метрик риска
        cursor.execute(
            """
            INSERT INTO risk_metrics (daily_loss, weekly_loss, portfolio_risk, max_drawdown)
            VALUES (0, 0, 0, 0)
        """
        )

        conn.commit()
        cursor.close()
        conn.close()

        logger.info("Initial data inserted successfully")

    except Exception as e:
        logger.error(f"Error inserting initial data: {e}")
        sys.exit(1)


def main():
    """Основная функция"""
    logger.info("Starting database initialization...")

    # Загрузка конфигурации
    config = load_config()

    # Создание базы данных
    create_database(config)

    # Создание таблиц
    create_tables(config)

    # Вставка начальных данных
    insert_initial_data(config)

    logger.info("Database initialization completed successfully!")


if __name__ == "__main__":
    main()
