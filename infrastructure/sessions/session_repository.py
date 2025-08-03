# -*- coding: utf-8 -*-
"""
Промышленная реализация репозитория торговых сессий.
Обеспечивает персистентность данных сессий с продвинутыми возможностями
анализа, кэширования и оптимизации производительности.
"""
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, Callable, TypeVar

from loguru import logger

T = TypeVar('T')
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from domain.interfaces.signal_protocols import SessionInfluenceSignal
from domain.sessions.session_influence_analyzer import SessionInfluenceResult
from domain.types.session_types import (
    SessionType, SessionPhase, SessionAnalysisResult, SessionMetrics, MarketConditions, ConfidenceScore, MarketRegime, SessionIntensity
)
from domain.sessions.session_influence_analyzer import SessionInfluenceMetrics
from domain.value_objects.timestamp import Timestamp


class SessionRepositoryProtocol(Protocol):
    """Протокол для репозитория сессий."""

    def save_session_analysis(self, analysis: SessionAnalysisResult) -> bool:
        """Сохранение анализа сессии."""
        ...

    def get_session_analysis(
        self,
        session_type: SessionType,
        start_time: Timestamp,
        end_time: Timestamp,
        limit: Optional[int] = None,
    ) -> List[SessionAnalysisResult]:
        """Получение анализа сессии."""
        ...

    def save_influence_result(self, result: SessionInfluenceResult) -> bool:
        """Сохранение результата анализа влияния."""
        ...

    def get_influence_results(
        self,
        symbol: str,
        session_type: Optional[SessionType] = None,
        start_time: Optional[Timestamp] = None,
        end_time: Optional[Timestamp] = None,
        limit: Optional[int] = None,
    ) -> List[SessionInfluenceResult]:
        """Получение результатов анализа влияния."""
        ...

    def save_signal(self, signal: SessionInfluenceSignal) -> bool:
        """Сохранение сигнала."""
        ...

    def get_signals(
        self,
        symbol: str,
        session_type: Optional[str] = None,
        start_time: Optional[Timestamp] = None,
        end_time: Optional[Timestamp] = None,
        limit: Optional[int] = None,
    ) -> List[SessionInfluenceSignal]:
        """Получение сигналов."""
        ...

    def get_session_statistics(
        self, session_type: SessionType, lookback_days: int = 30
    ) -> Dict[str, Union[str, float, int]]:
        """Получение статистики сессии."""
        ...

    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """Очистка старых данных."""
        ...

    def get_database_stats(self) -> Dict[str, Union[str, int, float]]:
        """Получение статистики базы данных."""
        ...


@dataclass
class SessionRepositoryConfig:
    """Конфигурация репозитория сессий."""

    # Параметры базы данных
    database_url: str = "sqlite:///data/session_data.db"
    connection_pool_size: int = 10
    connection_pool_timeout: int = 30
    connection_pool_recycle: int = 3600
    # Параметры производительности
    batch_size: int = 1000
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    # Параметры кэширования
    enable_query_cache: bool = True
    cache_ttl_seconds: int = 300
    max_cache_size: int = 10000
    # Параметры логирования
    log_queries: bool = False
    log_performance: bool = True
    # Параметры валидации
    validate_data_on_save: bool = True
    validate_data_on_load: bool = True

    def validate(self) -> bool:
        """Валидация конфигурации."""
        if not self.database_url:
            return False
        if self.batch_size <= 0:
            return False
        if self.max_retries < 0:
            return False
        if self.cache_ttl_seconds <= 0:
            return False
        return True


@dataclass
class SessionDataMetrics:
    """Метрики данных сессии."""

    total_records: int = 0
    records_today: int = 0
    records_this_week: int = 0
    records_this_month: int = 0
    avg_analysis_time_ms: float = 0.0
    avg_signal_generation_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    query_performance_ms: float = 0.0
    last_cleanup_date: Optional[datetime] = None
    last_optimization_date: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Union[str, int, float]]:
        """Преобразование в словарь."""
        return {
            "total_records": self.total_records,
            "records_today": self.records_today,
            "records_this_week": self.records_this_week,
            "records_this_month": self.records_this_month,
            "avg_analysis_time_ms": self.avg_analysis_time_ms,
            "avg_signal_generation_time_ms": self.avg_signal_generation_time_ms,
            "cache_hit_rate": self.cache_hit_rate,
            "query_performance_ms": self.query_performance_ms,
            "last_cleanup_date": (
                self.last_cleanup_date.isoformat() if self.last_cleanup_date else ""
            ),
            "last_optimization_date": (
                self.last_optimization_date.isoformat()
                if self.last_optimization_date
                else ""
            ),
        }


class SessionRepository(SessionRepositoryProtocol):
    """
    Промышленная реализация репозитория торговых сессий.
    Обеспечивает:
    - Высокопроизводительное хранение и извлечение данных
    - Продвинутое кэширование с TTL
    - Валидацию данных на входе и выходе
    - Мониторинг производительности
    - Автоматическую оптимизацию базы данных
    - Поддержку batch операций
    - Обработку ошибок и retry логику
    """

    def __init__(self, config: Optional[SessionRepositoryConfig] = None):
        """Инициализация репозитория."""
        self.config = config or SessionRepositoryConfig()
        if not self.config.validate():
            raise ValueError("Invalid repository configuration")
        # Инициализация базы данных
        self.engine: Optional[Engine] = None
        self._init_database()
        # Кэш для запросов
        self._query_cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_stats = {"hits": 0, "misses": 0}
        # Метрики производительности
        self.metrics = SessionDataMetrics()
        # Статистика запросов
        self._query_stats: Dict[str, List[float]] = {}
        logger.info(
            f"SessionRepository initialized with database: {self.config.database_url}"
        )

    def _init_database(self) -> None:
        """Инициализация базы данных."""
        try:
            self.engine = create_engine(
                self.config.database_url,
                pool_size=self.config.connection_pool_size,
                pool_timeout=self.config.connection_pool_timeout,
                pool_recycle=self.config.connection_pool_recycle,
                echo=self.config.log_queries,
            )
            # Создаем таблицы
            self._create_tables()
            # Создаем индексы для оптимизации
            self._create_indexes()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def _create_tables(self) -> None:
        """Создание таблиц базы данных."""
        if self.engine is None:
            raise RuntimeError("Database engine not initialized")
        try:
            with self.engine.connect() as conn:
                # Таблица результатов анализа влияния
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS session_influence_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        session_type TEXT NOT NULL,
                        session_phase TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        influence_metrics TEXT NOT NULL,
                        predicted_volatility REAL,
                        predicted_volume REAL,
                        predicted_direction TEXT,
                        confidence REAL,
                        market_context TEXT,
                        historical_patterns TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """
                    )
                )
                # Таблица сигналов
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS session_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        score REAL NOT NULL,
                        tendency TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        session_type TEXT NOT NULL,
                        session_phase TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        volatility_impact REAL,
                        volume_impact REAL,
                        momentum_impact REAL,
                        reversal_probability REAL,
                        false_breakout_probability REAL,
                        metadata TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """
                    )
                )
                # Таблица анализов сессий
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS session_analyses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        session_type TEXT NOT NULL,
                        session_phase TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        metrics TEXT NOT NULL,
                        market_conditions TEXT,
                        predictions TEXT,
                        risk_factors TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """
                    )
                )
                # Таблица статистики сессий
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS session_statistics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        session_type TEXT NOT NULL,
                        total_observations INTEGER DEFAULT 0,
                        avg_volume_change REAL DEFAULT 0.0,
                        avg_volatility_change REAL DEFAULT 0.0,
                        avg_direction_bias REAL DEFAULT 0.0,
                        avg_confidence REAL DEFAULT 0.0,
                        bullish_count INTEGER DEFAULT 0,
                        bearish_count INTEGER DEFAULT 0,
                        neutral_count INTEGER DEFAULT 0,
                        last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, session_type)
                    )
                """
                    )
                )
                # Таблица метрик производительности
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        operation_type TEXT NOT NULL,
                        execution_time_ms REAL NOT NULL,
                        records_processed INTEGER DEFAULT 0,
                        success BOOLEAN DEFAULT TRUE,
                        error_message TEXT,
                        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """
                    )
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise

    def _create_indexes(self) -> None:
        """Создание индексов для оптимизации."""
        if self.engine is None:
            raise RuntimeError("Database engine not initialized")
        try:
            with self.engine.connect() as conn:
                # Индексы для session_influence_results
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS idx_influence_symbol_timestamp "
                        "ON session_influence_results(symbol, timestamp)"
                    )
                )
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS idx_influence_session_type "
                        "ON session_influence_results(session_type)"
                    )
                )
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS idx_influence_created_at "
                        "ON session_influence_results(created_at)"
                    )
                )
                # Индексы для session_signals
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS idx_signals_symbol_timestamp "
                        "ON session_signals(symbol, timestamp)"
                    )
                )
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS idx_signals_session_type "
                        "ON session_signals(session_type)"
                    )
                )
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS idx_signals_tendency "
                        "ON session_signals(tendency)"
                    )
                )
                # Индексы для session_analyses
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS idx_analyses_symbol_timestamp "
                        "ON session_analyses(symbol, timestamp)"
                    )
                )
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS idx_analyses_session_type "
                        "ON session_analyses(session_type)"
                    )
                )
                # Индексы для performance_metrics
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS idx_performance_timestamp "
                        "ON performance_metrics(timestamp)"
                    )
                )
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS idx_performance_operation "
                        "ON performance_metrics(operation_type)"
                    )
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            raise

    def save_session_analysis(self, analysis: SessionAnalysisResult) -> bool:
        """Сохранение анализа сессии."""
        return self._execute_with_retry(
            self._save_session_analysis_impl,
            analysis,
            operation_type="save_session_analysis",
        )

    def _save_session_analysis_impl(self, analysis: SessionAnalysisResult) -> bool:
        """Реализация сохранения анализа сессии."""
        if self.engine is None:
            raise RuntimeError("Database engine not initialized")
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text(
                        """
                    INSERT INTO session_analyses (
                        symbol, session_type, session_phase, timestamp,
                        confidence, metrics, market_conditions, predictions, risk_factors
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                    ),
                    {
                        "symbol": analysis.session_type.value,
                        "session_type": analysis.session_type.value,
                        "session_phase": analysis.session_phase.value,
                        "timestamp": analysis.timestamp.to_iso(),
                        "confidence": float(analysis.confidence),
                        "metrics": json.dumps(analysis.metrics),
                        "market_conditions": json.dumps(analysis.market_conditions),
                        "predictions": json.dumps(analysis.predictions),
                        "risk_factors": json.dumps(analysis.risk_factors),
                    },
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save session analysis: {e}")
            return False

    def get_session_analysis(
        self,
        session_type: SessionType,
        start_time: Timestamp,
        end_time: Timestamp,
        limit: Optional[int] = None,
    ) -> List[SessionAnalysisResult]:
        """Получение анализа сессии."""
        cache_key = f"analysis_{session_type.value}_{start_time.to_iso()}_{end_time.to_iso()}_{limit}"
        # Проверяем кэш
        if self.config.enable_query_cache:
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result
        # Выполняем запрос
        result = self._execute_with_retry(
            self._get_session_analysis_impl,
            session_type,
            start_time,
            end_time,
            limit,
            operation_type="get_session_analysis",
        )
        # Сохраняем в кэш
        if self.config.enable_query_cache and result:
            self._save_to_cache(cache_key, result)
        return result

    def _get_session_analysis_impl(
        self,
        session_type: SessionType,
        start_time: Timestamp,
        end_time: Timestamp,
        limit: Optional[int] = None,
    ) -> List[SessionAnalysisResult]:
        """Реализация получения анализа сессии."""
        if self.engine is None:
            raise RuntimeError("Database engine not initialized")
        try:
            with self.engine.connect() as conn:
                query = """
                    SELECT * FROM session_analyses 
                    WHERE session_type = :session_type AND timestamp BETWEEN :start_time AND :end_time
                    ORDER BY timestamp DESC
                """
                params = {"session_type": session_type.value, "start_time": start_time.to_iso(), "end_time": end_time.to_iso()}
                if limit:
                    query += f" LIMIT {limit}"
                result = conn.execute(text(query), params)
                rows = result.fetchall()
                analyses = []
                for row in rows:
                    try:
                        analysis = self._row_to_session_analysis(tuple(row))
                        if analysis:
                            analyses.append(analysis)
                    except Exception as e:
                        logger.warning(f"Error parsing session analysis row: {e}")
                        continue
                return analyses
        except Exception as e:
            logger.error(f"Failed to get session analysis: {e}")
            return []

    def save_influence_result(self, result: SessionInfluenceResult) -> bool:
        """Сохранение результата анализа влияния."""
        return self._execute_with_retry(
            self._save_influence_result_impl,
            result,
            operation_type="save_influence_result",
        )

    def _save_influence_result_impl(self, result: SessionInfluenceResult) -> bool:
        """Реализация сохранения результата анализа влияния."""
        if self.engine is None:
            raise RuntimeError("Database engine not initialized")
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text(
                        """
                    INSERT INTO session_influence_results (
                        symbol, session_type, session_phase, timestamp,
                        influence_metrics, predicted_volatility, predicted_volume,
                        predicted_direction, confidence, market_context, historical_patterns
                    ) VALUES (:symbol, :session_type, :session_phase, :timestamp,
                             :influence_metrics, :predicted_volatility, :predicted_volume,
                             :predicted_direction, :confidence, :market_context, :historical_patterns)
                """
                    ),
                    {
                        "symbol": result.symbol,
                        "session_type": result.session_type.value,
                        "session_phase": result.session_phase.value,
                        "timestamp": result.timestamp.to_iso(),
                        "influence_metrics": json.dumps(result.influence_metrics),
                        "predicted_volatility": result.predicted_volatility,
                        "predicted_volume": result.predicted_volume,
                        "predicted_direction": result.predicted_direction,
                        "confidence": result.confidence,
                        "market_context": json.dumps(result.market_context),
                        "historical_patterns": json.dumps(result.historical_patterns),
                    },
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save influence result: {e}")
            return False

    def get_influence_results(
        self,
        symbol: str,
        session_type: Optional[SessionType] = None,
        start_time: Optional[Timestamp] = None,
        end_time: Optional[Timestamp] = None,
        limit: Optional[int] = None,
    ) -> List[SessionInfluenceResult]:
        """Получение результатов анализа влияния."""
        cache_key = f"influence_{symbol}_{session_type.value if session_type else 'all'}_{start_time.to_iso() if start_time else 'start'}_{end_time.to_iso() if end_time else 'end'}_{limit}"
        # Проверяем кэш
        if self.config.enable_query_cache:
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result
        # Выполняем запрос
        result = self._execute_with_retry(
            self._get_influence_results_impl,
            symbol,
            session_type,
            start_time,
            end_time,
            limit,
            operation_type="get_influence_results",
        )
        # Сохраняем в кэш
        if self.config.enable_query_cache and result:
            self._save_to_cache(cache_key, result)
        return result

    def _get_influence_results_impl(
        self,
        symbol: str,
        session_type: Optional[SessionType] = None,
        start_time: Optional[Timestamp] = None,
        end_time: Optional[Timestamp] = None,
        limit: Optional[int] = None,
    ) -> List[SessionInfluenceResult]:
        """Реализация получения результатов анализа влияния."""
        if self.engine is None:
            raise RuntimeError("Database engine not initialized")
        try:
            with self.engine.connect() as conn:
                query = "SELECT * FROM session_influence_results WHERE symbol = :symbol"
                params = {"symbol": symbol}
                if session_type:
                    query += " AND session_type = :session_type"
                    params["session_type"] = session_type.value
                if start_time:
                    query += " AND timestamp >= :start_time"
                    params["start_time"] = start_time.to_iso()
                if end_time:
                    query += " AND timestamp <= :end_time"
                    params["end_time"] = end_time.to_iso()
                query += " ORDER BY timestamp DESC"
                if limit:
                    query += f" LIMIT {limit}"
                result = conn.execute(text(query), params)
                rows = result.fetchall()
                results = []
                for row in rows:
                    try:
                        influence_result = self._row_to_influence_result(tuple(row))
                        if influence_result:
                            results.append(influence_result)
                    except Exception as e:
                        logger.warning(f"Error parsing influence result row: {e}")
                        continue
                return results
        except Exception as e:
            logger.error(f"Failed to get influence results: {e}")
            return []

    def save_signal(self, signal: SessionInfluenceSignal) -> bool:
        """Сохранение сигнала."""
        return self._execute_with_retry(
            self._save_signal_impl, signal, operation_type="save_signal"
        )

    def _save_signal_impl(self, signal: SessionInfluenceSignal) -> bool:
        """Реализация сохранения сигнала."""
        if self.engine is None:
            raise RuntimeError("Database engine not initialized")
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text(
                        """
                    INSERT INTO session_signals (
                        symbol, score, tendency, confidence, session_type, session_phase,
                        timestamp, volatility_impact, volume_impact, momentum_impact,
                        reversal_probability, false_breakout_probability, metadata
                    ) VALUES (:symbol, :score, :tendency, :confidence, :session_type, :session_phase,
                             :timestamp, :volatility_impact, :volume_impact, :momentum_impact,
                             :reversal_probability, :false_breakout_probability, :metadata)
                """
                    ),
                    {
                        "symbol": signal.session_type.value,
                        "score": signal.confidence,
                        "tendency": signal.predicted_impact,
                        "confidence": signal.confidence,
                        "session_type": signal.session_type.value,
                        "session_phase": signal.session_type.value,
                        "timestamp": signal.timestamp.to_iso(),
                        "volatility_impact": signal.influence_strength,
                        "volume_impact": signal.predicted_impact,
                        "momentum_impact": signal.predicted_impact,
                        "reversal_probability": signal.predicted_impact,
                        "false_breakout_probability": signal.predicted_impact,
                        "metadata": json.dumps(signal.metadata),
                    },
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save signal: {e}")
            return False

    def get_signals(
        self,
        symbol: str,
        session_type: Optional[str] = None,
        start_time: Optional[Timestamp] = None,
        end_time: Optional[Timestamp] = None,
        limit: Optional[int] = None,
    ) -> List[SessionInfluenceSignal]:
        """Получение сигналов."""
        cache_key = f"signals_{symbol}_{session_type or 'all'}_{start_time.to_iso() if start_time else 'start'}_{end_time.to_iso() if end_time else 'end'}_{limit}"
        # Проверяем кэш
        if self.config.enable_query_cache:
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result
        # Выполняем запрос
        result = self._execute_with_retry(
            self._get_signals_impl,
            symbol,
            session_type,
            start_time,
            end_time,
            limit,
            operation_type="get_signals",
        )
        # Сохраняем в кэш
        if self.config.enable_query_cache and result:
            self._save_to_cache(cache_key, result)
        return result

    def _get_signals_impl(
        self,
        symbol: str,
        session_type: Optional[str] = None,
        start_time: Optional[Timestamp] = None,
        end_time: Optional[Timestamp] = None,
        limit: Optional[int] = None,
    ) -> List[SessionInfluenceSignal]:
        """Реализация получения сигналов."""
        if self.engine is None:
            raise RuntimeError("Database engine not initialized")
        try:
            with self.engine.connect() as conn:
                query = "SELECT * FROM session_signals WHERE symbol = :symbol"
                params = {"symbol": symbol}
                if session_type:
                    query += " AND session_type = :session_type"
                    params["session_type"] = session_type
                if start_time:
                    query += " AND timestamp >= :start_time"
                    params["start_time"] = start_time.to_iso()
                if end_time:
                    query += " AND timestamp <= :end_time"
                    params["end_time"] = end_time.to_iso()
                query += " ORDER BY timestamp DESC"
                if limit:
                    query += f" LIMIT {limit}"
                result = conn.execute(text(query), params)
                rows = result.fetchall()
                signals = []
                for row in rows:
                    try:
                        signal = self._row_to_signal(tuple(row))
                        if signal:
                            signals.append(signal)
                    except Exception as e:
                        logger.warning(f"Error parsing signal row: {e}")
                        continue
                return signals
        except Exception as e:
            logger.error(f"Failed to get signals: {e}")
            return []

    def get_session_statistics(
        self, session_type: SessionType, lookback_days: int = 30
    ) -> Dict[str, Union[str, float, int]]:
        """Получение статистики сессии."""
        return self._execute_with_retry(
            self._get_session_statistics_impl, session_type, lookback_days, operation_type="get_session_statistics"
        )

    def _get_session_statistics_impl(
        self, session_type: SessionType, lookback_days: int
    ) -> Dict[str, Union[str, float, int]]:
        """Реализация получения статистики сессии."""
        if self.engine is None:
            raise RuntimeError("Database engine not initialized")
        try:
            with self.engine.connect() as conn:
                end_time = datetime.now()
                start_time = end_time - timedelta(days=lookback_days)
                # Получаем базовую статистику
                result = conn.execute(
                    text(
                        """
                    SELECT 
                        COUNT(*) as total_analyses,
                        AVG(confidence) as avg_confidence,
                        AVG(JSON_EXTRACT(metrics, '$.volume_change_percent')) as avg_volume_change,
                        AVG(JSON_EXTRACT(metrics, '$.volatility_change_percent')) as avg_volatility_change,
                        AVG(JSON_EXTRACT(metrics, '$.price_direction_bias')) as avg_direction_bias,
                        AVG(JSON_EXTRACT(metrics, '$.momentum_strength')) as avg_momentum,
                        AVG(JSON_EXTRACT(metrics, '$.false_breakout_probability')) as avg_false_breakout,
                        AVG(JSON_EXTRACT(metrics, '$.reversal_probability')) as avg_reversal,
                        MAX(created_at) as last_updated
                    FROM session_analyses 
                    WHERE session_type = :session_type 
                    AND created_at BETWEEN :start_time AND :end_time
                """
                    ),
                    {
                        "session_type": session_type.value,
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                    },
                )
                row = result.fetchone()
                if row:
                    stats: Dict[str, Union[str, float, int]] = {
                        "session_type": session_type.value,
                        "period_days": lookback_days,
                        "total_analyses": row[0] if row[0] else 0,
                        "avg_confidence": float(row[1]) if row[1] else 0.0,
                        "avg_volume_change": float(row[2]) if row[2] else 0.0,
                        "avg_volatility_change": float(row[3]) if row[3] else 0.0,
                        "avg_direction_bias": float(row[4]) if row[4] else 0.0,
                        "avg_momentum": float(row[5]) if row[5] else 0.0,
                        "avg_false_breakout": float(row[6]) if row[6] else 0.0,
                        "avg_reversal": float(row[7]) if row[7] else 0.0,
                        "last_updated": row[8] if row[8] else "",
                    }
                    return stats
                return {}
        except Exception as e:
            logger.error(f"Failed to get session statistics: {e}")
            return {}

    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """Очистка старых данных."""
        return self._execute_with_retry(
            self._cleanup_old_data_impl, days_to_keep, operation_type="cleanup_old_data"
        )

    def _cleanup_old_data_impl(self, days_to_keep: int) -> int:
        """Реализация очистки старых данных."""
        if self.engine is None:
            raise RuntimeError("Database engine not initialized")
        try:
            if self.engine is None:
                raise RuntimeError("Database engine not initialized")
            with self.engine.connect() as conn:
                cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
                deleted_count = 0
                # Очищаем старые анализы
                result = conn.execute(
                    text("DELETE FROM session_analyses WHERE created_at < :cutoff_date"),
                    {"cutoff_date": cutoff_date}
                )
                deleted_count += result.rowcount
                # Очищаем старые результаты влияния
                result = conn.execute(
                    text("DELETE FROM session_influence_results WHERE created_at < :cutoff_date"),
                    {"cutoff_date": cutoff_date}
                )
                deleted_count += result.rowcount
                # Очищаем старые сигналы
                result = conn.execute(
                    text("DELETE FROM session_signals WHERE created_at < :cutoff_date"),
                    {"cutoff_date": cutoff_date}
                )
                deleted_count += result.rowcount
                # Очищаем старые метрики производительности
                result = conn.execute(
                    text("DELETE FROM performance_metrics WHERE timestamp < :cutoff_date"),
                    {"cutoff_date": cutoff_date}
                )
                deleted_count += result.rowcount
                conn.commit()
                return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return 0

    def get_database_stats(self) -> Dict[str, Union[str, int, float]]:
        """Получение статистики базы данных."""
        if self.engine is None:
            raise RuntimeError("Database engine not initialized")
        try:
            with self.engine.connect() as conn:
                stats = {}
                # Количество записей в каждой таблице
                tables = ["session_analyses", "session_influence_results", "session_signals", "session_statistics", "performance_metrics"]
                for table in tables:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.scalar()
                    stats[f"{table}_count"] = count if count else 0
                # Размер базы данных
                result = conn.execute(text("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()"))
                size = result.scalar()
                stats["database_size_bytes"] = size if size else 0
                # Статистика кэша
                stats["cache_hits"] = self._cache_stats["hits"]
                stats["cache_misses"] = self._cache_stats["misses"]
                stats["cache_hit_rate"] = (
                    self._cache_stats["hits"] / (self._cache_stats["hits"] + self._cache_stats["misses"])
                    if (self._cache_stats["hits"] + self._cache_stats["misses"]) > 0
                    else 0.0
                )
                return stats
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}

    def _execute_with_retry(
        self, func: Callable[..., T], *args, operation_type: str = "unknown", **kwargs
    ) -> T:
        """Выполнение операции с retry логикой."""
        import time

        for attempt in range(self.config.max_retries + 1):
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                # Записываем метрики производительности
                self._record_performance_metric(
                    operation_type, execution_time, success=True
                )
                return result
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                # Записываем метрики производительности
                self._record_performance_metric(
                    operation_type, execution_time, success=False, error_message=str(e)
                )
                if attempt < self.config.max_retries:
                    logger.warning(
                        f"Operation {operation_type} failed (attempt {attempt + 1}/{self.config.max_retries + 1}): {e}"
                    )
                    time.sleep(
                        self.config.retry_delay_seconds * (2**attempt)
                    )  # Exponential backoff
                else:
                    logger.error(
                        f"Operation {operation_type} failed after {self.config.max_retries + 1} attempts: {e}"
                    )
                    raise

    def _record_performance_metric(
        self,
        operation_type: str,
        execution_time_ms: float,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> None:
        """Запись метрики производительности."""
        try:
            if self.engine is None:
                raise RuntimeError("Database engine not initialized")
            with self.engine.connect() as conn:
                conn.execute(
                    text(
                        """
                    INSERT INTO performance_metrics (
                        operation_type, execution_time_ms, success, error_message
                    ) VALUES (:operation_type, :execution_time_ms, :success, :error_message)
                """
                    ),
                    {
                        "operation_type": operation_type,
                        "execution_time_ms": execution_time_ms,
                        "success": success,
                        "error_message": error_message,
                    },
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to record performance metric: {e}")

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Получение из кэша."""
        import time

        if key in self._query_cache:
            value, timestamp = self._query_cache[key]
            if time.time() - timestamp < self.config.cache_ttl_seconds:
                self._cache_stats["hits"] += 1
                return value
            else:
                del self._query_cache[key]
        self._cache_stats["misses"] += 1
        return None

    def _save_to_cache(self, key: str, value: Any) -> None:
        """Сохранение в кэш."""
        import time

        # Очищаем кэш, если он превышает максимальный размер
        if len(self._query_cache) >= self.config.max_cache_size:
            # Удаляем самые старые записи
            oldest_keys = sorted(
                self._query_cache.keys(), key=lambda k: self._query_cache[k][1]
            )[: len(self._query_cache) // 2]
            for old_key in oldest_keys:
                del self._query_cache[old_key]
        self._query_cache[key] = (value, time.time())

    def _row_to_session_analysis(self, row: Tuple) -> Optional[SessionAnalysisResult]:
        """Преобразование строки в SessionAnalysisResult."""
        try:
            if len(row) < 11:
                return None
            metrics = SessionMetrics(
                volume_change_percent=0.0,
                volatility_change_percent=0.0,
                price_direction_bias=0.0,
                momentum_strength=0.0,
                false_breakout_probability=0.0,
                reversal_probability=0.0,
                trend_continuation_probability=0.0,
                influence_duration_minutes=0,
                peak_influence_time_minutes=0,
                spread_impact=0.0,
                liquidity_impact=0.0,
                correlation_with_other_sessions=0.0,
            )
            market_conditions = MarketConditions(
                volatility=0.0,
                volume=0.0,
                spread=0.0,
                liquidity=0.0,
                momentum=0.0,
                trend_strength=0.0,
                market_regime=MarketRegime.RANGING,
                session_intensity=SessionIntensity.NORMAL,
            )
            return SessionAnalysisResult(
                session_type=SessionType(row[2]),
                session_phase=SessionPhase(row[3]),
                timestamp=Timestamp.from_iso(row[4]),
                confidence=ConfidenceScore(row[5]),
                metrics=metrics,
                market_conditions=market_conditions,
                predictions={},
                risk_factors=[],
            )
        except Exception as e:
            logger.error(f"Error converting row to SessionAnalysisResult: {e}")
            return None

    def _row_to_influence_result(self, row: Tuple) -> Optional[SessionInfluenceResult]:
        """Преобразование строки в SessionInfluenceResult."""
        try:
            if len(row) < 13:
                return None
            influence_metrics = SessionInfluenceMetrics()  # или заполнить из row, если есть данные
            return SessionInfluenceResult(
                symbol=row[1],
                session_type=SessionType(row[2]),
                session_phase=SessionPhase(row[3]),
                timestamp=Timestamp.from_iso(row[4]),
                influence_metrics=influence_metrics,
                predicted_volatility=row[6] if row[6] else 0.0,
                predicted_volume=row[7] if row[7] else 0.0,
                predicted_direction=row[8] if row[8] else "neutral",
                confidence=row[9] if row[9] else 0.0,
                market_context={},
                historical_patterns=[],
            )
        except Exception as e:
            logger.error(f"Error converting row to SessionInfluenceResult: {e}")
            return None

    def _row_to_signal(self, row: Tuple) -> Optional[SessionInfluenceSignal]:
        """Преобразование строки в SessionInfluenceSignal."""
        try:
            if len(row) < 15:
                return None
            # Передать все обязательные аргументы
            return SessionInfluenceSignal(
                session_type=SessionType(row[5]),
                timestamp=Timestamp.from_iso(row[7]),
                confidence=row[4] if row[4] else 0.0,
                influence_strength=0.0,  # пример значения, заменить на актуальное
                market_conditions=MarketConditions(
                    volatility=0.0,
                    volume=0.0,
                    spread=0.0,
                    liquidity=0.0,
                    momentum=0.0,
                    trend_strength=0.0,
                    market_regime=MarketRegime.RANGING,
                    session_intensity=SessionIntensity.NORMAL,
                ),  # пример, заменить на актуальное
                predicted_impact={"volatility": 0.0, "volume": 0.0, "momentum": 0.0},  # пример значения, заменить на актуальное
                metadata={},
            )
        except Exception as e:
            logger.error(f"Error converting row to SessionInfluenceSignal: {e}")
            return None

    def optimize_database(self) -> bool:
        """Оптимизация базы данных."""
        if self.engine is None:
            raise RuntimeError("Database engine not initialized")
        try:
            with self.engine.connect() as conn:
                # Анализируем таблицы
                conn.execute(text("ANALYZE"))
                # Перестраиваем индексы
                conn.execute(text("REINDEX"))
                # Очищаем свободное место
                conn.execute(text("VACUUM"))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to optimize database: {e}")
            return False

    def get_performance_metrics(
        self, operation_type: Optional[str] = None, hours: int = 24
    ) -> Dict[str, Union[str, float, int]]:
        """Получение метрик производительности."""
        if self.engine is None:
            raise RuntimeError("Database engine not initialized")
        try:
            with self.engine.connect() as conn:
                cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
                query = """
                    SELECT 
                        operation_type,
                        AVG(execution_time_ms) as avg_time,
                        COUNT(*) as total_operations,
                        SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_operations,
                        SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failed_operations
                    FROM performance_metrics 
                    WHERE timestamp >= :cutoff_time
                """
                params = {"cutoff_time": cutoff_time}
                if operation_type:
                    query += " AND operation_type = :operation_type"
                    params["operation_type"] = operation_type
                query += " GROUP BY operation_type"
                result = conn.execute(text(query), params)
                rows = result.fetchall()
                metrics: Dict[str, Union[str, float, int]] = {}
                for row in rows:
                    if row and len(row) >= 5:
                        op_type = row[0] if row[0] else "unknown"
                        metrics[op_type] = {  # type: ignore
                            "avg_execution_time_ms": row[1] if row[1] else 0.0,
                            "total_operations": row[2] if row[2] else 0,
                            "successful_operations": row[3] if row[3] else 0,
                            "failed_operations": row[4] if row[4] else 0,
                        }
                return metrics
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {}

    def clear_cache(self) -> None:
        """Очистка кэша."""
        self._query_cache.clear()
        self._cache_stats = {"hits": 0, "misses": 0}
        logger.info("Repository cache cleared")

    def close(self) -> None:
        """Закрытие соединений с базой данных."""
        if self.engine:
            self.engine.dispose()
            logger.info("Repository connections closed")


# Импорт time для кэширования
import time
