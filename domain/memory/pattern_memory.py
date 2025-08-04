# -*- coding: utf-8 -*-
"""Система памяти паттернов для рыночной аналитики."""
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from shared.numpy_utils import np
from loguru import logger
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler
from uuid import UUID

from domain.memory.entities import PatternOutcome, PatternSnapshot, PredictionResult
from domain.memory.interfaces import (
    IPatternMatcher,
    IPatternMemoryRepository,
    IPatternMemoryService,
    IPatternPredictor,
)
from domain.memory.types import (
    MarketFeatures,
    MarketRegime,
    MemoryStatistics,
    OutcomeType,
    PatternMemoryConfig,
    PredictionDirection,
    SimilarityMetrics,
    VolumeProfile,
)
from domain.types.pattern_types import PatternType
from domain.value_objects.timestamp import Timestamp


# =============================================================================
# IMPLEMENTATIONS
# =============================================================================
class PatternMatcher(IPatternMatcher):
    """Реализация сопоставления паттернов."""

    def __init__(self, config: PatternMemoryConfig):
        self.config = config
        self.scaler = StandardScaler()
        self._fitted = False

    def calculate_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Вычисление сходства между векторами признаков."""
        try:
            # Нормализация векторов
            norm1 = np.linalg.norm(vector1)
            norm2 = np.linalg.norm(vector2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Косинусное сходство
            cosine_sim = np.dot(vector1, vector2) / (norm1 * norm2)
            
            # Евклидово расстояние (нормализованное)
            euclidean_dist = np.linalg.norm(vector1 - vector2)
            max_dist = np.linalg.norm(vector1) + np.linalg.norm(vector2)
            euclidean_sim = 1 - (euclidean_dist / max_dist) if max_dist > 0 else 0
            
            # Комбинированное сходство
            combined_sim = (cosine_sim + euclidean_sim) / 2
            # Исправляем возвращаемый тип - явно возвращаем float
            return float(max(0.0, min(1.0, combined_sim)))
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def find_similar_patterns(
        self,
        current_features: MarketFeatures,
        snapshots: List[PatternSnapshot],
        similarity_threshold: float = 0.9,
        max_results: int = 10,
    ) -> List[Tuple[PatternSnapshot, float]]:
        """Поиск похожих паттернов."""
        try:
            # Используем общий сервис для поиска паттернов
            from domain.services.pattern_service import (
                PatternSearchCriteria,
                PatternSearchService,
                TechnicalPatternAnalyzer,
            )

            # Создаем анализатор и сервис поиска
            analyzer = TechnicalPatternAnalyzer()
            search_service = PatternSearchService(analyzer)
            # Конвертируем snapshots в паттерны для поиска
            patterns = []
            for snapshot in snapshots:
                # Создаем паттерн из snapshot
                pattern_data = {
                    "prices": [snapshot.features.price],
                    "volumes": [snapshot.features.volume],
                    "timestamps": [snapshot.timestamp.to_datetime()],
                    "trading_pair_id": snapshot.symbol,
                }
                pattern = analyzer.analyze_pattern(pattern_data)
                pattern.id = UUID(snapshot.pattern_id)  # Преобразуем строку в UUID
                patterns.append(pattern)
                search_service.add_pattern(pattern)
            # Создаем целевой паттерн из текущих признаков
            target_pattern_data = {
                "prices": [current_features.price],
                "volumes": [current_features.volume],
                "timestamps": [datetime.now()],
                "trading_pair_id": "",  # MarketFeatures не имеет атрибута symbol
            }
            target_pattern = analyzer.analyze_pattern(target_pattern_data)
            # Критерии поиска
            criteria = PatternSearchCriteria(
                min_similarity=similarity_threshold,
                trading_pair_id="",  # MarketFeatures не имеет атрибута symbol
            )
            # Поиск похожих паттернов
            matches = search_service.find_similar_patterns(target_pattern, criteria)
            # Конвертируем результаты обратно в snapshots
            similar_patterns = []
            for match in matches[:max_results]:
                # Находим соответствующий snapshot
                for snapshot in snapshots:
                    if snapshot.pattern_id == match.pattern.id:
                        similar_patterns.append((snapshot, match.similarity_score))
                        break
            return similar_patterns
        except Exception as e:
            logger.error(f"Error finding similar patterns: {e}")
            return []

    def calculate_confidence_boost(
        self, similarity: float, snapshot: PatternSnapshot
    ) -> float:
        """Вычисление повышения уверенности."""
        try:
            # Базовое повышение от сходства
            base_boost = similarity * 0.4
            # Повышение от уверенности паттерна
            confidence_boost = snapshot.confidence * 0.3
            # Повышение от силы паттерна
            strength_boost = snapshot.strength * 0.2
            # Повышение от свежести данных
            age_hours = (
                datetime.now() - snapshot.timestamp.to_datetime()
            ).total_seconds() / 3600
            freshness_boost = max(0.0, 0.1 * (1 - age_hours / 24))
            total_boost = (
                base_boost + confidence_boost + strength_boost + freshness_boost
            )
            return min(1.0, total_boost)
        except Exception as e:
            logger.error(f"Error calculating confidence boost: {e}")
            return 0.0

    def calculate_signal_strength(self, snapshot: PatternSnapshot) -> float:
        """Вычисление силы сигнала."""
        try:
            # Базовая сила от уверенности и силы паттерна
            base_strength = snapshot.confidence * snapshot.strength
            # Модификатор от направления
            direction_modifier = {
                "up": 1.0,
                "down": -1.0,
                "neutral": 0.5,
                "bullish": 0.8,
                "bearish": -0.8,
            }.get(snapshot.direction.lower(), 0.5)
            # Модификатор от типа паттерна
            pattern_modifier = {
                PatternType.WHALE_ABSORPTION: 1.2,
                PatternType.MM_SPOOFING: 0.8,
                PatternType.ICEBERG_DETECTION: 1.1,
                PatternType.LIQUIDITY_GRAB: 1.0,
                PatternType.PUMP_AND_DUMP: 0.9,
                PatternType.STOP_HUNTING: 0.7,
                PatternType.ACCUMULATION: 1.3,
                PatternType.DISTRIBUTION: -1.3,
            }.get(snapshot.pattern_type, 1.0)
            signal_strength = base_strength * direction_modifier * pattern_modifier
            return max(-1.0, min(1.0, signal_strength))
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 0.0


class PatternPredictor(IPatternPredictor):
    """Реализация прогнозирования на основе паттернов."""

    def __init__(self, config: PatternMemoryConfig):
        self.config = config

    def generate_prediction(
        self,
        similar_cases: List[Tuple[PatternSnapshot, float]],
        outcomes: List[PatternOutcome],
        current_features: MarketFeatures,
        symbol: str,
    ) -> Optional[PredictionResult]:
        """Генерация прогноза на основе похожих случаев."""
        try:
            if not similar_cases or not outcomes:
                return None

            # Вычисляем веса на основе сходства
            similarities = [case[1] for case in similar_cases]
            weights = [sim / sum(similarities) for sim in similarities]

            # Вычисляем прогнозируемые метрики
            predicted_return = self.calculate_predicted_return(outcomes, weights)
            predicted_duration = self.calculate_predicted_duration(outcomes, weights)
            predicted_volatility = self._calculate_predicted_volatility(outcomes, weights)
            predicted_direction = self._calculate_predicted_direction(outcomes, weights)
            success_rate = self._calculate_success_rate(outcomes)

            # Вычисляем уверенность прогноза
            confidence = self.calculate_prediction_confidence(similar_cases, outcomes)

            # Создаем результат прогнозирования
            return PredictionResult(
                pattern_id=f"pred_{datetime.now().timestamp()}",
                symbol=symbol,
                confidence=confidence,
                predicted_direction=predicted_direction.value,
                predicted_return_percent=predicted_return,
                predicted_duration_minutes=predicted_duration,
                predicted_volatility=predicted_volatility,
                similar_cases_count=len(similar_cases),
                success_rate=success_rate,
                avg_return=predicted_return,
                avg_duration=predicted_duration,
                metadata={
                    "similarities": similarities,
                    "weights": weights,
                    "outcomes_count": len(outcomes),
                },
            )
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            return None

    def calculate_prediction_confidence(
        self,
        similar_cases: List[Tuple[PatternSnapshot, float]],
        outcomes: List[PatternOutcome],
    ) -> float:
        """Вычисление уверенности в прогнозе."""
        try:
            if not similar_cases or not outcomes:
                return 0.0

            # Уверенность от схожести паттернов
            avg_similarity = sum(similarity for _, similarity in similar_cases) / len(similar_cases)
            similarity_confidence = avg_similarity * 0.4

            # Уверенность от количества случаев
            cases_confidence = min(len(outcomes) / 10.0, 1.0) * 0.3

            # Уверенность от согласованности исходов
            success_rate = self._calculate_success_rate(outcomes)
            consistency_confidence = success_rate * 0.3

            total_confidence = (
                similarity_confidence + cases_confidence + consistency_confidence
            )
            return float(min(1.0, total_confidence))  # Исправляем: явно возвращаем float
        except Exception as e:
            logger.error(f"Error calculating prediction confidence: {e}")
            return 0.0

    def calculate_predicted_return(
        self, outcomes: List[PatternOutcome], weights: Optional[List[float]] = None
    ) -> float:
        """Вычисление прогнозируемой доходности."""
        try:
            if not outcomes:
                return 0.0
            if weights is None:
                weights = [1.0 / len(outcomes)] * len(outcomes)
            weighted_return = sum(
                outcome.final_return_percent * weight
                for outcome, weight in zip(outcomes, weights)
            )
            return weighted_return
        except Exception as e:
            logger.error(f"Error calculating predicted return: {e}")
            return 0.0

    def calculate_predicted_duration(
        self, outcomes: List[PatternOutcome], weights: Optional[List[float]] = None
    ) -> int:
        """Вычисление прогнозируемой длительности."""
        try:
            if not outcomes:
                return 0
            if weights is None:
                weights = [1.0 / len(outcomes)] * len(outcomes)
            weighted_duration = sum(
                outcome.duration_minutes * weight
                for outcome, weight in zip(outcomes, weights)
            )
            return int(weighted_duration)
        except Exception as e:
            logger.error(f"Error calculating predicted duration: {e}")
            return 0

    def _calculate_predicted_volatility(
        self, outcomes: List[PatternOutcome], weights: Optional[List[float]] = None
    ) -> float:
        """Вычисление прогнозируемой волатильности."""
        try:
            if not outcomes:
                return 0.0
            if weights is None:
                weights = [1.0 / len(outcomes)] * len(outcomes)
            weighted_volatility = sum(
                outcome.volatility_during * weight
                for outcome, weight in zip(outcomes, weights)
            )
            return weighted_volatility
        except Exception as e:
            logger.error(f"Error calculating predicted volatility: {e}")
            return 0.0

    def _calculate_predicted_direction(
        self, outcomes: List[PatternOutcome], weights: Optional[List[float]] = None
    ) -> PredictionDirection:
        """Вычисление прогнозируемого направления."""
        try:
            if not outcomes:
                return PredictionDirection.NEUTRAL
            if weights is None:
                weights = [1.0 / len(outcomes)] * len(outcomes)
            # Взвешенное голосование по направлениям
            direction_scores = {
                PredictionDirection.UP: 0.0,
                PredictionDirection.DOWN: 0.0,
                PredictionDirection.NEUTRAL: 0.0,
            }
            for outcome, weight in zip(outcomes, weights):
                # Используем outcome_type вместо direction
                if outcome.outcome_type == OutcomeType.PROFITABLE:
                    direction_scores[PredictionDirection.UP] += weight
                elif outcome.outcome_type == OutcomeType.UNPROFITABLE:
                    direction_scores[PredictionDirection.DOWN] += weight
                else:
                    direction_scores[PredictionDirection.NEUTRAL] += weight
            # Возвращаем направление с максимальным весом
            return max(direction_scores, key=lambda k: direction_scores[k])
        except Exception as e:
            logger.error(f"Error calculating predicted direction: {e}")
            return PredictionDirection.NEUTRAL

    def _calculate_success_rate(self, outcomes: List[PatternOutcome]) -> float:
        """Вычисление процента успешных прогнозов."""
        try:
            if not outcomes:
                return 0.0
            successful = sum(1 for outcome in outcomes if outcome.outcome_type == OutcomeType.PROFITABLE)
            return successful / len(outcomes)
        except Exception as e:
            logger.error(f"Error calculating success rate: {e}")
            return 0.0


class SQLitePatternMemoryRepository(IPatternMemoryRepository):
    """SQLite репозиторий для хранения паттернов в памяти."""

    def __init__(self, config: PatternMemoryConfig):
        self.config = config
        self._init_database()

    def _init_database(self) -> None:
        """Инициализация базы данных."""
        try:
            db_path = Path(self.config.db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(db_path) as conn:
                # Таблица снимков паттернов
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS pattern_snapshots (
                        pattern_id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        pattern_type TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        strength REAL NOT NULL,
                        price REAL NOT NULL,
                        volume REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        features_data TEXT NOT NULL,
                        metadata TEXT
                    )
                """)
                
                # Таблица исходов паттернов
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS pattern_outcomes (
                        pattern_id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        outcome_type TEXT NOT NULL,
                        final_return_percent REAL NOT NULL,
                        duration_minutes INTEGER NOT NULL,
                        volatility_during REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        metadata TEXT
                    )
                """)
                
                # Индексы для быстрого поиска
                conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_symbol ON pattern_snapshots(symbol)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_type ON pattern_snapshots(pattern_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON pattern_snapshots(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_symbol ON pattern_outcomes(symbol)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_timestamp ON pattern_outcomes(timestamp)")
                
                conn.commit()
        except Exception as e:
            logger.error(f"Error initializing database: {e}")

    def save_snapshot(self, pattern_id: str, snapshot: PatternSnapshot) -> bool:
        """Сохранение снимка паттерна."""
        try:
            db_path = Path(self.config.db_path)
            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO pattern_snapshots 
                    (pattern_id, symbol, pattern_type, direction, confidence, strength, 
                     price, volume, timestamp, features_data, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern_id,
                    snapshot.symbol,
                    snapshot.pattern_type.value,  # Исправляем: используем .value для enum
                    snapshot.direction,
                    snapshot.confidence,
                    snapshot.strength,
                    snapshot.features.price,
                    snapshot.features.volume,
                    snapshot.timestamp.to_datetime().isoformat(),
                    json.dumps(snapshot.features.to_dict()),
                    json.dumps(snapshot.metadata) if snapshot.metadata else None
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")
            return False

    def save_outcome(self, pattern_id: str, outcome: PatternOutcome) -> bool:
        """Сохранение исхода паттерна."""
        try:
            db_path = Path(self.config.db_path)
            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO pattern_outcomes 
                    (pattern_id, symbol, outcome_type, final_return_percent,
                     duration_minutes, volatility_during, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern_id,
                    outcome.symbol,
                    outcome.outcome_type.value,
                    outcome.final_return_percent,
                    outcome.duration_minutes,
                    outcome.volatility_during,
                    outcome.timestamp.to_datetime().isoformat(),
                    json.dumps(outcome.metadata) if outcome.metadata else {}
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving outcome: {e}")
            return False

    def get_snapshots(
        self,
        symbol: str,
        pattern_type: Optional[PatternType] = None,
        limit: Optional[int] = None,
    ) -> List[PatternSnapshot]:
        """Получение снимков паттернов."""
        try:
            db_path = Path(self.config.db_path)
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                query = """
                    SELECT pattern_id, symbol, pattern_type, direction, confidence, strength,
                           price, volume, timestamp, features_data, metadata
                    FROM pattern_snapshots
                    WHERE symbol = ?
                """
                params = [symbol]
                if pattern_type:
                    query += " AND pattern_type = ?"
                    params.append(pattern_type.value)
                query += " ORDER BY timestamp DESC"
                if limit:
                    query += f" LIMIT {limit}"
                cursor.execute(query, params)
                snapshots = []
                for row in cursor.fetchall():
                    try:
                        snapshot = PatternSnapshot(
                            pattern_id=row[0],
                            timestamp=Timestamp.from_iso(row[8]),
                            symbol=row[1],
                            pattern_type=PatternType(row[2]),
                            confidence=row[4],
                            strength=row[5],
                            direction=row[3],
                            features=MarketFeatures.from_dict(json.loads(row[9])),
                            metadata=json.loads(row[10]) if row[10] else {}
                        )
                        snapshots.append(snapshot)
                    except Exception as e:
                        logger.warning(f"Error parsing snapshot {row[0]}: {e}")
                        continue
                return snapshots
        except Exception as e:
            logger.error(f"Error getting snapshots: {e}")
            return []

    def get_outcomes(self, pattern_ids: List[str]) -> List[PatternOutcome]:
        """Получение исходов паттернов."""
        try:
            if not pattern_ids:
                return []
            db_path = Path(self.config.db_path)
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                placeholders = ",".join("?" * len(pattern_ids))
                cursor.execute(
                    f"""
                    SELECT pattern_id, symbol, outcome_type, final_return_percent,
                           duration_minutes, volatility_during, timestamp, metadata
                    FROM pattern_outcomes
                    WHERE pattern_id IN ({placeholders})
                """,
                    pattern_ids,
                )
                outcomes = []
                for row in cursor.fetchall():
                    try:
                        outcome = PatternOutcome(
                            pattern_id=row[0],
                            symbol=row[1],
                            outcome_type=OutcomeType(row[2]),
                            timestamp=Timestamp.from_iso(row[6]),
                            price_change_percent=0.0,  # Добавляем недостающие поля
                            volume_change_percent=0.0,
                            duration_minutes=row[4],
                            max_profit_percent=0.0,
                            max_loss_percent=0.0,
                            final_return_percent=row[3],
                            volatility_during=row[5],
                            volume_profile="stable",
                            market_regime="trending",
                            metadata=json.loads(row[7]) if row[7] else {}
                        )
                        outcomes.append(outcome)
                    except Exception as e:
                        logger.warning(f"Error parsing outcome {row[0]}: {e}")
                        continue
                return outcomes
        except Exception as e:
            logger.error(f"Error getting outcomes: {e}")
            return []

    def get_statistics(self) -> MemoryStatistics:
        """Получение статистики."""
        try:
            db_path = Path(self.config.db_path)
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                # Общая статистика
                cursor.execute("SELECT COUNT(*) FROM pattern_snapshots")
                total_snapshots = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM pattern_outcomes")
                total_outcomes = cursor.fetchone()[0]
                # Статистика по типам паттернов
                cursor.execute(
                    """
                    SELECT pattern_type, COUNT(*) 
                    FROM pattern_snapshots 
                    GROUP BY pattern_type
                """
                )
                pattern_type_stats = dict(cursor.fetchall())
                # Статистика по исходам
                cursor.execute(
                    """
                    SELECT outcome_type, COUNT(*) 
                    FROM pattern_outcomes 
                    GROUP BY outcome_type
                """
                )
                outcome_type_stats = dict(cursor.fetchall())
                # Статистика по символам
                cursor.execute(
                    """
                    SELECT symbol, COUNT(*) 
                    FROM pattern_snapshots 
                    GROUP BY symbol
                """
                )
                symbol_stats = dict(cursor.fetchall())
                # Средняя уверенность
                cursor.execute("SELECT AVG(confidence) FROM pattern_snapshots")
                avg_confidence = cursor.fetchone()[0] or 0.0
                # Средняя успешность
                cursor.execute(
                    """
                    SELECT COUNT(*) * 1.0 / (SELECT COUNT(*) FROM pattern_outcomes)
                    FROM pattern_outcomes 
                    WHERE outcome_type = 'profitable'
                """
                )
                avg_success_rate = cursor.fetchone()[0] or 0.0
                return MemoryStatistics(
                    total_snapshots=total_snapshots,
                    total_outcomes=total_outcomes,
                    pattern_type_stats=pattern_type_stats,
                    outcome_type_stats=outcome_type_stats,
                    symbol_stats=symbol_stats,
                    avg_confidence=avg_confidence,
                    avg_success_rate=avg_success_rate,
                    last_cleanup=datetime.now(),
                )
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return MemoryStatistics(
                total_snapshots=0,
                total_outcomes=0,
                pattern_type_stats={},
                outcome_type_stats={},
                symbol_stats={},
                avg_confidence=0.0,
                avg_success_rate=0.0,
            )

    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """Очистка старых данных."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cutoff_iso = cutoff_date.isoformat()
            db_path = Path(self.config.db_path)
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                # Удаляем старые снимки
                cursor.execute(
                    """
                    DELETE FROM pattern_snapshots 
                    WHERE timestamp < ?
                """,
                    (cutoff_iso,),
                )
                snapshots_deleted = cursor.rowcount
                # Удаляем старые исходы
                cursor.execute(
                    """
                    DELETE FROM pattern_outcomes 
                    WHERE timestamp < ?
                """,
                    (cutoff_iso,),
                )
                outcomes_deleted = cursor.rowcount
                conn.commit()
                total_deleted = snapshots_deleted + outcomes_deleted
                logger.info(f"Cleaned up {total_deleted} old records")
                return total_deleted
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return 0


class PatternMemory(IPatternMemoryService):
    """Система памяти паттернов с SQLite хранилищем."""

    def __init__(self, config: Optional[PatternMemoryConfig] = None):
        self.config = config or PatternMemoryConfig()
        self.repository = SQLitePatternMemoryRepository(self.config)
        self.matcher = PatternMatcher(self.config)
        self.predictor = PatternPredictor(self.config)
        logger.info(f"PatternMemory initialized with config: {self.config}")

    def match_snapshot(
        self,
        current_features: MarketFeatures,
        symbol: str,
        pattern_type: Optional[PatternType] = None,
    ) -> Optional[PredictionResult]:
        """Сопоставление снимка с историческими паттернами."""
        try:
            # Получаем исторические снимки
            snapshots = self.repository.get_snapshots(symbol, pattern_type)
            if not snapshots:
                return None
            # Ищем похожие случаи
            similar_cases = self.matcher.find_similar_patterns(
                current_features,
                snapshots,
                self.config.similarity_threshold,
                self.config.max_similar_cases,
            )
            if not similar_cases:
                return None
            # Получаем исходы для похожих случаев
            pattern_ids = [case[0].pattern_id for case in similar_cases]
            outcomes = self.repository.get_outcomes(pattern_ids)
            if not outcomes:
                return None
            # Генерируем прогноз
            prediction = self.predictor.generate_prediction(
                similar_cases, outcomes, current_features, symbol
            )
            return prediction
        except Exception as e:
            logger.error(f"Error matching snapshot: {e}")
            return None

    def save_pattern_data(self, pattern_id: str, snapshot: PatternSnapshot) -> bool:
        """Сохранение данных паттерна."""
        return self.repository.save_snapshot(pattern_id, snapshot)

    def update_pattern_outcome(self, pattern_id: str, outcome: PatternOutcome) -> bool:
        """Обновление исхода паттерна."""
        return self.repository.save_outcome(pattern_id, outcome)

    def get_pattern_statistics(
        self, symbol: str, pattern_type: Optional[PatternType] = None
    ) -> Dict[str, Any]:
        """Получение статистики паттернов."""
        try:
            snapshots = self.repository.get_snapshots(symbol, pattern_type)
            if not snapshots:
                return {
                    "total_patterns": 0,
                    "avg_confidence": 0.0,
                    "avg_strength": 0.0,
                    "pattern_types": {},
                    "directions": {},
                }
            # Базовая статистика
            total_patterns = len(snapshots)
            avg_confidence = np.mean([s.confidence for s in snapshots])
            avg_strength = np.mean([s.strength for s in snapshots])
            # Статистика по типам паттернов
            pattern_types = {}
            for snapshot in snapshots:
                pattern_type_str = snapshot.pattern_type.value if snapshot.pattern_type else "unknown"
                if pattern_type_str not in pattern_types:
                    pattern_types[pattern_type_str] = 0
                pattern_types[pattern_type_str] += 1
            # Статистика по направлениям
            directions = {}
            for snapshot in snapshots:
                direction = snapshot.direction
                if direction not in directions:
                    directions[direction] = 0
                directions[direction] += 1
            return {
                "total_patterns": total_patterns,
                "avg_confidence": avg_confidence,
                "avg_strength": avg_strength,
                "pattern_types": pattern_types,
                "directions": directions,
                "recent_patterns": [
                    {
                        "pattern_id": s.pattern_id,
                        "timestamp": s.timestamp.to_iso(),
                        "pattern_type": s.pattern_type.value if hasattr(s.pattern_type, 'value') else str(s.pattern_type) if s.pattern_type else None,
                        "confidence": s.confidence,
                        "strength": s.strength,
                        "direction": s.direction,
                    }
                    for s in snapshots[:10]  # Последние 10 паттернов
                ],
            }
        except Exception as e:
            logger.error(f"Error getting pattern statistics: {e}")
            return {}

    def cleanup_old_patterns(self, days_to_keep: int = 30) -> int:
        """Очистка старых паттернов."""
        return self.repository.cleanup_old_data(days_to_keep)

    def get_memory_statistics(self) -> MemoryStatistics:
        """Получение общей статистики памяти."""
        return self.repository.get_statistics()
