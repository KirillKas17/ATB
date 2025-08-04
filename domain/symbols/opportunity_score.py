# -*- coding: utf-8 -*-
"""Калькулятор метрики пригодности торговой пары к анализу."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from shared.numpy_utils import np
import pandas as pd
from loguru import logger

from domain.type_definitions import (
    ConfidenceValue,
    ConfigurationError,
    DataInsufficientError,
    MarketDataFrame,
    MarketPhase,
    OpportunityScoreCalculatorProtocol,
    OpportunityScoreConfigData,
    OpportunityScoreResult,
    OpportunityScoreValue,
    OrderBookData,
    PatternMemoryData,
    SessionData,
    ValidationError,
)


@dataclass
class OpportunityScoreConfig:
    """Конфигурация для расчета opportunity score."""

    # Веса компонентов (должны суммироваться в 1.0)
    alpha1_liquidity_score: float = 0.20
    alpha2_volume_stability: float = 0.15
    alpha3_structural_predictability: float = 0.20
    alpha4_orderbook_symmetry: float = 0.15
    alpha5_session_alignment: float = 0.15
    alpha6_historical_pattern_match: float = 0.15
    # Пороги для нормализации
    min_volume_threshold: float = 1000000  # Минимальный объем в USD
    max_spread_threshold: float = 0.005  # Максимальный спред 0.5%
    min_atr_threshold: float = 0.001  # Минимальный ATR 0.1%
    max_entropy_threshold: float = 0.8  # Максимальная энтропия
    # Пороги для паттернов
    min_pattern_confidence: float = 0.6
    min_historical_match: float = 0.7
    # Множители для фаз рынка
    phase_multipliers: Optional[Dict[MarketPhase, float]] = None

    def __post_init__(self) -> None:
        """Проверка и установка значений по умолчанию."""
        # Валидация весов
        total_weight = (
            self.alpha1_liquidity_score
            + self.alpha2_volume_stability
            + self.alpha3_structural_predictability
            + self.alpha4_orderbook_symmetry
            + self.alpha5_session_alignment
            + self.alpha6_historical_pattern_match
        )
        if abs(total_weight - 1.0) > 1e-6:
            raise ConfigurationError(f"Weights must sum to 1.0, got {total_weight}")
        # Валидация порогов
        if self.min_volume_threshold <= 0:
            raise ConfigurationError("Min volume threshold must be positive")
        if self.max_spread_threshold <= 0:
            raise ConfigurationError("Max spread threshold must be positive")
        if self.min_atr_threshold <= 0:
            raise ConfigurationError("Min ATR threshold must be positive")
        if not 0.0 <= self.max_entropy_threshold <= 1.0:
            raise ConfigurationError(
                "Max entropy threshold must be between 0.0 and 1.0"
            )
        if not 0.0 <= self.min_pattern_confidence <= 1.0:
            raise ConfigurationError(
                "Min pattern confidence must be between 0.0 and 1.0"
            )
        if not 0.0 <= self.min_historical_match <= 1.0:
            raise ConfigurationError("Min historical match must be between 0.0 and 1.0")
        if self.phase_multipliers is None:
            self.phase_multipliers = {
                MarketPhase.ACCUMULATION: 0.8,
                MarketPhase.BREAKOUT_SETUP: 1.2,
                MarketPhase.BREAKOUT_ACTIVE: 1.0,
                MarketPhase.EXHAUSTION: 0.6,
                MarketPhase.REVERSION_POTENTIAL: 1.1,
                MarketPhase.NO_STRUCTURE: 0.3,
            }


@dataclass
class OpportunityScore:
    """Результат расчета opportunity score."""

    symbol: str
    total_score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    # Компоненты score
    liquidity_score: float = 0.0
    volume_stability: float = 0.0
    structural_predictability: float = 0.0
    orderbook_symmetry: float = 0.0
    session_alignment: float = 0.0
    historical_pattern_match: float = 0.0
    # Дополнительные метрики
    market_phase: MarketPhase = MarketPhase.NO_STRUCTURE
    phase_confidence: float = 0.0
    volume_metrics: Dict[str, float] = field(default_factory=dict)
    price_metrics: Dict[str, float] = field(default_factory=dict)
    pattern_metrics: Dict[str, float] = field(default_factory=dict)
    # Метаданные
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Инициализация метаданных."""
        # Валидация scores
        if not 0.0 <= self.total_score <= 1.0:
            raise ValidationError("value", "", "validation", "Total score must be between 0.0 and 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError("value", "", "validation", "Confidence must be between 0.0 and 1.0")
        if not 0.0 <= self.phase_confidence <= 1.0:
            raise ValidationError("value", "", "validation", "Phase confidence must be between 0.0 and 1.0")

    def is_opportunity(self, min_score: float = 0.78) -> bool:
        """Проверка, является ли символ торговой возможностью."""
        if not 0.0 <= min_score <= 1.0:
            raise ValidationError("value", "", "validation", "Minimum score must be between 0.0 and 1.0")
        return self.total_score >= min_score and self.confidence >= 0.6

    def get_score_breakdown(self) -> Dict[str, float]:
        """Получение разбивки score по компонентам."""
        return {
            "total_score": self.total_score,
            "liquidity_score": self.liquidity_score,
            "volume_stability": self.volume_stability,
            "structural_predictability": self.structural_predictability,
            "orderbook_symmetry": self.orderbook_symmetry,
            "session_alignment": self.session_alignment,
            "historical_pattern_match": self.historical_pattern_match,
            "market_phase": float(self.market_phase.value),
            "phase_confidence": self.phase_confidence,
        }


class OpportunityScoreCalculator(OpportunityScoreCalculatorProtocol):
    """Калькулятор метрики пригодности торговой пары к анализу."""

    def __init__(self, config: Optional[OpportunityScoreConfig] = None):
        """Инициализация калькулятора."""
        self.config = config or OpportunityScoreConfig()
        self.logger = logger.bind(name=self.__class__.__name__)

    def calculate_opportunity_score(
        self,
        symbol: str,
        market_data: MarketDataFrame,
        order_book: OrderBookData,
        pattern_memory: Optional[PatternMemoryData] = None,
        session_data: Optional[SessionData] = None,
    ) -> OpportunityScoreResult:
        """
        Расчет opportunity score для торговой пары.
        Args:
            symbol: Торговый символ
            market_data: OHLCV данные
            order_book: Данные стакана заявок
            pattern_memory: Данные из PatternMemory (опционально)
            session_data: Данные сессии (опционально)
        Returns:
            OpportunityScoreResult: Результат расчета
        """
        try:
            # Валидация входных данных
            self._validate_input_data(symbol, market_data, order_book)
            if len(market_data) < 50:  # Минимальное количество данных
                return self._create_default_result(symbol)
            # Расчет компонентов score
            liquidity_score = self._calculate_liquidity_score(market_data, order_book)
            volume_stability = self._calculate_volume_stability(market_data)
            structural_predictability = self._calculate_structural_predictability(
                market_data
            )
            orderbook_symmetry = self._calculate_orderbook_symmetry(order_book)
            session_alignment = self._calculate_session_alignment(session_data)
            historical_pattern_match = self._calculate_historical_pattern_match(
                symbol, pattern_memory
            )
            # Определение фазы рынка
            market_phase, phase_confidence = self._determine_market_phase(market_data)
            # Расчет общего score
            total_score = self._calculate_total_score(
                liquidity_score,
                volume_stability,
                structural_predictability,
                orderbook_symmetry,
                session_alignment,
                historical_pattern_match,
                market_phase,
            )
            # Расчет уверенности
            confidence = self._calculate_confidence(
                liquidity_score,
                volume_stability,
                structural_predictability,
                orderbook_symmetry,
                session_alignment,
                historical_pattern_match,
                phase_confidence,
            )
            # Извлечение дополнительных метрик
            volume_metrics = self._extract_volume_metrics(market_data)
            price_metrics = self._extract_price_metrics(market_data)
            pattern_metrics = self._extract_pattern_metrics(pattern_memory)
            # Создание метаданных
            metadata = self._create_metadata(market_data, order_book)
            # Создание результата
            return OpportunityScoreResult(
                symbol=symbol,
                total_score=OpportunityScoreValue(total_score),
                confidence=ConfidenceValue(confidence),
                market_phase=market_phase,
                phase_confidence=ConfidenceValue(phase_confidence),
                components={
                    "liquidity_score": liquidity_score,
                    "volume_stability": volume_stability,
                    "structural_predictability": structural_predictability,
                    "orderbook_symmetry": orderbook_symmetry,
                    "session_alignment": session_alignment,
                    "historical_pattern_match": historical_pattern_match,
                },
                metrics={
                    "volume_metrics": volume_metrics,
                    "price_metrics": price_metrics,
                    "pattern_metrics": pattern_metrics,
                },
                metadata=metadata,
            )
        except Exception as e:
            self.logger.error(f"Error calculating opportunity score for {symbol}: {e}")
            return self._create_default_result(symbol)

    def calculate_score(
        self,
        volume_profile: Optional[Any] = None,
        price_structure: Optional[Any] = None,
        orderbook_metrics: Optional[Any] = None,
        pattern_metrics: Optional[Any] = None,
        session_metrics: Optional[Any] = None,
        market_phase: Optional[Any] = None,
    ) -> float:
        """
        Упрощенный расчет opportunity score.
        Args:
            volume_profile: Профиль объема
            price_structure: Структура цен
            orderbook_metrics: Метрики ордербука
            pattern_metrics: Метрики паттернов
            session_metrics: Метрики сессии
            market_phase: Фаза рынка
        Returns:
            float: Opportunity score
        """
        try:
            # Простой расчет на основе доступных данных
            score = 0.5  # Базовый score
            
            # Корректировка по объему
            if volume_profile:
                if hasattr(volume_profile, 'volume_trend'):
                    score += volume_profile.volume_trend * 0.1
            
            # Корректировка по структуре цен
            if price_structure:
                if hasattr(price_structure, 'price_trend'):
                    score += price_structure.price_trend * 0.1
            
            # Корректировка по ордербуку
            if orderbook_metrics:
                if hasattr(orderbook_metrics, 'bid_ask_ratio'):
                    ratio = orderbook_metrics.bid_ask_ratio
                    if ratio > 1.0:
                        score += 0.1
                    elif ratio < 0.5:
                        score -= 0.1
            
            # Корректировка по паттернам
            if pattern_metrics:
                if hasattr(pattern_metrics, 'pattern_confidence'):
                    score += pattern_metrics.pattern_confidence * 0.1
            
            # Корректировка по сессии
            if session_metrics:
                if hasattr(session_metrics, 'session_momentum'):
                    score += session_metrics.session_momentum * 0.1
            
            # Корректировка по фазе рынка
            if market_phase:
                phase_multipliers = {
                    MarketPhase.ACCUMULATION: 1.1,
                    MarketPhase.BREAKOUT_SETUP: 1.2,
                    MarketPhase.BREAKOUT_ACTIVE: 1.0,
                    MarketPhase.EXHAUSTION: 0.8,
                    MarketPhase.REVERSION_POTENTIAL: 1.1,
                    MarketPhase.NO_STRUCTURE: 0.9,
                }
                multiplier = phase_multipliers.get(market_phase, 1.0)
                score *= multiplier
            
            return max(0.0, min(1.0, score))
        except Exception as e:
            self.logger.error(f"Error calculating simplified score: {e}")
            return 0.5

    def is_opportunity(
        self, score: float, min_score: float = 0.78
    ) -> bool:
        """Проверка, является ли символ торговой возможностью."""
        if not 0.0 <= min_score <= 1.0:
            raise ValidationError("value", "", "validation", "Minimum score must be between 0.0 and 1.0")
        return score >= min_score

    def _validate_input_data(
        self, symbol: str, market_data: MarketDataFrame, order_book: OrderBookData
    ) -> None:
        """Валидация входных данных."""
        if not symbol or not isinstance(symbol, str):
            raise ValidationError("value", "", "validation", "Symbol must be a non-empty string")
        if not isinstance(market_data, pd.DataFrame):
            raise ValidationError("value", "", "validation", "Market data must be a pandas DataFrame")
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [
            col for col in required_columns if col not in market_data.columns
        ]
        if missing_columns:
            raise ValidationError("value", "", "format", f"Missing required columns: {missing_columns}")
        if not isinstance(order_book, dict):
            raise ValidationError("value", "", "validation", "Order book must be a dictionary")
        if len(market_data) == 0:
            raise DataInsufficientError("Market data is empty")

    def _calculate_liquidity_score(
        self, market_data: MarketDataFrame, order_book: OrderBookData
    ) -> float:
        """Расчет score ликвидности."""
        try:
            # Объем торгов
            current_volume = market_data["volume"].iloc[-1]
            avg_volume = market_data["volume"].rolling(window=20).mean().iloc[-1]
            volume_score = min(current_volume / self.config.min_volume_threshold, 1.0)
            volume_trend_score = min(avg_volume / self.config.min_volume_threshold, 1.0)
            # Спред
            spread_percent = order_book.get("spread_percent", 0.01)
            spread_score = max(
                0.0, 1.0 - spread_percent / self.config.max_spread_threshold
            )
            # Глубина стакана
            depth_score = self._calculate_orderbook_depth(order_book)
            # Взвешенный score
            liquidity_score = (
                volume_score * 0.4
                + volume_trend_score * 0.3
                + spread_score * 0.2
                + depth_score * 0.1
            )
            return float(np.clip(liquidity_score, 0.0, 1.0))
        except Exception as e:
            self.logger.warning(f"Error calculating liquidity score: {e}")
            return 0.0

    def _calculate_volume_stability(self, market_data: MarketDataFrame) -> float:
        """Расчет стабильности объема."""
        try:
            volume = market_data["volume"]
            if len(volume) < 20:
                return 0.0
            # Коэффициент вариации
            cv = volume.std() / volume.mean() if volume.mean() > 0 else 1.0
            # Тренд объема
            volume_trend = self._calculate_volume_trend(volume)
            # Аномалии объема
            volume_anomaly = self._detect_volume_anomalies(volume)
            # Взвешенный score
            stability_score = (
                (1.0 / (1.0 + cv)) * 0.5
                + (1.0 - abs(volume_trend)) * 0.3
                + (1.0 - volume_anomaly) * 0.2
            )
            return float(np.clip(stability_score, 0.0, 1.0))
        except Exception as e:
            self.logger.warning(f"Error calculating volume stability: {e}")
            return 0.0

    def _calculate_structural_predictability(
        self, market_data: MarketDataFrame
    ) -> float:
        """Расчет структурной предсказуемости."""
        try:
            close = market_data["close"]
            if len(close) < 50:
                return 0.0
            # Энтропия цены
            price_entropy = self._calculate_price_entropy(close.pct_change().dropna())
            entropy_score = max(
                0.0, 1.0 - price_entropy / self.config.max_entropy_threshold
            )
            # ATR
            atr = self._calculate_atr(market_data)
            atr_score = min(1.0, atr / self.config.min_atr_threshold)
            # Сила тренда
            trend_strength = self._calculate_trend_strength(close)
            # Уровни поддержки/сопротивления
            support_resistance_score = self._calculate_support_resistance_score(
                market_data
            )
            # Взвешенный score
            predictability_score = (
                entropy_score * 0.3
                + atr_score * 0.2
                + trend_strength * 0.3
                + support_resistance_score * 0.2
            )
            return float(np.clip(predictability_score, 0.0, 1.0))
        except Exception as e:
            self.logger.warning(f"Error calculating structural predictability: {e}")
            return 0.0

    def _calculate_orderbook_symmetry(self, order_book: OrderBookData) -> float:
        """Расчет симметрии стакана заявок."""
        try:
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            if not bids or not asks:
                return 0.0
            # Симметрия цен
            price_symmetry = self._calculate_price_symmetry(bids, asks)
            # Симметрия глубины
            depth_symmetry = self._calculate_depth_symmetry(bids, asks)
            # Взвешенный score
            symmetry_score = price_symmetry * 0.6 + depth_symmetry * 0.4
            return float(np.clip(symmetry_score, 0.0, 1.0))
        except Exception as e:
            self.logger.warning(f"Error calculating orderbook symmetry: {e}")
            return 0.0

    def _calculate_session_alignment(
        self, session_data: Optional[SessionData]
    ) -> float:
        """Расчет выравнивания сессии."""
        try:
            if not session_data:
                return 0.5  # Нейтральное значение при отсутствии данных
            # Извлечение метрик сессии
            session_alignment = session_data.get("alignment", 0.5)
            session_activity = session_data.get("activity", 0.5)
            session_volatility = session_data.get("volatility", 0.5)
            # Нормализация
            alignment_score = float(np.clip(session_alignment, 0.0, 1.0))
            activity_score = float(np.clip(session_activity, 0.0, 1.0))
            volatility_score = 1.0 - float(np.clip(session_volatility, 0.0, 1.0))
            # Взвешенный score
            session_score = (
                alignment_score * 0.5 + activity_score * 0.3 + volatility_score * 0.2
            )
            return float(np.clip(session_score, 0.0, 1.0))
        except Exception as e:
            self.logger.warning(f"Error calculating session alignment: {e}")
            return 0.5

    def _calculate_historical_pattern_match(
        self, symbol: str, pattern_memory: Optional[PatternMemoryData]
    ) -> float:
        """Расчет соответствия историческим паттернам."""
        try:
            if not pattern_memory:
                return 0.5  # Нейтральное значение при отсутствии данных
            # Извлечение данных о паттернах
            symbol_patterns = pattern_memory.get(symbol, {})
            if not symbol_patterns:
                return 0.5
            # Метрики паттернов
            pattern_confidence = symbol_patterns.get("confidence", 0.5)
            historical_match = symbol_patterns.get("historical_match", 0.5)
            pattern_complexity = symbol_patterns.get("complexity", 0.5)
            # Нормализация
            confidence_score = float(np.clip(pattern_confidence, 0.0, 1.0))
            match_score = float(np.clip(historical_match, 0.0, 1.0))
            complexity_score = 1.0 - float(np.clip(pattern_complexity, 0.0, 1.0))
            # Взвешенный score
            pattern_score = (
                confidence_score * 0.4 + match_score * 0.4 + complexity_score * 0.2
            )
            return float(np.clip(pattern_score, 0.0, 1.0))
        except Exception as e:
            self.logger.warning(f"Error calculating historical pattern match: {e}")
            return 0.5

    def _determine_market_phase(
        self, market_data: MarketDataFrame
    ) -> Tuple[MarketPhase, float]:
        """Определение фазы рынка на основе данных."""
        try:
            close = market_data["close"]
            volume = market_data["volume"]
            # Простой анализ тренда
            if len(close) < 20:
                return MarketPhase.NO_STRUCTURE, 0.0
            # Расчет тренда
            recent_close = close.tail(20)
            trend_slope = float(
                np.polyfit(range(len(recent_close)), recent_close, 1)[0]
            )
            # Расчет волатильности
            returns = close.pct_change().dropna()
            volatility = float(returns.std())
            # Анализ объема
            recent_volume = volume.tail(20)
            volume_trend = float(
                np.polyfit(range(len(recent_volume)), recent_volume, 1)[0]
            )
            # Определение фазы на основе метрик
            if abs(trend_slope) < 0.001 and volatility < 0.02:
                return MarketPhase.ACCUMULATION, 0.7
            elif trend_slope > 0.002 and volume_trend > 0:
                return MarketPhase.BREAKOUT_ACTIVE, 0.8
            elif trend_slope > 0.001 and volatility > 0.03:
                return MarketPhase.BREAKOUT_SETUP, 0.6
            elif abs(trend_slope) < 0.001 and volume_trend < 0:
                return MarketPhase.EXHAUSTION, 0.5
            elif trend_slope < -0.001 and volatility > 0.04:
                return MarketPhase.REVERSION_POTENTIAL, 0.6
            else:
                return MarketPhase.NO_STRUCTURE, 0.3
        except Exception as e:
            self.logger.error(f"Error determining market phase: {e}")
            return MarketPhase.NO_STRUCTURE, 0.0

    def _calculate_total_score(
        self,
        liquidity_score: float,
        volume_stability: float,
        structural_predictability: float,
        orderbook_symmetry: float,
        session_alignment: float,
        historical_pattern_match: float,
        market_phase: MarketPhase,
    ) -> float:
        """Расчет общего score."""
        try:
            # Взвешенная сумма компонентов
            base_score = (
                liquidity_score * self.config.alpha1_liquidity_score
                + volume_stability * self.config.alpha2_volume_stability
                + structural_predictability
                * self.config.alpha3_structural_predictability
                + orderbook_symmetry * self.config.alpha4_orderbook_symmetry
                + session_alignment * self.config.alpha5_session_alignment
                + historical_pattern_match * self.config.alpha6_historical_pattern_match
            )
            # Множитель фазы рынка
            phase_multiplier = (self.config.phase_multipliers or {}).get(market_phase, 1.0)
            total_score = base_score * phase_multiplier
            return float(np.clip(total_score, 0.0, 1.0))
        except Exception as e:
            self.logger.warning(f"Error calculating total score: {e}")
            return 0.0

    def _calculate_confidence(
        self,
        liquidity_score: float,
        volume_stability: float,
        structural_predictability: float,
        orderbook_symmetry: float,
        session_alignment: float,
        historical_pattern_match: float,
        phase_confidence: float,
    ) -> float:
        """Расчет уверенности в результате."""
        try:
            # Среднее значение всех компонентов
            component_confidence = (
                liquidity_score
                + volume_stability
                + structural_predictability
                + orderbook_symmetry
                + session_alignment
                + historical_pattern_match
            ) / 6.0
            # Взвешенная уверенность
            confidence = component_confidence * 0.7 + phase_confidence * 0.3
            return float(np.clip(confidence, 0.0, 1.0))
        except Exception as e:
            self.logger.warning(f"Error calculating confidence: {e}")
            return 0.0

    def _create_default_result(self, symbol: str) -> OpportunityScoreResult:
        """Создание результата по умолчанию."""
        return OpportunityScoreResult(
            symbol=symbol,
            total_score=OpportunityScoreValue(0.0),
            confidence=ConfidenceValue(0.0),
            market_phase=MarketPhase.NO_STRUCTURE,
            phase_confidence=ConfidenceValue(0.0),
            components={},
            metrics={},
            metadata={"error": "Insufficient data"},
        )

    def _calculate_orderbook_depth(self, order_book: OrderBookData) -> float:
        """Расчет глубины стакана заявок."""
        try:
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            if not bids or not asks:
                return 0.0
            # Суммарный объем в пределах 1% от цены
            total_volume = sum(bid[1] for bid in bids[:5]) + sum(
                ask[1] for ask in asks[:5]
            )
            # Нормализация
            depth_score = min(1.0, total_volume / 1000000)  # 1M USD как эталон
            return float(depth_score)
        except Exception as e:
            self.logger.warning(f"Error calculating orderbook depth: {e}")
            return 0.0

    def _calculate_volume_trend(self, volume: pd.Series) -> float:
        """Расчет тренда объема."""
        try:
            if len(volume) < 10:
                return 0.0
            # Линейная регрессия
            x = np.arange(len(volume), dtype=np.float64)
            y = volume.values.astype(np.float64)
            coeffs, residuals, _, _, _ = np.polyfit(x, y, 1, full=True)
            slope = coeffs[0]
            # Корреляция через np.corrcoef
            correlation = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else 0.0
            trend_strength = abs(slope) / float(volume.mean()) if volume.mean() > 0 else 0.0
            return float(trend_strength * abs(correlation))
        except Exception as e:
            self.logger.warning(f"Error calculating volume trend: {e}")
            return 0.0

    def _detect_volume_anomalies(self, volume: pd.Series) -> float:
        """Обнаружение аномалий объема."""
        try:
            if len(volume) < 20:
                return 1.0
            # Рассчитываем статистики объема
            mean_volume: float = float(volume.rolling(window=20).mean().iloc[-1])
            std_volume: float = float(volume.rolling(window=20).std().iloc[-1])
            
            if std_volume == 0:
                return 0.0
            
            # Текущий объем
            current_volume: float = float(volume.iloc[-1])
            
            # Z-score аномалии
            z_score: float = abs(current_volume - mean_volume) / std_volume
            return min(1.0, z_score / 3.0)  # Нормализация к [0, 1]
        except Exception as e:
            self.logger.warning(f"Error detecting volume anomalies: {e}")
            return 0.0

    def _calculate_price_entropy(self, price_changes: pd.Series) -> float:
        """Расчет энтропии цены."""
        try:
            if len(price_changes) < 10:
                return 0.5
            
            # Рассчитываем энтропию распределения изменений цен
            hist, _ = np.histogram(price_changes, bins=10)
            hist = hist[hist > 0]  # Убираем пустые корзины
            if len(hist) == 0:
                return 0.0
            
            # Нормализация гистограммы
            hist = hist / hist.sum()
            
            # Расчет энтропии
            entropy = -np.sum(hist * np.log2(hist))
            max_entropy = np.log2(len(hist))
            
            # Нормализация к [0, 1]
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            return float(normalized_entropy)
        except Exception as e:
            self.logger.warning(f"Error calculating price entropy: {e}")
            return 0.5

    def _calculate_atr(self, market_data: MarketDataFrame) -> float:
        """Расчет ATR."""
        try:
            if len(market_data) < 14:
                return 0.0
            
            high = market_data["high"]
            low = market_data["low"]
            close = market_data["close"]
            
            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # ATR
            atr = tr.rolling(window=14).mean().iloc[-1]
            return float(atr / close.iloc[-1]) if close.iloc[-1] > 0 else 0.0
        except Exception as e:
            self.logger.warning(f"Error calculating ATR: {e}")
            return 0.0

    def _calculate_trend_strength(self, close: pd.Series) -> float:
        """Расчет силы тренда."""
        try:
            if len(close) < 20:
                return 0.0
            # Линейная регрессия для определения тренда
            x = np.arange(len(close), dtype=np.float64)
            y = close.values.astype(np.float64)
            coeffs, residuals, _, _, _ = np.polyfit(x, y, 1, full=True)
            slope = coeffs[0]
            # R^2 через np.corrcoef
            r_value = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else 0.0
            trend_direction = np.sign(slope)
            return float(trend_direction * (r_value ** 2))
        except Exception as e:
            self.logger.warning(f"Error calculating trend strength: {e}")
            return 0.0

    def _calculate_support_resistance_score(
        self, market_data: MarketDataFrame
    ) -> float:
        """Расчет score уровней поддержки/сопротивления."""
        try:
            high = market_data["high"]
            low = market_data["low"]
            close = market_data["close"]
            current_price = close.iloc[-1]
            # Поиск ближайших уровней
            recent_highs = high.tail(20)
            recent_lows = low.tail(20)
            resistance_level = recent_highs.max()
            support_level = recent_lows.min()
            # Расстояние до уровней
            resistance_distance = abs(resistance_level - current_price) / current_price
            support_distance = abs(current_price - support_level) / current_price
            # Score на основе близости к уровням
            resistance_score = max(0.0, 1.0 - resistance_distance / 0.1)
            support_score = max(0.0, 1.0 - support_distance / 0.1)
            return float((resistance_score + support_score) / 2.0)
        except Exception as e:
            self.logger.warning(f"Error calculating support/resistance score: {e}")
            return 0.0

    def _calculate_price_symmetry(self, bids: List, asks: List) -> float:
        """Расчет симметрии цен."""
        try:
            if not bids or not asks:
                return 0.0
            # Средние цены
            avg_bid = sum(bid[0] for bid in bids[:5]) / len(bids[:5])
            avg_ask = sum(ask[0] for ask in asks[:5]) / len(asks[:5])
            # Симметрия относительно спреда
            spread = avg_ask - avg_bid
            if spread == 0:
                return 1.0
            mid_price = (avg_bid + avg_ask) / 2
            symmetry = 1.0 - abs(avg_bid - mid_price) / spread
            return float(np.clip(symmetry, 0.0, 1.0))
        except Exception as e:
            self.logger.warning(f"Error calculating price symmetry: {e}")
            return 0.0

    def _calculate_depth_symmetry(self, bids: List, asks: List) -> float:
        """Расчет симметрии глубины."""
        try:
            if not bids or not asks:
                return 0.0
            # Объемы на первых 5 уровнях
            bid_volume = sum(bid[1] for bid in bids[:5])
            ask_volume = sum(ask[1] for ask in asks[:5])
            total_volume = bid_volume + ask_volume
            if total_volume == 0:
                return 0.0
            # Симметрия объемов
            symmetry = 1.0 - abs(bid_volume - ask_volume) / total_volume
            return float(np.clip(symmetry, 0.0, 1.0))
        except Exception as e:
            self.logger.warning(f"Error calculating depth symmetry: {e}")
            return 0.0

    def _extract_volume_metrics(self, market_data: MarketDataFrame) -> Dict[str, float]:
        """Извлечение метрик объема."""
        try:
            volume = market_data["volume"]
            x = np.arange(len(volume.tail(20)), dtype=np.float64)
            y = volume.tail(20).values.astype(np.float64)
            slope = float(np.polyfit(x, y, 1)[0]) if len(x) > 1 else 0.0
            return {
                "current_volume": float(volume.iloc[-1]),
                "avg_volume_20": float(volume.tail(20).mean()),
                "volume_std": float(volume.tail(20).std()),
                "volume_trend": slope,
                "volume_anomaly_ratio": (
                    float(volume.iloc[-1] / volume.tail(20).mean())
                    if volume.tail(20).mean() > 0
                    else 1.0
                ),
            }
        except Exception as e:
            self.logger.error(f"Error extracting volume metrics: {e}")
            return {
                "current_volume": 0.0,
                "avg_volume_20": 0.0,
                "volume_std": 0.0,
                "volume_trend": 0.0,
                "volume_anomaly_ratio": 1.0,
            }

    def _extract_price_metrics(self, market_data: MarketDataFrame) -> Dict[str, float]:
        """Извлечение метрик цены."""
        try:
            close = market_data["close"]
            high = market_data["high"]
            low = market_data["low"]
            return {
                "current_price": float(close.iloc[-1]),
                "atr": self._calculate_atr(market_data),
                "price_entropy": self._calculate_price_entropy(
                    close.pct_change().dropna()
                ),
                "trend_strength": self._calculate_trend_strength(close),
                "support_resistance_score": self._calculate_support_resistance_score(
                    market_data
                ),
            }
        except Exception as e:
            self.logger.error(f"Error extracting price metrics: {e}")
            return {
                "current_price": 0.0,
                "atr": 0.0,
                "price_entropy": 0.5,
                "trend_strength": 0.0,
                "support_resistance_score": 0.0,
            }

    def _extract_pattern_metrics(
        self, pattern_memory: Optional[PatternMemoryData]
    ) -> Dict[str, float]:
        """Извлечение метрик паттернов."""
        try:
            if pattern_memory is None:
                return {
                    "pattern_confidence": 0.0,
                    "historical_match": 0.0,
                    "pattern_complexity": 0.0,
                }
            # Агрегация метрик по всем символам
            confidences = []
            matches = []
            complexities = []
            for symbol_data in pattern_memory.values():
                if isinstance(symbol_data, dict):
                    confidences.append(symbol_data.get("confidence", 0.0))
                    matches.append(symbol_data.get("historical_match", 0.0))
                    complexities.append(symbol_data.get("complexity", 0.0))
            return {
                "pattern_confidence": (
                    float(np.mean(confidences)) if confidences else 0.0
                ),
                "historical_match": float(np.mean(matches)) if matches else 0.0,
                "pattern_complexity": (
                    float(np.mean(complexities)) if complexities else 0.0
                ),
            }
        except Exception as e:
            self.logger.error(f"Error extracting pattern metrics: {e}")
            return {
                "pattern_confidence": 0.0,
                "historical_match": 0.0,
                "pattern_complexity": 0.0,
            }

    def _create_metadata(
        self, market_data: MarketDataFrame, order_book: OrderBookData
    ) -> Dict[str, Any]:
        """Создание метаданных для результата."""
        try:
            from datetime import datetime
            return {
                "data_points": len(market_data),
                "calculation_timestamp": datetime.now().isoformat(),
                "config_weights": {
                    "liquidity_score": self.config.alpha1_liquidity_score,
                    "volume_stability": self.config.alpha2_volume_stability,
                    "structural_predictability": self.config.alpha3_structural_predictability,
                    "orderbook_symmetry": self.config.alpha4_orderbook_symmetry,
                    "session_alignment": self.config.alpha5_session_alignment,
                    "historical_pattern_match": self.config.alpha6_historical_pattern_match,
                },
                "thresholds": {
                    "min_volume": self.config.min_volume_threshold,
                    "max_spread": self.config.max_spread_threshold,
                    "min_atr": self.config.min_atr_threshold,
                    "max_entropy": self.config.max_entropy_threshold,
                },
            }
        except Exception as e:
            self.logger.error(f"Error creating metadata: {e}")
            from datetime import datetime
            return {
                "error": str(e),
                "calculation_timestamp": datetime.now().isoformat(),
            }
