"""
Интерфейс стратегии в домене.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    cast,
    runtime_checkable,
)
from uuid import UUID, uuid4

from domain.entities.market import MarketData, MarketState
from domain.entities.signal import Signal, SignalType, SignalStrength
from domain.entities.strategy import StrategyStatus, StrategyType
from domain.entities.strategy_performance import StrategyPerformance
from domain.type_definitions import (
    ConfidenceLevel,
    MetadataDict,
    PerformanceScore,
    RiskLevel,
    SignalId,
    StrategyConfig,
    StrategyId,
    TradingPair,
)
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency


@runtime_checkable
class MarketDataProviderProtocol(Protocol):
    """Протокол для поставщика рыночных данных."""

    def get_market_data(
        self, symbol: str, timeframe: str, limit: int
    ) -> List[MarketData]:
        """Получить рыночные данные."""
        ...

    def get_market_state(self, symbol: str) -> Optional[MarketState]:
        """Получить состояние рынка."""
        ...


@runtime_checkable
class SignalGeneratorProtocol(Protocol):
    """Протокол для генератора сигналов."""

    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """Генерировать сигнал."""
        ...

    def validate_signal(self, signal: Signal) -> bool:
        """Валидировать сигнал."""
        ...


@runtime_checkable
class RiskManagerProtocol(Protocol):
    """Протокол для управления рисками."""

    def calculate_position_size(
        self, signal: Signal, account_balance: Decimal
    ) -> Decimal:
        """Рассчитать размер позиции."""
        ...

    def validate_risk_limits(self, signal: Signal) -> bool:
        """Проверить лимиты риска."""
        ...


@dataclass
class StrategyAnalysisResult:
    """Результат анализа стратегии."""

    confidence_score: float
    trend_direction: str
    trend_strength: float
    volatility_level: float
    volume_analysis: Dict[str, Any]
    technical_indicators: Dict[str, Any]
    market_regime: str
    risk_assessment: Dict[str, Any]
    support_resistance: Tuple[Optional[float], Optional[float]]
    momentum_indicators: Dict[str, float]
    pattern_recognition: List[str]
    market_sentiment: str
    timestamp: datetime = field(default_factory=datetime.now)


class StrategyInterface(ABC):
    """
    Интерфейс стратегии - абстрактный базовый класс для всех стратегий.
    Определяет контракт для реализации торговых стратегий с поддержкой
    анализа рынка, генерации сигналов, управления рисками и производительности.
    """

    def __init__(
        self,
        strategy_id: StrategyId,
        name: str,
        strategy_type: StrategyType,
        trading_pairs: List[str],
        parameters: Dict[str, Any],
        risk_level: RiskLevel = RiskLevel(Decimal("0.5")),
        confidence_threshold: ConfidenceLevel = ConfidenceLevel(Decimal("0.6")),
    ):
        """
        Инициализация интерфейса стратегии.
        Args:
            strategy_id: Уникальный идентификатор стратегии
            name: Название стратегии
            strategy_type: Тип стратегии
            trading_pairs: Список торговых пар
            parameters: Параметры стратегии
            risk_level: Уровень риска
            confidence_threshold: Порог уверенности
        """
        self._strategy_id = strategy_id
        self._name = name
        self._strategy_type = strategy_type
        self._trading_pairs = [TradingPair(pair) for pair in trading_pairs]
        self._parameters = parameters
        self._risk_level = risk_level
        self._confidence_threshold = confidence_threshold
        self._status = StrategyStatus.INACTIVE
        self._performance = StrategyPerformance()
        self._created_at = datetime.now()
        self._updated_at = datetime.now()
        self._metadata: Dict[str, Any] = {}
        self._last_analysis: Optional[StrategyAnalysisResult] = None
        self._signal_history: List[Signal] = []
        self._execution_count = 0
        self._success_count = 0
        self._error_count = 0

    def get_strategy_id(self) -> StrategyId:
        """
        Получить ID стратегии.
        Returns:
            StrategyId: Уникальный идентификатор стратегии
        """
        return self._strategy_id

    def get_strategy_type(self) -> StrategyType:
        """
        Получить тип стратегии.
        Returns:
            StrategyType: Тип стратегии
        """
        return self._strategy_type

    def analyze_market(self, market_data: MarketData) -> StrategyAnalysisResult:
        """
        Анализировать рынок и возвращать результаты анализа.
        Args:
            market_data: Рыночные данные для анализа
        Returns:
            StrategyAnalysisResult: Результаты анализа рынка
        Raises:
            ValueError: Если данные некорректны
            RuntimeError: Если анализ не может быть выполнен
        """
        if not market_data:
            raise ValueError("Market data cannot be None")
        if not self._is_trading_pair_supported(market_data.symbol):
            raise ValueError(
                f"Trading pair {market_data.symbol} not supported by strategy"
            )
        # Выполняем специализированный анализ
        analysis_result = self._perform_market_analysis(market_data)
        # Сохраняем результат анализа
        self._last_analysis = analysis_result
        return analysis_result

    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """
        Генерировать торговый сигнал на основе рыночных данных.
        Args:
            market_data: Рыночные данные для генерации сигнала
        Returns:
            Optional[Signal]: Сгенерированный сигнал или None, если сигнал не генерируется
        Raises:
            ValueError: Если данные некорректны
            RuntimeError: Если генерация сигнала не может быть выполнена
        """
        if not market_data:
            raise ValueError("Market data cannot be None")
        if not self._is_trading_pair_supported(market_data.symbol):
            raise ValueError(
                f"Trading pair {market_data.symbol} not supported by strategy"
            )
        # Проверка активности стратегии
        if not self.is_active():
            return None
        # Анализ рынка
        analysis = self.analyze_market(market_data)
        confidence_score = analysis.confidence_score
        # Проверка порога уверенности
        if confidence_score < self._confidence_threshold:
            return None
        # Генерация сигнала на основе типа стратегии
        signal = self._generate_signal_by_type(market_data, analysis)
        if signal:
            signal.strategy_id = self._strategy_id
            signal.trading_pair = str(market_data.symbol)
            signal.confidence = Decimal(str(confidence_score))
            signal.metadata.update(
                {
                    "strategy_type": self._strategy_type.value,
                    "risk_level": str(self._risk_level),
                    "analysis": {
                        "confidence_score": confidence_score,
                        "trend_direction": analysis.trend_direction,
                        "trend_strength": analysis.trend_strength,
                        "volatility_level": analysis.volatility_level,
                        "market_regime": analysis.market_regime,
                    },
                }
            )
            # Добавляем сигнал в историю
            self._signal_history.append(signal)
        return signal

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Валидировать параметры стратегии.
        Args:
            parameters: Параметры для валидации
        Returns:
            bool: True если параметры валидны
        Raises:
            ValueError: Если параметры некорректны
        """
        if not parameters:
            raise ValueError("Parameters cannot be None or empty")
        # Проверка обязательных параметров
        required_params = self._get_required_parameters()
        for param in required_params:
            if param not in parameters:
                raise ValueError(f"Required parameter '{param}' is missing")
        # Проверка типов и диапазонов параметров
        if not self._validate_parameter_types_and_ranges(parameters):
            raise ValueError("Parameter types or ranges are invalid")
        return True

    def get_required_features(self) -> List[str]:
        """
        Получить список требуемых функций для стратегии.
        Returns:
            List[str]: Список требуемых функций
        """
        base_features = ["price_data", "volume_data", "timestamp_data"]
        # Добавляем специфичные функции в зависимости от типа стратегии
        if self._strategy_type == StrategyType.TREND_FOLLOWING:
            base_features.extend(
                ["moving_averages", "trend_indicators", "momentum_indicators"]
            )
        elif self._strategy_type == StrategyType.MEAN_REVERSION:
            base_features.extend(
                ["mean_calculation", "deviation_indicators", "bollinger_bands"]
            )
        elif self._strategy_type == StrategyType.BREAKOUT:
            base_features.extend(
                ["support_resistance", "volume_analysis", "volatility_indicators"]
            )
        elif self._strategy_type == StrategyType.SCALPING:
            base_features.extend(
                ["micro_price_movements", "order_book_analysis", "execution_speed"]
            )
        elif self._strategy_type == StrategyType.ARBITRAGE:
            base_features.extend(
                ["multi_exchange_data", "spread_analysis", "execution_latency"]
            )
        elif self._strategy_type == StrategyType.GRID:
            base_features.extend(
                ["price_levels", "position_sizing", "rebalancing_logic"]
            )
        elif self._strategy_type == StrategyType.MOMENTUM:
            base_features.extend(
                ["momentum_indicators", "relative_strength", "trend_following"]
            )
        elif self._strategy_type == StrategyType.VOLATILITY:
            base_features.extend(
                ["volatility_indicators", "regime_detection", "risk_adjustment"]
            )
        return base_features

    def is_active(self) -> bool:
        """
        Проверить, активна ли стратегия.
        Returns:
            bool: True если стратегия активна
        """
        return self._status == StrategyStatus.ACTIVE

    def activate(self) -> None:
        """Активировать стратегию."""
        self._status = StrategyStatus.ACTIVE
        self._updated_at = datetime.now()

    def deactivate(self) -> None:
        """Деактивировать стратегию."""
        self._status = StrategyStatus.INACTIVE
        self._updated_at = datetime.now()

    def pause(self) -> None:
        """Приостановить стратегию."""
        self._status = StrategyStatus.PAUSED
        self._updated_at = datetime.now()

    def get_performance(self) -> StrategyPerformance:
        """
        Получить производительность стратегии.
        Returns:
            StrategyPerformance: Производительность стратегии
        """
        return self._performance

    def update_performance(self, performance: StrategyPerformance) -> None:
        """
        Обновить производительность стратегии.
        Args:
            performance: Новая производительность
        """
        self._performance = performance
        self._updated_at = datetime.now()

    def get_parameters(self) -> Dict[str, Any]:
        """
        Получить параметры стратегии.
        Returns:
            Dict[str, Any]: Параметры стратегии
        """
        return self._parameters.copy()

    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Обновить параметры стратегии.
        Args:
            parameters: Новые параметры
        """
        if self.validate_parameters(parameters):
            self._parameters.update(parameters)
            self._updated_at = datetime.now()

    def get_trading_pairs(self) -> List[TradingPair]:
        """
        Получить торговые пары стратегии.
        Returns:
            List[TradingPair]: Список торговых пар
        """
        return self._trading_pairs.copy()

    def add_trading_pair(self, trading_pair: str) -> None:
        """
        Добавить торговую пару.
        Args:
            trading_pair: Торговая пара для добавления
        """
        pair = TradingPair(trading_pair)
        if pair not in self._trading_pairs:
            self._trading_pairs.append(pair)
            self._updated_at = datetime.now()

    def remove_trading_pair(self, trading_pair: str) -> None:
        """
        Удалить торговую пару.
        Args:
            trading_pair: Торговая пара для удаления
        """
        pair = TradingPair(trading_pair)
        if pair in self._trading_pairs:
            self._trading_pairs.remove(pair)
            self._updated_at = datetime.now()

    def get_metadata(self) -> Dict[str, Any]:
        """
        Получить метаданные стратегии.
        Returns:
            Dict[str, Any]: Метаданные стратегии
        """
        return self._metadata.copy()

    def set_metadata(self, key: str, value: Any) -> None:
        """
        Установить метаданные стратегии.
        Args:
            key: Ключ метаданных
            value: Значение метаданных
        """
        self._metadata[key] = value
        self._updated_at = datetime.now()

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Получить статистику выполнения стратегии.
        Returns:
            Dict[str, Any]: Статистика выполнения
        """
        return {
            "execution_count": self._execution_count,
            "success_count": self._success_count,
            "error_count": self._error_count,
            "success_rate": self._success_count / max(self._execution_count, 1),
            "last_analysis": (
                self._last_analysis.timestamp if self._last_analysis else None
            ),
            "signal_count": len(self._signal_history),
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразовать стратегию в словарь.
        Returns:
            Dict[str, Any]: Словарь с данными стратегии
        """
        return {
            "strategy_id": str(self._strategy_id),
            "name": self._name,
            "strategy_type": self._strategy_type.value,
            "trading_pairs": [str(pair) for pair in self._trading_pairs],
            "parameters": self._parameters,
            "risk_level": str(self._risk_level),
            "confidence_threshold": str(self._confidence_threshold),
            "status": self._status.value,
            "performance": self._performance.to_dict(),
            "created_at": self._created_at.isoformat(),
            "updated_at": self._updated_at.isoformat(),
            "metadata": self._metadata,
            "execution_stats": self.get_execution_stats(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyInterface":
        """
        Создать стратегию из словаря.
        Args:
            data: Данные стратегии
        Returns:
            StrategyInterface: Созданная стратегия
        """
        # Базовая реализация для создания стратегии из словаря
        strategy_id = StrategyId(UUID(data["strategy_id"]))
        name = data["name"]
        strategy_type = StrategyType(data["strategy_type"])
        trading_pairs = data["trading_pairs"]
        parameters = data["parameters"]
        risk_level = RiskLevel(Decimal(data["risk_level"]))
        confidence_threshold = ConfidenceLevel(Decimal(data["confidence_threshold"]))
        # Создаем экземпляр стратегии
        strategy = cls(
            strategy_id=strategy_id,
            name=name,
            strategy_type=strategy_type,
            trading_pairs=trading_pairs,
            parameters=parameters,
            risk_level=risk_level,
            confidence_threshold=confidence_threshold,
        )
        # Восстанавливаем дополнительные данные
        if "metadata" in data:
            strategy._metadata = data["metadata"]
        if "created_at" in data:
            strategy._created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            strategy._updated_at = datetime.fromisoformat(data["updated_at"])
        return strategy

    def _is_trading_pair_supported(self, symbol: str) -> bool:
        """Проверить, поддерживается ли торговая пара."""
        return TradingPair(symbol) in self._trading_pairs

    def _perform_market_analysis(
        self, market_data: MarketData
    ) -> StrategyAnalysisResult:
        """
        Выполнить анализ рынка.
        Args:
            market_data: Рыночные данные
        Returns:
            StrategyAnalysisResult: Результат анализа
        """
        # Базовая реализация анализа рынка
        price_analysis = self._analyze_price(market_data)
        volume_analysis = self._analyze_volume(market_data)
        trend_analysis = self._analyze_trend(market_data)
        volatility_analysis = self._analyze_volatility(market_data)
        risk_assessment = self._assess_risk(market_data)
        # Определяем режим рынка
        market_regime = self._determine_market_regime(market_data)
        # Рассчитываем уверенность
        confidence_score = self._calculate_confidence_score(market_data)
        # Технические индикаторы (базовая реализация)
        technical_indicators = {
            "sma_20": float(market_data.close.amount),  # Упрощенно
            "rsi": 50.0,  # Упрощенно
            "macd": 0.0,  # Упрощенно
            "bollinger_upper": float(market_data.high.amount),
            "bollinger_lower": float(market_data.low.amount),
        }
        # Поддержка и сопротивление (базовая реализация)
        support_resistance = (
            float(market_data.low.amount),
            float(market_data.high.amount),
        )
        # Индикаторы импульса
        momentum_indicators = {
            "price_momentum": price_analysis["price_change"],
            "volume_momentum": 0.0,  # Упрощенно
            "trend_strength": trend_analysis["trend_strength"],
        }
        # Распознавание паттернов (базовая реализация)
        pattern_recognition = []
        if price_analysis["price_change"] > 0.02:
            pattern_recognition.append("bullish_candle")
        elif price_analysis["price_change"] < -0.02:
            pattern_recognition.append("bearish_candle")
        # Настроения рынка
        market_sentiment = "neutral"
        if price_analysis["price_change"] > 0.01:
            market_sentiment = "bullish"
        elif price_analysis["price_change"] < -0.01:
            market_sentiment = "bearish"
        return StrategyAnalysisResult(
            confidence_score=confidence_score,
            trend_direction=trend_analysis["trend_direction"],
            trend_strength=trend_analysis["trend_strength"],
            volatility_level=volatility_analysis["volatility"],
            volume_analysis=volume_analysis,
            technical_indicators=technical_indicators,
            market_regime=market_regime,
            risk_assessment=risk_assessment,
            support_resistance=support_resistance,
            momentum_indicators=momentum_indicators,
            pattern_recognition=pattern_recognition,
            market_sentiment=market_sentiment,
        )

    def _generate_signal_by_type(
        self, market_data: MarketData, analysis: StrategyAnalysisResult
    ) -> Optional[Signal]:
        """
        Генерировать сигнал на основе типа стратегии.
        Args:
            market_data: Рыночные данные
            analysis: Результат анализа
        Returns:
            Optional[Signal]: Сгенерированный сигнал
        """
        # Базовая реализация генерации сигналов
        if analysis.confidence_score < self._confidence_threshold:
            return None
        # Определяем направление сигнала на основе анализа
        signal_type = SignalType.HOLD
        signal_strength = SignalStrength.MEDIUM
        if analysis.trend_direction == "up" and analysis.trend_strength > 0.6:
            signal_type = SignalType.BUY
            if analysis.trend_strength > 0.8:
                signal_type = SignalType.BUY
                signal_strength = SignalStrength.STRONG
        elif analysis.trend_direction == "down" and analysis.trend_strength > 0.6:
            signal_type = SignalType.SELL
            if analysis.trend_strength > 0.8:
                signal_type = SignalType.SELL
                signal_strength = SignalStrength.STRONG
        # Создаем сигнал
        signal = Signal(
            id=SignalId(uuid4()),
            strategy_id=self._strategy_id,
            trading_pair=market_data.symbol,
            signal_type=signal_type,
            price=Money(Decimal(str(market_data.close.amount)), Currency.USD),
            timestamp=market_data.timestamp,
            confidence=ConfidenceLevel(Decimal(str(analysis.confidence_score))),
            strength=signal_strength,
            metadata={
                "strategy_id": str(self._strategy_id),
                "strategy_type": self._strategy_type.value,
                "analysis": {
                    "trend_direction": analysis.trend_direction,
                    "trend_strength": analysis.trend_strength,
                    "volatility_level": analysis.volatility_level,
                    "market_regime": analysis.market_regime,
                    "market_sentiment": analysis.market_sentiment,
                },
            },
        )
        return signal

    def _get_required_parameters(self) -> List[str]:
        """Получить обязательные параметры стратегии."""
        return ["stop_loss", "take_profit", "position_size"]

    def _validate_parameter_types_and_ranges(self, parameters: Dict[str, Any]) -> bool:
        """Валидировать типы и диапазоны параметров."""
        try:
            # Проверка stop_loss
            if "stop_loss" in parameters:
                stop_loss = parameters["stop_loss"]
                if not isinstance(stop_loss, (int, float, Decimal)):
                    return False
                if stop_loss <= 0 or stop_loss > 1:
                    return False
            # Проверка take_profit
            if "take_profit" in parameters:
                take_profit = parameters["take_profit"]
                if not isinstance(take_profit, (int, float, Decimal)):
                    return False
                if take_profit <= 0 or take_profit > 10:
                    return False
            # Проверка position_size
            if "position_size" in parameters:
                position_size = parameters["position_size"]
                if not isinstance(position_size, (int, float, Decimal)):
                    return False
                if position_size <= 0 or position_size > 1:
                    return False
            return True
        except Exception:
            return False

    def _analyze_price(self, market_data: MarketData) -> Dict[str, Any]:
        """Анализ цены."""
        price_change = (
            market_data.close.amount - market_data.open.amount
        ) / market_data.open.amount
        return {
            "price_change": float(price_change),
            "price_range": float(market_data.high.amount - market_data.low.amount),
            "open_close_ratio": float(
                market_data.close.amount / market_data.open.amount
            ),
        }

    def _analyze_volume(self, market_data: MarketData) -> Dict[str, Any]:
        """Анализ объема."""
        return {
            "volume": float(market_data.volume.amount),
            "volume_sma_ratio": 1.0,  # Упрощенно
            "volume_trend": "stable",
        }

    def _analyze_trend(self, market_data: MarketData) -> Dict[str, Any]:
        """Анализ тренда."""
        price_change = (
            market_data.close.amount - market_data.open.amount
        ) / market_data.open.amount
        if price_change > 0.01:
            trend_direction = "up"
            trend_strength = min(abs(price_change) * 10, 1.0)
        elif price_change < -0.01:
            trend_direction = "down"
            trend_strength = min(abs(price_change) * 10, 1.0)
        else:
            trend_direction = "sideways"
            trend_strength = 0.1
        return {
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "price_momentum": float(price_change),
        }

    def _analyze_volatility(self, market_data: MarketData) -> Dict[str, Any]:
        """Анализ волатильности."""
        price_range = (
            market_data.high.amount - market_data.low.amount
        ) / market_data.open.amount
        return {
            "volatility": float(price_range),
            "volatility_level": (
                "high"
                if price_range > 0.05
                else "medium" if price_range > 0.02 else "low"
            ),
        }

    def _determine_market_regime(self, market_data: MarketData) -> str:
        """Определить режим рынка."""
        price_change = abs(
            (market_data.close.amount - market_data.open.amount)
            / market_data.open.amount
        )
        price_range = (
            market_data.high.amount - market_data.low.amount
        ) / market_data.open.amount
        if price_range > 0.05:
            return "volatile"
        elif price_change > 0.02:
            return "trending"
        else:
            return "ranging"

    def _calculate_confidence_score(self, market_data: MarketData) -> float:
        """Рассчитать оценку уверенности."""
        # Базовая реализация - должна быть переопределена в дочерних классах
        return 0.5

    def _assess_risk(self, market_data: MarketData) -> Dict[str, Any]:
        """Оценить риски."""
        volatility = (
            market_data.high.amount - market_data.low.amount
        ) / market_data.open.amount
        return {
            "volatility_risk": float(volatility),
            "price_risk": "medium",
            "volume_risk": "low",
        }
