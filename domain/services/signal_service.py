"""
Доменный сервис для работы с сигналами.
Содержит бизнес-логику для генерации, валидации, агрегации
и анализа торговых сигналов.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
)

from domain.entities.signal import Signal, SignalStrength, SignalType
from domain.entities.strategy import Strategy
from domain.types import MarketDataFrame
from domain.types.signal_types import (
    SignalAggregationResult,
    SignalAnalysisResult,
    SignalValidationResult,
)

# Определяем тип для асинхронных генераторов сигналов
AsyncSignalGenerator = Callable[
    [Strategy, MarketDataFrame], Coroutine[Any, Any, List[Signal]]
]


class SignalService(ABC):
    """
    Абстрактный сервис для работы с сигналами.
    Определяет интерфейс для всех операций с сигналами,
    включая генерацию, валидацию, агрегацию и анализ.
    """

    @abstractmethod
    async def generate_signals(
        self, strategy: Strategy, market_data: MarketDataFrame
    ) -> List[Signal]:
        """
        Генерация сигналов для стратегии.
        Args:
            strategy: Стратегия для генерации сигналов
            market_data: Рыночные данные
        Returns:
            List[Signal]: Список сгенерированных сигналов
        """
        pass

    @abstractmethod
    async def validate_signal(self, signal: Signal) -> SignalValidationResult:
        """
        Валидация сигнала.
        Args:
            signal: Сигнал для валидации
        Returns:
            SignalValidationResult: Результат валидации
        """
        pass

    @abstractmethod
    async def aggregate_signals(self, signals: List[Signal]) -> SignalAggregationResult:
        """
        Агрегация множественных сигналов.
        Args:
            signals: Список сигналов для агрегации
        Returns:
            SignalAggregationResult: Результат агрегации
        """
        pass

    @abstractmethod
    async def analyze_signals(
        self, signals: List[Signal], period: timedelta
    ) -> SignalAnalysisResult:
        """
        Анализ сигналов.
        Args:
            signals: Список сигналов для анализа
            period: Период анализа
        Returns:
            SignalAnalysisResult: Результат анализа
        """
        pass


@dataclass
class SignalGenerationContext:
    """Контекст для генерации сигналов."""

    strategy: Strategy
    market_data: MarketDataFrame
    current_price: float
    volume: float
    volatility: float
    trend_direction: str
    support_level: float
    resistance_level: float


@dataclass
class AggregationWeights:
    """Веса для агрегации сигналов."""

    confidence_weight: float = 0.4
    strength_weight: float = 0.3
    recency_weight: float = 0.2
    volume_weight: float = 0.1
    min_confidence: float = 0.3
    max_signals: int = 10


class DefaultSignalService(SignalService):
    """
    Промышленная реализация сервиса сигналов.
    Предоставляет полную реализацию всех методов для работы
    с сигналами с использованием строгой типизации.
    """

    def __init__(self) -> None:
        """Инициализация сервиса."""
        self._signal_generators: Dict[str, AsyncSignalGenerator] = (
            self._setup_signal_generators()
        )
        self._aggregation_weights: AggregationWeights = AggregationWeights()

    def _setup_signal_generators(self) -> Dict[str, AsyncSignalGenerator]:
        """Настройка генераторов сигналов."""
        return {
            "trend_following": self._generate_trend_signals,
            "mean_reversion": self._generate_mean_reversion_signals,
            "breakout": self._generate_breakout_signals,
            "scalping": self._generate_scalping_signals,
            "arbitrage": self._generate_arbitrage_signals,
            "grid": self._generate_grid_signals,
            "momentum": self._generate_momentum_signals,
            "volatility": self._generate_volatility_signals,
        }

    async def generate_signals(
        self, strategy: Strategy, market_data: MarketDataFrame
    ) -> List[Signal]:
        """
        Генерация сигналов для стратегии.
        Args:
            strategy: Стратегия для генерации сигналов
            market_data: Рыночные данные
        Returns:
            List[Signal]: Список сгенерированных сигналов
        """
        strategy_type = strategy.strategy_type.value
        if strategy_type not in self._signal_generators:
            return []
        generator = self._signal_generators[strategy_type]
        signals = await generator(strategy, market_data)
        # Фильтрация сигналов по параметрам стратегии
        filtered_signals = await self._filter_signals(signals, strategy)
        return filtered_signals

    async def _generate_trend_signals(
        self, strategy: Strategy, market_data: MarketDataFrame
    ) -> List[Signal]:
        """Генерация трендовых сигналов."""
        signals: List[Signal] = []
        parameters = strategy.parameters.parameters
        # Безопасное извлечение параметров с приведением типов
        trend_strength_param = parameters.get("trend_strength", 0.7)
        trend_period_param = parameters.get("trend_period", 20)
        if isinstance(trend_strength_param, (int, float, str)):
            trend_strength = float(trend_strength_param)
        else:
            trend_strength = 0.7
        if isinstance(trend_period_param, (int, str)):
            trend_period = int(trend_period_param)
        else:
            trend_period = 20
        # Анализ тренда на основе рыночных данных
        if hasattr(market_data, 'columns') and 'close' in market_data.columns:
            price_data = market_data['close'].tolist()
            if len(price_data) >= trend_period:
                recent_prices = price_data[-trend_period:]
                trend_direction = self._calculate_trend_direction(recent_prices)
                if trend_direction == "up" and trend_strength > 0.6:
                    signal = Signal(
                        strategy_id=strategy.id,
                        trading_pair=strategy.trading_pairs[0],
                        signal_type=SignalType.BUY,
                        strength=(
                            SignalStrength.STRONG
                            if trend_strength > 0.8
                            else SignalStrength.MEDIUM
                        ),
                        confidence=Decimal(str(trend_strength)),
                    )
                    signals.append(signal)
                elif trend_direction == "down" and trend_strength > 0.6:
                    signal = Signal(
                        strategy_id=strategy.id,
                        trading_pair=strategy.trading_pairs[0],
                        signal_type=SignalType.SELL,
                        strength=(
                            SignalStrength.STRONG
                            if trend_strength > 0.8
                            else SignalStrength.MEDIUM
                        ),
                        confidence=Decimal(str(trend_strength)),
                    )
                    signals.append(signal)
        return signals

    def _calculate_trend_direction(self, prices: List[float]) -> str:
        """Расчет направления тренда."""
        if len(prices) < 2:
            return "sideways"
        first_price = prices[0]
        last_price = prices[-1]
        if last_price > first_price * 1.01:
            return "up"
        elif last_price < first_price * 0.99:
            return "down"
        else:
            return "sideways"

    async def _generate_mean_reversion_signals(
        self, strategy: Strategy, market_data: MarketDataFrame
    ) -> List[Signal]:
        """Генерация сигналов возврата к среднему."""
        signals: List[Signal] = []
        parameters = strategy.parameters.parameters
        # Безопасное извлечение параметров
        reversion_strength_param = parameters.get("reversion_strength", 0.6)
        reversion_period_param = parameters.get("reversion_period", 20)
        if isinstance(reversion_strength_param, (int, float, str)):
            reversion_strength = float(reversion_strength_param)
        else:
            reversion_strength = 0.6
        if isinstance(reversion_period_param, (int, str)):
            reversion_period = int(reversion_period_param)
        else:
            reversion_period = 20
        # Анализ возврата к среднему
        if hasattr(market_data, 'columns') and 'close' in market_data.columns:
            price_data = market_data['close'].tolist()
            if len(price_data) >= reversion_period:
                recent_prices = price_data[-reversion_period:]
                mean_price = sum(recent_prices) / len(recent_prices)
                current_price = recent_prices[-1]
                deviation = abs(current_price - mean_price) / mean_price
                if deviation > 0.02:  # 2% отклонение
                    if current_price > mean_price:
                        signal = Signal(
                            strategy_id=strategy.id,
                            trading_pair=strategy.trading_pairs[0],
                            signal_type=SignalType.SELL,
                            strength=SignalStrength.MEDIUM,
                            confidence=Decimal(str(reversion_strength)),
                        )
                        signals.append(signal)
                    else:
                        signal = Signal(
                            strategy_id=strategy.id,
                            trading_pair=strategy.trading_pairs[0],
                            signal_type=SignalType.BUY,
                            strength=SignalStrength.MEDIUM,
                            confidence=Decimal(str(reversion_strength)),
                        )
                        signals.append(signal)
        return signals

    async def _generate_breakout_signals(
        self, strategy: Strategy, market_data: MarketDataFrame
    ) -> List[Signal]:
        """Генерация сигналов пробоя."""
        signals: List[Signal] = []
        parameters = strategy.parameters.parameters
        # Безопасное извлечение параметров
        breakout_strength_param = parameters.get("breakout_strength", 0.7)
        breakout_period_param = parameters.get("breakout_period", 20)
        if isinstance(breakout_strength_param, (int, float, str)):
            breakout_strength = float(breakout_strength_param)
        else:
            breakout_strength = 0.7
        if isinstance(breakout_period_param, (int, str)):
            breakout_period = int(breakout_period_param)
        else:
            breakout_period = 20
        # Анализ пробоев
        if hasattr(market_data, 'columns') and 'close' in market_data.columns:
            price_data = market_data['close'].tolist()
            if len(price_data) >= breakout_period:
                recent_prices = price_data[-breakout_period:]
                resistance = max(recent_prices[:-1])
                support = min(recent_prices[:-1])
                current_price = recent_prices[-1]
                if current_price > resistance * 1.01:
                    signal = Signal(
                        strategy_id=strategy.id,
                        trading_pair=strategy.trading_pairs[0],
                        signal_type=SignalType.BUY,
                        strength=SignalStrength.STRONG,
                        confidence=Decimal(str(breakout_strength)),
                    )
                    signals.append(signal)
                elif current_price < support * 0.99:
                    signal = Signal(
                        strategy_id=strategy.id,
                        trading_pair=strategy.trading_pairs[0],
                        signal_type=SignalType.SELL,
                        strength=SignalStrength.STRONG,
                        confidence=Decimal(str(breakout_strength)),
                    )
                    signals.append(signal)
        return signals

    async def _generate_scalping_signals(
        self, strategy: Strategy, market_data: MarketDataFrame
    ) -> List[Signal]:
        """Генерация скальпинг сигналов."""
        signals: List[Signal] = []
        parameters = strategy.parameters.parameters
        # Безопасное извлечение параметров
        scalping_strength_param = parameters.get("scalping_strength", 0.5)
        scalping_period_param = parameters.get("scalping_period", 5)
        if isinstance(scalping_strength_param, (int, float, str)):
            scalping_strength = float(scalping_strength_param)
        else:
            scalping_strength = 0.5
        if isinstance(scalping_period_param, (int, str)):
            scalping_period = int(scalping_period_param)
        else:
            scalping_period = 5
        # Анализ для скальпинга
        if hasattr(market_data, 'columns') and 'close' in market_data.columns:
            price_data = market_data['close'].tolist()
            if len(price_data) >= scalping_period:
                recent_prices = price_data[-scalping_period:]
                price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                if abs(price_change) > 0.001:  # 0.1% изменение
                    if price_change > 0:
                        signal = Signal(
                            strategy_id=strategy.id,
                            trading_pair=strategy.trading_pairs[0],
                            signal_type=SignalType.BUY,
                            strength=SignalStrength.WEAK,
                            confidence=Decimal(str(scalping_strength)),
                        )
                        signals.append(signal)
                    else:
                        signal = Signal(
                            strategy_id=strategy.id,
                            trading_pair=strategy.trading_pairs[0],
                            signal_type=SignalType.SELL,
                            strength=SignalStrength.WEAK,
                            confidence=Decimal(str(scalping_strength)),
                        )
                        signals.append(signal)
        return signals

    async def _generate_arbitrage_signals(
        self, strategy: Strategy, market_data: MarketDataFrame
    ) -> List[Signal]:
        """Генерация арбитражных сигналов."""
        signals: List[Signal] = []
        parameters = strategy.parameters.parameters
        # Безопасное извлечение параметров
        arbitrage_strength_param = parameters.get("arbitrage_strength", 0.8)
        min_spread_param = parameters.get("min_spread", 0.001)
        if isinstance(arbitrage_strength_param, (int, float, str)):
            arbitrage_strength = float(arbitrage_strength_param)
        else:
            arbitrage_strength = 0.8
        if isinstance(min_spread_param, (int, float, str)):
            min_spread = float(min_spread_param)
        else:
            min_spread = 0.001
        # Анализ арбитража (упрощенно)
        if hasattr(market_data, 'columns') and 'close' in market_data.columns:
            price_data = market_data['close'].tolist()
            if len(price_data) >= 2:
                current_price = price_data[-1]
                prev_price = price_data[-2]
                spread = abs(current_price - prev_price) / prev_price
                if spread > min_spread:
                    signal = Signal(
                        strategy_id=strategy.id,
                        trading_pair=strategy.trading_pairs[0],
                        signal_type=SignalType.BUY,
                        strength=SignalStrength.STRONG,
                        confidence=Decimal(str(arbitrage_strength)),
                    )
                    signals.append(signal)
        return signals

    async def _generate_grid_signals(
        self, strategy: Strategy, market_data: MarketDataFrame
    ) -> List[Signal]:
        """Генерация сеточных сигналов."""
        signals: List[Signal] = []
        parameters = strategy.parameters.parameters
        # Безопасное извлечение параметров
        grid_strength_param = parameters.get("grid_strength", 0.6)
        grid_levels_param = parameters.get("grid_levels", 5)
        if isinstance(grid_strength_param, (int, float, str)):
            grid_strength = float(grid_strength_param)
        else:
            grid_strength = 0.6
        if isinstance(grid_levels_param, (int, str)):
            grid_levels = int(grid_levels_param)
        else:
            grid_levels = 5
        # Анализ для сеточной торговли
        if hasattr(market_data, 'columns') and 'close' in market_data.columns:
            price_data = market_data['close'].tolist()
            if len(price_data) >= grid_levels:
                current_price = price_data[-1]
                avg_price = sum(price_data[-grid_levels:]) / grid_levels
                if current_price < avg_price:
                    signal = Signal(
                        strategy_id=strategy.id,
                        trading_pair=strategy.trading_pairs[0],
                        signal_type=SignalType.BUY,
                        strength=SignalStrength.MEDIUM,
                        confidence=Decimal(str(grid_strength)),
                    )
                    signals.append(signal)
                elif current_price > avg_price:
                    signal = Signal(
                        strategy_id=strategy.id,
                        trading_pair=strategy.trading_pairs[0],
                        signal_type=SignalType.SELL,
                        strength=SignalStrength.MEDIUM,
                        confidence=Decimal(str(grid_strength)),
                    )
                    signals.append(signal)
        return signals

    async def _generate_momentum_signals(
        self, strategy: Strategy, market_data: MarketDataFrame
    ) -> List[Signal]:
        """Генерация импульсных сигналов."""
        signals: List[Signal] = []
        parameters = strategy.parameters.parameters
        # Безопасное извлечение параметров
        momentum_strength_param = parameters.get("momentum_strength", 0.7)
        momentum_period_param = parameters.get("momentum_period", 14)
        if isinstance(momentum_strength_param, (int, float, str)):
            momentum_strength = float(momentum_strength_param)
        else:
            momentum_strength = 0.7
        if isinstance(momentum_period_param, (int, str)):
            momentum_period = int(momentum_period_param)
        else:
            momentum_period = 14
        # Анализ импульса
        if hasattr(market_data, 'columns') and 'close' in market_data.columns:
            price_data = market_data['close'].tolist()
            if len(price_data) >= momentum_period:
                recent_prices = price_data[-momentum_period:]
                momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                if momentum > 0.02:  # 2% импульс
                    signal = Signal(
                        strategy_id=strategy.id,
                        trading_pair=strategy.trading_pairs[0],
                        signal_type=SignalType.BUY,
                        strength=SignalStrength.STRONG,
                        confidence=Decimal(str(momentum_strength)),
                    )
                    signals.append(signal)
                elif momentum < -0.02:
                    signal = Signal(
                        strategy_id=strategy.id,
                        trading_pair=strategy.trading_pairs[0],
                        signal_type=SignalType.SELL,
                        strength=SignalStrength.STRONG,
                        confidence=Decimal(str(momentum_strength)),
                    )
                    signals.append(signal)
        return signals

    async def _generate_volatility_signals(
        self, strategy: Strategy, market_data: MarketDataFrame
    ) -> List[Signal]:
        """Генерация сигналов на основе волатильности."""
        signals: List[Signal] = []
        parameters = strategy.parameters.parameters
        # Безопасное извлечение параметров
        volatility_strength_param = parameters.get("volatility_strength", 0.6)
        volatility_period_param = parameters.get("volatility_period", 20)
        if isinstance(volatility_strength_param, (int, float, str)):
            volatility_strength = float(volatility_strength_param)
        else:
            volatility_strength = 0.6
        if isinstance(volatility_period_param, (int, str)):
            volatility_period = int(volatility_period_param)
        else:
            volatility_period = 20
        # Анализ волатильности
        if hasattr(market_data, 'columns') and 'close' in market_data.columns:
            price_data = market_data['close'].tolist()
            if len(price_data) >= volatility_period:
                recent_prices = price_data[-volatility_period:]
                returns = [
                    (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
                    for i in range(1, len(recent_prices))
                ]
                volatility = (sum(r*r for r in returns) / len(returns)) ** 0.5
                if volatility > 0.02:  # 2% волатильность
                    signal = Signal(
                        strategy_id=strategy.id,
                        trading_pair=strategy.trading_pairs[0],
                        signal_type=SignalType.BUY,
                        strength=SignalStrength.MEDIUM,
                        confidence=Decimal(str(volatility_strength)),
                    )
                    signals.append(signal)
        return signals

    async def _filter_signals(
        self, signals: List[Signal], strategy: Strategy
    ) -> List[Signal]:
        """Фильтрация сигналов по параметрам стратегии."""
        filtered_signals = []
        parameters = strategy.parameters.parameters
        
        # Безопасное извлечение параметров
        min_confidence_param = parameters.get("min_confidence", 0.3)
        max_signals_param = parameters.get("max_signals", 10)
        
        if isinstance(min_confidence_param, (int, float, str)):
            min_confidence = float(min_confidence_param)
        else:
            min_confidence = 0.3
            
        if isinstance(max_signals_param, (int, str)):
            max_signals = int(max_signals_param)
        else:
            max_signals = 10

        for signal in signals:
            if signal.confidence >= Decimal(str(min_confidence)):
                filtered_signals.append(signal)
                if len(filtered_signals) >= max_signals:
                    break

        return filtered_signals

    def _validate_signal_data(self, signal_data: Dict[str, Any]) -> bool:
        """Валидация данных сигнала."""
        return True

    def _process_signal(self, signal: Any) -> Dict[str, Any]:
        """Обработка сигнала."""
        return {}

    async def validate_signal(self, signal: Signal) -> SignalValidationResult:
        """
        Валидация сигнала.
        Args:
            signal: Сигнал для валидации
        Returns:
            SignalValidationResult: Результат валидации
        """
        # Базовая валидация
        if not signal.strategy_id:
            return SignalValidationResult(
                is_valid=False,
                errors=["Missing strategy_id"],
                warnings=[],
            )

        if not signal.trading_pair:
            return SignalValidationResult(
                is_valid=False,
                errors=["Missing trading_pair"],
                warnings=[],
            )

        if signal.confidence < 0 or signal.confidence > 1:
            return SignalValidationResult(
                is_valid=False,
                errors=["Invalid confidence value"],
                warnings=[],
            )

        # Проверка силы сигнала
        warnings = []
        if signal.strength == SignalStrength.WEAK and signal.confidence > 0.8:
            warnings.append("High confidence with weak signal strength")

        if signal.strength == SignalStrength.STRONG and signal.confidence < 0.5:
            warnings.append("Low confidence with strong signal strength")

        return SignalValidationResult(
            is_valid=True,
            errors=[],
            warnings=warnings,
        )

    async def aggregate_signals(self, signals: List[Signal]) -> SignalAggregationResult:
        """
        Агрегация сигналов.
        Args:
            signals: Список сигналов для агрегации
        Returns:
            SignalAggregationResult: Результат агрегации
        """
        if not signals:
            return SignalAggregationResult(
                aggregated_signal=None,
                confidence=0.0,
                strength=SignalStrength.WEAK,
                metadata={},
            )

        # Фильтрация по минимальной уверенности
        valid_signals = [
            s for s in signals if s.confidence >= self._aggregation_weights.min_confidence
        ]

        if not valid_signals:
            return SignalAggregationResult(
                aggregated_signal=None,
                confidence=0.0,
                strength=SignalStrength.WEAK,
                metadata={"reason": "No signals meet minimum confidence"},
            )

        # Ограничение количества сигналов
        if len(valid_signals) > self._aggregation_weights.max_signals:
            valid_signals = sorted(
                valid_signals, key=lambda s: s.confidence, reverse=True
            )[: self._aggregation_weights.max_signals]

        # Расчет взвешенной уверенности
        weighted_confidence = 0.0
        total_weight = 0.0

        for signal in valid_signals:
            weight = (
                float(signal.confidence) * self._aggregation_weights.confidence_weight
                + self._get_strength_value(signal.strength)
                * self._aggregation_weights.strength_weight
            )
            weighted_confidence += float(signal.confidence) * weight
            total_weight += weight

        if total_weight > 0:
            final_confidence = float(weighted_confidence / total_weight)
        else:
            final_confidence = 0.0

        # Определение итоговой силы сигнала
        final_strength = self._determine_aggregated_strength(valid_signals)

        # Создание агрегированного сигнала
        if valid_signals:
            base_signal = valid_signals[0]
            aggregated_signal = Signal(
                strategy_id=base_signal.strategy_id,
                trading_pair=base_signal.trading_pair,
                signal_type=self._determine_aggregated_type(valid_signals),
                strength=final_strength,
                confidence=Decimal(str(final_confidence)),
            )
        else:
            aggregated_signal = None

        metadata = {
            "total_signals": len(signals),
            "valid_signals": len(valid_signals),
            "signal_types": list(set(s.signal_type.value for s in valid_signals)),
            "strength_distribution": self._get_strength_distribution(valid_signals),
        }

        return SignalAggregationResult(
            aggregated_signal=aggregated_signal,
            confidence=final_confidence,
            strength=final_strength,
            metadata=metadata,
        )

    def _get_strength_value(self, strength: SignalStrength) -> float:
        """Получение числового значения силы сигнала."""
        strength_values = {
            SignalStrength.WEAK: 0.3,
            SignalStrength.MEDIUM: 0.6,
            SignalStrength.STRONG: 1.0,
        }
        return strength_values.get(strength, 0.5)

    def _determine_aggregated_strength(self, signals: List[Signal]) -> SignalStrength:
        """Определение итоговой силы сигнала на основе списка сигналов."""
        if not signals:
            return SignalStrength.WEAK
        
        strength_counts = {
            SignalStrength.WEAK: 0,
            SignalStrength.MEDIUM: 0,
            SignalStrength.STRONG: 0,
        }
        
        for signal in signals:
            strength_counts[signal.strength] += 1
        
        # Возвращаем наиболее часто встречающуюся силу
        max_count = max(strength_counts.values())
        for strength, count in strength_counts.items():
            if count == max_count:
                return strength
        
        return SignalStrength.MEDIUM

    def _determine_aggregated_type(self, signals: List[Signal]) -> SignalType:
        """Определение итогового типа сигнала на основе списка сигналов."""
        if not signals:
            return SignalType.HOLD
        
        type_counts = {
            SignalType.BUY: 0,
            SignalType.SELL: 0,
            SignalType.HOLD: 0,
            SignalType.CLOSE: 0,
        }
        
        for signal in signals:
            type_counts[signal.signal_type] += 1
        
        # Возвращаем наиболее часто встречающийся тип
        max_count = max(type_counts.values())
        for signal_type, count in type_counts.items():
            if count == max_count:
                return signal_type
        
        return SignalType.HOLD

    def _get_strength_distribution(self, signals: List[Signal]) -> Dict[str, int]:
        """Получение распределения силы сигналов."""
        distribution = {
            SignalStrength.WEAK.value: 0,
            SignalStrength.MEDIUM.value: 0,
            SignalStrength.STRONG.value: 0,
        }
        
        for signal in signals:
            distribution[signal.strength.value] += 1
        
        return distribution

    async def analyze_signals(
        self, signals: List[Signal], period: timedelta
    ) -> SignalAnalysisResult:
        """
        Анализ сигналов.
        Args:
            signals: Список сигналов для анализа
            period: Период анализа
        Returns:
            SignalAnalysisResult: Результат анализа
        """
        if not signals:
            return SignalAnalysisResult(
                total_signals=0,
                recent_signals=0,
                signal_distribution={},
                avg_confidence=0.0,
                strength_distribution={},
                period_analysis={},
                stats={},
                recommendations=[],
                distribution={}
            )

        # Базовый анализ
        total_signals = len(signals)
        
        # Распределение по типам сигналов
        signal_distribution: Dict[str, int] = {}
        for signal in signals:
            signal_type = signal.signal_type.value
            signal_distribution[signal_type] = signal_distribution.get(signal_type, 0) + 1

        # Средняя уверенность
        total_confidence = sum(float(s.confidence) for s in signals)
        average_confidence = total_confidence / total_signals

        # Распределение по силе сигналов
        strength_distribution: Dict[str, int] = {}
        for signal in signals:
            strength = signal.strength.value
            strength_distribution[strength] = strength_distribution.get(strength, 0) + 1

        # Анализ по периодам
        now = datetime.now()
        recent_signals = [
            s for s in signals 
            if (now - s.timestamp) <= period
        ]

        period_analysis = {
            "recent_signals": len(recent_signals),
            "period_duration_hours": period.total_seconds() / 3600,
            "signals_per_hour": len(recent_signals) / (period.total_seconds() / 3600) if period.total_seconds() > 0 else 0
        }

        # Генерация рекомендаций
        recommendations = self._generate_signal_recommendations(
            signals, signal_distribution, average_confidence
        )

        # Дополнительная статистика
        stats = {
            "total_signals": total_signals,
            "recent_signals": len(recent_signals),
            "average_confidence": average_confidence,
            "period_duration_hours": period.total_seconds() / 3600,
        }

        return SignalAnalysisResult(
            total_signals=total_signals,
            recent_signals=len(recent_signals),
            signal_distribution=signal_distribution,
            avg_confidence=average_confidence,
            strength_distribution=strength_distribution,
            period_analysis=period_analysis,
            stats=stats,
            recommendations=recommendations,
            distribution=signal_distribution
        )

    def _generate_signal_recommendations(
        self, signals: List[Signal], distribution: Dict[str, int], avg_confidence: float
    ) -> List[str]:
        """Генерация рекомендаций на основе анализа сигналов."""
        recommendations = []

        if avg_confidence < 0.5:
            recommendations.append("Consider reducing position sizes due to low confidence")

        if distribution.get("buy", 0) > distribution.get("sell", 0) * 2:
            recommendations.append("Strong bullish bias detected - consider taking profits")

        if distribution.get("sell", 0) > distribution.get("buy", 0) * 2:
            recommendations.append("Strong bearish bias detected - consider covering shorts")

        if len(signals) > 20:
            recommendations.append("High signal frequency - consider filtering criteria")

        return recommendations


# Экспорт интерфейса для обратной совместимости
ISignalService = SignalService
