"""Модуль для обработки торговых сигналов.

Этот модуль содержит классы для обработки и анализа торговых сигналов,
включая расчет уверенности, размера позиции и метрик риска.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

import pandas as pd  # type: ignore
from loguru import logger


@dataclass
class Signal:
    """Расширенный класс сигнала."""

    timestamp: datetime
    symbol: str
    direction: str  # buy, sell, hold
    strength: float  # -1 to 1
    source: str
    confidence: float  # 0 to 1
    timeframe: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    expected_duration: Optional[int] = None  # в барах
    priority: int = 0  # приоритет сигнала
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Инициализация после создания объекта."""
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразование в словарь.

        Returns:
            Dict[str, Any]: Словарь с данными сигнала
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "direction": self.direction,
            "strength": self.strength,
            "source": self.source,
            "confidence": self.confidence,
            "timeframe": self.timeframe,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_reward_ratio": self.risk_reward_ratio,
            "expected_duration": self.expected_duration,
            "priority": self.priority,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Signal":
        """
        Создание из словаря с проверкой ключей.

        Args:
            data: Словарь с данными сигнала

        Returns:
            Signal: Объект сигнала

        Raises:
            ValueError: При отсутствии обязательных полей
        """
        required_fields = ["entry_price", "stop_loss", "take_profit"]
        missing = [f for f in required_fields if f not in data or data[f] is None]
        if missing:
            logger.warning(f"Пропуск сигнала: отсутствуют ключи {missing} в {data}")
            raise ValueError(f"Signal missing required fields: {missing}")
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class MarketContext:
    """Расширенный контекст рынка."""

    volatility: float
    trend: str
    volume: float
    indicators: Dict[str, float]
    market_regime: str
    liquidity: float
    momentum: float
    sentiment: float
    support_levels: List[float]
    resistance_levels: List[float]
    market_depth: Dict[str, float]
    correlation_matrix: Dict[str, Dict[str, float]]
    market_impact: float
    volume_profile: Dict[float, float]

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразование в словарь.

        Returns:
            Dict[str, Any]: Словарь с данными контекста
        """
        return {
            "volatility": self.volatility,
            "trend": self.trend,
            "volume": self.volume,
            "indicators": self.indicators,
            "market_regime": self.market_regime,
            "liquidity": self.liquidity,
            "momentum": self.momentum,
            "sentiment": self.sentiment,
            "support_levels": self.support_levels,
            "resistance_levels": self.resistance_levels,
            "market_depth": self.market_depth,
            "correlation_matrix": self.correlation_matrix,
            "market_impact": self.market_impact,
            "volume_profile": self.volume_profile,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketContext":
        """
        Создание из словаря.

        Args:
            data: Словарь с данными контекста

        Returns:
            MarketContext: Объект контекста рынка
        """
        return cls(**data)


@dataclass
class ProcessedSignal:
    """Расширенный обработанный сигнал."""

    signal: Signal
    context: MarketContext
    confidence: float
    position_size: float
    risk_metrics: Dict[str, float]
    execution_priority: int
    expected_impact: float
    correlation_impact: Dict[str, float]
    market_impact: float
    execution_time: Optional[datetime] = None
    execution_price: Optional[float] = None
    execution_status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразование в словарь.

        Returns:
            Dict[str, Any]: Словарь с данными обработанного сигнала
        """
        return {
            "signal": self.signal.to_dict(),
            "context": self.context.to_dict(),
            "confidence": self.confidence,
            "position_size": self.position_size,
            "risk_metrics": self.risk_metrics,
            "execution_priority": self.execution_priority,
            "expected_impact": self.expected_impact,
            "correlation_impact": self.correlation_impact,
            "market_impact": self.market_impact,
            "execution_time": (
                self.execution_time.isoformat() if self.execution_time else None
            ),
            "execution_price": self.execution_price,
            "execution_status": self.execution_status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessedSignal":
        """
        Создание из словаря.

        Args:
            data: Словарь с данными обработанного сигнала

        Returns:
            ProcessedSignal: Объект обработанного сигнала
        """
        data["signal"] = Signal.from_dict(data["signal"])
        data["context"] = MarketContext.from_dict(data["context"])
        if data["execution_time"]:
            data["execution_time"] = datetime.fromisoformat(data["execution_time"])
        return cls(**data)


class SignalProcessor:
    """Расширенный процессор сигналов."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация процессора сигналов.

        Args:
            config: Конфигурация процессора
        """
        self.config: Dict[str, Any] = {}
        if config is not None:
            self.config = config
        self.signals: List[Signal] = []
        self.metadata: Dict[str, Any] = {}
        self.last_update: Optional[datetime] = None
        self.min_confidence = self.config.get("min_confidence", 0.6)
        self.max_position_size = self.config.get("max_position_size", 1.0)
        self.risk_per_trade = self.config.get("risk_per_trade", 0.02)
        self.correlation_threshold = self.config.get("correlation_threshold", 0.7)
        self.impact_threshold = self.config.get("impact_threshold", 0.1)
        self.priority_weights = self.config.get(
            "priority_weights",
            {
                "strength": 0.3,
                "confidence": 0.3,
                "risk_reward": 0.2,
                "market_regime": 0.1,
                "liquidity": 0.1,
            },
        )
        self._initialize()

    def _initialize(self) -> None:
        """Дополнительная инициализация."""

    def process_signal(
        self, signal: Signal, context: MarketContext
    ) -> Optional[ProcessedSignal]:
        """
        Обработка сигнала с расширенной логикой.

        Args:
            signal: Входной сигнал
            context: Контекст рынка

        Returns:
            Optional[ProcessedSignal]: Обработанный сигнал или None
        """
        try:
            # Расчет базовой уверенности
            confidence = self._calculate_confidence(signal, context)

            # Проверка минимальной уверенности
            if confidence < self.min_confidence:
                logger.debug(
                    f"Signal rejected: confidence {confidence} < "
                    f"{self.min_confidence}"
                )
                return None

            # Расчет размера позиции
            position_size = self._calculate_position_size(signal, context, confidence)

            # Расчет метрик риска
            risk_metrics = self._calculate_risk_metrics(signal, context, position_size)

            # Расчет приоритета исполнения
            execution_priority = self._calculate_execution_priority(
                signal, context, confidence
            )

            # Расчет ожидаемого влияния
            expected_impact = self._calculate_expected_impact(
                signal, context, position_size
            )

            # Расчет влияния на коррелированные инструменты
            correlation_impact = self._calculate_correlation_impact(signal, context)

            # Расчет влияния на рынок
            market_impact = self._calculate_market_impact(
                signal, context, position_size
            )

            # Создание обработанного сигнала
            return ProcessedSignal(
                signal=signal,
                context=context,
                confidence=confidence,
                position_size=position_size,
                risk_metrics=risk_metrics,
                execution_priority=execution_priority,
                expected_impact=expected_impact,
                correlation_impact=correlation_impact,
                market_impact=market_impact,
            )

        except Exception as e:
            logger.error(f"Error processing signal: {str(e)}")
            return None

    def _calculate_confidence(self, signal: Signal, context: MarketContext) -> float:
        """
        Расчет уверенности в сигнале.

        Args:
            signal: Входной сигнал
            context: Контекст рынка

        Returns:
            float: Уверенность в сигнале (0-1)
        """
        # Базовая уверенность из сигнала
        base_confidence = signal.confidence

        # Корректировка на основе силы сигнала
        strength_factor = abs(signal.strength)

        # Корректировка на основе режима рынка
        regime_factor = self._get_regime_factor(signal.direction, context.market_regime)

        # Корректировка на основе ликвидности
        liquidity_factor = min(
            context.liquidity / 1000000, 1.0
        )  # нормализация ликвидности

        # Корректировка на основе волатильности
        volatility_factor = 1.0 - min(context.volatility, 1.0)

        # Итоговая уверенность
        confidence = (
            base_confidence
            * (
                self.priority_weights["strength"] * strength_factor
                + self.priority_weights["market_regime"] * regime_factor
                + self.priority_weights["liquidity"] * liquidity_factor
            )
            * volatility_factor
        )

        return float(min(max(confidence, 0.0), 1.0))

    def _get_regime_factor(self, direction: str, regime: str) -> float:
        """
        Получение фактора режима рынка.

        Args:
            direction: Направление сигнала
            regime: Режим рынка

        Returns:
            float: Фактор режима рынка
        """
        if regime == "bull" and direction == "buy":
            return 1.0
        elif regime == "bear" and direction == "sell":
            return 1.0
        elif regime == "sideways":
            return 0.5
        elif regime == "volatile":
            return 0.3
        else:
            return 0.1

    def _calculate_position_size(
        self, signal: Signal, context: MarketContext, confidence: float
    ) -> float:
        """
        Расчет размера позиции.

        Args:
            signal: Входной сигнал
            context: Контекст рынка
            confidence: Уверенность в сигнале

        Returns:
            float: Рекомендуемый размер позиции
        """
        # Базовый размер на основе риска
        base_size = self.risk_per_trade * confidence

        # Корректировка на основе волатильности
        volatility_factor = 1.0 - min(context.volatility, 0.5)

        # Корректировка на основе ликвидности
        liquidity_factor = min(context.liquidity / 1000000, 1.0)

        # Корректировка на основе силы сигнала
        strength_factor = abs(signal.strength)

        # Итоговый размер
        position_size = (
            base_size * volatility_factor * liquidity_factor * strength_factor
        )

        return min(position_size, self.max_position_size)

    def _calculate_risk_metrics(
        self, signal: Signal, context: MarketContext, position_size: float
    ) -> Dict[str, float]:
        """
        Расчет метрик риска.

        Args:
            signal: Входной сигнал
            context: Контекст рынка
            position_size: Размер позиции

        Returns:
            Dict[str, float]: Метрики риска
        """
        return {
            "var_95": self._calculate_var(context.volatility, position_size),
            "expected_drawdown": self._calculate_expected_drawdown(
                context.volatility, position_size
            ),
            "sharpe_ratio": self._calculate_sharpe_ratio(
                context.momentum, context.volatility
            ),
            "sortino_ratio": self._calculate_sortino_ratio(
                context.momentum, context.volatility
            ),
            "max_position_risk": position_size * context.volatility,
            "correlation_risk": self._calculate_correlation_risk(signal, context),
        }

    def _calculate_var(self, volatility: float, position_size: float) -> float:
        """
        Расчет Value at Risk.

        Args:
            volatility: Волатильность
            position_size: Размер позиции

        Returns:
            float: Value at Risk
        """
        return position_size * volatility * 1.645

    def _calculate_expected_drawdown(
        self, volatility: float, position_size: float
    ) -> float:
        """
        Расчет ожидаемой просадки.

        Args:
            volatility: Волатильность
            position_size: Размер позиции

        Returns:
            float: Ожидаемая просадка
        """
        return position_size * volatility * 0.5

    def _calculate_sharpe_ratio(self, momentum: float, volatility: float) -> float:
        """
        Расчет коэффициента Шарпа.

        Args:
            momentum: Моментум
            volatility: Волатильность

        Returns:
            float: Коэффициент Шарпа
        """
        if volatility > 0:
            return momentum / volatility
        return 0.0

    def _calculate_sortino_ratio(self, momentum: float, volatility: float) -> float:
        """
        Расчет коэффициента Сортино.

        Args:
            momentum: Моментум
            volatility: Волатильность

        Returns:
            float: Коэффициент Сортино
        """
        if volatility > 0:
            return momentum / volatility
        return 0.0

    def _calculate_correlation_risk(
        self, signal: Signal, context: MarketContext
    ) -> float:
        """
        Расчет корреляционного риска.

        Args:
            signal: Входной сигнал
            context: Контекст рынка

        Returns:
            float: Корреляционный риск
        """
        symbol = signal.symbol
        correlations = context.correlation_matrix.get(symbol, {})

        # Расчет среднего корреляционного риска
        if correlations:
            avg_correlation = sum(abs(corr) for corr in correlations.values()) / len(
                correlations
            )
            return min(avg_correlation, 1.0)

        return 0.0

    def _calculate_execution_priority(
        self, signal: Signal, context: MarketContext, confidence: float
    ) -> int:
        """
        Расчет приоритета исполнения.

        Args:
            signal: Входной сигнал
            context: Контекст рынка
            confidence: Уверенность в сигнале

        Returns:
            int: Приоритет исполнения
        """
        # Базовый приоритет
        base_priority = int(confidence * 100)

        # Корректировка на основе силы сигнала
        strength_bonus = int(abs(signal.strength) * 20)

        # Корректировка на основе ликвидности
        liquidity_bonus = int(min(context.liquidity / 1000000, 1.0) * 10)

        # Корректировка на основе волатильности
        volatility_penalty = int(context.volatility * 20)

        # Итоговый приоритет
        priority = base_priority + strength_bonus + liquidity_bonus - volatility_penalty

        return max(min(priority, 100), 1)

    def _calculate_expected_impact(
        self, signal: Signal, context: MarketContext, position_size: float
    ) -> float:
        """
        Расчет ожидаемого влияния.

        Args:
            signal: Входной сигнал
            context: Контекст рынка
            position_size: Размер позиции

        Returns:
            float: Ожидаемое влияние
        """
        # Базовое влияние на основе размера позиции
        base_impact = position_size * context.market_impact

        # Корректировка на основе ликвидности
        liquidity_factor = 1.0 / max(context.liquidity / 1000000, 0.1)

        # Корректировка на основе волатильности
        volatility_factor = 1.0 + context.volatility

        return base_impact * liquidity_factor * volatility_factor

    def _calculate_correlation_impact(
        self, signal: Signal, context: MarketContext
    ) -> Dict[str, float]:
        """
        Расчет влияния на коррелированные инструменты.

        Args:
            signal: Входной сигнал
            context: Контекст рынка

        Returns:
            Dict[str, float]: Влияние на коррелированные инструменты
        """
        symbol = signal.symbol
        correlations = context.correlation_matrix.get(symbol, {})

        impact: Dict[str, float] = {}

        for correlated_symbol, correlation in correlations.items():
            if abs(correlation) > self.correlation_threshold:
                impact[correlated_symbol] = correlation * signal.strength

        return impact

    def _calculate_market_impact(
        self, signal: Signal, context: MarketContext, position_size: float
    ) -> float:
        """
        Расчет влияния на рынок.

        Args:
            signal: Входной сигнал
            context: Контекст рынка
            position_size: Размер позиции

        Returns:
            float: Влияние на рынок
        """
        # Базовое влияние на основе размера позиции
        base_impact = position_size * context.market_impact

        # Корректировка на основе объема
        volume_factor = 1.0 / max(context.volume / 1000000, 0.1)

        # Корректировка на основе глубины рынка
        depth_factor = 1.0 / max(sum(context.market_depth.values()) / 1000000, 0.1)

        result = base_impact * volume_factor * depth_factor
        return float(result)

    async def process_signals(self, signals: List[Signal]) -> List[ProcessedSignal]:
        """
        Обработка списка сигналов.

        Args:
            signals: Список сигналов для обработки

        Returns:
            List[ProcessedSignal]: Список обработанных сигналов
        """
        processed_signals = []

        for signal in signals:
            # Создание контекста рынка (упрощенная версия)
            context = MarketContext(
                volatility=0.02,
                trend="neutral",
                volume=1000000,
                indicators={},
                market_regime="sideways",
                liquidity=1000000,
                momentum=0.0,
                sentiment=0.0,
                support_levels=[],
                resistance_levels=[],
                market_depth={},
                correlation_matrix={},
                market_impact=0.001,
                volume_profile={},
            )

            processed_signal = self.process_signal(signal, context)
            if processed_signal:
                processed_signals.append(processed_signal)

        return processed_signals

    def generate_rsi_signals(self, data: pd.DataFrame, period: int = 14) -> List[Dict[str, Any]]:
        """Генерация RSI сигналов."""
        signals: List[Dict[str, Any]] = []
        try:
            # Простой расчет RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Генерация сигналов
            for i in range(len(data)):
                if pd.isna(rsi.iloc[i]):
                    continue
                    
                if rsi.iloc[i] < 30:  # Перепроданность
                    signals.append({
                        "type": "buy",
                        "strength": 0.8,
                        "timestamp": data.index[i].to_pydatetime(),
                        "price": Decimal(str(data['close'].iloc[i])),
                        "confidence": 0.75
                    })
                elif rsi.iloc[i] > 70:  # Перекупленность
                    signals.append({
                        "type": "sell",
                        "strength": 0.7,
                        "timestamp": data.index[i].to_pydatetime(),
                        "price": Decimal(str(data['close'].iloc[i])),
                        "confidence": 0.65
                    })
        except Exception as e:
            logger.error(f"Error generating RSI signals: {str(e)}")
        return signals

    def generate_macd_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Генерация MACD сигналов."""
        signals: List[Dict[str, Any]] = []
        try:
            # Простой расчет MACD
            ema12 = data['close'].ewm(span=12).mean()
            ema26 = data['close'].ewm(span=26).mean()
            macd = ema12 - ema26
            signal_line = macd.ewm(span=9).mean()
            
            # Генерация сигналов
            for i in range(1, len(data)):
                if macd.iloc[i] > signal_line.iloc[i] and macd.iloc[i-1] <= signal_line.iloc[i-1]:
                    signals.append({
                        "type": "buy",
                        "strength": 0.6,
                        "timestamp": data.index[i].to_pydatetime(),
                        "price": Decimal(str(data['close'].iloc[i])),
                        "confidence": 0.6
                    })
                elif macd.iloc[i] < signal_line.iloc[i] and macd.iloc[i-1] >= signal_line.iloc[i-1]:
                    signals.append({
                        "type": "sell",
                        "strength": 0.5,
                        "timestamp": data.index[i].to_pydatetime(),
                        "price": Decimal(str(data['close'].iloc[i])),
                        "confidence": 0.55
                    })
        except Exception as e:
            logger.error(f"Error generating MACD signals: {str(e)}")
        return signals

    def generate_moving_average_signals(self, data: pd.DataFrame, short_period: int = 10, long_period: int = 50) -> List[Dict[str, Any]]:
        """Генерация сигналов скользящих средних."""
        signals: List[Dict[str, Any]] = []
        try:
            sma_short = data['close'].rolling(window=short_period).mean()
            sma_long = data['close'].rolling(window=long_period).mean()
            
            for i in range(1, len(data)):
                if pd.isna(sma_short.iloc[i]) or pd.isna(sma_long.iloc[i]):
                    continue
                    
                if sma_short.iloc[i] > sma_long.iloc[i] and sma_short.iloc[i-1] <= sma_long.iloc[i-1]:
                    signals.append({
                        "type": "buy",
                        "strength": 0.7,
                        "timestamp": data.index[i].to_pydatetime(),
                        "price": Decimal(str(data['close'].iloc[i])),
                        "confidence": 0.65
                    })
                elif sma_short.iloc[i] < sma_long.iloc[i] and sma_short.iloc[i-1] >= sma_long.iloc[i-1]:
                    signals.append({
                        "type": "sell",
                        "strength": 0.6,
                        "timestamp": data.index[i].to_pydatetime(),
                        "price": Decimal(str(data['close'].iloc[i])),
                        "confidence": 0.6
                    })
        except Exception as e:
            logger.error(f"Error generating MA signals: {str(e)}")
        return signals

    def generate_bollinger_bands_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Генерация сигналов полос Боллинджера."""
        signals: List[Dict[str, Any]] = []
        try:
            sma = data['close'].rolling(window=20).mean()
            std = data['close'].rolling(window=20).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            
            for i in range(len(data)):
                if pd.isna(upper_band.iloc[i]) or pd.isna(lower_band.iloc[i]):
                    continue
                    
                if data['close'].iloc[i] <= lower_band.iloc[i]:
                    signals.append({
                        "type": "buy",
                        "strength": 0.8,
                        "timestamp": data.index[i].to_pydatetime(),
                        "price": Decimal(str(data['close'].iloc[i])),
                        "confidence": 0.75
                    })
                elif data['close'].iloc[i] >= upper_band.iloc[i]:
                    signals.append({
                        "type": "sell",
                        "strength": 0.7,
                        "timestamp": data.index[i].to_pydatetime(),
                        "price": Decimal(str(data['close'].iloc[i])),
                        "confidence": 0.7
                    })
        except Exception as e:
            logger.error(f"Error generating BB signals: {str(e)}")
        return signals

    def generate_volume_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Генерация объемных сигналов."""
        signals: List[Dict[str, Any]] = []
        try:
            volume_sma = data['volume'].rolling(window=20).mean()
            
            for i in range(len(data)):
                if pd.isna(volume_sma.iloc[i]):
                    continue
                    
                if data['volume'].iloc[i] > volume_sma.iloc[i] * 1.5:
                    # Высокий объем - потенциальный сигнал
                    price_change = (data['close'].iloc[i] - data['open'].iloc[i]) / data['open'].iloc[i]
                    if price_change > 0.01:
                        signals.append({
                            "type": "buy",
                            "strength": 0.6,
                            "timestamp": data.index[i].to_pydatetime(),
                            "price": Decimal(str(data['close'].iloc[i])),
                            "confidence": 0.6
                        })
                    elif price_change < -0.01:
                        signals.append({
                            "type": "sell",
                            "strength": 0.5,
                            "timestamp": data.index[i].to_pydatetime(),
                            "price": Decimal(str(data['close'].iloc[i])),
                            "confidence": 0.55
                        })
        except Exception as e:
            logger.error(f"Error generating volume signals: {str(e)}")
        return signals

    def filter_signals(self, signals: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Фильтрация сигналов."""
        filtered_signals: List[Dict[str, Any]] = []
        min_strength = filters.get("min_strength", 0.0)
        min_confidence = filters.get("min_confidence", 0.0)
        signal_types = filters.get("signal_types", ["buy", "sell", "hold"])
        
        for signal in signals:
            if (signal.get("strength", 0) >= min_strength and
                signal.get("confidence", 0) >= min_confidence and
                signal.get("type") in signal_types):
                filtered_signals.append(signal)
        
        return filtered_signals

    def aggregate_signals(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Агрегация сигналов."""
        if not signals:
            return {
                "aggregated_signal": {},
                "signal_consensus": "neutral",
                "aggregation_metrics": {}
            }
        
        buy_signals = [s for s in signals if s.get("type") == "buy"]
        sell_signals = [s for s in signals if s.get("type") == "sell"]
        
        buy_strength = sum(s.get("strength", 0) for s in buy_signals)
        sell_strength = sum(s.get("strength", 0) for s in sell_signals)
        
        if buy_strength > sell_strength:
            consensus = "buy"
            strength = buy_strength / len(buy_signals) if buy_signals else 0
        elif sell_strength > buy_strength:
            consensus = "sell"
            strength = sell_strength / len(sell_signals) if sell_signals else 0
        else:
            consensus = "neutral"
            strength = 0
        
        return {
            "aggregated_signal": {
                "type": consensus,
                "strength": strength,
                "confidence": sum(s.get("confidence", 0) for s in signals) / len(signals)
            },
            "signal_consensus": consensus,
            "aggregation_metrics": {
                "total_signals": len(signals),
                "buy_signals": len(buy_signals),
                "sell_signals": len(sell_signals)
            }
        }

    def analyze_signal_quality(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Анализ качества сигналов."""
        if not signals:
            return {
                "overall_quality": 0.0,
                "signal_reliability": 0.0,
                "signal_consistency": 0.0,
                "quality_metrics": {}
            }
        
        avg_confidence = sum(s.get("confidence", 0) for s in signals) / len(signals)
        avg_strength = sum(s.get("strength", 0) for s in signals) / len(signals)
        
        return {
            "overall_quality": (avg_confidence + avg_strength) / 2,
            "signal_reliability": avg_confidence,
            "signal_consistency": avg_strength,
            "quality_metrics": {
                "avg_confidence": avg_confidence,
                "avg_strength": avg_strength,
                "signal_count": len(signals)
            }
        }

    def validate_signals(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Валидация сигналов."""
        errors = []
        valid_signals = []
        
        for signal in signals:
            if not all(key in signal for key in ["type", "strength", "confidence"]):
                errors.append(f"Missing required fields in signal: {signal}")
                continue
            
            if signal["type"] not in ["buy", "sell", "hold"]:
                errors.append(f"Invalid signal type: {signal['type']}")
                continue
            
            if not (0 <= signal["strength"] <= 1):
                errors.append(f"Invalid strength value: {signal['strength']}")
                continue
            
            if not (0 <= signal["confidence"] <= 1):
                errors.append(f"Invalid confidence value: {signal['confidence']}")
                continue
            
            valid_signals.append(signal)
        
        return {
            "is_valid": len(errors) == 0,
            "validation_errors": errors,
            "validation_score": len(valid_signals) / len(signals) if signals else 0.0,
            "validated_signals": valid_signals
        }

    def calculate_signal_statistics(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Расчет статистики сигналов."""
        if not signals:
            return {
                "total_signals": 0,
                "buy_signals": 0,
                "sell_signals": 0,
                "avg_strength": 0.0,
                "avg_confidence": 0.0,
                "signal_distribution": {}
            }
        
        buy_signals = [s for s in signals if s.get("type") == "buy"]
        sell_signals = [s for s in signals if s.get("type") == "sell"]
        
        return {
            "total_signals": len(signals),
            "buy_signals": len(buy_signals),
            "sell_signals": len(sell_signals),
            "avg_strength": sum(s.get("strength", 0) for s in signals) / len(signals),
            "avg_confidence": sum(s.get("confidence", 0) for s in signals) / len(signals),
            "signal_distribution": {
                "buy": len(buy_signals),
                "sell": len(sell_signals),
                "hold": len(signals) - len(buy_signals) - len(sell_signals)
            }
        }

    def backtest_signals(self, signals: List[Dict[str, Any]], market_data: pd.DataFrame) -> Dict[str, Any]:
        """Бэктестинг сигналов."""
        if not signals or market_data.empty:
            return {
                "total_return": 0.0,
                "signal_accuracy": 0.0,
                "profit_factor": 0.0,
                "win_rate": 0.0,
                "trades": []
            }
        
        # Простая симуляция бэктестинга
        trades = []
        total_return = 0.0
        wins = 0
        
        for signal in signals[:10]:  # Ограничиваем для демонстрации
            if signal.get("type") in ["buy", "sell"]:
                # Простая логика: если цена выросла после buy сигнала - выигрыш
                trade_return = 0.05 if signal.get("type") == "buy" else -0.03
                total_return += trade_return
                
                if trade_return > 0:
                    wins += 1
                
                trades.append({
                    "signal": signal,
                    "return": trade_return,
                    "timestamp": signal.get("timestamp")
                })
        
        # Расчет продвинутых метрик производительности
        
        profit_factor = self._calculate_profit_factor(trades)
        sharpe_ratio = self._calculate_sharpe_ratio([t["return"] for t in trades])
        max_drawdown = self._calculate_max_drawdown([t["return"] for t in trades])
        sortino_ratio = self._calculate_sortino_ratio([t["return"] for t in trades])
        calmar_ratio = self._calculate_calmar_ratio([t["return"] for t in trades])
        
        return {
            "total_return": total_return,
            "signal_accuracy": wins / len(trades) if trades else 0.0,
            "profit_factor": profit_factor,
            "win_rate": wins / len(trades) if trades else 0.0,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "trades": trades,
            "avg_win": np.mean([t["return"] for t in trades if t["return"] > 0]) if any(t["return"] > 0 for t in trades) else 0.0,
            "avg_loss": np.mean([t["return"] for t in trades if t["return"] < 0]) if any(t["return"] < 0 for t in trades) else 0.0
        }

    def optimize_signal_parameters(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Оптимизация параметров сигналов."""
        # Простая оптимизация RSI периода
        best_period = 14
        best_score = 0.0
        
        for period in [10, 14, 20, 30]:
            signals = self.generate_rsi_signals(market_data, period)
            if signals:
                quality = self.analyze_signal_quality(signals)
                score = quality["overall_quality"]
                if score > best_score:
                    best_score = score
                    best_period = period
        
        return {
            "optimal_rsi_period": best_period,
            "best_score": best_score,
            "optimization_metrics": {
                "tested_periods": [10, 14, 20, 30],
                "best_period": best_period
            }
        }

    def generate_signal_alerts(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Генерация алертов для сигналов."""
        alerts = []
        
        for signal in signals:
            if signal.get("strength", 0) > 0.8 and signal.get("confidence", 0) > 0.8:
                alerts.append({
                    "type": "high_confidence_signal",
                    "message": f"High confidence {signal.get('type')} signal detected",
                    "signal": signal,
                    "priority": "high",
                    "timestamp": datetime.now()
                })
        
        return alerts

    def get_signal_history(self, symbol: str, timeframe: str = "1h") -> List[Dict[str, Any]]:
        """Получение продвинутой истории сигналов с аналитикой."""
        try:
            # Фильтрация сигналов по символу и времени
            filtered_signals = []
            current_time = datetime.now()
            
            for signal in self.signals:
                signal_time = signal.get("timestamp", current_time)
                if isinstance(signal_time, str):
                    try:
                        signal_time = datetime.fromisoformat(signal_time.replace('Z', '+00:00'))
                    except:
                        signal_time = current_time
                
                # Фильтр по символу
                if signal.get("symbol") == symbol:
                    # Добавляем дополнительную аналитику
                    enhanced_signal = signal.copy()
                    enhanced_signal.update({
                        "age_hours": (current_time - signal_time).total_seconds() / 3600,
                        "relevance_score": self._calculate_signal_relevance(signal, current_time),
                        "confidence_trend": self._calculate_confidence_trend(signal),
                        "performance_impact": self._estimate_performance_impact(signal)
                    })
                    filtered_signals.append(enhanced_signal)
            
            # Сортировка по времени (новые первыми)
            filtered_signals.sort(key=lambda x: x.get("timestamp", current_time), reverse=True)
            
            # Ограничение количества для производительности
            return filtered_signals[:100]
            
        except Exception as e:
            logger.error(f"Error getting signal history: {e}")
            return []

    def _calculate_signal_relevance(self, signal: Dict[str, Any], current_time: datetime) -> float:
        """Расчет релевантности сигнала."""
        try:
            signal_time = signal.get("timestamp", current_time)
            if isinstance(signal_time, str):
                signal_time = datetime.fromisoformat(signal_time.replace('Z', '+00:00'))
            
            # Время разложения релевантности (сигнал теряет актуальность)
            hours_passed = (current_time - signal_time).total_seconds() / 3600
            
            # Экспоненциальное затухание релевантности
            decay_rate = 0.1  # Половина релевантности теряется за ~7 часов
            relevance = np.exp(-decay_rate * hours_passed)
            
            # Учет силы сигнала
            strength = signal.get("strength", 0.5)
            confidence = signal.get("confidence", 0.5)
            
            # Комбинированная релевантность
            combined_relevance = relevance * (0.6 * strength + 0.4 * confidence)
            
            return float(max(0.0, min(1.0, combined_relevance)))
            
        except Exception:
            return 0.5

    def _calculate_confidence_trend(self, signal: Dict[str, Any]) -> str:
        """Определение тренда уверенности в сигнале."""
        try:
            confidence = signal.get("confidence", 0.5)
            
            if confidence > 0.8:
                return "high"
            elif confidence > 0.6:
                return "medium"
            elif confidence > 0.4:
                return "low"
            else:
                return "very_low"
                
        except Exception:
            return "unknown"

    def _estimate_performance_impact(self, signal: Dict[str, Any]) -> Dict[str, float]:
        """Оценка потенциального влияния сигнала на производительность."""
        try:
            signal_type = signal.get("type", "unknown")
            strength = signal.get("strength", 0.5)
            confidence = signal.get("confidence", 0.5)
            
            # Базовые оценки влияния для разных типов сигналов
            impact_multipliers = {
                "buy": 1.0,
                "sell": -1.0,
                "strong_buy": 1.5,
                "strong_sell": -1.5,
                "hold": 0.0
            }
            
            base_impact = impact_multipliers.get(signal_type, 0.0)
            
            # Модификация влияния на основе силы и уверенности
            adjusted_impact = base_impact * strength * confidence
            
            # Оценка риска
            risk_score = 1.0 - confidence  # Чем меньше уверенности, тем больше риск
            
            # Ожидаемая доходность (упрощенная модель)
            expected_return = adjusted_impact * 0.02  # До 2% ожидаемой доходности
            
            return {
                "expected_return": float(expected_return),
                "risk_score": float(risk_score),
                "impact_magnitude": float(abs(adjusted_impact)),
                "direction": 1.0 if adjusted_impact > 0 else -1.0 if adjusted_impact < 0 else 0.0
            }
            
        except Exception:
            return {
                "expected_return": 0.0,
                "risk_score": 0.5,
                "impact_magnitude": 0.0,
                "direction": 0.0
            }

    def _calculate_profit_factor(self, trades: List[Dict[str, Any]]) -> float:
        """Расчет профит-фактора."""
        try:
            profits = [t["return"] for t in trades if t["return"] > 0]
            losses = [abs(t["return"]) for t in trades if t["return"] < 0]
            
            total_profit = sum(profits) if profits else 0.0
            total_loss = sum(losses) if losses else 0.001  # Избегаем деления на ноль
            
            return total_profit / total_loss if total_loss > 0 else 0.0
            
        except Exception:
            return 1.0

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Расчет коэффициента Шарпа."""
        try:
            if not returns or len(returns) < 2:
                return 0.0
            
            returns_array = np.array(returns)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            
            # Предполагаем безрисковую ставку 0% для упрощения
            sharpe = mean_return / std_return if std_return > 0 else 0.0
            
            return float(sharpe)
            
        except Exception:
            return 0.0

    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Расчет коэффициента Сортино."""
        try:
            if not returns or len(returns) < 2:
                return 0.0
            
            returns_array = np.array(returns)
            mean_return = np.mean(returns_array)
            
            # Только отрицательные отклонения (downside deviation)
            negative_returns = returns_array[returns_array < 0]
            if len(negative_returns) == 0:
                return float('inf') if mean_return > 0 else 0.0
            
            downside_deviation = np.std(negative_returns)
            sortino = mean_return / downside_deviation if downside_deviation > 0 else 0.0
            
            return float(sortino)
            
        except Exception:
            return 0.0

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Расчет максимальной просадки."""
        try:
            if not returns:
                return 0.0
            
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = cumulative_returns - running_max
            
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
            
            return float(max_drawdown)
            
        except Exception:
            return 0.0

    def _calculate_calmar_ratio(self, returns: List[float]) -> float:
        """Расчет коэффициента Кальмара."""
        try:
            if not returns:
                return 0.0
            
            annual_return = np.mean(returns) * 252  # Предполагаем 252 торговых дня
            max_drawdown = abs(self._calculate_max_drawdown(returns))
            
            calmar = annual_return / max_drawdown if max_drawdown > 0 else 0.0
            
            return float(calmar)
            
        except Exception:
            return 0.0
    
    def cleanup(self) -> None:
        """Очистка ресурсов процессора сигналов."""
        self.signals.clear()
        self.metadata.clear()
        self.last_update = None

    # Добавляем атрибуты для совместимости с тестами
    @property
    def signal_generators(self) -> Dict[str, Any]:
        """Генераторы сигналов."""
        return {
            "rsi": self.generate_rsi_signals,
            "macd": self.generate_macd_signals,
            "ma": self.generate_moving_average_signals,
            "bb": self.generate_bollinger_bands_signals,
            "volume": self.generate_volume_signals
        }

    @property
    def signal_filters(self) -> Dict[str, Any]:
        """Фильтры сигналов."""
        return {
            "strength": lambda s, min_strength: s.get("strength", 0) >= min_strength,
            "confidence": lambda s, min_confidence: s.get("confidence", 0) >= min_confidence,
            "type": lambda s, types: s.get("type") in types
        }

    @property
    def signal_aggregators(self) -> Dict[str, Any]:
        """Агрегаторы сигналов."""
        return {
            "simple": self.aggregate_signals,
            "weighted": self.aggregate_signals  # Используем ту же функцию
        }
