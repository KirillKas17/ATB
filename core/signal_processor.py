from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class Signal:
    """Расширенный класс сигнала"""

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
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
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
        """Создание из словаря с проверкой ключей"""
        required_fields = ["entry_price", "stop_loss", "take_profit"]
        missing = [f for f in required_fields if f not in data or data[f] is None]
        if missing:
            logger.warning(f"Пропуск сигнала: отсутствуют ключи {missing} в {data}")
            raise ValueError(f"Signal missing required fields: {missing}")
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class MarketContext:
    """Расширенный контекст рынка"""

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
        """Преобразование в словарь"""
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
        """Создание из словаря"""
        return cls(**data)


@dataclass
class ProcessedSignal:
    """Расширенный обработанный сигнал"""

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
        """Преобразование в словарь"""
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
        """Создание из словаря"""
        data["signal"] = Signal.from_dict(data["signal"])
        data["context"] = MarketContext.from_dict(data["context"])
        if data["execution_time"]:
            data["execution_time"] = datetime.fromisoformat(data["execution_time"])
        return cls(**data)


class SignalProcessor:
    """Расширенный процессор сигналов"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Инициализация процессора сигналов."""
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

    def _initialize(self):
        # Additional initialization logic if needed
        pass

    def process_signal(
        self, signal: Signal, context: MarketContext
    ) -> Optional[ProcessedSignal]:
        """Обработка сигнала с расширенной логикой"""
        try:
            # Расчет базовой уверенности
            confidence = self._calculate_confidence(signal, context)

            # Проверка минимальной уверенности
            if confidence < self.min_confidence:
                logger.debug(
                    f"Signal rejected: confidence {confidence} < {self.min_confidence}"
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
        """Расчет уверенности в сигнале"""
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

        return min(max(confidence, 0.0), 1.0)

    def _get_regime_factor(self, direction: str, regime: str) -> float:
        """Получение фактора режима рынка"""
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
        """Расчет размера позиции"""
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
        """Расчет метрик риска"""
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
        """Расчет Value at Risk"""
        return position_size * volatility * 1.645  # 95% VaR

    def _calculate_expected_drawdown(
        self, volatility: float, position_size: float
    ) -> float:
        """Расчет ожидаемой просадки"""
        return position_size * volatility * 2.0  # примерная оценка

    def _calculate_sharpe_ratio(self, momentum: float, volatility: float) -> float:
        """Расчет коэффициента Шарпа"""
        if volatility == 0:
            return 0.0
        return momentum / volatility

    def _calculate_sortino_ratio(self, momentum: float, volatility: float) -> float:
        """Расчет коэффициента Сортино"""
        if volatility == 0:
            return 0.0
        return momentum / (
            volatility * 0.5
        )  # используем только отрицательную волатильность

    def _calculate_correlation_risk(
        self, signal: Signal, context: MarketContext
    ) -> float:
        """Расчет риска корреляции"""
        if not context.correlation_matrix:
            return 0.0

        correlations = []
        for pair, corr_dict in context.correlation_matrix.items():
            if pair != signal.symbol:
                correlations.append(max(abs(corr) for corr in corr_dict.values()))

        return max(correlations) if correlations else 0.0

    def _calculate_execution_priority(
        self, signal: Signal, context: MarketContext, confidence: float
    ) -> int:
        """Расчет приоритета исполнения"""
        # Базовый приоритет
        priority = signal.priority

        # Корректировка на основе уверенности
        priority += int(confidence * 10)

        # Корректировка на основе силы сигнала
        priority += int(abs(signal.strength) * 5)

        # Корректировка на основе режима рынка
        if (context.market_regime == "bull" and signal.direction == "buy") or (
            context.market_regime == "bear" and signal.direction == "sell"
        ):
            priority += 5

        # Корректировка на основе ликвидности
        priority += int(min(context.liquidity / 1000000, 1.0) * 3)

        return priority

    def _calculate_expected_impact(
        self, signal: Signal, context: MarketContext, position_size: float
    ) -> float:
        """Расчет ожидаемого влияния на рынок"""
        # Базовое влияние
        base_impact = position_size * context.market_impact

        # Корректировка на основе ликвидности
        liquidity_factor = 1.0 - min(context.liquidity / 1000000, 1.0)

        # Корректировка на основе глубины рынка
        depth_factor = 1.0 - min(sum(context.market_depth.values()) / 1000000, 1.0)

        return base_impact * (liquidity_factor + depth_factor) / 2

    def _calculate_correlation_impact(
        self, signal: Signal, context: MarketContext
    ) -> Dict[str, float]:
        """Расчет влияния на коррелированные инструменты"""
        impact = {}

        if not context.correlation_matrix:
            return impact

        for pair, corr_dict in context.correlation_matrix.items():
            if pair != signal.symbol:
                max_corr = max(abs(corr) for corr in corr_dict.values())
                if max_corr > self.correlation_threshold:
                    impact[pair] = max_corr

        return impact

    def _calculate_market_impact(
        self, signal: Signal, context: MarketContext, position_size: float
    ) -> float:
        """Расчет влияния на рынок"""
        # Базовое влияние
        base_impact = position_size * context.market_impact

        # Корректировка на основе объема
        volume_factor = min(position_size / context.volume, 1.0)

        # Корректировка на основе ликвидности
        liquidity_factor = 1.0 - min(context.liquidity / 1000000, 1.0)

        return base_impact * volume_factor * liquidity_factor

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
            try:
                # Создаем контекст рынка
                context = MarketContext(
                    volatility=0.0,  # Будет рассчитано на основе данных
                    trend="neutral",
                    volume=0.0,
                    indicators={},
                    market_regime="neutral",
                    liquidity=0.0,
                    momentum=0.0,
                    sentiment=0.0,
                    support_levels=[],
                    resistance_levels=[],
                    market_depth={},
                    correlation_matrix={},
                    market_impact=0.0,
                    volume_profile={},
                )

                # Обрабатываем сигнал
                processed_signal = self.process_signal(signal, context)
                if processed_signal:
                    processed_signals.append(processed_signal)

            except Exception as e:
                logger.error(f"Error processing signal: {str(e)}")
                continue

        return processed_signals
