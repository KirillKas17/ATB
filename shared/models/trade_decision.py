"""
Единая модель торгового решения.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.value_objects.price import Price


class TradeAction(Enum):
    """Действия торгового решения."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"
    OPEN = "open"


class TradeDirection(Enum):
    """Направления торговли."""

    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class TradeSource(Enum):
    """Источники торговых решений."""

    STRATEGY = "strategy"
    ML_MODEL = "ml_model"
    TECHNICAL_ANALYSIS = "technical_analysis"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    WHALE_ACTIVITY = "whale_activity"
    NEWS_SENTIMENT = "news_sentiment"
    MARKET_MAKER = "market_maker"
    ENTANGLEMENT_DETECTOR = "entanglement_detector"
    ENSEMBLE = "ensemble"
    MANUAL = "manual"


@dataclass
class TradeDecision:
    """
    Единая модель торгового решения для всего проекта.

    Этот класс заменяет все дублирующиеся определения TradeDecision
    в различных модулях проекта.
    """

    # Основные параметры
    symbol: str
    action: TradeAction
    direction: TradeDirection = TradeDirection.NEUTRAL
    confidence: Decimal = Decimal("0")

    # Ценовые параметры
    price: Optional[Price] = None
    stop_loss: Optional[Price] = None
    take_profit: Optional[Price] = None

    # Объемные параметры
    size: Optional[Decimal] = None
    volume: Optional[Decimal] = None
    position_size: Optional[Decimal] = None

    # Временные параметры
    timestamp: datetime = field(default_factory=datetime.now)
    expiration_time: Optional[datetime] = None

    # Источники и метаданные
    source: TradeSource = TradeSource.STRATEGY
    sources: List[str] = field(default_factory=list)
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Дополнительные параметры
    risk_score: Decimal = Decimal("0")
    market_regime: str = ""
    signal_strength: Decimal = Decimal("0")
    priority: int = 0

    def __post_init__(self):
        """Пост-инициализация с валидацией типов."""
        # Конвертация типов для совместимости
        if isinstance(self.action, str):
            self.action = TradeAction(self.action)

        if isinstance(self.direction, str):
            self.direction = TradeDirection(self.direction)

        if isinstance(self.source, str):
            self.source = TradeSource(self.source)

        if isinstance(self.confidence, (int, float, str)):
            self.confidence = Decimal(str(self.confidence))

        if isinstance(self.risk_score, (int, float, str)):
            self.risk_score = Decimal(str(self.risk_score))

        if isinstance(self.signal_strength, (int, float, str)):
            self.signal_strength = Decimal(str(self.signal_strength))

        if isinstance(self.size, (int, float, str)):
            self.size = Decimal(str(self.size))

        if isinstance(self.volume, (int, float, str)):
            self.volume = Decimal(str(self.volume))

        if isinstance(self.position_size, (int, float, str)):
            self.position_size = Decimal(str(self.position_size))

        if isinstance(self.price, (int, float, str)):
            self.price = Price(Decimal(str(self.price)), Currency.USD)

        if isinstance(self.stop_loss, (int, float, str)):
            self.stop_loss = Price(Decimal(str(self.stop_loss)), Currency.USD)

        if isinstance(self.take_profit, (int, float, str)):
            self.take_profit = Price(Decimal(str(self.take_profit)), Currency.USD)

    @property
    def is_actionable(self) -> bool:
        """Можно ли действовать по решению."""
        return (
            self.action
            in [TradeAction.BUY, TradeAction.SELL, TradeAction.OPEN, TradeAction.CLOSE]
            and self.confidence > Decimal("0.3")
            and self.risk_score < Decimal("0.7")
        )

    @property
    def is_high_confidence(self) -> bool:
        """Высокая ли уверенность в решении."""
        return self.confidence > Decimal("0.8")

    @property
    def is_low_risk(self) -> bool:
        """Низкий ли риск решения."""
        return self.risk_score < Decimal("0.3")

    @property
    def is_urgent(self) -> bool:
        """Срочное ли решение."""
        return self.priority > 7

    def calculate_expected_return(self, current_price: Price) -> Optional[Money]:
        """
        Расчет ожидаемой доходности.

        Args:
            current_price: Текущая цена

        Returns:
            Money: Ожидаемая доходность
        """
        if not self.price or not self.take_profit:
            return None

        if self.action == TradeAction.BUY:
            return Money(
                (self.take_profit.value - current_price.value)
                * (self.size or Decimal("1")),
                Currency.USD,
            )
        elif self.action == TradeAction.SELL:
            return Money(
                (current_price.value - self.take_profit.value)
                * (self.size or Decimal("1")),
                Currency.USD,
            )

        return None

    def calculate_risk_reward_ratio(self) -> Optional[Decimal]:
        """
        Расчет соотношения риск/доходность.

        Returns:
            Decimal: Соотношение риск/доходность
        """
        if not self.stop_loss or not self.take_profit or not self.price:
            return None

        if self.action == TradeAction.BUY:
            # Для покупки: риск = цена входа - стоп-лосс, доходность = тейк-профит - цена входа
            risk = self.price.value - self.stop_loss.value
            reward = self.take_profit.value - self.price.value
        elif self.action == TradeAction.SELL:
            # Для продажи: риск = стоп-лосс - цена входа, доходность = цена входа - тейк-профит
            risk = self.stop_loss.value - self.price.value
            reward = self.price.value - self.take_profit.value
        else:
            return None

        # Проверяем что риск и доходность положительные
        if risk <= 0 or reward <= 0:
            return None

        return reward / risk

    def validate(self) -> List[str]:
        """
        Валидация торгового решения.

        Returns:
            List[str]: Список ошибок валидации
        """
        errors = []

        # Проверка обязательных полей
        if not self.symbol:
            errors.append("Symbol is required")

        if not self.action:
            errors.append("Action is required")

        if self.confidence < Decimal("0") or self.confidence > Decimal("1"):
            errors.append("Confidence must be between 0 and 1")

        if self.risk_score < Decimal("0") or self.risk_score > Decimal("1"):
            errors.append("Risk score must be between 0 and 1")

        # Проверка ценовых параметров
        if self.action in [TradeAction.BUY, TradeAction.SELL, TradeAction.OPEN]:
            if not self.price:
                errors.append("Price is required for trading actions")

            if not self.stop_loss:
                errors.append("Stop loss is required for trading actions")

            if not self.take_profit:
                errors.append("Take profit is required for trading actions")

            if self.price and self.stop_loss and self.take_profit:
                if self.action == TradeAction.BUY:
                    if self.stop_loss.value >= self.price.value:
                        errors.append(
                            "Stop loss must be below current price for buy action"
                        )
                    if self.take_profit.value <= self.price.value:
                        errors.append(
                            "Take profit must be above current price for buy action"
                        )
                elif self.action == TradeAction.SELL:
                    if self.stop_loss.value <= self.price.value:
                        errors.append(
                            "Stop loss must be above current price for sell action"
                        )
                    if self.take_profit.value >= self.price.value:
                        errors.append(
                            "Take profit must be below current price for sell action"
                        )

        # Проверка объемных параметров
        if self.action in [TradeAction.BUY, TradeAction.SELL, TradeAction.OPEN]:
            if not self.size and not self.volume and not self.position_size:
                errors.append(
                    "Size, volume, or position_size is required for trading actions"
                )

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "symbol": self.symbol,
            "action": self.action.value,
            "direction": self.direction.value,
            "confidence": str(self.confidence),
            "price": str(self.price.value) if self.price else None,
            "stop_loss": str(self.stop_loss.value) if self.stop_loss else None,
            "take_profit": str(self.take_profit.value) if self.take_profit else None,
            "size": str(self.size) if self.size else None,
            "volume": str(self.volume) if self.volume else None,
            "position_size": str(self.position_size) if self.position_size else None,
            "timestamp": self.timestamp.isoformat(),
            "expiration_time": (
                self.expiration_time.isoformat() if self.expiration_time else None
            ),
            "source": self.source.value,
            "sources": self.sources,
            "explanation": self.explanation,
            "risk_score": str(self.risk_score),
            "market_regime": self.market_regime,
            "signal_strength": str(self.signal_strength),
            "priority": self.priority,
            "is_actionable": self.is_actionable,
            "is_high_confidence": self.is_high_confidence,
            "is_low_risk": self.is_low_risk,
            "is_urgent": self.is_urgent,
            "risk_reward_ratio": (
                str(self.calculate_risk_reward_ratio())
                if self.calculate_risk_reward_ratio()
                else None
            ),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeDecision":
        """Создание из словаря."""
        return cls(
            symbol=data["symbol"],
            action=TradeAction(data["action"]),
            direction=TradeDirection(data.get("direction", "neutral")),
            confidence=Decimal(data["confidence"]),
            price=(
                Price(Decimal(data["price"]), Currency.USD)
                if data.get("price")
                else None
            ),
            stop_loss=(
                Price(Decimal(data["stop_loss"]), Currency.USD)
                if data.get("stop_loss")
                else None
            ),
            take_profit=(
                Price(Decimal(data["take_profit"]), Currency.USD)
                if data.get("take_profit")
                else None
            ),
            size=Decimal(data["size"]) if data.get("size") else None,
            volume=Decimal(data["volume"]) if data.get("volume") else None,
            position_size=(
                Decimal(data["position_size"]) if data.get("position_size") else None
            ),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            expiration_time=(
                datetime.fromisoformat(data["expiration_time"])
                if data.get("expiration_time")
                else None
            ),
            source=TradeSource(data.get("source", "strategy")),
            sources=data.get("sources", []),
            explanation=data.get("explanation", ""),
            risk_score=Decimal(data.get("risk_score", "0")),
            market_regime=data.get("market_regime", ""),
            signal_strength=Decimal(data.get("signal_strength", "0")),
            priority=data.get("priority", 0),
            metadata=data.get("metadata", {}),
        )

    def __str__(self) -> str:
        """Строковое представление."""
        return (
            f"TradeDecision({self.symbol}, {self.action.value}, "
            f"confidence={self.confidence:.3f}, risk={self.risk_score:.3f})"
        )

    def __repr__(self) -> str:
        """Представление для отладки."""
        return (
            f"TradeDecision(symbol='{self.symbol}', action={self.action.value}, "
            f"direction={self.direction.value}, confidence={self.confidence}, "
            f"price={self.price}, stop_loss={self.stop_loss}, "
            f"take_profit={self.take_profit}, source={self.source.value})"
        )
