"""
События для торгового репозитория.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from domain.entities.account import Account
from domain.entities.order import Order
from domain.entities.position import Position
from domain.entities.trading_pair import TradingPair


class EventType(Enum):
    """Типы событий."""

    ORDER_CREATED = "order_created"
    ORDER_UPDATED = "order_updated"
    ORDER_DELETED = "order_deleted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    POSITION_CREATED = "position_created"
    POSITION_UPDATED = "position_updated"
    POSITION_DELETED = "position_deleted"
    POSITION_CLOSED = "position_closed"
    TRADING_PAIR_CREATED = "trading_pair_created"
    TRADING_PAIR_UPDATED = "trading_pair_updated"
    TRADING_PAIR_DELETED = "trading_pair_deleted"
    ACCOUNT_CREATED = "account_created"
    ACCOUNT_UPDATED = "account_updated"
    ACCOUNT_DELETED = "account_deleted"
    METRICS_UPDATED = "metrics_updated"
    PATTERN_DETECTED = "pattern_detected"
    LIQUIDITY_ANALYZED = "liquidity_analyzed"
    CACHE_CLEARED = "cache_cleared"
    CACHE_INVALIDATED = "cache_invalidated"
    VALIDATION_FAILED = "validation_failed"
    BUSINESS_RULE_VIOLATION = "business_rule_violation"
    ERROR_OCCURRED = "error_occurred"
    WARNING_RAISED = "warning_raised"


@dataclass
class TradingEvent:
    """Базовое событие торгового репозитория."""

    event_type: EventType
    entity_id: str
    entity_type: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderEvent(TradingEvent):
    """Событие, связанное с ордером."""

    order: Optional[Order] = None
    previous_state: Optional[Dict[str, Any]] = None
    changes: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None


@dataclass
class PositionEvent(TradingEvent):
    """Событие, связанное с позицией."""

    position: Optional[Position] = None
    previous_state: Optional[Dict[str, Any]] = None
    changes: Optional[Dict[str, Any]] = None
    pnl_change: Optional[float] = None


@dataclass
class TradingPairEvent(TradingEvent):
    """Событие, связанное с торговой парой."""

    trading_pair: Optional[TradingPair] = None
    previous_state: Optional[Dict[str, Any]] = None
    changes: Optional[Dict[str, Any]] = None


@dataclass
class AccountEvent(TradingEvent):
    """Событие, связанное с аккаунтом."""

    account: Optional[Account] = None
    previous_state: Optional[Dict[str, Any]] = None
    changes: Optional[Dict[str, Any]] = None
    balance_changes: Optional[Dict[str, float]] = None


@dataclass
class MetricsEvent(TradingEvent):
    """Событие с метриками."""

    metrics: Dict[str, Any] = field(default_factory=dict)
    period: Optional[str] = None
    account_id: Optional[str] = None


@dataclass
class PatternEvent(TradingEvent):
    """Событие с паттерном."""

    pattern_type: str = ""
    confidence: float = 0.0
    risk_score: float = 0.0
    prediction: Optional[str] = None
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LiquidityEvent(TradingEvent):
    """Событие с анализом ликвидности."""

    trading_pair_id: str = ""
    sufficient: bool = True
    slippage: float = 0.0
    depth: int = 0
    spread: float = 0.0
    volume_24h: float = 0.0
    market_impact: float = 0.0


@dataclass
class CacheEvent(TradingEvent):
    """Событие кэша."""

    cache_type: str = ""
    keys_affected: List[str] = field(default_factory=list)
    reason: Optional[str] = None


@dataclass
class ValidationEvent(TradingEvent):
    """Событие валидации."""

    validation_type: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    field_name: Optional[str] = None


@dataclass
class ErrorEvent(TradingEvent):
    """Событие ошибки."""

    error_type: str = ""
    error_message: str = ""
    stack_trace: Optional[str] = None
    severity: str = "ERROR"


class TradingEventBus:
    """Шина событий для торгового репозитория."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_history: List[TradingEvent] = []
        self._max_history_size = 10000
        self._enabled = True

    def subscribe(self, event_type: EventType, callback: Callable) -> None:
        """Подписка на события."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)
            self.logger.debug(f"Subscribed to {event_type.value}")

    def unsubscribe(self, event_type: EventType, callback: Callable) -> None:
        """Отписка от событий."""
        if event_type in self._subscribers:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
                self.logger.debug(f"Unsubscribed from {event_type.value}")

    async def publish(self, event: TradingEvent) -> None:
        """Публикация события."""
        if not self._enabled:
            return
        try:
            # Добавляем в историю
            self._event_history.append(event)
            if len(self._event_history) > self._max_history_size:
                self._event_history.pop(0)
            # Уведомляем подписчиков
            if event.event_type in self._subscribers:
                for callback in self._subscribers[event.event_type]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(event)
                        else:
                            callback(event)
                    except Exception as e:
                        self.logger.error(f"Error in event callback: {e}")
            self.logger.debug(
                f"Published event: {event.event_type.value} for {event.entity_id}"
            )
        except Exception as e:
            self.logger.error(f"Error publishing event: {e}")

    def get_event_history(
        self, event_type: Optional[EventType] = None, limit: Optional[int] = None
    ) -> List[TradingEvent]:
        """Получение истории событий."""
        events = self._event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if limit:
            events = events[-limit:]
        return events

    def clear_history(self) -> None:
        """Очистка истории событий."""
        self._event_history.clear()

    def enable(self) -> None:
        """Включение шины событий."""
        self._enabled = True

    def disable(self) -> None:
        """Отключение шины событий."""
        self._enabled = False

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики шины событий."""
        event_counts = {}
        for event_type in EventType:
            event_counts[event_type.value] = len(
                [e for e in self._event_history if e.event_type == event_type]
            )
        return {
            "total_events": len(self._event_history),
            "event_counts": event_counts,
            "subscribers": {
                et.value: len(subs) for et, subs in self._subscribers.items()
            },
            "enabled": self._enabled,
            "max_history_size": self._max_history_size,
        }


class TradingEventFactory:
    """Фабрика событий торгового репозитория."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def create_order_created_event(
        self,
        order: Order,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> OrderEvent:
        """Создание события создания ордера."""
        return OrderEvent(
            event_type=EventType.ORDER_CREATED,
            entity_id=str(order.id),
            entity_type="order",
            user_id=user_id,
            session_id=session_id,
            order=order,
            data={
                "symbol": str(order.trading_pair),
                "side": order.side.value,
                "type": order.order_type.value,
                "quantity": float(order.quantity),
                "price": float(order.price.amount) if order.price else None,
            },
        )

    def create_order_updated_event(
        self, order: Order, changes: Dict[str, Any], user_id: Optional[str] = None
    ) -> OrderEvent:
        """Создание события обновления ордера."""
        return OrderEvent(
            event_type=EventType.ORDER_UPDATED,
            entity_id=str(order.id),
            entity_type="order",
            user_id=user_id,
            order=order,
            changes=changes,
            data={"symbol": str(order.trading_pair), "changes": changes},
        )

    def create_order_filled_event(
        self, order: Order, filled_quantity: float, fill_price: float
    ) -> OrderEvent:
        """Создание события исполнения ордера."""
        return OrderEvent(
            event_type=EventType.ORDER_FILLED,
            entity_id=str(order.id),
            entity_type="order",
            order=order,
            data={
                "symbol": str(order.trading_pair),
                "filled_quantity": filled_quantity,
                "fill_price": fill_price,
                "total_value": filled_quantity * fill_price,
            },
        )

    def create_position_created_event(
        self, position: Position, user_id: Optional[str] = None
    ) -> PositionEvent:
        """Создание события создания позиции."""
        return PositionEvent(
            event_type=EventType.POSITION_CREATED,
            entity_id=str(position.id),
            entity_type="position",
            user_id=user_id,
            position=position,
            data={
                "symbol": str(position.trading_pair),
                "side": position.side.value,
                "quantity": float(position.volume.to_decimal()),
                "average_price": float(position.entry_price.amount),
            },
        )

    def create_position_updated_event(
        self,
        position: Position,
        changes: Dict[str, Any],
        pnl_change: Optional[float] = None,
    ) -> PositionEvent:
        """Создание события обновления позиции."""
        return PositionEvent(
            event_type=EventType.POSITION_UPDATED,
            entity_id=str(position.id),
            entity_type="position",
            position=position,
            changes=changes,
            pnl_change=pnl_change,
            data={
                "symbol": str(position.trading_pair),
                "changes": changes,
                "pnl_change": pnl_change,
            },
        )

    def create_metrics_updated_event(
        self,
        metrics: Dict[str, Any],
        account_id: Optional[str] = None,
        period: Optional[str] = None,
    ) -> MetricsEvent:
        """Создание события обновления метрик."""
        return MetricsEvent(
            event_type=EventType.METRICS_UPDATED,
            entity_id=account_id or "global",
            entity_type="metrics",
            account_id=account_id,
            period=period,
            metrics=metrics,
            data=metrics,
        )

    def create_pattern_detected_event(
        self,
        pattern_type: str,
        confidence: float,
        risk_score: float,
        entity_id: str,
        features: Dict[str, Any],
    ) -> PatternEvent:
        """Создание события обнаружения паттерна."""
        return PatternEvent(
            event_type=EventType.PATTERN_DETECTED,
            entity_id=entity_id,
            entity_type="pattern",
            pattern_type=pattern_type,
            confidence=confidence,
            risk_score=risk_score,
            features=features,
            data={
                "pattern_type": pattern_type,
                "confidence": confidence,
                "risk_score": risk_score,
                "features": features,
            },
        )

    def create_liquidity_analyzed_event(
        self, trading_pair_id: str, analysis: Dict[str, Any]
    ) -> LiquidityEvent:
        """Создание события анализа ликвидности."""
        return LiquidityEvent(
            event_type=EventType.LIQUIDITY_ANALYZED,
            entity_id=trading_pair_id,
            entity_type="liquidity",
            trading_pair_id=trading_pair_id,
            sufficient=analysis.get("sufficient", True),
            slippage=analysis.get("slippage", 0.0),
            depth=analysis.get("depth", 0),
            spread=analysis.get("spread", 0.0),
            volume_24h=analysis.get("volume_24h", 0.0),
            market_impact=analysis.get("market_impact", 0.0),
            data=analysis,
        )

    def create_validation_failed_event(
        self,
        entity_id: str,
        entity_type: str,
        errors: List[str],
        field_name: Optional[str] = None,
    ) -> ValidationEvent:
        """Создание события ошибки валидации."""
        return ValidationEvent(
            event_type=EventType.VALIDATION_FAILED,
            entity_id=entity_id,
            entity_type=entity_type,
            validation_type="data_validation",
            errors=errors,
            field_name=field_name,
            data={"errors": errors, "field_name": field_name},
        )

    def create_error_event(
        self,
        entity_id: str,
        entity_type: str,
        error_type: str,
        error_message: str,
        severity: str = "ERROR",
    ) -> ErrorEvent:
        """Создание события ошибки."""
        return ErrorEvent(
            event_type=EventType.ERROR_OCCURRED,
            entity_id=entity_id,
            entity_type=entity_type,
            error_type=error_type,
            error_message=error_message,
            severity=severity,
            data={
                "error_type": error_type,
                "error_message": error_message,
                "severity": severity,
            },
        )

    def validate_input(self, data: Any) -> bool:
        """Валидация входных данных."""
        return True

    def process_data(self, data: Any) -> Any:
        """Обработка данных."""
        return data
