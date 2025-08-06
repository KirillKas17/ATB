"""
Промышленный сервис уведомлений для application слоя.
Реализует протокол NotificationServiceProtocol из application/protocols.py
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import UUID, uuid4

from application.types import (
    NotificationChannel,
    NotificationConfig,
    NotificationLevel,
    NotificationPriority,
    NotificationTemplate,
    NotificationType,
)
from domain.entities.order import Order
from domain.entities.position import Position
from domain.entities.trading import Trade


class NotificationError(Exception):
    """Ошибка сервиса уведомлений."""

    def __init__(self, *args, **kwargs) -> Any:
        super().__init__(message)
        self.message = message
        self.details = details or {}


@dataclass
class Notification:
    """Уведомление."""

    id: UUID = field(default_factory=uuid4)
    user_id: str = ""
    title: str = ""
    message: str = ""
    level: NotificationLevel = NotificationLevel.INFO
    type: NotificationType = NotificationType.SYSTEM
    priority: NotificationPriority = NotificationPriority.NORMAL
    channels: List[NotificationChannel] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    delivered: bool = False
    read: bool = False


@dataclass
class Alert:
    """Алерт."""

    id: UUID = field(default_factory=uuid4)
    alert_type: str = ""
    title: str = ""
    message: str = ""
    severity: NotificationLevel = NotificationLevel.WARNING
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None


class NotificationChannelProtocol(ABC):
    """Протокол для каналов уведомлений."""

    @abstractmethod
    async def send(self, notification: Notification) -> bool:
        """Отправка уведомления."""
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        """Проверка доступности канала."""
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Получение имени канала."""
        ...


class EmailNotificationChannel(NotificationChannelProtocol):
    """Канал email уведомлений."""

    def __init__(self, *args, **kwargs) -> Any:
        self.smtp_config = smtp_config
        self.logger = logging.getLogger(__name__)

    async def send(self, notification: Notification) -> bool:
        """Отправка email уведомления."""
        try:
            # Здесь должна быть реальная реализация отправки email
            # Пока что просто логируем
            self.logger.info(
                f"Email notification sent to {notification.user_id}: {notification.title}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
            return False

    async def is_available(self) -> bool:
        """Проверка доступности SMTP."""
        return True  # Упрощенная проверка

    def get_name(self) -> str:
        return "email"


class WebhookNotificationChannel(NotificationChannelProtocol):
    """Канал webhook уведомлений."""

    def __init__(self, *args, **kwargs) -> Any:
        self.webhook_url = webhook_url
        self.headers = headers or {}
        self.logger = logging.getLogger(__name__)

    async def send(self, notification: Notification) -> bool:
        """Отправка webhook уведомления."""
        try:
            # Здесь должна быть реальная реализация отправки webhook
            # Пока что просто логируем
            self.logger.info(f"Webhook notification sent: {notification.title}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}")
            return False

    async def is_available(self) -> bool:
        """Проверка доступности webhook."""
        return True  # Упрощенная проверка

    def get_name(self) -> str:
        return "webhook"


class NotificationTemplateManager:
    """Менеджер шаблонов уведомлений."""

    def __init__(self) -> None:
        self.templates: Dict[str, NotificationTemplate] = {}
        self._load_default_templates()

    def _load_default_templates(self) -> None:
        """Загрузка шаблонов по умолчанию."""
        self.templates = {
            "trade_executed": NotificationTemplate(
                name="trade_executed",
                title="Trade Executed",
                message="Trade {symbol} {side} {quantity} @ {price}",
                level=NotificationLevel.INFO,
                type=NotificationType.TRADE,
                priority=NotificationPriority.NORMAL,
            ),
            "order_filled": NotificationTemplate(
                name="order_filled",
                title="Order Filled",
                message="Order {order_id} for {symbol} has been filled",
                level=NotificationLevel.INFO,
                type=NotificationType.ORDER,
                priority=NotificationPriority.NORMAL,
            ),
            "risk_alert": NotificationTemplate(
                name="risk_alert",
                title="Risk Alert",
                message="Risk level exceeded for {portfolio_id}: {risk_metric}",
                level=NotificationLevel.WARNING,
                type=NotificationType.RISK,
                priority=NotificationPriority.HIGH,
            ),
            "position_closed": NotificationTemplate(
                name="position_closed",
                title="Position Closed",
                message="Position {position_id} for {symbol} has been closed with P&L: {pnl}",
                level=NotificationLevel.INFO,
                type=NotificationType.POSITION,
                priority=NotificationPriority.NORMAL,
            ),
            "system_error": NotificationTemplate(
                name="system_error",
                title="System Error",
                message="System error occurred: {error_message}",
                level=NotificationLevel.ERROR,
                type=NotificationType.SYSTEM,
                priority=NotificationPriority.CRITICAL,
            ),
        }

    def get_template(self, template_name: str) -> Optional[NotificationTemplate]:
        """Получение шаблона по имени."""
        return self.templates.get(template_name)

    def register_template(self, template: NotificationTemplate) -> None:
        """Регистрация нового шаблона."""
        self.templates[template.name] = template

    def format_message(self, template_name: str, **kwargs) -> Optional[str]:
        """Форматирование сообщения по шаблону."""
        template = self.get_template(template_name)
        if not template:
            return None
        try:
            return template.message.format(**kwargs)
        except KeyError as e:
            raise NotificationError(f"Missing template parameter: {e}")


class NotificationService:
    """Промышленный сервис уведомлений."""

    def __init__(self, *args, **kwargs) -> Any:
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.template_manager = NotificationTemplateManager()
        self.channels: Dict[str, NotificationChannelProtocol] = {}
        self.notification_queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        self._setup_channels()

    def _setup_channels(self) -> None:
        """Настройка каналов уведомлений."""
        if self.config.email_enabled:
            self.channels["email"] = EmailNotificationChannel(self.config.email_config)
        if self.config.webhook_enabled:
            self.channels["webhook"] = WebhookNotificationChannel(
                self.config.webhook_url, self.config.webhook_headers
            )

    async def start(self) -> None:
        """Запуск сервиса уведомлений."""
        if self.is_running:
            return
        self.is_running = True
        asyncio.create_task(self._process_notification_queue())
        self.logger.info("Notification service started")

    async def stop(self) -> Any:
        """Остановка сервиса уведомлений."""
        self.is_running = False
        self.logger.info("Notification service stopped")

    async def send_notification(
        self,
        user_id: str,
        title: str,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
        channels: Optional[List[NotificationChannel]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Отправка уведомления."""
        try:
            notification = Notification(
                user_id=user_id,
                title=title,
                message=message,
                level=level,
                channels=channels or [NotificationChannel.EMAIL],
                metadata=metadata or {},
            )
            await self.notification_queue.put(notification)
            return True
        except Exception as e:
            self.logger.error(f"Failed to queue notification: {e}")
            return False

    async def send_alert(
        self,
        user_id: str,
        alert_type: str,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Отправка алерта."""
        try:
            alert = Alert(
                alert_type=alert_type, title=title, message=message, data=data or {}
            )
            # Создаем уведомление из алерта
            notification = Notification(
                user_id=user_id,
                title=alert.title,
                message=alert.message,
                level=NotificationLevel.WARNING,
                type=NotificationType.ALERT,
                priority=NotificationPriority.HIGH,
                metadata={
                    "alert_id": str(alert.id),
                    "alert_type": alert_type,
                    **alert.data,
                },
            )
            await self.notification_queue.put(notification)
            return True
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")
            return False

    async def send_trade_notification(self, trade: Trade) -> bool:
        """Отправка уведомления о сделке."""
        try:
            message = self.template_manager.format_message(
                "trade_executed",
                symbol=trade.trading_pair,
                side=trade.side.value,
                quantity=str(trade.quantity),
                price=str(trade.price),
            )
            if not message:
                raise NotificationError("Failed to format trade notification message")
            return await self.send_notification(
                user_id=str(trade.order_id),
                title="Trade Executed",
                message=message,
                level=NotificationLevel.INFO,
            )
        except Exception as e:
            self.logger.error(f"Failed to send trade notification: {e}")
            return False

    async def send_risk_alert(
        self, portfolio_id: str, risk_level: str, details: Dict[str, Any]
    ) -> bool:
        """Отправка алерта о риске."""
        try:
            message = self.template_manager.format_message(
                "risk_alert", portfolio_id=portfolio_id, risk_metric=risk_level
            )
            if not message:
                raise NotificationError("Failed to format risk alert message")
            return await self.send_alert(
                user_id=portfolio_id,
                alert_type="risk",
                title="Risk Alert",
                message=message,
                data=details,
            )
        except Exception as e:
            self.logger.error(f"Failed to send risk alert: {e}")
            return False

    async def send_order_notification(self, order: Order, event_type: str) -> bool:
        """Отправка уведомления об ордере."""
        try:
            if event_type == "filled":
                message = self.template_manager.format_message(
                    "order_filled", order_id=str(order.id), symbol=order.trading_pair
                )
                title = "Order Filled"
            else:
                message = f"Order {str(order.id)} status changed to {order.status}"
                title = "Order Status Update"
            if not message:
                raise NotificationError("Failed to format order notification message")
            return await self.send_notification(
                user_id=str(order.portfolio_id),
                title=title,
                message=message,
                level=NotificationLevel.INFO,
            )
        except Exception as e:
            self.logger.error(f"Failed to send order notification: {e}")
            return False

    async def send_position_notification(
        self, position: Position, event_type: str
    ) -> bool:
        """Отправка уведомления о позиции."""
        try:
            if event_type == "closed":
                message = self.template_manager.format_message(
                    "position_closed",
                    position_id=str(position.id),
                    symbol=position.trading_pair.symbol,
                    pnl=(
                        str(position.unrealized_pnl)
                        if position.unrealized_pnl
                        else "N/A"
                    ),
                )
                title = "Position Closed"
            else:
                message = f"Position {str(position.id)} for {position.trading_pair.symbol} updated"
                title = "Position Update"
            if not message:
                raise NotificationError(
                    "Failed to format position notification message"
                )
            return await self.send_notification(
                user_id=str(position.portfolio_id),
                title=title,
                message=message,
                level=NotificationLevel.INFO,
            )
        except Exception as e:
            self.logger.error(f"Failed to send position notification: {e}")
            return False

    async def _process_notification_queue(self) -> Any:
        """Обработка очереди уведомлений."""
        while self.is_running:
            try:
                notification = await asyncio.wait_for(
                    self.notification_queue.get(), timeout=1.0
                )
                await self._send_notification_to_channels(notification)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing notification queue: {e}")

    async def _send_notification_to_channels(self, *args, **kwargs) -> Any:
        """Отправка уведомления по каналам."""
        sent_channels = []
        for channel_name in notification.channels:
            channel = self.channels.get(channel_name.value)
            if not channel:
                self.logger.warning(f"Channel {channel_name} not available")
                continue
            if not await channel.is_available():
                self.logger.warning(f"Channel {channel_name} is not available")
                continue
            try:
                success = await channel.send(notification)
                if success:
                    sent_channels.append(channel_name)
                else:
                    self.logger.error(f"Failed to send notification via {channel_name}")
            except Exception as e:
                self.logger.error(f"Error sending notification via {channel_name}: {e}")
        if sent_channels:
            notification.sent_at = datetime.now()
            notification.delivered = True
            self.logger.info(f"Notification sent via channels: {sent_channels}")
        else:
            self.logger.error("Failed to send notification via any channel")

    def add_channel(self, name: str, channel: NotificationChannelProtocol) -> None:
        """Добавление нового канала."""
        self.channels[name] = channel
        self.logger.info(f"Added notification channel: {name}")

    def remove_channel(self, name: str) -> None:
        """Удаление канала."""
        if name in self.channels:
            del self.channels[name]
            self.logger.info(f"Removed notification channel: {name}")

    def get_available_channels(self) -> List[str]:
        """Получение списка доступных каналов."""
        return list(self.channels.keys())

    async def get_notification_stats(self) -> Dict[str, Any]:
        """Получение статистики уведомлений."""
        return {
            "queue_size": self.notification_queue.qsize(),
            "available_channels": self.get_available_channels(),
            "is_running": self.is_running,
            "templates_count": len(self.template_manager.templates),
        }
