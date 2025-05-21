import asyncio
import json
import os
import queue
import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import telegram
from loguru import logger


@dataclass
class NotificationConfig:
    """Конфигурация уведомлений"""

    # Параметры уведомлений
    enabled: bool = True  # Включены ли уведомления
    channels: List[str] = None  # Каналы уведомлений
    log_dir: str = "logs"  # Директория для логов

    # Email
    smtp_server: str = None  # SMTP сервер
    smtp_port: int = None  # SMTP порт
    smtp_username: str = None  # SMTP логин
    smtp_password: str = None  # SMTP пароль
    email_from: str = None  # Email отправителя
    email_to: List[str] = None  # Email получателей

    # Telegram
    telegram_token: str = None  # Telegram токен
    telegram_chat_id: str = None  # Telegram chat ID

    # Webhook
    webhook_url: str = None  # Webhook URL

    def __post_init__(self):
        """Инициализация параметров по умолчанию"""
        if self.channels is None:
            self.channels = ["email", "telegram", "webhook"]

        if self.email_to is None:
            self.email_to = []


class NotificationSystem:
    """Система уведомлений"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация системы уведомлений.

        Args:
            config: Словарь с параметрами уведомлений
        """
        self.config = NotificationConfig(**config) if config else NotificationConfig()
        self._setup_logger()
        self._setup_notifications()

    def _setup_logger(self):
        """Настройка логгера"""
        logger.add(
            f"{self.config.log_dir}/notifications_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
        )

    def _setup_notifications(self):
        """Настройка уведомлений"""
        try:
            # Создаем директорию для логов
            Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)

            # Инициализируем очередь для уведомлений
            self.notification_queue = queue.Queue()

            # Запускаем поток обработки уведомлений
            self.notification_thread = Thread(target=self._notification_loop, daemon=True)
            self.notification_thread.start()

        except Exception as e:
            logger.error(f"Error setting up notifications: {str(e)}")

    def _notification_loop(self):
        """Цикл обработки уведомлений"""
        try:
            while True:
                # Получаем уведомление из очереди
                notification = self.notification_queue.get()

                # Отправляем уведомление
                self._send_notification(notification)

                # Отмечаем задачу как выполненную
                self.notification_queue.task_done()

        except Exception as e:
            logger.error(f"Error in notification loop: {str(e)}")

    def _send_notification(self, notification: Dict[str, Any]):
        """
        Отправка уведомления.

        Args:
            notification: Словарь с данными уведомления
        """
        try:
            # Проверяем, включены ли уведомления
            if not self.config.enabled:
                return

            # Отправляем по каждому каналу
            for channel in self.config.channels:
                if channel == "email":
                    self._send_email(notification)
                elif channel == "telegram":
                    self._send_telegram(notification)
                elif channel == "webhook":
                    self._send_webhook(notification)

        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")

    def _send_email(self, notification: Dict[str, Any]):
        """
        Отправка email.

        Args:
            notification: Словарь с данными уведомления
        """
        try:
            # Проверяем наличие необходимых параметров
            if not all(
                [
                    self.config.smtp_server,
                    self.config.smtp_port,
                    self.config.smtp_username,
                    self.config.smtp_password,
                    self.config.email_from,
                    self.config.email_to,
                ]
            ):
                logger.warning("Email notification skipped: missing parameters")
                return

            # Создаем сообщение
            msg = MIMEMultipart()
            msg["From"] = self.config.email_from
            msg["To"] = ", ".join(self.config.email_to)
            msg["Subject"] = notification.get("subject", "Notification")

            # Добавляем текст
            msg.attach(MIMEText(notification.get("message", ""), "plain"))

            # Отправляем сообщение
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_username, self.config.smtp_password)
                server.send_message(msg)

            logger.info("Email notification sent")

        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")

    def _send_telegram(self, notification: Dict[str, Any]):
        """
        Отправка сообщения в Telegram.

        Args:
            notification: Словарь с данными уведомления
        """
        try:
            # Проверяем наличие необходимых параметров
            if not all([self.config.telegram_token, self.config.telegram_chat_id]):
                logger.warning("Telegram notification skipped: missing parameters")
                return

            # Создаем бота
            bot = telegram.Bot(token=self.config.telegram_token)

            # Формируем сообщение
            message = f"*{notification.get('subject', 'Notification')}*\n\n"
            message += notification.get("message", "")

            # Отправляем сообщение
            asyncio.run(
                bot.send_message(
                    chat_id=self.config.telegram_chat_id, text=message, parse_mode="Markdown"
                )
            )

            logger.info("Telegram notification sent")

        except Exception as e:
            logger.error(f"Error sending telegram message: {str(e)}")

    def _send_webhook(self, notification: Dict[str, Any]):
        """
        Отправка webhook.

        Args:
            notification: Словарь с данными уведомления
        """
        try:
            # Проверяем наличие необходимых параметров
            if not self.config.webhook_url:
                logger.warning("Webhook notification skipped: missing URL")
                return

            # Отправляем запрос
            response = requests.post(
                self.config.webhook_url,
                json=notification,
                headers={"Content-Type": "application/json"},
            )

            # Проверяем ответ
            if response.status_code == 200:
                logger.info("Webhook notification sent")
            else:
                logger.warning(f"Webhook notification failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Error sending webhook: {str(e)}")

    def send(self, subject: str, message: str, level: str = "info"):
        """
        Отправка уведомления.

        Args:
            subject: Тема уведомления
            message: Текст уведомления
            level: Уровень уведомления (info, warning, error)
        """
        try:
            # Формируем уведомление
            notification = {
                "subject": subject,
                "message": message,
                "level": level,
                "timestamp": datetime.now().isoformat(),
            }

            # Добавляем в очередь
            self.notification_queue.put(notification)

        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")

    def send_info(self, subject: str, message: str):
        """
        Отправка информационного уведомления.

        Args:
            subject: Тема уведомления
            message: Текст уведомления
        """
        self.send(subject, message, "info")

    def send_warning(self, subject: str, message: str):
        """
        Отправка предупреждения.

        Args:
            subject: Тема уведомления
            message: Текст уведомления
        """
        self.send(subject, message, "warning")

    def send_error(self, subject: str, message: str):
        """
        Отправка уведомления об ошибке.

        Args:
            subject: Тема уведомления
            message: Текст уведомления
        """
        self.send(subject, message, "error")

    def __enter__(self):
        """Контекстный менеджер: вход"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер: выход"""
        pass
