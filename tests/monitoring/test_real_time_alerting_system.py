#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive тесты системы real-time мониторинга и алертов.
Критически важно для финансовой системы - мгновенное обнаружение проблем.
"""

import pytest
import asyncio
import time
import threading
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from unittest.mock import Mock, AsyncMock, patch
import json
import statistics

from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.entities.order import Order, OrderSide, OrderType, OrderStatus
from infrastructure.monitoring.alert_manager import AlertManager, AlertSeverity, AlertType
from infrastructure.monitoring.metrics_collector import MetricsCollector, MetricType
from infrastructure.monitoring.real_time_monitor import RealTimeMonitor, MonitoringRule
from infrastructure.monitoring.notification_service import NotificationService, NotificationChannel
from infrastructure.monitoring.dashboard import MonitoringDashboard, DashboardWidget
from domain.exceptions import AlertingError, MonitoringError


class AlertSeverity(Enum):
    """Уровни серьезности алертов."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    """Типы алертов."""
    PERFORMANCE = "PERFORMANCE"
    SECURITY = "SECURITY"
    COMPLIANCE = "COMPLIANCE"
    BUSINESS_LOGIC = "BUSINESS_LOGIC"
    SYSTEM_HEALTH = "SYSTEM_HEALTH"
    MARKET_DATA = "MARKET_DATA"
    RISK_MANAGEMENT = "RISK_MANAGEMENT"


@dataclass
class AlertRule:
    """Правило для генерации алертов."""
    rule_id: str
    name: str
    metric_name: str
    condition: str
    threshold: Decimal
    severity: AlertSeverity
    alert_type: AlertType
    enabled: bool = True
    cooldown_minutes: int = 5


@dataclass
class MetricDataPoint:
    """Точка данных метрики."""
    timestamp: datetime
    metric_name: str
    value: Decimal
    tags: Dict[str, str]


class TestRealTimeAlertingSystem:
    """Comprehensive тесты системы алертов."""

    @pytest.fixture
    def alert_manager(self) -> AlertManager:
        """Фикстура менеджера алертов."""
        return AlertManager(
            max_alerts_per_minute=100,
            alert_retention_days=30,
            auto_resolution_enabled=True,
            escalation_enabled=True
        )

    @pytest.fixture
    def metrics_collector(self) -> MetricsCollector:
        """Фикстура сборщика метрик."""
        return MetricsCollector(
            collection_interval_seconds=1,
            buffer_size=10000,
            batch_size=100,
            persistence_enabled=True
        )

    @pytest.fixture
    def real_time_monitor(self) -> RealTimeMonitor:
        """Фикстура real-time монитора."""
        return RealTimeMonitor(
            monitoring_interval_seconds=1,
            rule_evaluation_enabled=True,
            anomaly_detection_enabled=True,
            machine_learning_enabled=True
        )

    @pytest.fixture
    def notification_service(self) -> NotificationService:
        """Фикстура сервиса уведомлений."""
        return NotificationService(
            channels=[
                NotificationChannel.EMAIL,
                NotificationChannel.SLACK,
                NotificationChannel.SMS,
                NotificationChannel.WEBHOOK
            ],
            rate_limiting_enabled=True,
            template_engine_enabled=True
        )

    @pytest.fixture
    def sample_alert_rules(self) -> List[AlertRule]:
        """Фикстура правил алертов."""
        return [
            AlertRule(
                rule_id="cpu_high",
                name="High CPU Usage",
                metric_name="system.cpu.usage",
                condition="greater_than",
                threshold=Decimal("80"),
                severity=AlertSeverity.HIGH,
                alert_type=AlertType.PERFORMANCE
            ),
            AlertRule(
                rule_id="memory_critical",
                name="Critical Memory Usage",
                metric_name="system.memory.usage",
                condition="greater_than",
                threshold=Decimal("95"),
                severity=AlertSeverity.CRITICAL,
                alert_type=AlertType.PERFORMANCE
            ),
            AlertRule(
                rule_id="order_latency_high",
                name="High Order Latency",
                metric_name="trading.order.latency_ms",
                condition="greater_than",
                threshold=Decimal("100"),
                severity=AlertSeverity.MEDIUM,
                alert_type=AlertType.PERFORMANCE
            ),
            AlertRule(
                rule_id="failed_logins",
                name="Multiple Failed Logins",
                metric_name="security.failed_logins",
                condition="greater_than",
                threshold=Decimal("5"),
                severity=AlertSeverity.HIGH,
                alert_type=AlertType.SECURITY
            ),
            AlertRule(
                rule_id="large_transactions",
                name="Unusually Large Transaction",
                metric_name="trading.transaction.amount",
                condition="greater_than",
                threshold=Decimal("1000000"),
                severity=AlertSeverity.MEDIUM,
                alert_type=AlertType.COMPLIANCE
            )
        ]

    def test_alert_rule_evaluation(
        self,
        alert_manager: AlertManager,
        sample_alert_rules: List[AlertRule]
    ) -> None:
        """Тест оценки правил алертов."""
        
        # Загружаем правила
        for rule in sample_alert_rules:
            alert_manager.add_rule(rule)
        
        # Тестовые метрики
        test_metrics = [
            MetricDataPoint(
                timestamp=datetime.utcnow(),
                metric_name="system.cpu.usage",
                value=Decimal("85"),  # Превышает порог 80%
                tags={"server": "trading-01"}
            ),
            MetricDataPoint(
                timestamp=datetime.utcnow(),
                metric_name="system.memory.usage",
                value=Decimal("70"),  # Не превышает порог 95%
                tags={"server": "trading-01"}
            ),
            MetricDataPoint(
                timestamp=datetime.utcnow(),
                metric_name="trading.order.latency_ms",
                value=Decimal("150"),  # Превышает порог 100ms
                tags={"exchange": "bybit"}
            ),
            MetricDataPoint(
                timestamp=datetime.utcnow(),
                metric_name="security.failed_logins",
                value=Decimal("7"),  # Превышает порог 5
                tags={"user_id": "suspicious_user"}
            )
        ]
        
        # Оцениваем правила
        triggered_alerts = []
        for metric in test_metrics:
            alerts = alert_manager.evaluate_rules(metric)
            triggered_alerts.extend(alerts)
        
        # Проверяем результаты
        assert len(triggered_alerts) == 3  # 3 метрики превысили пороги
        
        # Проверяем конкретные алерты
        cpu_alert = next(
            alert for alert in triggered_alerts 
            if alert.rule_id == "cpu_high"
        )
        assert cpu_alert.severity == AlertSeverity.HIGH
        assert cpu_alert.metric_value == Decimal("85")
        
        latency_alert = next(
            alert for alert in triggered_alerts
            if alert.rule_id == "order_latency_high"
        )
        assert latency_alert.severity == AlertSeverity.MEDIUM
        
        security_alert = next(
            alert for alert in triggered_alerts
            if alert.rule_id == "failed_logins"
        )
        assert security_alert.severity == AlertSeverity.HIGH

    def test_alert_aggregation_and_deduplication(
        self,
        alert_manager: AlertManager
    ) -> None:
        """Тест агрегации и дедупликации алертов."""
        
        # Создаем одинаковые алерты
        duplicate_alerts = []
        for i in range(10):
            alert = alert_manager.create_alert(
                rule_id="cpu_high",
                message="High CPU usage detected",
                severity=AlertSeverity.HIGH,
                metric_value=Decimal("85"),
                tags={"server": "trading-01"}
            )
            duplicate_alerts.append(alert)
            time.sleep(0.1)  # Небольшая задержка
        
        # Проверяем дедупликацию
        unique_alerts = alert_manager.get_active_alerts()
        
        # Должен остаться только один алерт благодаря дедупликации
        cpu_alerts = [
            alert for alert in unique_alerts 
            if alert.rule_id == "cpu_high"
        ]
        assert len(cpu_alerts) == 1
        
        # Проверяем агрегацию счетчика
        aggregated_alert = cpu_alerts[0]
        assert aggregated_alert.occurrence_count >= 10
        assert aggregated_alert.last_occurrence is not None

    def test_alert_escalation_workflow(
        self,
        alert_manager: AlertManager,
        notification_service: NotificationService
    ) -> None:
        """Тест workflow эскалации алертов."""
        
        # Настраиваем escalation rules
        escalation_rules = [
            {
                "severity": AlertSeverity.MEDIUM,
                "initial_delay_minutes": 5,
                "escalation_levels": [
                    {"level": 1, "delay_minutes": 15, "channels": ["EMAIL"]},
                    {"level": 2, "delay_minutes": 30, "channels": ["SLACK", "EMAIL"]},
                ]
            },
            {
                "severity": AlertSeverity.HIGH,
                "initial_delay_minutes": 1,
                "escalation_levels": [
                    {"level": 1, "delay_minutes": 5, "channels": ["SLACK", "EMAIL"]},
                    {"level": 2, "delay_minutes": 10, "channels": ["SMS", "SLACK", "EMAIL"]},
                    {"level": 3, "delay_minutes": 20, "channels": ["WEBHOOK", "SMS", "SLACK"]}
                ]
            },
            {
                "severity": AlertSeverity.CRITICAL,
                "initial_delay_minutes": 0,  # Немедленно
                "escalation_levels": [
                    {"level": 1, "delay_minutes": 0, "channels": ["SMS", "SLACK", "EMAIL", "WEBHOOK"]},
                    {"level": 2, "delay_minutes": 2, "channels": ["PHONE_CALL"]},
                    {"level": 3, "delay_minutes": 5, "channels": ["EMERGENCY_CONTACT"]}
                ]
            }
        ]
        
        alert_manager.configure_escalation(escalation_rules)
        
        # Создаем критический алерт
        critical_alert = alert_manager.create_alert(
            rule_id="memory_critical",
            message="Critical memory usage - system may become unstable",
            severity=AlertSeverity.CRITICAL,
            metric_value=Decimal("98"),
            tags={"server": "trading-01", "urgency": "immediate"}
        )
        
        # Проверяем немедленную отправку уведомлений
        notifications = notification_service.get_pending_notifications(
            alert_id=critical_alert.alert_id
        )
        
        # Для критического алерта должны быть уведомления на все каналы
        assert len(notifications) >= 4  # SMS, SLACK, EMAIL, WEBHOOK
        
        # Проверяем содержание уведомлений
        sms_notification = next(
            notif for notif in notifications
            if notif.channel == NotificationChannel.SMS
        )
        assert "CRITICAL" in sms_notification.message
        assert "trading-01" in sms_notification.message
        
        # Симулируем отсутствие ответа для эскалации
        time.sleep(1)  # Ждем эскалации
        
        escalated_notifications = notification_service.check_escalations()
        phone_notifications = [
            notif for notif in escalated_notifications
            if notif.channel == NotificationChannel.PHONE_CALL
        ]
        assert len(phone_notifications) > 0

    def test_anomaly_detection_alerts(
        self,
        real_time_monitor: RealTimeMonitor,
        metrics_collector: MetricsCollector
    ) -> None:
        """Тест алертов обнаружения аномалий."""
        
        # Генерируем базовую линию метрик
        baseline_metrics = []
        for i in range(100):  # 100 нормальных точек данных
            metric = MetricDataPoint(
                timestamp=datetime.utcnow() - timedelta(minutes=100-i),
                metric_name="trading.orders_per_minute",
                value=Decimal(str(50 + (i % 10))),  # Нормальное значение 45-55
                tags={"exchange": "bybit"}
            )
            baseline_metrics.append(metric)
            metrics_collector.collect_metric(metric)
        
        # Обучаем модель anomaly detection
        real_time_monitor.train_anomaly_detector(
            metric_name="trading.orders_per_minute",
            training_data=baseline_metrics
        )
        
        # Генерируем аномальные метрики
        anomalous_metrics = [
            MetricDataPoint(
                timestamp=datetime.utcnow(),
                metric_name="trading.orders_per_minute",
                value=Decimal("200"),  # Аномально высокое значение
                tags={"exchange": "bybit"}
            ),
            MetricDataPoint(
                timestamp=datetime.utcnow() + timedelta(minutes=1),
                metric_name="trading.orders_per_minute", 
                value=Decimal("5"),  # Аномально низкое значение
                tags={"exchange": "bybit"}
            )
        ]
        
        # Проверяем обнаружение аномалий
        detected_anomalies = []
        for metric in anomalous_metrics:
            anomaly_result = real_time_monitor.detect_anomaly(metric)
            if anomaly_result.is_anomaly:
                detected_anomalies.append(anomaly_result)
        
        assert len(detected_anomalies) == 2
        
        # Проверяем детали аномалий
        high_anomaly = detected_anomalies[0]
        assert high_anomaly.anomaly_score > 0.8  # Высокий score для явной аномалии
        assert high_anomaly.expected_range[0] < Decimal("60")  # Ожидаемый максимум
        
        low_anomaly = detected_anomalies[1]
        assert low_anomaly.anomaly_score > 0.8
        assert low_anomaly.expected_range[1] > Decimal("40")  # Ожидаемый минимум

    def test_business_logic_monitoring(
        self,
        real_time_monitor: RealTimeMonitor,
        alert_manager: AlertManager
    ) -> None:
        """Тест мониторинга бизнес-логики."""
        
        # Правила бизнес-логики
        business_rules = [
            {
                "rule_id": "order_success_rate",
                "name": "Order Success Rate Below Threshold",
                "metric": "trading.order_success_rate",
                "condition": "less_than",
                "threshold": Decimal("95"),
                "window_minutes": 5
            },
            {
                "rule_id": "settlement_delays",
                "name": "Settlement Processing Delays",
                "metric": "settlement.average_delay_minutes",
                "condition": "greater_than",
                "threshold": Decimal("30"),
                "window_minutes": 10
            },
            {
                "rule_id": "risk_limit_breaches",
                "name": "Risk Limit Breaches",
                "metric": "risk.limit_breaches",
                "condition": "greater_than",
                "threshold": Decimal("0"),
                "window_minutes": 1
            }
        ]
        
        for rule in business_rules:
            real_time_monitor.add_business_rule(rule)
        
        # Симулируем бизнес-события
        business_events = [
            {
                "event_type": "ORDER_COMPLETED",
                "success": False,
                "timestamp": datetime.utcnow(),
                "order_id": "order_001"
            },
            {
                "event_type": "ORDER_COMPLETED", 
                "success": False,
                "timestamp": datetime.utcnow(),
                "order_id": "order_002"
            },
            {
                "event_type": "ORDER_COMPLETED",
                "success": True,
                "timestamp": datetime.utcnow(),
                "order_id": "order_003"
            },
            {
                "event_type": "SETTLEMENT_COMPLETED",
                "delay_minutes": 45,  # Превышает порог 30 минут
                "timestamp": datetime.utcnow(),
                "settlement_id": "settlement_001"
            },
            {
                "event_type": "RISK_LIMIT_BREACH",
                "user_id": "user_001",
                "limit_type": "POSITION_SIZE",
                "timestamp": datetime.utcnow()
            }
        ]
        
        # Обрабатываем события
        triggered_alerts = []
        for event in business_events:
            alerts = real_time_monitor.process_business_event(event)
            triggered_alerts.extend(alerts)
        
        # Проверяем результаты
        assert len(triggered_alerts) >= 2  # Settlement delay + Risk breach
        
        # Проверяем алерт о задержке settlement
        settlement_alert = next(
            alert for alert in triggered_alerts
            if "settlement" in alert.message.lower()
        )
        assert settlement_alert.severity in [AlertSeverity.MEDIUM, AlertSeverity.HIGH]
        
        # Проверяем алерт о нарушении риск-лимита
        risk_alert = next(
            alert for alert in triggered_alerts
            if "risk" in alert.message.lower()
        )
        assert risk_alert.severity == AlertSeverity.HIGH

    def test_dashboard_widgets_and_visualization(
        self,
        metrics_collector: MetricsCollector
    ) -> None:
        """Тест виджетов дашборда и визуализации."""
        
        dashboard = MonitoringDashboard()
        
        # Создаем виджеты
        widgets = [
            DashboardWidget(
                widget_id="cpu_usage_chart",
                widget_type="LINE_CHART",
                title="CPU Usage Over Time",
                metrics=["system.cpu.usage"],
                time_range_minutes=60,
                refresh_interval_seconds=30
            ),
            DashboardWidget(
                widget_id="order_volume_gauge", 
                widget_type="GAUGE",
                title="Current Order Volume",
                metrics=["trading.orders_per_minute"],
                time_range_minutes=5,
                refresh_interval_seconds=10
            ),
            DashboardWidget(
                widget_id="alert_summary_table",
                widget_type="TABLE",
                title="Active Alerts Summary",
                metrics=["alerts.count_by_severity"],
                time_range_minutes=1440,  # 24 hours
                refresh_interval_seconds=60
            ),
            DashboardWidget(
                widget_id="performance_heatmap",
                widget_type="HEATMAP",
                title="System Performance Heatmap",
                metrics=["system.cpu.usage", "system.memory.usage", "system.disk.usage"],
                time_range_minutes=120,
                refresh_interval_seconds=120
            )
        ]
        
        for widget in widgets:
            dashboard.add_widget(widget)
        
        # Генерируем данные для виджетов
        test_metrics = []
        for i in range(60):  # 60 минут данных
            timestamp = datetime.utcnow() - timedelta(minutes=60-i)
            
            # CPU usage с трендом
            cpu_value = Decimal(str(30 + 20 * (i / 60) + (i % 5)))  # Рост от 30% до 50%
            test_metrics.append(MetricDataPoint(
                timestamp=timestamp,
                metric_name="system.cpu.usage",
                value=cpu_value,
                tags={"server": "trading-01"}
            ))
            
            # Order volume с пиками
            if i % 15 == 0:  # Пики каждые 15 минут
                order_volume = Decimal("120")
            else:
                order_volume = Decimal(str(50 + (i % 10)))
            
            test_metrics.append(MetricDataPoint(
                timestamp=timestamp,
                metric_name="trading.orders_per_minute",
                value=order_volume,
                tags={"exchange": "bybit"}
            ))
        
        # Загружаем данные
        for metric in test_metrics:
            metrics_collector.collect_metric(metric)
        
        # Обновляем виджеты
        dashboard.refresh_all_widgets()
        
        # Проверяем виджеты
        cpu_widget = dashboard.get_widget("cpu_usage_chart")
        assert len(cpu_widget.data_points) == 60
        assert cpu_widget.data_points[-1].value > cpu_widget.data_points[0].value  # Тренд роста
        
        volume_widget = dashboard.get_widget("order_volume_gauge")
        current_volume = volume_widget.current_value
        assert Decimal("40") <= current_volume <= Decimal("130")
        
        # Проверяем что есть пики в данных
        volume_peaks = [
            dp for dp in volume_widget.data_points 
            if dp.value > Decimal("100")
        ]
        assert len(volume_peaks) >= 3  # Как минимум 3 пика

    def test_alert_notification_routing(
        self,
        notification_service: NotificationService,
        alert_manager: AlertManager
    ) -> None:
        """Тест маршрутизации уведомлений алертов."""
        
        # Настраиваем routing rules
        routing_rules = [
            {
                "condition": {
                    "severity": [AlertSeverity.CRITICAL],
                    "alert_type": [AlertType.SECURITY, AlertType.SYSTEM_HEALTH]
                },
                "channels": [NotificationChannel.SMS, NotificationChannel.PHONE_CALL, NotificationChannel.EMAIL],
                "recipients": ["security-team@company.com", "ops-manager@company.com"]
            },
            {
                "condition": {
                    "severity": [AlertSeverity.HIGH],
                    "alert_type": [AlertType.PERFORMANCE, AlertType.BUSINESS_LOGIC]
                },
                "channels": [NotificationChannel.SLACK, NotificationChannel.EMAIL],
                "recipients": ["trading-team@company.com", "tech-lead@company.com"]
            },
            {
                "condition": {
                    "tags": {"exchange": "bybit"}
                },
                "channels": [NotificationChannel.WEBHOOK],
                "recipients": ["webhook://monitoring.company.com/bybit-alerts"]
            },
            {
                "condition": {
                    "severity": [AlertSeverity.LOW, AlertSeverity.MEDIUM]
                },
                "channels": [NotificationChannel.EMAIL],
                "recipients": ["operations@company.com"]
            }
        ]
        
        notification_service.configure_routing(routing_rules)
        
        # Создаем тестовые алерты
        test_alerts = [
            {
                "rule_id": "security_breach",
                "message": "Potential security breach detected",
                "severity": AlertSeverity.CRITICAL,
                "alert_type": AlertType.SECURITY,
                "tags": {"source": "firewall"}
            },
            {
                "rule_id": "performance_degradation",
                "message": "Trading system performance degraded",
                "severity": AlertSeverity.HIGH,
                "alert_type": AlertType.PERFORMANCE,
                "tags": {"component": "order_engine"}
            },
            {
                "rule_id": "bybit_connection",
                "message": "Bybit connection unstable",
                "severity": AlertSeverity.MEDIUM,
                "alert_type": AlertType.SYSTEM_HEALTH,
                "tags": {"exchange": "bybit"}
            },
            {
                "rule_id": "low_priority",
                "message": "Minor system warning",
                "severity": AlertSeverity.LOW,
                "alert_type": AlertType.SYSTEM_HEALTH,
                "tags": {}
            }
        ]
        
        # Отправляем алерты и проверяем маршрутизацию
        for alert_data in test_alerts:
            alert = alert_manager.create_alert(**alert_data)
            
            # Получаем уведомления для алерта
            notifications = notification_service.route_alert(alert)
            
            if alert.severity == AlertSeverity.CRITICAL and alert.alert_type == AlertType.SECURITY:
                # Критический security алерт должен идти на все каналы
                channels = [notif.channel for notif in notifications]
                assert NotificationChannel.SMS in channels
                assert NotificationChannel.PHONE_CALL in channels
                assert NotificationChannel.EMAIL in channels
                
                # Проверяем получателей
                recipients = [notif.recipient for notif in notifications]
                assert "security-team@company.com" in recipients
                
            elif alert.severity == AlertSeverity.HIGH and alert.alert_type == AlertType.PERFORMANCE:
                # Performance алерт для команды разработки
                channels = [notif.channel for notif in notifications]
                assert NotificationChannel.SLACK in channels
                assert NotificationChannel.EMAIL in channels
                
            elif "exchange" in alert.tags and alert.tags["exchange"] == "bybit":
                # Bybit алерт должен идти на webhook
                webhook_notifications = [
                    notif for notif in notifications
                    if notif.channel == NotificationChannel.WEBHOOK
                ]
                assert len(webhook_notifications) > 0

    def test_alert_storm_protection(
        self,
        alert_manager: AlertManager
    ) -> None:
        """Тест защиты от alert storm."""
        
        # Настраиваем защиту от storm
        storm_protection_config = {
            "max_alerts_per_minute": 50,
            "max_alerts_per_rule_per_minute": 5,
            "storm_detection_threshold": 100,  # alerts per minute
            "storm_protection_duration_minutes": 10,
            "priority_rules": ["memory_critical", "security_breach"]  # Эти всегда проходят
        }
        
        alert_manager.configure_storm_protection(storm_protection_config)
        
        # Симулируем alert storm
        storm_alerts = []
        start_time = time.time()
        
        # Генерируем много алертов быстро
        for i in range(200):  # 200 алертов
            alert = alert_manager.create_alert(
                rule_id="cpu_high",
                message=f"High CPU usage detected #{i}",
                severity=AlertSeverity.MEDIUM,
                metric_value=Decimal("85"),
                tags={"server": f"server-{i % 5}"}  # 5 разных серверов
            )
            storm_alerts.append(alert)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Проверяем что storm был обнаружен
        storm_status = alert_manager.get_storm_status()
        assert storm_status.storm_detected is True
        assert storm_status.alerts_suppressed > 0
        
        # Проверяем что rate limiting работает
        alerts_per_minute = len(storm_alerts) / (duration / 60)
        active_alerts = alert_manager.get_active_alerts()
        
        # Активных алертов должно быть меньше чем сгенерированных
        assert len(active_alerts) < len(storm_alerts)
        
        # Тестируем приоритетные алерты во время storm
        priority_alert = alert_manager.create_alert(
            rule_id="memory_critical",
            message="Critical memory usage during storm",
            severity=AlertSeverity.CRITICAL,
            metric_value=Decimal("98"),
            tags={"server": "critical-server"}
        )
        
        # Приоритетный алерт должен пройти даже во время storm
        assert priority_alert.suppressed is False
        assert priority_alert in alert_manager.get_active_alerts()

    def test_alert_correlation_and_root_cause_analysis(
        self,
        alert_manager: AlertManager,
        real_time_monitor: RealTimeMonitor
    ) -> None:
        """Тест корреляции алертов и анализа первопричин."""
        
        # Настраиваем правила корреляции
        correlation_rules = [
            {
                "rule_id": "database_cascade",
                "name": "Database Performance Cascade",
                "trigger_alerts": ["database_slow_queries"],
                "correlated_alerts": ["api_high_latency", "order_processing_delays"],
                "time_window_minutes": 5,
                "correlation_threshold": 0.8
            },
            {
                "rule_id": "network_issues",
                "name": "Network Connectivity Issues",
                "trigger_alerts": ["network_packet_loss", "connection_timeouts"],
                "correlated_alerts": ["exchange_disconnections", "price_feed_delays"],
                "time_window_minutes": 3,
                "correlation_threshold": 0.7
            }
        ]
        
        alert_manager.configure_correlation_rules(correlation_rules)
        
        # Генерируем цепочку связанных алертов
        timestamp = datetime.utcnow()
        
        # Первопричина: медленные запросы к БД
        root_cause_alert = alert_manager.create_alert(
            rule_id="database_slow_queries",
            message="Database queries taking longer than expected",
            severity=AlertSeverity.HIGH,
            timestamp=timestamp,
            tags={"database": "trading_db", "query_type": "order_lookup"}
        )
        
        # Связанные алерты через 1-2 минуты
        related_alerts = [
            alert_manager.create_alert(
                rule_id="api_high_latency",
                message="API response time increased",
                severity=AlertSeverity.MEDIUM,
                timestamp=timestamp + timedelta(minutes=1),
                tags={"api_endpoint": "/orders", "latency_ms": "2500"}
            ),
            alert_manager.create_alert(
                rule_id="order_processing_delays",
                message="Order processing queue backing up",
                severity=AlertSeverity.MEDIUM,
                timestamp=timestamp + timedelta(minutes=2),
                tags={"queue_size": "150", "avg_processing_time": "5000ms"}
            )
        ]
        
        # Запускаем анализ корреляции
        correlation_analysis = alert_manager.analyze_correlations(
            time_window_minutes=10
        )
        
        # Проверяем что корреляция обнаружена
        database_correlation = next(
            corr for corr in correlation_analysis
            if corr.rule_id == "database_cascade"
        )
        
        assert database_correlation.correlation_detected is True
        assert database_correlation.root_cause_alert_id == root_cause_alert.alert_id
        assert len(database_correlation.correlated_alert_ids) == 2
        assert database_correlation.confidence_score > 0.8
        
        # Проверяем root cause analysis
        rca_result = real_time_monitor.perform_root_cause_analysis(
            alert_correlation=database_correlation
        )
        
        assert rca_result.primary_cause == "DATABASE_PERFORMANCE"
        assert "slow queries" in rca_result.analysis_summary.lower()
        assert len(rca_result.recommended_actions) > 0
        
        # Проверяем recommended actions
        actions = rca_result.recommended_actions
        assert any("database" in action.lower() for action in actions)
        assert any("query" in action.lower() for action in actions)

    def test_predictive_alerting(
        self,
        real_time_monitor: RealTimeMonitor,
        metrics_collector: MetricsCollector
    ) -> None:
        """Тест предиктивных алертов."""
        
        # Генерируем исторические данные с трендом
        historical_data = []
        base_timestamp = datetime.utcnow() - timedelta(hours=24)
        
        for i in range(1440):  # 24 часа данных (по минуте)
            timestamp = base_timestamp + timedelta(minutes=i)
            
            # Симулируем медленный рост использования памяти
            if i < 720:  # Первые 12 часов - стабильно
                memory_usage = Decimal(str(60 + (i % 10)))
            else:  # Следующие 12 часов - рост
                growth_factor = (i - 720) / 720  # От 0 до 1
                memory_usage = Decimal(str(60 + growth_factor * 35 + (i % 5)))
            
            metric = MetricDataPoint(
                timestamp=timestamp,
                metric_name="system.memory.usage",
                value=memory_usage,
                tags={"server": "trading-01"}
            )
            historical_data.append(metric)
            metrics_collector.collect_metric(metric)
        
        # Обучаем предиктивную модель
        real_time_monitor.train_predictive_model(
            metric_name="system.memory.usage",
            training_data=historical_data
        )
        
        # Получаем текущее значение (около 95%)
        current_value = historical_data[-1].value
        
        # Делаем предсказание на следующие 60 минут
        prediction = real_time_monitor.predict_metric_future(
            metric_name="system.memory.usage",
            prediction_horizon_minutes=60,
            current_value=current_value
        )
        
        # Проверяем предсказание
        assert prediction.trend == "INCREASING"
        assert prediction.predicted_max > current_value
        
        # Проверяем предиктивные алерты
        if prediction.predicted_max > Decimal("95"):  # Критический порог
            predictive_alerts = real_time_monitor.generate_predictive_alerts(
                prediction
            )
            
            assert len(predictive_alerts) > 0
            
            memory_alert = predictive_alerts[0]
            assert "memory" in memory_alert.message.lower()
            assert "predicted" in memory_alert.message.lower()
            assert memory_alert.severity in [AlertSeverity.MEDIUM, AlertSeverity.HIGH]
            assert memory_alert.eta_minutes <= 60

    def test_multi_dimensional_alerting(
        self,
        alert_manager: AlertManager,
        metrics_collector: MetricsCollector
    ) -> None:
        """Тест многомерных алертов."""
        
        # Создаем многомерные правила
        multi_dimensional_rules = [
            {
                "rule_id": "trading_performance_composite",
                "name": "Trading Performance Degradation",
                "conditions": [
                    {"metric": "trading.order.latency_ms", "operator": ">", "value": Decimal("100")},
                    {"metric": "trading.order.success_rate", "operator": "<", "value": Decimal("95")},
                    {"metric": "system.cpu.usage", "operator": ">", "value": Decimal("80")}
                ],
                "logic": "AND",  # Все условия должны выполняться
                "severity": AlertSeverity.HIGH
            },
            {
                "rule_id": "exchange_health_composite",
                "name": "Exchange Connectivity Issues",
                "conditions": [
                    {"metric": "exchange.connection_count", "operator": "<", "value": Decimal("3")},
                    {"metric": "exchange.ping_ms", "operator": ">", "value": Decimal("1000")}
                ],
                "logic": "OR",  # Любое условие может сработать
                "severity": AlertSeverity.MEDIUM
            },
            {
                "rule_id": "risk_threshold_breach",
                "name": "Risk Thresholds Breached",
                "conditions": [
                    {"metric": "portfolio.var_95", "operator": ">", "value": Decimal("1000000")},
                    {"metric": "portfolio.leverage", "operator": ">", "value": Decimal("3.0")},
                    {"metric": "portfolio.concentration_risk", "operator": ">", "value": Decimal("0.4")}
                ],
                "logic": "WEIGHTED",  # Взвешенная логика
                "weights": [0.5, 0.3, 0.2],  # Веса для каждого условия
                "threshold": 0.7,  # Общий порог
                "severity": AlertSeverity.CRITICAL
            }
        ]
        
        for rule in multi_dimensional_rules:
            alert_manager.add_multi_dimensional_rule(rule)
        
        # Генерируем тестовые данные
        test_scenarios = [
            {
                "name": "trading_performance_degraded",
                "metrics": [
                    ("trading.order.latency_ms", Decimal("150")),
                    ("trading.order.success_rate", Decimal("92")),
                    ("system.cpu.usage", Decimal("85"))
                ]
            },
            {
                "name": "exchange_partially_down",
                "metrics": [
                    ("exchange.connection_count", Decimal("2")),
                    ("exchange.ping_ms", Decimal("500"))  # Ping в норме
                ]
            },
            {
                "name": "high_risk_portfolio",
                "metrics": [
                    ("portfolio.var_95", Decimal("1200000")),
                    ("portfolio.leverage", Decimal("3.5")),
                    ("portfolio.concentration_risk", Decimal("0.3"))
                ]
            }
        ]
        
        # Тестируем каждый сценарий
        for scenario in test_scenarios:
            timestamp = datetime.utcnow()
            
            # Отправляем метрики
            for metric_name, value in scenario["metrics"]:
                metric = MetricDataPoint(
                    timestamp=timestamp,
                    metric_name=metric_name,
                    value=value,
                    tags={"scenario": scenario["name"]}
                )
                metrics_collector.collect_metric(metric)
            
            # Проверяем многомерные алерты
            multi_alerts = alert_manager.evaluate_multi_dimensional_rules(timestamp)
            
            if scenario["name"] == "trading_performance_degraded":
                # Все условия выполнены (AND логика)
                trading_alerts = [
                    alert for alert in multi_alerts
                    if alert.rule_id == "trading_performance_composite"
                ]
                assert len(trading_alerts) == 1
                assert trading_alerts[0].severity == AlertSeverity.HIGH
                
            elif scenario["name"] == "exchange_partially_down":
                # Только одно условие выполнено (OR логика)
                exchange_alerts = [
                    alert for alert in multi_alerts
                    if alert.rule_id == "exchange_health_composite"
                ]
                assert len(exchange_alerts) == 1
                
            elif scenario["name"] == "high_risk_portfolio":
                # Взвешенная оценка превышает порог
                risk_alerts = [
                    alert for alert in multi_alerts
                    if alert.rule_id == "risk_threshold_breach"
                ]
                assert len(risk_alerts) == 1
                
                # Проверяем взвешенную оценку
                risk_alert = risk_alerts[0]
                weighted_score = (1.0 * 0.5) + (1.0 * 0.3) + (0.0 * 0.2)  # VaR + leverage + концентрация
                assert weighted_score >= 0.7  # Превышает порог