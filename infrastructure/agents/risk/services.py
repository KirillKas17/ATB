"""
Сервисы для risk agent.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd

from .analyzers import DefaultRiskCalculator, RiskMetricsCalculator
from .types import RiskConfig, RiskLevel, RiskLimits, RiskMetrics

# Type aliases для pandas
DataFrame = pd.DataFrame
Series = pd.Series


@dataclass
class AlertConfig:
    """Конфигурация алертов."""

    var_threshold: float = 0.05
    drawdown_threshold: float = 0.1
    volatility_threshold: float = 0.03
    correlation_threshold: float = 0.8
    position_size_threshold: float = 0.2
    enable_email_alerts: bool = True
    enable_slack_alerts: bool = False
    alert_cooldown_minutes: int = 30


@dataclass
class RiskAlert:
    """Алерт о риске."""

    alert_id: str
    alert_type: str
    severity: str
    message: str
    timestamp: datetime
    portfolio_id: Optional[str] = None
    position_id: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False


class RiskMonitoringService:
    """Сервис мониторинга рисков."""

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.calculator = DefaultRiskCalculator(config)
        self.metrics_calculator = RiskMetricsCalculator(config)
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.risk_history: List[RiskMetrics] = []
        self.max_history_size = 1000

    async def start_monitoring(
        self, portfolio_id: str, update_interval: int = 60
    ) -> None:
        """Запуск мониторинга рисков."""
        if self.monitoring_active:
            return
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(portfolio_id, update_interval)
        )

    async def stop_monitoring(self) -> None:
        """Остановка мониторинга рисков."""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

    async def _monitoring_loop(self, portfolio_id: str, update_interval: int) -> None:
        """Основной цикл мониторинга."""
        while self.monitoring_active:
            try:
                # Получение данных портфеля
                portfolio_data = await self._get_portfolio_data(portfolio_id)
                # Расчет рисков
                risk_metrics = self.calculator.calculate_portfolio_risk(portfolio_data)
                # Сохранение в историю
                self._update_risk_history(risk_metrics)
                # Проверка на аномалии
                await self._check_risk_anomalies(portfolio_id, risk_metrics)
                # Ожидание следующего обновления
                await asyncio.sleep(update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in risk monitoring: {e}")
                await asyncio.sleep(update_interval)

    async def _get_portfolio_data(self, portfolio_id: str) -> Dict[str, Any]:
        """Получение данных портфеля."""
        # В реальной системе здесь был бы запрос к репозиторию
        return {
            "portfolio_id": portfolio_id,
            "positions": [],
            "market_data": {},
            "equity_curve": [],
            "returns": [],
            "market_returns": [],
        }

    def _update_risk_history(self, risk_metrics: RiskMetrics) -> None:
        """Обновление истории рисков."""
        self.risk_history.append(risk_metrics)
        # Ограничение размера истории
        if len(self.risk_history) > self.max_history_size:
            self.risk_history = self.risk_history[-self.max_history_size :]

    async def _check_risk_anomalies(
        self, portfolio_id: str, risk_metrics: RiskMetrics
    ) -> None:
        """Проверка на аномалии рисков."""
        details = risk_metrics.details
        if details is None:
            return
            
        # Проверка VaR
        var_95 = details.get("var_95", 0)
        if var_95 > self.config.threshold:
            await self._trigger_alert(
                portfolio_id=portfolio_id,
                alert_type="high_var",
                severity="high",
                message=f"VaR превышает порог: {var_95:.2%}",
                metrics=details,
            )
        # Проверка просадки
        max_dd = details.get("max_drawdown", 0)
        if max_dd > 0.1:  # 10% просадка
            await self._trigger_alert(
                portfolio_id=portfolio_id,
                alert_type="high_drawdown",
                severity="high",
                message=f"Высокая просадка: {max_dd:.2%}",
                metrics=details,
            )
        # Проверка волатильности
        volatility = details.get("volatility", 0)
        if volatility > 0.03:  # 3% дневная волатильность
            await self._trigger_alert(
                portfolio_id=portfolio_id,
                alert_type="high_volatility",
                severity="medium",
                message=f"Высокая волатильность: {volatility:.2%}",
                metrics=details,
            )

    async def _trigger_alert(
        self,
        portfolio_id: str,
        alert_type: str,
        severity: str,
        message: str,
        metrics: Dict[str, Any],
    ) -> None:
        """Создание алерта."""
        alert = RiskAlert(
            alert_id=f"{alert_type}_{datetime.now().timestamp()}",
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            portfolio_id=portfolio_id,
            metrics=metrics,
        )
        # Отправка алерта
        await self._send_alert(alert)

    async def _send_alert(self, alert: RiskAlert) -> None:
        """Отправка алерта."""
        # В реальной системе здесь была бы интеграция с системами уведомлений
        print(f"Risk Alert: {alert.severity.upper()} - {alert.message}")

    def get_risk_history(self, hours: int = 24) -> List[RiskMetrics]:
        """Получение истории рисков за указанный период."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            rm
            for rm in self.risk_history
            if hasattr(rm, "timestamp") and rm.timestamp > cutoff_time
        ]

    def get_risk_trend(self, hours: int = 24) -> Dict[str, float]:
        """Получение тренда рисков."""
        history = self.get_risk_history(hours)
        if not history:
            return {}
        risk_values = [rm.value for rm in history]
        return {
            "current": risk_values[-1] if risk_values else 0.0,
            "average": float(np.mean(risk_values)),
            "max": float(np.max(risk_values)),
            "min": float(np.min(risk_values)),
            "trend": self._calculate_trend(risk_values),
        }

    def _calculate_trend(self, values: List[float]) -> float:
        """Расчет тренда (наклон линии тренда)."""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)


class RiskAlertService:
    """Сервис алертов рисков."""

    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self.alerts: List[RiskAlert] = []
        self.alert_handlers: Dict[str, Callable] = {}
        self.alert_cooldowns: Dict[str, datetime] = {}

    def register_alert_handler(self, alert_type: str, handler: Callable) -> None:
        """Регистрация обработчика алертов."""
        self.alert_handlers[alert_type] = handler

    async def process_alert(self, alert: RiskAlert) -> None:
        """Обработка алерта."""
        # Проверка кулдауна
        if self._is_in_cooldown(alert.alert_type):
            return
        # Добавление алерта в список
        self.alerts.append(alert)
        # Вызов обработчика
        if alert.alert_type in self.alert_handlers:
            try:
                await self.alert_handlers[alert.alert_type](alert)
            except Exception as e:
                print(f"Error in alert handler: {e}")
        # Установка кулдауна
        self._set_cooldown(alert.alert_type)
        # Очистка старых алертов
        self._cleanup_old_alerts()

    def _is_in_cooldown(self, alert_type: str) -> bool:
        """Проверка кулдауна для типа алерта."""
        if alert_type not in self.alert_cooldowns:
            return False
        cooldown_time = self.alert_cooldowns[alert_type]
        return datetime.now() < cooldown_time

    def _set_cooldown(self, alert_type: str) -> None:
        """Установка кулдауна для типа алерта."""
        cooldown_duration = timedelta(minutes=self.config.alert_cooldown_minutes)
        self.alert_cooldowns[alert_type] = datetime.now() + cooldown_duration

    def _cleanup_old_alerts(self) -> None:
        """Очистка старых алертов."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_time]

    def get_alerts(
        self,
        portfolio_id: Optional[str] = None,
        alert_type: Optional[str] = None,
        severity: Optional[str] = None,
        hours: int = 24,
    ) -> List[RiskAlert]:
        """Получение алертов с фильтрацией."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        filtered_alerts = []
        for alert in self.alerts:
            if alert.timestamp < cutoff_time:
                continue
            if portfolio_id and alert.portfolio_id != portfolio_id:
                continue
            if alert_type and alert.alert_type != alert_type:
                continue
            if severity and alert.severity != severity:
                continue
            filtered_alerts.append(alert)
        return sorted(filtered_alerts, key=lambda x: x.timestamp, reverse=True)

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Подтверждение алерта."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def get_alert_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Получение статистики алертов."""
        alerts = self.get_alerts(hours=hours)
        if not alerts:
            return {
                "total_alerts": 0,
                "by_type": {},
                "by_severity": {},
                "acknowledged_rate": 0.0,
            }
        # Статистика по типам
        by_type: dict[str, int] = {}
        for alert in alerts:
            by_type[alert.alert_type] = by_type.get(alert.alert_type, 0) + 1
        # Статистика по серьезности
        by_severity: dict[str, int] = {}
        for alert in alerts:
            by_severity[alert.severity] = by_severity.get(alert.severity, 0) + 1
        # Процент подтвержденных
        acknowledged = sum(1 for alert in alerts if alert.acknowledged)
        acknowledged_rate = acknowledged / len(alerts)
        return {
            "total_alerts": len(alerts),
            "by_type": by_type,
            "by_severity": by_severity,
            "acknowledged_rate": acknowledged_rate,
        }
