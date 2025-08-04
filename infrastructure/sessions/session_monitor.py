# -*- coding: utf-8 -*-
"""
Промышленный модуль мониторинга и алертов по торговым сессиям.
"""
from typing import Any, Callable, Dict, List, Protocol
from domain.type_definitions.session_types import SessionAnalysisResult


class SessionMonitorProtocol(Protocol):
    """Протокол для мониторинга и алертов по сессиям."""

    def register_alert(
        self, alert_name: str, condition: Callable[[SessionAnalysisResult], bool]
    ) -> None: ...
    def check_alerts(self, analysis: SessionAnalysisResult) -> List[str]: ...
class SessionMonitor(SessionMonitorProtocol):
    """
    Промышленная реализация мониторинга и алертов по сессиям.
    - Строгая типизация
    - Расширяемость стратегий
    """

    def __init__(self) -> None:
        self._alerts: Dict[str, Callable[[SessionAnalysisResult], bool]] = {}

    def register_alert(
        self, alert_name: str, condition: Callable[[SessionAnalysisResult], bool]
    ) -> None:
        self._alerts[alert_name] = condition

    def check_alerts(self, analysis: SessionAnalysisResult) -> List[str]:
        triggered = []
        for name, cond in self._alerts.items():
            try:
                if cond(analysis):
                    triggered.append(name)
            except Exception:
                continue
        return triggered
