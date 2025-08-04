# -*- coding: utf-8 -*-
"""
Промышленный модуль анализа рисков для торговых сессий.
"""
from typing import Any, Dict, List, Protocol

from domain.type_definitions.session_types import SessionAnalysisResult


class SessionRiskAnalyzerProtocol(Protocol):
    """Протокол для анализа рисков сессий."""

    def assess_risk(self, analysis: SessionAnalysisResult) -> Dict[str, Any]: ...
    def risk_alerts(self, analysis: SessionAnalysisResult) -> List[str]: ...
class SessionRiskAnalyzer(SessionRiskAnalyzerProtocol):
    """
    Промышленная реализация анализа рисков сессий.
    - Строгая типизация
    - Расширяемость стратегий
    """

    def assess_risk(self, analysis: SessionAnalysisResult) -> Dict[str, Any]:
        # Пример: простая оценка риска по метрикам
        risk = {}
        if getattr(analysis.metrics, "volatility_change_percent", 0) > 5:
            risk["volatility_risk"] = "high"
        if abs(getattr(analysis.metrics, "price_direction_bias", 0)) > 0.5:
            risk["direction_risk"] = "high"
        if getattr(analysis.metrics, "volume_change_percent", 0) < 0.1:
            risk["liquidity_risk"] = "low"
        return risk

    def risk_alerts(self, analysis: SessionAnalysisResult) -> List[str]:
        alerts = []
        if getattr(analysis.metrics, "volatility_change_percent", 0) > 10:
            alerts.append("volatility_spike")
        if getattr(analysis.metrics, "reversal_probability", 0) > 0.7:
            alerts.append("reversal_risk")
        return alerts
