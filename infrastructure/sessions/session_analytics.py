# -*- coding: utf-8 -*-
"""
Промышленный модуль аналитики и построения отчётов по торговым сессиям.
"""
from typing import Any, Dict, List, Protocol
import pandas as pd
from domain.type_definitions.session_types import SessionAnalysisResult


class SessionAnalyticsProtocol(Protocol):
    """Протокол для аналитики и отчётов по сессиям."""

    def build_report(self, analyses: List[SessionAnalysisResult]) -> pd.DataFrame: ...
    def summary_statistics(
        self, analyses: List[SessionAnalysisResult]
    ) -> Dict[str, Any]: ...
class SessionAnalytics(SessionAnalyticsProtocol):
    """
    Промышленная реализация аналитики и отчётов по сессиям.
    - Строгая типизация
    - Расширяемость стратегий
    """

    def build_report(self, analyses: List[SessionAnalysisResult]) -> pd.DataFrame:
        # Пример: строим DataFrame по основным метрикам
        data = [
            {
                "session_type": a.session_type.value,
                "session_phase": a.session_phase.value,
                "timestamp": a.timestamp.to_iso(),
                "confidence": float(a.confidence),
                "direction_bias": getattr(a.metrics, "price_direction_bias", 0.0),
                "volatility": getattr(a.metrics, "volatility_change_percent", 0.0),
                "volume": getattr(a.metrics, "volume_change_percent", 0.0),
            }
            for a in analyses
        ]
        return pd.DataFrame(data)

    def summary_statistics(
        self, analyses: List[SessionAnalysisResult]
    ) -> Dict[str, Any]:
        df = self.build_report(analyses)
        if df.empty:
            return {}
        return {
            "mean_confidence": df["confidence"].mean(),
            "mean_direction_bias": df["direction_bias"].mean(),
            "mean_volatility": df["volatility"].mean(),
            "mean_volume": df["volume"].mean(),
            "count": len(df),
        }
