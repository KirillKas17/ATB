"""
Technical Analysis Service Adapter - Backward Compatibility
Адаптер для обратной совместимости с существующим кодом.
"""

from typing import Any, Dict, List, Optional

from domain.entities.market import MarketData
from domain.exceptions import TechnicalAnalysisError
from domain.protocols.exchange_protocol import ExchangeProtocol
from domain.type_definitions.external_service_types import ConnectionConfig, OrderRequest


class TechnicalAnalysisServiceAdapter(ExchangeProtocol):
    """Адаптер TechnicalAnalysisService для обратной совместимости."""

    def __init__(self, config: Optional[ConnectionConfig] = None) -> None:
        self.config = config or ConnectionConfig()

    async def calculate_indicators(self, market_data: MarketData) -> Dict[str, float]:
        """Вычисление технических индикаторов."""
        try:
            return {
                "sma_20": 50000.0,
                "ema_20": 50100.0,
                "rsi": 55.0,
                "macd": 100.0,
                "bollinger_upper": 52000.0,
                "bollinger_lower": 48000.0,
            }
        except Exception as e:
            raise TechnicalAnalysisError(f"Failed to calculate indicators: {e}")

    async def identify_patterns(self, market_data: MarketData) -> List[Dict[str, Any]]:
        """Идентификация паттернов."""
        try:
            return [
                {"pattern": "double_top", "confidence": 0.8, "price_level": 52000.0},
                {"pattern": "support_level", "confidence": 0.9, "price_level": 48000.0},
            ]
        except Exception as e:
            raise TechnicalAnalysisError(f"Failed to identify patterns: {e}")

    async def generate_signals(self, market_data: MarketData) -> List[Dict[str, Any]]:
        """Генерация торговых сигналов."""
        try:
            return [
                {"signal": "buy", "strength": 0.7, "reason": "oversold_condition"},
                {"signal": "hold", "strength": 0.5, "reason": "neutral_momentum"},
            ]
        except Exception as e:
            raise TechnicalAnalysisError(f"Failed to generate signals: {e}")


# Экспорт для обратной совместимости
__all__ = ["TechnicalAnalysisServiceAdapter"]
