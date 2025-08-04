"""
Аналитика для выбора символов.
"""

from shared.numpy_utils import np
import pandas as pd
from typing import Any, Dict, List

from loguru import logger

from domain.symbols import SymbolProfile
from domain.type_definitions.symbol_types import MarketPhase

from .types import SymbolSelectionResult


class SymbolAnalytics:
    """Класс аналитики для выбора символов."""

    def __init__(self):
        self.logger = logger.bind(name=self.__class__.__name__)

    def add_advanced_analytics(
        self, result: SymbolSelectionResult, profiles: List[SymbolProfile]
    ) -> None:
        """Добавление продвинутой аналитики к результату."""
        try:
            symbols = [p.symbol for p in profiles]
            # Мок-данные для продвинутой аналитики
            result.correlation_matrix = pd.DataFrame(
                np.random.rand(len(symbols), len(symbols)),
                index=symbols,
                columns=symbols,
            )
            result.entanglement_groups = [symbols[:3]] if len(symbols) >= 3 else []
            result.pattern_memory_insights = {
                symbol: {
                    "success_rate": 0.7 + np.random.rand() * 0.2,
                    "pattern_count": np.random.randint(5, 20),
                    "avg_confidence": 0.6 + np.random.rand() * 0.3,
                }
                for symbol in symbols
            }
            result.session_alignment_scores = {
                symbol: 0.6 + np.random.rand() * 0.3 for symbol in symbols
            }
            result.liquidity_gravity_scores = {
                symbol: 0.5 + np.random.rand() * 0.4 for symbol in symbols
            }
            result.reversal_probabilities = {
                symbol: np.random.rand() * 0.5 for symbol in symbols
            }
        except Exception as e:
            self.logger.error(f"Error adding advanced analytics: {e}")

    def calculate_opportunity_score_from_metrics(
        self, metrics: Dict[str, Any], market_phase: MarketPhase
    ) -> float:
        """Расчет opportunity score из метрик."""
        try:
            # Базовый score
            base_score = 0.5
            # Корректировка по фазе рынка
            phase_multiplier = {
                MarketPhase.ACCUMULATION: 1.2,
                MarketPhase.BREAKOUT_SETUP: 1.1,
                MarketPhase.BREAKOUT_ACTIVE: 1.0,
                MarketPhase.EXHAUSTION: 0.8,
                MarketPhase.REVERSION_POTENTIAL: 0.7,
                MarketPhase.NO_STRUCTURE: 0.9,
            }.get(market_phase, 1.0)
            # Корректировка по волатильности
            volatility = metrics.get("volatility", 0.02)
            volatility_score = min(volatility * 10, 1.0)  # Нормализуем волатильность
            # Корректировка по объему
            volume = metrics.get("volume", 0)
            volume_score = min(volume / 1000000, 1.0)  # Нормализуем объем
            # Корректировка по спреду
            spread = metrics.get("spread", 0.001)
            spread_score = max(0, 1.0 - spread * 100)  # Меньший спред = лучший score
            # Итоговый расчет
            final_score = (
                base_score
                * phase_multiplier
                * (0.3 * volatility_score + 0.3 * volume_score + 0.4 * spread_score)
            )
            return min(max(final_score, 0.0), 1.0)
        except Exception as e:
            self.logger.error(f"Error calculating opportunity score: {e}")
            return 0.5

    def get_market_data_for_phase(self, symbol: str) -> pd.DataFrame:
        """Получение рыночных данных для анализа фазы."""
        try:
            # В реальной системе получаем данные из репозитория
            # Пока возвращаем мок-данные
            dates = pd.date_range(start="2024-01-01", periods=100, freq="H")
            data = pd.DataFrame(
                {
                    "timestamp": dates,
                    "open": np.random.randn(100).cumsum() + 100,
                    "high": np.random.randn(100).cumsum() + 102,
                    "low": np.random.randn(100).cumsum() + 98,
                    "close": np.random.randn(100).cumsum() + 100,
                    "volume": np.random.randint(1000, 10000, 100),
                }
            )
            return data
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return pd.DataFrame()
