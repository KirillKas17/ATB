"""
Протокол для извлечения паттернов цен.
"""

from decimal import Decimal
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import pandas as pd  # type: ignore


@runtime_checkable
class PricePattern(Protocol):
    """Протокол для паттерна цены."""

    pattern_type: str
    confidence: Decimal
    start_index: int
    end_index: int
    features: Dict[str, Any]
    metadata: Dict[str, Any]


@runtime_checkable
class PricePatternExtractorProtocol(Protocol):
    """Протокол для извлечения паттернов цен."""

    def extract_patterns(self, price_data: pd.DataFrame) -> List[PricePattern]:
        """Извлечение паттернов из данных цен."""
        ...

    def detect_support_resistance(
        self, price_data: pd.DataFrame
    ) -> Tuple[List[Decimal], List[Decimal]]:
        """Обнаружение уровней поддержки и сопротивления."""
        ...

    def find_pivot_points(self, price_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Поиск точек разворота."""
        ...

    def detect_trend_changes(self, price_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Обнаружение изменений тренда."""
        ...

    def extract_volatility_patterns(
        self, price_data: pd.DataFrame
    ) -> List[PricePattern]:
        """Извлечение паттернов волатильности."""
        ...

    def detect_breakouts(self, price_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Обнаружение пробоев."""
        ...

    def find_divergences(
        self, price_data: pd.DataFrame, indicator_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Поиск дивергенций."""
        ...

    def calculate_pattern_metrics(self, pattern: PricePattern) -> Dict[str, Decimal]:
        """Расчет метрик паттерна."""
        ...

    def validate_pattern(self, pattern: PricePattern) -> bool:
        """Валидация паттерна."""
        ...

    def get_pattern_statistics(self, patterns: List[PricePattern]) -> Dict[str, Any]:
        """Получение статистики паттернов."""
        ...


# Алиас для обратной совместимости
PricePatternExtractor = PricePatternExtractorProtocol
