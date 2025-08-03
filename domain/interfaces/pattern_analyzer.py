"""
Интерфейсы для анализа паттернов в domain слое.
"""

from datetime import datetime
from typing import Any, Dict, List, Protocol, runtime_checkable

from domain.market_maker.mm_pattern import (
    MarketMakerPattern, PatternMemory, PatternResult
)
from domain.types.market_maker_types import (
    Confidence, SimilarityScore
)


@runtime_checkable
class IPatternAnalyzer(Protocol):
    """
    Интерфейс для анализа паттернов маркет-мейкера.
    """

    async def analyze_pattern_similarity(
        self, pattern1: MarketMakerPattern, pattern2: MarketMakerPattern
    ) -> SimilarityScore:
        """
        Анализ схожести паттернов.
        Args:
            pattern1: Первый паттерн
            pattern2: Второй паттерн
        Returns:
            Оценка схожести
        """
        ...

    async def calculate_pattern_confidence(
        self, pattern: MarketMakerPattern, historical_patterns: List[PatternMemory]
    ) -> Confidence:
        """
        Расчет уверенности в паттерне.
        Args:
            pattern: Анализируемый паттерн
            historical_patterns: Исторические паттерны
        Returns:
            Уверенность в паттерне
        """
        ...

    async def predict_pattern_outcome(
        self, pattern: MarketMakerPattern, historical_patterns: List[PatternMemory]
    ) -> PatternResult:
        """
        Предсказание исхода паттерна.
        Args:
            pattern: Анализируемый паттерн
            historical_patterns: Исторические паттерны
        Returns:
            Предсказанный результат
        """
        ...

    async def analyze_market_context(
        self, symbol: str, timestamp: datetime
    ) -> Dict[str, Any]:
        """
        Анализ рыночного контекста.
        Args:
            symbol: Символ торговой пары
            timestamp: Временная метка
        Returns:
            Рыночный контекст
        """
        ...

    async def calculate_pattern_effectiveness(
        self, pattern: MarketMakerPattern, historical_patterns: List[PatternMemory]
    ) -> float:
        """
        Расчет эффективности паттерна.
        Args:
            pattern: Анализируемый паттерн
            historical_patterns: Исторические паттерны
        Returns:
            Эффективность паттерна
        """
        ...

    async def get_pattern_recommendations(
        self, symbol: str, current_patterns: List[MarketMakerPattern]
    ) -> List[Dict[str, Any]]:
        """
        Получение рекомендаций по паттернам.
        Args:
            symbol: Символ торговой пары
            current_patterns: Текущие паттерны
        Returns:
            Список рекомендаций
        """
        ... 