"""
Интерфейсы для хранения паттернов маркет-мейкера.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from domain.market_maker.mm_pattern import (
    MarketMakerPattern,
    PatternMemory,
    PatternOutcome,
    PatternResult,
)
from domain.type_definitions.market_maker_types import (
    Accuracy,
    Confidence,
    MarketMakerPatternType,
    SimilarityScore,
)

from ..models.storage_models import (
    BehaviorRecord,
    PatternMetadata,
    StorageStatistics,
    SuccessMapEntry,
)


@runtime_checkable
class IPatternStorage(Protocol):
    """
    Интерфейс для хранения паттернов маркет-мейкера.
    Этот интерфейс определяет контракт для всех реализаций хранения паттернов,
    обеспечивая единообразный API для работы с различными бэкендами хранения.
    """

    async def save_pattern(self, symbol: str, pattern: MarketMakerPattern) -> bool:
        """
        Сохранение паттерна.
        Args:
            symbol: Символ торговой пары
            pattern: Паттерн для сохранения
        Returns:
            True если сохранение успешно, False в противном случае
        """
        ...

    async def update_pattern_result(
        self, symbol: str, pattern_id: str, result: PatternResult
    ) -> bool:
        """
        Обновление результата паттерна.
        Args:
            symbol: Символ торговой пары
            pattern_id: Идентификатор паттерна
            result: Результат паттерна
        Returns:
            True если обновление успешно, False в противном случае
        """
        ...

    async def get_patterns_by_symbol(
        self, symbol: str, limit: int = 100
    ) -> List[PatternMemory]:
        """
        Получение паттернов по символу.
        Args:
            symbol: Символ торговой пары
            limit: Максимальное количество паттернов
        Returns:
            Список паттернов
        """
        ...

    async def get_successful_patterns(
        self, symbol: str, min_accuracy: float = 0.7
    ) -> List[PatternMemory]:
        """
        Получение успешных паттернов.
        Args:
            symbol: Символ торговой пары
            min_accuracy: Минимальная точность
        Returns:
            Список успешных паттернов
        """
        ...

    async def find_similar_patterns(
        self, symbol: str, features: Dict[str, Any], similarity_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Поиск похожих паттернов.
        Args:
            symbol: Символ торговой пары
            features: Признаки для поиска
            similarity_threshold: Порог схожести
        Returns:
            Список похожих паттернов с метаданными
        """
        ...

    async def get_success_map(self, symbol: str) -> Dict[str, float]:
        """
        Получение карты успешности паттернов.
        Args:
            symbol: Символ торговой пары
        Returns:
            Карта успешности по типам паттернов
        """
        ...

    async def update_success_map(
        self, symbol: str, pattern_type: str, success_rate: float
    ) -> bool:
        """
        Обновление карты успешности.
        Args:
            symbol: Символ торговой пары
            pattern_type: Тип паттерна
            success_rate: Коэффициент успешности
        Returns:
            True если обновление успешно, False в противном случае
        """
        ...

    async def cleanup_old_data(self, symbol: str, days: int = 30) -> int:
        """
        Очистка старых данных.
        Args:
            symbol: Символ торговой пары
            days: Количество дней для сохранения
        Returns:
            Количество удаленных записей
        """
        ...

    async def get_storage_statistics(self) -> StorageStatistics:
        """
        Получение статистики хранилища.
        Returns:
            Статистика хранилища
        """
        ...

    async def backup_data(self, symbol: str) -> bool:
        """
        Создание резервной копии данных.
        Args:
            symbol: Символ торговой пары
        Returns:
            True если резервное копирование успешно, False в противном случае
        """
        ...

    async def restore_data(self, symbol: str, backup_timestamp: str) -> bool:
        """
        Восстановление данных из резервной копии.
        Args:
            symbol: Символ торговой пары
            backup_timestamp: Временная метка резервной копии
        Returns:
            True если восстановление успешно, False в противном случае
        """
        ...

    async def validate_data_integrity(self, symbol: str) -> bool:
        """
        Проверка целостности данных.
        Args:
            symbol: Символ торговой пары
        Returns:
            True если данные целостны, False в противном случае
        """
        ...

    async def get_pattern_metadata(self, symbol: str) -> List[PatternMetadata]:
        """
        Получение метаданных паттернов.
        Args:
            symbol: Символ торговой пары
        Returns:
            Список метаданных паттернов
        """
        ...


@runtime_checkable
class IBehaviorHistoryStorage(Protocol):
    """
    Интерфейс для хранения истории поведения маркет-мейкера.
    """

    async def save_behavior_record(self, record: BehaviorRecord) -> bool:
        """
        Сохранение записи поведения.
        Args:
            record: Запись поведения
        Returns:
            True если сохранение успешно, False в противном случае
        """
        ...

    async def get_behavior_history(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        pattern_type: Optional[MarketMakerPatternType] = None,
    ) -> List[BehaviorRecord]:
        """
        Получение истории поведения.
        Args:
            symbol: Символ торговой пары
            start_date: Начальная дата
            end_date: Конечная дата
            pattern_type: Тип паттерна для фильтрации
        Returns:
            Список записей поведения
        """
        ...

    async def get_behavior_statistics(
        self, symbol: str, days: int = 30
    ) -> Dict[str, Any]:
        """
        Получение статистики поведения.
        Args:
            symbol: Символ торговой пары
            days: Количество дней для анализа
        Returns:
            Статистика поведения
        """
        ...

    async def cleanup_old_behavior_data(self, symbol: str, days: int = 90) -> int:
        """
        Очистка старых данных поведения.
        Args:
            symbol: Символ торговой пары
            days: Количество дней для сохранения
        Returns:
            Количество удаленных записей
        """
        ...


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
