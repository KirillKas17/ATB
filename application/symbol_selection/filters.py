"""
Фильтры для выбора символов.
"""

from typing import List

from loguru import logger

from domain.symbols import SymbolProfile

from .types import DOASSConfig


class SymbolFilters:
    """Класс фильтров для выбора символов."""

    def __init__(self, config: DOASSConfig):
        self.config = config
        self.logger = logger.bind(name=self.__class__.__name__)

    def passes_basic_filters(self, profile: SymbolProfile) -> bool:
        """Проверка базовых фильтров."""
        try:
            # Проверяем минимальный opportunity score
            if hasattr(profile.opportunity_score, "score"):
                score_value = float(profile.opportunity_score.score)
            else:
                score_value = (
                    float(profile.opportunity_score)
                    if isinstance(profile.opportunity_score, (int, float))
                    else 0.0
                )

            if score_value < self.config.min_opportunity_score:
                return False

            # Проверяем минимальный confidence threshold
            if hasattr(profile.opportunity_score, "confidence"):
                confidence_value = float(profile.opportunity_score.confidence)
            else:
                confidence_value = 1.0  # Дефолтное значение

            if confidence_value < self.config.min_confidence_threshold:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error in basic filters: {e}")
            return False

    async def passes_correlation_filter(
        self, profile: SymbolProfile, selected_profiles: List[SymbolProfile]
    ) -> bool:
        """Проверка корреляционного фильтра."""
        try:
            if not selected_profiles:
                return True

            # Простая проверка корреляции (в реальной системе используем CorrelationChain)
            # Пока пропускаем все
            return True

        except Exception as e:
            self.logger.error(f"Error in correlation filter: {e}")
            return True

    async def passes_pattern_memory_filter(self, profile: SymbolProfile) -> bool:
        """Проверка фильтра памяти паттернов."""
        try:
            # В реальной системе проверяем PatternMemory
            # Пока пропускаем все
            return True

        except Exception as e:
            self.logger.error(f"Error in pattern memory filter: {e}")
            return True

    async def passes_session_filter(self, profile: SymbolProfile) -> bool:
        """Проверка фильтра сессий."""
        try:
            # В реальной системе проверяем SessionEngine
            # Пока пропускаем все
            return True

        except Exception as e:
            self.logger.error(f"Error in session filter: {e}")
            return True

    async def passes_liquidity_filter(self, profile: SymbolProfile) -> bool:
        """Проверка фильтра ликвидности."""
        try:
            # Проверяем score ликвидности из профиля
            liquidity_score = float(getattr(profile.order_book_metrics, "symmetry", 0.5))
            return liquidity_score >= self.config.min_liquidity_score

        except Exception as e:
            self.logger.error(f"Error in liquidity filter: {e}")
            return True

    async def passes_reversal_filter(self, profile: SymbolProfile) -> bool:
        """Проверка фильтра предсказания разворотов."""
        try:
            # В реальной системе проверяем ReversalPredictor
            # Пока пропускаем все
            return True

        except Exception as e:
            self.logger.error(f"Error in reversal filter: {e}")
            return True

    async def apply_advanced_filters(
        self, profiles: List[SymbolProfile]
    ) -> List[SymbolProfile]:
        """Применение всех продвинутых фильтров."""
        try:
            filtered_profiles: List[SymbolProfile] = []

            for profile in profiles:
                # Базовые фильтры
                if not self.passes_basic_filters(profile):
                    continue

                # Корреляционный фильтр
                if self.config.enable_correlation_filtering:
                    if not await self.passes_correlation_filter(
                        profile, filtered_profiles
                    ):
                        continue

                # Фильтр памяти паттернов
                if self.config.enable_pattern_memory_integration:
                    if not await self.passes_pattern_memory_filter(profile):
                        continue

                # Фильтр сессий
                if self.config.enable_session_alignment:
                    if not await self.passes_session_filter(profile):
                        continue

                # Фильтр ликвидности
                if self.config.enable_liquidity_gravity:
                    if not await self.passes_liquidity_filter(profile):
                        continue

                # Фильтр предсказания разворотов
                if self.config.enable_reversal_prediction:
                    if not await self.passes_reversal_filter(profile):
                        continue

                filtered_profiles.append(profile)

            return filtered_profiles

        except Exception as e:
            self.logger.error(f"Error applying advanced filters: {e}")
            return profiles

    def apply_filters(self, profiles: dict) -> dict:
        """Применение фильтров к профилям символов."""
        try:
            # Простая реализация - возвращаем профили как есть
            # В реальной системе здесь должна быть логика фильтрации
            return profiles
        except Exception as e:
            self.logger.error(f"Error applying filters: {e}")
            return profiles
