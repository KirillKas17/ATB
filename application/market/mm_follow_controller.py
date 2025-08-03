from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from domain.market_maker.mm_pattern import (
    MarketMakerPattern,
    MatchedPattern,
    PatternFeatures,
    PatternOutcome,
    PatternResult,
)
from domain.market_maker.mm_pattern_classifier import (
    IPatternClassifier,
    OrderBookSnapshot,
    TradeSnapshot,
)
from domain.market_maker.mm_pattern_memory import IPatternMemoryRepository
from domain.types import Symbol


@dataclass
class FollowSignal:
    """Сигнал для следования за паттерном"""

    symbol: str
    pattern_type: str
    confidence: float
    expected_direction: str  # "buy", "sell", "hold"
    expected_return: float
    position_size_modifier: float
    risk_modifier: float
    entry_timing: str  # "immediate", "wait", "gradual"
    stop_loss_modifier: float
    take_profit_modifier: float
    timestamp: datetime
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь"""
        return {
            "symbol": self.symbol,
            "pattern_type": self.pattern_type,
            "confidence": self.confidence,
            "expected_direction": self.expected_direction,
            "expected_return": self.expected_return,
            "position_size_modifier": self.position_size_modifier,
            "risk_modifier": self.risk_modifier,
            "entry_timing": self.entry_timing,
            "stop_loss_modifier": self.stop_loss_modifier,
            "take_profit_modifier": self.take_profit_modifier,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class FollowResult:
    """Результат следования за паттерном"""

    signal: FollowSignal
    actual_outcome: PatternOutcome
    actual_return: float
    success: bool
    execution_time: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь"""
        return {
            "signal": self.signal.to_dict(),
            "actual_outcome": self.actual_outcome.value,
            "actual_return": self.actual_return,
            "success": self.success,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
        }


class IMMFollowController(ABC):
    """Интерфейс контроллера следования за ММ"""

    @abstractmethod
    async def process_pattern(
        self, symbol: str, order_book: OrderBookSnapshot, trades: TradeSnapshot
    ) -> Optional[FollowSignal]:
        """Обработка паттерна и генерация сигнала следования"""
        ...

    @abstractmethod
    async def record_pattern_result(
        self, symbol: str, pattern: MarketMakerPattern, result: PatternResult
    ) -> bool:
        """Запись результата паттерна"""
        ...

    @abstractmethod
    async def get_follow_recommendations(self, symbol: str) -> List[FollowSignal]:
        """Получение рекомендаций по следованию"""
        ...

    @abstractmethod
    async def update_follow_result(
        self, signal: FollowSignal, result: FollowResult
    ) -> bool:
        """Обновление результата следования"""
        ...


class MarketMakerFollowController(IMMFollowController):
    """Контроллер следования за паттернами маркет-мейкера"""

    def __init__(
        self,
        pattern_classifier: IPatternClassifier,
        pattern_memory: IPatternMemoryRepository,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.pattern_classifier = pattern_classifier
        self.pattern_memory = pattern_memory
        self.config = config or {
            "min_similarity_threshold": 0.8,
            "min_accuracy_threshold": 0.7,
            "min_confidence_threshold": 0.6,
            "max_position_size_modifier": 2.0,
            "min_position_size_modifier": 0.5,
            "max_risk_modifier": 1.5,
            "min_risk_modifier": 0.5,
            "signal_validity_hours": 2,
            "max_active_signals": 5,
        }
        # Активные сигналы
        self.active_signals: Dict[str, FollowSignal] = {}
        self.signal_history: List[FollowSignal] = []
        self.follow_results: List[FollowResult] = []
        # Статистика
        self.total_signals = 0
        self.successful_signals = 0
        self.total_return = 0.0

    async def process_pattern(
        self, symbol: str, order_book: OrderBookSnapshot, trades: TradeSnapshot
    ) -> Optional[FollowSignal]:
        """Обработка паттерна и генерация сигнала следования"""
        try:
            # Классифицируем паттерн
            from domain.types.market_maker_types import Symbol as MarketMakerSymbol
            pattern = self.pattern_classifier.classify_pattern(
                MarketMakerSymbol(symbol), order_book, trades
            )
            if not pattern:
                return None
            # Ищем похожие исторические паттерны
            features = self.pattern_classifier.extract_features(order_book, trades)
            features_dict: Dict[str, Any] = (
                features.__dict__ if hasattr(features, "__dict__") else dict(features) if hasattr(features, "__iter__") else {}
            )
            similar_patterns = await self.pattern_memory.find_similar_patterns(
                symbol, features_dict, self.config["min_similarity_threshold"]
            )
            if not similar_patterns:
                # Сохраняем новый паттерн для будущего анализа
                await self.pattern_memory.save_pattern(symbol, pattern)
                return None
            # Выбираем лучший совпадающий паттерн
            best_match_data = similar_patterns[0]
            # Создаем MatchedPattern из данных
            # Исправлено: приводим типы к ожидаемым
            pattern_memory = best_match_data.get("pattern_memory")
            expected_outcome = best_match_data.get("expected_outcome")
            if pattern_memory is None:
                return None
            best_match = MatchedPattern(
                pattern_memory=pattern_memory,
                similarity_score=best_match_data.get("similarity_score", 0.0),
                confidence_boost=best_match_data.get("confidence_boost", 0.0),
                signal_strength=best_match_data.get("signal_strength", 0.0),
                expected_outcome=expected_outcome,
            )
            # Проверяем критерии для генерации сигнала
            if not self._should_generate_signal(best_match):
                return None
            # Генерируем сигнал следования
            follow_signal = self._generate_follow_signal(symbol, pattern, best_match)
            # Проверяем, что сигнал был успешно создан
            if not follow_signal:
                return None
            # Сохраняем активный сигнал
            self.active_signals[symbol] = follow_signal
            self.signal_history.append(follow_signal)
            self.total_signals = int(self.total_signals) + 1
            return follow_signal
        except Exception as e:
            print(f"Error processing pattern: {e}")
            return None

    async def record_pattern_result(
        self, symbol: str, pattern: MarketMakerPattern, result: PatternResult
    ) -> bool:
        """Запись результата паттерна"""
        try:
            # Сохраняем результат в память
            pattern_id = f"{pattern.symbol}_{pattern.pattern_type.value}_{pattern.timestamp.strftime('%Y%m%d_%H%M%S')}"
            success = await self.pattern_memory.update_pattern_result(
                symbol, pattern_id, result
            )
            # Обновляем статистику
            if result.outcome == PatternOutcome.SUCCESS:
                self.successful_signals = int(self.successful_signals) + 1
            self.total_return = float(self.total_return) + float(result.price_change_15min)
            return success
        except Exception as e:
            print(f"Error recording pattern result: {e}")
            return False

    async def get_follow_recommendations(self, symbol: str) -> List[FollowSignal]:
        """Получение рекомендаций по следованию"""
        try:
            recommendations = []
            # Проверяем активные сигналы
            if symbol in self.active_signals:
                signal = self.active_signals[symbol]
                if self._is_signal_valid(signal):
                    recommendations.append(signal)
            # Получаем успешные исторические паттерны
            successful_patterns = await self.pattern_memory.get_successful_patterns(
                symbol, self.config["min_accuracy_threshold"]
            )
            # Генерируем рекомендации на основе истории
            for pattern_memory in successful_patterns[:3]:  # Топ-3 паттерна
                if self._should_recommend_pattern(pattern_memory):
                    recommendation = self._create_recommendation_from_history(
                        symbol, pattern_memory
                    )
                    if recommendation:  # Проверяем, что рекомендация не None
                        recommendations.append(recommendation)
            return recommendations
        except Exception as e:
            print(f"Error getting follow recommendations: {e}")
            return []

    async def update_follow_result(
        self, signal: FollowSignal, result: FollowResult
    ) -> bool:
        """Обновление результата следования"""
        try:
            # Сохраняем результат
            self.follow_results.append(result)
            # Обновляем статистику
            if result.success:
                self.successful_signals = int(self.successful_signals) + 1
            self.total_return += result.actual_return
            # Удаляем сигнал из активных
            if signal.symbol in self.active_signals:
                del self.active_signals[signal.symbol]
            return True
        except Exception as e:
            print(f"Error updating follow result: {e}")
            return False

    def _should_generate_signal(self, matched_pattern: MatchedPattern) -> bool:
        """Проверка критериев для генерации сигнала"""
        try:
            # Проверяем минимальную точность
            if (
                matched_pattern.pattern_memory.accuracy
                < self.config["min_accuracy_threshold"]
            ):
                return False
            # Проверяем минимальную уверенность
            if (
                matched_pattern.confidence_boost
                < self.config["min_confidence_threshold"]
            ):
                return False
            # Проверяем количество активных сигналов
            if len(self.active_signals) >= self.config["max_active_signals"]:
                return False
            # Проверяем силу сигнала
            if (
                matched_pattern.signal_strength < 0.01
            ):  # Минимальная ожидаемая доходность
                return False
            return True
        except Exception as e:
            print(f"Error checking signal criteria: {e}")
            return False

    def _generate_follow_signal(
        self, symbol: str, pattern: MarketMakerPattern, matched_pattern: MatchedPattern
    ) -> Optional[FollowSignal]:
        """Генерация сигнала следования"""
        try:
            # Определяем направление
            expected_direction = self._determine_direction(matched_pattern)
            # Рассчитываем модификаторы
            position_size_modifier = self._calculate_position_size_modifier(
                matched_pattern
            )
            risk_modifier = self._calculate_risk_modifier(matched_pattern)
            stop_loss_modifier = self._calculate_stop_loss_modifier(matched_pattern)
            take_profit_modifier = self._calculate_take_profit_modifier(matched_pattern)
            # Определяем время входа
            entry_timing = self._determine_entry_timing(matched_pattern)
            # Создаем сигнал
            signal = FollowSignal(
                symbol=symbol,
                pattern_type=pattern.pattern_type.value,
                confidence=matched_pattern.confidence_boost,
                expected_direction=expected_direction,
                expected_return=matched_pattern.signal_strength,
                position_size_modifier=position_size_modifier,
                risk_modifier=risk_modifier,
                entry_timing=entry_timing,
                stop_loss_modifier=stop_loss_modifier,
                take_profit_modifier=take_profit_modifier,
                timestamp=datetime.now(),
                metadata={
                    "pattern_id": f"{pattern.symbol}_{pattern.pattern_type.value}_{pattern.timestamp.strftime('%Y%m%d_%H%M%S')}",
                    "similarity_score": matched_pattern.similarity_score,
                    "historical_accuracy": matched_pattern.pattern_memory.accuracy,
                    "avg_historical_return": matched_pattern.pattern_memory.avg_return,
                    "pattern_features": pattern.features.__dict__,
                },
            )
            return signal
        except Exception as e:
            print(f"Error generating follow signal: {e}")
            return None

    def _determine_direction(self, matched_pattern: MatchedPattern) -> str:
        """Определение направления торговли"""
        try:
            expected_outcome = matched_pattern.expected_outcome
            if expected_outcome.price_change_15min > 0.005:  # Ожидается рост >0.5%
                return "buy"
            elif (
                expected_outcome.price_change_15min < -0.005
            ):  # Ожидается падение >0.5%
                return "sell"
            else:
                return "hold"
        except Exception as e:
            print(f"Error determining direction: {e}")
            return "hold"

    def _calculate_position_size_modifier(
        self, matched_pattern: MatchedPattern
    ) -> float:
        """Расчет модификатора размера позиции"""
        try:
            base_modifier = (
                matched_pattern.confidence_boost
                * matched_pattern.pattern_memory.accuracy
            )
            # Ограничиваем модификатор
            modifier = max(
                self.config["min_position_size_modifier"],
                min(self.config["max_position_size_modifier"], base_modifier),
            )
            return modifier
        except Exception as e:
            print(f"Error calculating position size modifier: {e}")
            return float(1.0)

    def _calculate_risk_modifier(self, matched_pattern: MatchedPattern) -> float:
        """Расчет модификатора риска"""
        try:
            # Высокая точность снижает риск
            risk_modifier = 1.0 - (matched_pattern.pattern_memory.accuracy * 0.3)
            # Ограничиваем модификатор
            modifier = max(
                self.config["min_risk_modifier"],
                min(self.config["max_risk_modifier"], risk_modifier),
            )
            return modifier
        except Exception as e:
            print(f"Error calculating risk modifier: {e}")
            return float(1.0)

    def _calculate_stop_loss_modifier(self, matched_pattern: MatchedPattern) -> float:
        """Расчет модификатора стоп-лосса"""
        try:
            # Высокая точность позволяет более широкий стоп-лосс
            volatility = (
                matched_pattern.pattern_memory.pattern.features.price_volatility
            )
            accuracy = matched_pattern.pattern_memory.accuracy
            # Базовый модификатор на основе волатильности
            base_modifier = 1.0 + (volatility * 10)
            # Корректировка на основе точности
            if accuracy > 0.8:
                base_modifier *= 1.2  # Более широкий стоп-лосс для точных паттернов
            elif accuracy < 0.6:
                base_modifier *= 0.8  # Более узкий стоп-лосс для неточных паттернов
            return base_modifier
        except Exception as e:
            print(f"Error calculating stop loss modifier: {e}")
            return float(1.0)

    def _calculate_take_profit_modifier(self, matched_pattern: MatchedPattern) -> float:
        """Расчет модификатора тейк-профита"""
        try:
            # Высокая точность позволяет более высокий тейк-профит
            accuracy = matched_pattern.pattern_memory.accuracy
            avg_return = matched_pattern.pattern_memory.avg_return
            # Базовый модификатор на основе средней доходности
            base_modifier = 1.0 + abs(avg_return) * 10
            # Корректировка на основе точности
            if accuracy > 0.8:
                base_modifier *= 1.3  # Более высокий тейк-профит для точных паттернов
            elif accuracy < 0.6:
                base_modifier *= 0.7  # Более низкий тейк-профит для неточных паттернов
            return base_modifier
        except Exception as e:
            print(f"Error calculating take profit modifier: {e}")
            return float(1.0)

    def _determine_entry_timing(self, matched_pattern: MatchedPattern) -> str:
        """Определение времени входа"""
        try:
            pattern_type = matched_pattern.pattern_memory.pattern.pattern_type.value
            accuracy = matched_pattern.pattern_memory.accuracy
            if pattern_type in ["spoofing", "exit"]:
                return "immediate"  # Быстрые паттерны
            elif pattern_type in ["accumulation", "absorption"]:
                return "gradual"  # Медленные паттерны
            elif accuracy > 0.8:
                return "immediate"  # Высокоточные паттерны
            else:
                return "wait"  # Ожидание подтверждения
        except Exception as e:
            print(f"Error determining entry timing: {e}")
            return "wait"

    def _is_signal_valid(self, signal: FollowSignal) -> bool:
        """Проверка валидности сигнала"""
        try:
            # Проверяем время жизни сигнала
            signal_age = datetime.now() - signal.timestamp
            max_age = timedelta(hours=self.config["signal_validity_hours"])
            return signal_age <= max_age
        except Exception as e:
            print(f"Error checking signal validity: {e}")
            return False

    def _should_recommend_pattern(self, pattern_memory: Any) -> bool:
        """Проверка необходимости рекомендации паттерна"""
        try:
            # Проверяем последнее появление
            if hasattr(pattern_memory, 'last_seen') and pattern_memory.last_seen:
                time_since_last = datetime.now() - pattern_memory.last_seen
                if time_since_last < timedelta(hours=1):  # Недавно уже был
                    return False
            # Проверяем точность и количество наблюдений
            return (
                hasattr(pattern_memory, 'accuracy') and 
                pattern_memory.accuracy >= self.config["min_accuracy_threshold"]
                and hasattr(pattern_memory, 'total_count') and 
                pattern_memory.total_count >= 3
            )
        except Exception as e:
            print(f"Error checking pattern recommendation: {e}")
            return False

    def _create_recommendation_from_history(
        self, symbol: str, pattern_memory: Any
    ) -> Optional[FollowSignal]:
        """Создание рекомендации на основе исторического паттерна"""
        try:
            if not hasattr(pattern_memory, 'pattern') or not hasattr(pattern_memory, 'accuracy') or not hasattr(pattern_memory, 'avg_return'):
                return None
            pattern = pattern_memory.pattern
            # Определяем направление на основе средней доходности
            if pattern_memory.avg_return > 0.005:
                expected_direction = "buy"
            elif pattern_memory.avg_return < -0.005:
                expected_direction = "sell"
            else:
                expected_direction = "hold"
            # Создаем сигнал
            signal = FollowSignal(
                symbol=symbol,
                pattern_type=pattern.pattern_type.value,
                confidence=pattern_memory.accuracy,
                expected_direction=expected_direction,
                expected_return=abs(pattern_memory.avg_return),
                position_size_modifier=pattern_memory.accuracy,
                risk_modifier=1.0 - (pattern_memory.accuracy * 0.2),
                entry_timing="wait",  # Исторические паттерны требуют подтверждения
                stop_loss_modifier=1.0,
                take_profit_modifier=1.0 + abs(pattern_memory.avg_return) * 5,
                timestamp=datetime.now(),
                metadata={
                    "source": "historical_pattern",
                    "historical_accuracy": pattern_memory.accuracy,
                    "avg_historical_return": pattern_memory.avg_return,
                    "total_occurrences": getattr(pattern_memory, 'total_count', 0),
                    "last_seen": (
                        pattern_memory.last_seen.isoformat()
                        if hasattr(pattern_memory, 'last_seen') and pattern_memory.last_seen
                        else None
                    ),
                },
            )
            return signal
        except Exception as e:
            print(f"Error creating recommendation from history: {e}")
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики контроллера"""
        try:
            success_rate = (
                (self.successful_signals / self.total_signals * 100)
                if self.total_signals > 0
                else 0
            )
            return {
                "total_signals": self.total_signals,
                "successful_signals": self.successful_signals,
                "success_rate": success_rate,
                "total_return": self.total_return,
                "active_signals": len(self.active_signals),
                "signal_history_size": len(self.signal_history),
                "follow_results_size": len(self.follow_results),
            }
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}
