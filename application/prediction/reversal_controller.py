# -*- coding: utf-8 -*-
"""Reversal Controller for Integration with AgentContext and Global Predictions."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from domain.prediction.reversal_predictor import PredictionConfig, ReversalPredictor
from domain.prediction.reversal_signal import ReversalSignal
from domain.protocols.agent_protocols import AgentContextProtocol
from domain.types.prediction_types import OrderBookData

# from infrastructure.core.analysis.global_prediction_engine import GlobalPredictionEngine  # Временно закомментировано
from shared.logging import get_logger


@dataclass
class ControllerConfig:
    """Конфигурация контроллера разворотов."""

    # Параметры интеграции
    update_interval: float = 30.0  # секунды
    max_signals_per_symbol: int = 5
    signal_lifetime: timedelta = timedelta(hours=2)

    # Пороги для интеграции
    min_agreement_score: float = 0.3
    max_controversy_threshold: float = 0.7
    confidence_boost_factor: float = 0.2
    confidence_reduction_factor: float = 0.15

    # Параметры фильтрации
    enable_signal_filtering: bool = True
    enable_controversy_detection: bool = True
    enable_agreement_scoring: bool = True

    # Параметры логирования
    log_detailed_signals: bool = True
    log_integration_events: bool = True


class ReversalController:
    """Контроллер интеграции разворотов с системой прогнозирования."""

    def __init__(
        self,
        agent_context: AgentContextProtocol,
        # global_predictor: GlobalPredictionEngine,
        config: Optional[ControllerConfig] = None,
    ):
        """
        Инициализация контроллера.

        Args:
            agent_context: Контекст агента для интеграции
            # global_predictor: Глобальный движок прогнозирования
            config: Конфигурация контроллера
        """
        self.agent_context = agent_context
        # self.global_predictor = global_predictor
        self.config = config or ControllerConfig()

        # Инициализация прогнозатора
        prediction_config = PredictionConfig(
            lookback_period=100,
            min_confidence=0.3,
            min_signal_strength=0.4,
            prediction_horizon=timedelta(hours=4),
        )
        self.reversal_predictor = ReversalPredictor(prediction_config)

        # Состояние контроллера
        self.active_signals: Dict[str, List[ReversalSignal]] = {}
        self.signal_history: List[ReversalSignal] = []
        self.integration_stats = {
            "signals_generated": 0,
            "signals_integrated": 0,
            "signals_filtered": 0,
            "controversy_detected": 0,
            "agreement_boosted": 0,
        }

        self.logger = get_logger(__name__)
        self.logger.info(f"ReversalController initialized with config: {self.config}")

    async def start_monitoring(self) -> None:
        """Запуск мониторинга и интеграции разворотов."""
        try:
            self.logger.info("Starting reversal monitoring and integration")

            while True:
                await self._process_market_data()
                await asyncio.sleep(self.config.update_interval)

        except Exception as e:
            self.logger.error(f"Error in reversal monitoring: {e}")
            raise

    async def _process_market_data(self) -> None:
        """Обработка рыночных данных и генерация сигналов."""
        try:
            # Получаем активные символы из контекста агента
            active_symbols = await self._get_active_symbols()

            for symbol in active_symbols:
                # Получаем рыночные данные
                market_data = await self._get_market_data(symbol)
                if market_data is None or len(market_data) < 100:
                    continue

                # Получаем данные ордербука
                order_book = await self._get_order_book(symbol)

                # Генерируем сигнал разворота
                signal = self.reversal_predictor.predict_reversal(
                    symbol,
                    market_data,
                    OrderBookData(order_book) if order_book else None,
                )

                if signal:
                    # Интегрируем сигнал
                    await self._integrate_signal(signal)

        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")

    async def _get_active_symbols(self) -> List[str]:
        """Получение активных торговых символов."""
        try:
            # Получаем из контекста агента
            trading_config = self.agent_context.get_trading_config()
            return trading_config.get("active_symbols", ["BTCUSDT", "ETHUSDT"])
        except Exception as e:
            self.logger.error(f"Error getting active symbols: {e}")
            return ["BTCUSDT"]

    async def _get_market_data(self, symbol: str) -> Optional[Any]:
        """Получение рыночных данных для символа."""
        try:
            # Получаем из контекста агента
            market_service = self.agent_context.get_market_service()
            data = await market_service.get_ohlcv_data(symbol, "1h", limit=200)
            return data
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return None

    async def _get_order_book(self, symbol: str) -> Optional[Dict]:
        """Получение данных ордербука."""
        try:
            # Получаем из контекста агента
            market_service = self.agent_context.get_market_service()
            order_book = await market_service.get_order_book(symbol, depth=20)
            return order_book
        except Exception as e:
            self.logger.error(f"Error getting order book for {symbol}: {e}")
            return None

    async def _integrate_signal(self, signal: ReversalSignal) -> None:
        """Интеграция сигнала разворота с системой."""
        try:
            self.integration_stats["signals_generated"] = int(self.integration_stats.get("signals_generated", 0)) + 1

            # Фильтрация сигнала
            if self.config.enable_signal_filtering:
                if not self._should_accept_signal(signal):
                    self.integration_stats["signals_filtered"] = int(self.integration_stats.get("signals_filtered", 0)) + 1
                    self.logger.debug(f"Signal filtered for {signal.symbol}: {signal}")
                    return

            # Анализ согласованности с глобальными прогнозами
            if self.config.enable_agreement_scoring:
                agreement_score = await self._calculate_agreement_score(signal)
                signal.update_agreement_score(agreement_score)

                # Усиление/ослабление уверенности на основе согласованности
                if agreement_score > 0.7:
                    signal.enhance_confidence(self.config.confidence_boost_factor)
                    self.integration_stats["agreement_boosted"] = int(self.integration_stats.get("agreement_boosted", 0)) + 1
                    self.logger.info(
                        f"Signal confidence boosted for {signal.symbol}: {agreement_score:.3f}"
                    )
                elif agreement_score < 0.3:
                    signal.reduce_confidence(self.config.confidence_reduction_factor)
                    self.logger.info(
                        f"Signal confidence reduced for {signal.symbol}: {agreement_score:.3f}"
                    )

            # Обнаружение спорных сигналов
            if self.config.enable_controversy_detection:
                controversy_reasons = await self._detect_controversy(signal)
                if controversy_reasons:
                    signal.mark_controversial(
                        reason="Multiple controversy factors detected",
                        details={"reasons": controversy_reasons},
                    )
                    self.integration_stats["controversy_detected"] = int(self.integration_stats.get("controversy_detected", 0)) + 1
                    self.logger.warning(
                        f"Controversy detected for {signal.symbol}: {controversy_reasons}"
                    )

            # Добавление сигнала в активные
            if signal.symbol not in self.active_signals:
                self.active_signals[signal.symbol] = []

            self.active_signals[signal.symbol].append(signal)

            # Ограничение количества сигналов на символ
            if (
                len(self.active_signals[signal.symbol])
                > self.config.max_signals_per_symbol
            ):
                self.active_signals[signal.symbol] = self.active_signals[signal.symbol][
                    -self.config.max_signals_per_symbol :
                ]

            # Добавление в историю
            self.signal_history.append(signal)

            # Очистка устаревших сигналов
            await self._cleanup_expired_signals()

            # Интеграция с глобальным прогнозом
            await self._integrate_with_global_prediction(signal)

            self.integration_stats["signals_integrated"] = int(self.integration_stats.get("signals_integrated", 0)) + 1

            if self.config.log_detailed_signals:
                self.logger.info(f"Signal integrated: {signal}")

        except Exception as e:
            self.logger.error(f"Error integrating signal for {signal.symbol}: {e}")

    def _should_accept_signal(self, signal: ReversalSignal) -> bool:
        """Проверка, следует ли принять сигнал."""
        try:
            # Проверка минимальной уверенности
            if signal.confidence < 0.3:
                return False

            # Проверка минимальной силы сигнала
            if signal.signal_strength < 0.4:
                return False

            # Проверка времени жизни
            if signal.is_expired:
                return False

            # Проверка на дублирование
            if signal.symbol in self.active_signals:
                for existing_signal in self.active_signals[signal.symbol]:
                    if (
                        existing_signal.direction == signal.direction
                        and abs(
                            existing_signal.pivot_price.value - signal.pivot_price.value
                        )
                        / signal.pivot_price.value
                        < 0.01
                    ):
                        return False

            return True

        except Exception as e:
            self.logger.error(f"Error in signal acceptance check: {e}")
            return False

    async def _calculate_agreement_score(self, signal: ReversalSignal) -> float:
        """Вычисление оценки согласованности с глобальными прогнозами."""
        try:
            agreement_factors = []

            # Получаем глобальный прогноз для символа
            # global_prediction = await self.global_predictor.get_prediction(
            #     signal.symbol
            # )
            # if global_prediction:
            #     # Сравнение направления
            #     if global_prediction.get("direction") == signal.direction.value:
            #         agreement_factors.append(0.4)
            #     elif global_prediction.get("direction") == "neutral":
            #         agreement_factors.append(0.2)
            #     else:
            #         agreement_factors.append(0.0)

            #     # Сравнение уровней
            #     global_level = global_prediction.get("target_price")
            #     if global_level:
            #         price_diff = (
            #             abs(global_level - signal.pivot_price.value)
            #             / signal.pivot_price.value
            #         )
            #         if price_diff < 0.02:  # 2%
            #             agreement_factors.append(0.3)
            #         elif price_diff < 0.05:  # 5%
            #             agreement_factors.append(0.2)
            #         else:
            #             agreement_factors.append(0.0)

            #     # Сравнение временного горизонта
            #     global_horizon = global_prediction.get("horizon_hours", 24)
            #     signal_horizon = signal.horizon.total_seconds() / 3600
            #     horizon_diff = abs(global_horizon - signal_horizon) / max(
            #         global_horizon, signal_horizon
            #     )
            #     if horizon_diff < 0.3:
            #         agreement_factors.append(0.3)
            #     else:
            #         agreement_factors.append(0.0)

            # Анализ согласованности с другими сигналами
            if signal.symbol in self.active_signals:
                other_signals = [
                    s for s in self.active_signals[signal.symbol] if s != signal
                ]
                if other_signals:
                    same_direction_count = sum(
                        1 for s in other_signals if s.direction == signal.direction
                    )
                    agreement_factors.append(
                        same_direction_count / len(other_signals) * 0.2
                    )

            if agreement_factors:
                return sum(agreement_factors)
            else:
                return 0.5  # Нейтральная оценка

        except Exception as e:
            self.logger.error(f"Error calculating agreement score: {e}")
            return 0.5

    async def _detect_controversy(self, signal: ReversalSignal) -> List[str]:
        """Обнаружение спорных аспектов сигнала."""
        try:
            controversy_reasons = []

            # Проверка противоречий с глобальным прогнозом
            # global_prediction = await self.global_predictor.get_prediction(
            #     signal.symbol
            # )
            # if global_prediction:
            #     if (
            #         global_prediction.get("direction") != signal.direction.value
            #         and global_prediction.get("direction") != "neutral"
            #     ):
            #         controversy_reasons.append("conflicts_with_global_prediction")

            #     global_confidence = global_prediction.get("confidence", 0.5)
            #     if abs(signal.confidence - global_confidence) > 0.4:
            #         controversy_reasons.append("confidence_mismatch")

            # Проверка противоречий с другими сигналами
            if signal.symbol in self.active_signals:
                other_signals = [
                    s for s in self.active_signals[signal.symbol] if s != signal
                ]
                if other_signals:
                    opposite_signals = [
                        s for s in other_signals if s.direction != signal.direction
                    ]
                    if len(opposite_signals) > len(other_signals) * 0.6:
                        controversy_reasons.append("multiple_opposite_signals")

            # Проверка временных аспектов
            if signal.time_to_expiry.total_seconds() < 300:  # 5 минут
                controversy_reasons.append("short_time_to_expiry")

            # Проверка силы сигнала
            if signal.signal_strength < 0.5:
                controversy_reasons.append("weak_signal_strength")

            return controversy_reasons

        except Exception as e:
            self.logger.error(f"Error detecting controversy: {e}")
            return []

    async def _integrate_with_global_prediction(self, signal: ReversalSignal) -> None:
        """Интеграция с глобальным прогнозированием."""
        try:
            # Временно закомментировано из-за отсутствия GlobalPredictionEngine
            # if self.global_predictor:
            #     global_prediction = await self.global_predictor.get_prediction(signal.symbol)
            #     if global_prediction:
            #         signal.integrate_global_prediction(global_prediction)

            # Сохраняем сигнал
            if signal.symbol not in self.active_signals:
                self.active_signals[signal.symbol] = []
            self.active_signals[signal.symbol].append(signal)
            self.signal_history.append(signal)

            self.integration_stats["signals_integrated"] = int(self.integration_stats.get("signals_integrated", 0)) + 1

            if self.config.log_integration_events:
                self.logger.info(f"Signal integrated for {signal.symbol}: {signal}")

        except Exception as e:
            self.logger.error(f"Error integrating signal: {e}")

    async def _cleanup_expired_signals(self) -> None:
        """Очистка устаревших сигналов."""
        try:
            current_time = datetime.now()

            # Очистка активных сигналов
            for symbol in list(self.active_signals.keys()):
                self.active_signals[symbol] = [
                    signal
                    for signal in self.active_signals[symbol]
                    if not signal.is_expired
                ]

                if not self.active_signals[symbol]:
                    del self.active_signals[symbol]

            # Очистка истории
            cutoff_time = current_time - self.config.signal_lifetime
            self.signal_history = [
                signal
                for signal in self.signal_history
                if signal.timestamp.to_datetime() > cutoff_time
            ]

        except Exception as e:
            self.logger.error(f"Error cleaning up expired signals: {e}")

    async def get_active_signals(
        self, symbol: Optional[str] = None
    ) -> List[ReversalSignal]:
        """Получение активных сигналов."""
        try:
            if symbol:
                return self.active_signals.get(symbol, [])
            else:
                all_signals = []
                for signals in self.active_signals.values():
                    all_signals.extend(signals)
                return all_signals
        except Exception as e:
            self.logger.error(f"Error getting active signals: {e}")
            return []

    async def get_signal_statistics(self) -> Dict[str, Any]:
        """Получение статистики сигналов."""
        try:
            stats = {
                "integration_stats": self.integration_stats.copy(),
                "active_signals_count": sum(
                    len(signals) for signals in self.active_signals.values()
                ),
                "symbols_with_signals": len(self.active_signals),
                "history_size": len(self.signal_history),
                "controller_config": {
                    "update_interval": self.config.update_interval,
                    "max_signals_per_symbol": self.config.max_signals_per_symbol,
                    "signal_lifetime_hours": self.config.signal_lifetime.total_seconds()
                    / 3600,
                },
            }

            # Статистика по направлениям
            direction_stats = {"bullish": 0, "bearish": 0, "neutral": 0}
            for signals in self.active_signals.values():
                for signal in signals:
                    direction_stats[signal.direction.value] += 1

            stats["direction_distribution"] = direction_stats

            # Статистика по силе сигналов
            strength_stats = {"weak": 0, "moderate": 0, "strong": 0, "very_strong": 0}
            for signals in self.active_signals.values():
                for signal in signals:
                    strength_stats[signal.strength_category.value] += 1

            stats["strength_distribution"] = strength_stats

            return stats

        except Exception as e:
            self.logger.error(f"Error getting signal statistics: {e}")
            return {}

    async def stop_monitoring(self) -> None:
        """Остановка мониторинга."""
        try:
            self.logger.info("Stopping reversal monitoring")
            # Очистка ресурсов
            self.active_signals.clear()
            self.signal_history.clear()

        except Exception as e:
            self.logger.error(f"Error stopping monitoring: {e}")

    def __str__(self) -> str:
        """Строковое представление контроллера."""
        return (
            f"ReversalController(active_symbols={len(self.active_signals)}, "
            f"total_signals={sum(len(s) for s in self.active_signals.values())}, "
            f"history_size={len(self.signal_history)})"
        )

    def __repr__(self) -> str:
        """Представление для отладки."""
        return self.__str__()
