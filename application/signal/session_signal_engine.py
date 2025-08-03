# -*- coding: utf-8 -*-
"""Движок генерации сигналов влияния торговых сессий."""
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
import pandas as pd

from loguru import logger

from domain.sessions.session_influence_analyzer import (
    SessionInfluenceAnalyzer,
    SessionInfluenceResult,
)
from domain.sessions.session_marker import MarketSessionContext, SessionMarker
from domain.value_objects.timestamp import Timestamp


@dataclass
class SessionInfluenceSignal:
    """Сигнал влияния торговой сессии."""

    symbol: str
    score: float  # -1.0 to 1.0
    tendency: Literal["bullish", "bearish", "neutral"]
    confidence: float  # 0.0 to 1.0
    session_type: str
    session_phase: str
    timestamp: Timestamp
    # Дополнительные характеристики
    volatility_impact: float = 0.0
    volume_impact: float = 0.0
    momentum_impact: float = 0.0
    reversal_probability: float = 0.0
    false_breakout_probability: float = 0.0
    # Метаданные
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "symbol": self.symbol,
            "score": self.score,
            "tendency": self.tendency,
            "confidence": self.confidence,
            "session_type": self.session_type,
            "session_phase": self.session_phase,
            "timestamp": self.timestamp.to_iso(),
            "volatility_impact": self.volatility_impact,
            "volume_impact": self.volume_impact,
            "momentum_impact": self.momentum_impact,
            "reversal_probability": self.reversal_probability,
            "false_breakout_probability": self.false_breakout_probability,
            "metadata": self.metadata,
        }


class SessionSignalEngine:
    """Движок генерации сигналов влияния торговых сессий."""

    def __init__(
        self,
        session_analyzer: Optional[SessionInfluenceAnalyzer] = None,
        session_marker: Optional[SessionMarker] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.session_marker = session_marker or SessionMarker()
        # Создаем заглушку для registry, если не передана
        registry: Dict[str, Any] = {}  # Заглушка для registry
        self.session_analyzer = session_analyzer or SessionInfluenceAnalyzer(
            registry=registry, session_marker=self.session_marker  # type: ignore
        )
        self.config = config or {
            "signal_update_interval_seconds": 900,  # 15 минут
            "min_confidence_threshold": 0.6,
            "max_signals_per_symbol": 10,
            "signal_ttl_minutes": 60,
            "enable_real_time_updates": True,
            "enable_historical_analysis": True,
        }
        # Хранилище сигналов
        self.signals: Dict[str, List[SessionInfluenceSignal]] = {}
        self.signal_history: Dict[str, List[SessionInfluenceSignal]] = {}
        # Статистика
        self.stats = {
            "total_signals_generated": 0,
            "high_confidence_signals": 0,
            "bullish_signals": 0,
            "bearish_signals": 0,
            "neutral_signals": 0,
        }
        # Флаг работы
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        logger.info("SessionSignalEngine initialized")

    async def start(self) -> None:
        """Запуск движка сигналов."""
        if self._running:
            logger.warning("SessionSignalEngine is already running")
            return
        self._running = True
        if self.config["enable_real_time_updates"]:
            self._update_task = asyncio.create_task(self._signal_update_loop())
            logger.info("SessionSignalEngine started with real-time updates")
        else:
            logger.info("SessionSignalEngine started without real-time updates")

    async def stop(self) -> None:
        """Остановка движка сигналов."""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                # Корректная обработка отмены задачи
                logger.debug("Update task cancelled successfully")
        logger.info("SessionSignalEngine stopped")

    async def generate_signal(
        self,
        symbol: str,
        market_data: Optional[Dict[str, Any]] = None,
        timestamp: Optional[Timestamp] = None,
    ) -> Optional[SessionInfluenceSignal]:
        """
        Генерация сигнала влияния сессии для символа.
        Args:
            symbol: Торговая пара
            market_data: Рыночные данные
            timestamp: Время генерации сигнала
        Returns:
            Сигнал влияния сессии
        """
        try:
            if timestamp is None:
                timestamp = Timestamp.now()
            # Получаем контекст сессий
            session_context = self.session_marker.get_session_context(timestamp)
            if not session_context.primary_session:
                logger.debug(f"No active session for {symbol} at {timestamp}")
                return None
            # Анализируем влияние сессии
            if market_data:
                # Преобразуем в DataFrame если нужно
                if isinstance(market_data, dict):
                    market_df = pd.DataFrame([market_data])
                else:
                    market_df = market_data
                influence_result = self.session_analyzer.analyze_session_influence(
                    symbol, market_df, session_context, timestamp
                )
            else:
                # Используем базовый анализ без рыночных данных
                influence_result = self._generate_basic_influence_result(
                    symbol, session_context, timestamp
                )
            if not influence_result:
                return None
            # Создаем сигнал
            signal = self._create_signal_from_influence_result(influence_result)
            if signal and signal.confidence >= self.config["min_confidence_threshold"]:
                # Сохраняем сигнал
                self._store_signal(symbol, signal)
                # Обновляем статистику
                self._update_stats(signal)
                logger.info(
                    f"Generated session signal for {symbol}: "
                    f"tendency={signal.tendency}, confidence={signal.confidence:.3f}, "
                    f"session={signal.session_type}, phase={signal.session_phase}"
                )
            return signal
        except Exception as e:
            logger.error(f"Error generating session signal for {symbol}: {e}")
            return None

    async def get_current_signals(self, symbol: str) -> List[SessionInfluenceSignal]:
        """Получение текущих сигналов для символа."""
        if symbol not in self.signals:
            return []
        # Фильтруем актуальные сигналы
        current_time = Timestamp.now()
        ttl_minutes = self.config["signal_ttl_minutes"]
        active_signals = []
        for signal in self.signals[symbol]:
            signal_age_minutes = current_time.time_difference_minutes(signal.timestamp)
            if signal_age_minutes <= ttl_minutes:
                active_signals.append(signal)
        return active_signals

    async def get_aggregated_signal(
        self, symbol: str
    ) -> Optional[SessionInfluenceSignal]:
        """Получение агрегированного сигнала для символа."""
        current_signals = await self.get_current_signals(symbol)
        if not current_signals:
            return None
        # Агрегируем сигналы по весу уверенности
        total_weight = 0.0
        weighted_score = 0.0
        weighted_confidence = 0.0
        for signal in current_signals:
            weight = signal.confidence
            total_weight += weight
            weighted_score += signal.score * weight
            weighted_confidence += signal.confidence * weight
        if total_weight == 0:
            return None
        # Вычисляем средневзвешенные значения
        avg_score = weighted_score / total_weight
        avg_confidence = weighted_confidence / total_weight
        # Определяем общую тенденцию
        if avg_score > 0.1:
            aggregated_tendency: Literal["bullish", "bearish", "neutral"] = "bullish"
        elif avg_score < -0.1:
            aggregated_tendency = "bearish"
        else:
            aggregated_tendency = "neutral"
        # Создаем агрегированный сигнал
        aggregated_signal = SessionInfluenceSignal(
            symbol=symbol,
            score=avg_score,
            tendency=aggregated_tendency,
            confidence=avg_confidence,
            session_type=current_signals[0].session_type,
            session_phase=current_signals[0].session_phase,
            timestamp=Timestamp.now(),
            volatility_impact=sum(s.volatility_impact for s in current_signals) / len(current_signals),
            volume_impact=sum(s.volume_impact for s in current_signals) / len(current_signals),
            momentum_impact=sum(s.momentum_impact for s in current_signals) / len(current_signals),
            reversal_probability=sum(s.reversal_probability for s in current_signals) / len(current_signals),
            false_breakout_probability=sum(s.false_breakout_probability for s in current_signals) / len(current_signals),
        )
        return aggregated_signal

    def get_session_analysis(self, symbol: str) -> Dict[str, Any]:
        """Получение анализа сессии для символа."""
        try:
            current_signals = self.signals.get(symbol, [])
            if not current_signals:
                return {
                    "symbol": symbol,
                    "has_signals": False,
                    "analysis": None,
                }
            # Получаем последний сигнал
            latest_signal = max(current_signals, key=lambda s: s.timestamp)
            # Получаем сводку влияния
            influence_summary = self.session_analyzer.get_influence_summary(symbol)  # type: ignore
            analysis = {
                "symbol": symbol,
                "has_signals": True,
                "latest_signal": latest_signal.to_dict(),
                "total_signals": len(current_signals),
                "avg_confidence": sum(s.confidence for s in current_signals) / len(current_signals),
                "tendency_distribution": {
                    "bullish": len([s for s in current_signals if s.tendency == "bullish"]),
                    "bearish": len([s for s in current_signals if s.tendency == "bearish"]),
                    "neutral": len([s for s in current_signals if s.tendency == "neutral"]),
                },
                "influence_summary": influence_summary,
            }
            return analysis
        except Exception as e:
            logger.error(f"Error getting session analysis for {symbol}: {e}")
            return {
                "symbol": symbol,
                "has_signals": False,
                "analysis": None,
                "error": str(e),
            }

    def _create_signal_from_influence_result(
        self, influence_result: SessionInfluenceResult
    ) -> Optional[SessionInfluenceSignal]:
        """Создание сигнала из результата анализа влияния."""
        try:
            # Вычисляем score на основе метрик влияния
            score = (
                influence_result.influence_metrics.price_direction_bias * 0.4 +
                influence_result.influence_metrics.momentum_strength * 0.3 +
                (influence_result.confidence - 0.5) * 0.3
            )
            # Ограничиваем score в диапазоне [-1, 1]
            score = max(-1.0, min(1.0, score))
            
            # Определяем тенденцию на основе score
            if score > 0.1:
                signal_tendency: Literal["bullish", "bearish", "neutral"] = "bullish"
            elif score < -0.1:
                signal_tendency = "bearish"
            else:
                signal_tendency = "neutral"
            # Создаем сигнал
            signal = SessionInfluenceSignal(
                symbol=influence_result.symbol,
                score=score,
                tendency=signal_tendency,
                confidence=influence_result.confidence,
                session_type=influence_result.session_type.value,
                session_phase=influence_result.session_phase.value,
                timestamp=influence_result.timestamp,
                volatility_impact=influence_result.influence_metrics.volatility_change_percent,
                volume_impact=influence_result.influence_metrics.volume_change_percent,
                momentum_impact=influence_result.influence_metrics.momentum_strength,
                reversal_probability=influence_result.influence_metrics.reversal_probability,
                false_breakout_probability=influence_result.influence_metrics.false_breakout_probability,
                metadata=influence_result.market_context,
            )
            return signal
        except Exception as e:
            logger.error(f"Error creating signal from influence result: {e}")
            return None

    def _generate_basic_influence_result(
        self, symbol: str, session_context: MarketSessionContext, timestamp: Timestamp
    ) -> Optional[SessionInfluenceResult]:
        """Генерация базового результата влияния без рыночных данных."""
        try:
            # Создаем базовый результат влияния
            from domain.sessions.session_influence_analyzer import SessionInfluenceResult, SessionInfluenceMetrics
            from domain.types.session_types import SessionType, SessionPhase
            
            # Определяем тип сессии
            session_type = SessionType.ASIAN  # По умолчанию
            if session_context.primary_session:
                try:
                    session_type = SessionType(session_context.primary_session)
                except ValueError:
                    pass
            
            # Определяем фазу сессии
            session_phase = SessionPhase.MID_SESSION  # Используем существующую фазу
            
            # Создаем базовые метрики влияния
            influence_metrics = SessionInfluenceMetrics(
                volatility_change_percent=0.1,
                volume_change_percent=0.1,
                price_direction_bias=0.0,
                momentum_strength=0.1,
                reversal_probability=0.3,
                false_breakout_probability=0.2,
                trend_continuation_probability=0.6,
                influence_duration_minutes=30,
                peak_influence_time_minutes=15,
                spread_impact=0.1,
                liquidity_impact=0.2,
                correlation_with_other_sessions=0.5,
            )
            
            # Создаем результат влияния
            result = SessionInfluenceResult(
                symbol=symbol,
                session_type=session_type,
                session_phase=session_phase,
                timestamp=timestamp,
                influence_metrics=influence_metrics,
                predicted_volatility=0.1,
                predicted_volume=0.1,
                predicted_direction="neutral",
                confidence=0.5,
                market_context={},
                historical_patterns=[],
            )
            return result
        except Exception as e:
            logger.error(f"Error generating basic influence result: {e}")
            return None

    def _store_signal(self, symbol: str, signal: SessionInfluenceSignal) -> None:
        """Сохранение сигнала."""
        if symbol not in self.signals:
            self.signals[symbol] = []
        # Добавляем сигнал
        self.signals[symbol].append(signal)
        # Ограничиваем количество сигналов
        max_signals = self.config["max_signals_per_symbol"]
        if len(self.signals[symbol]) > max_signals:
            self.signals[symbol] = self.signals[symbol][-max_signals:]
        # Добавляем в историю
        if symbol not in self.signal_history:
            self.signal_history[symbol] = []
        self.signal_history[symbol].append(signal)

    def _update_stats(self, signal: SessionInfluenceSignal) -> None:
        """Обновление статистики сигналов."""
        self.stats["total_signals_generated"] = int(self.stats.get("total_signals_generated", 0)) + 1
        
        if signal.confidence >= 0.8:
            self.stats["high_confidence_signals"] = int(self.stats.get("high_confidence_signals", 0)) + 1
        
        if signal.tendency == "bullish":
            self.stats["bullish_signals"] = int(self.stats.get("bullish_signals", 0)) + 1
        elif signal.tendency == "bearish":
            self.stats["bearish_signals"] = int(self.stats.get("bearish_signals", 0)) + 1
        else:
            self.stats["neutral_signals"] = int(self.stats.get("neutral_signals", 0)) + 1

    def _get_signal_statistics(self, symbol: str) -> Dict[str, Any]:
        """Получение статистики сигналов для символа."""
        signals = self.signals.get(symbol, [])
        if not signals:
            return {
                "symbol": symbol,
                "total_signals": 0,
                "avg_confidence": 0.0,
                "tendency_distribution": {"bullish": 0, "bearish": 0, "neutral": 0},
            }
        return {
            "symbol": symbol,
            "total_signals": len(signals),
            "avg_confidence": sum(s.confidence for s in signals) / len(signals),
            "tendency_distribution": {
                "bullish": len([s for s in signals if s.tendency == "bullish"]),
                "bearish": len([s for s in signals if s.tendency == "bearish"]),
                "neutral": len([s for s in signals if s.tendency == "neutral"]),
            },
        }

    async def _signal_update_loop(self) -> None:
        """Цикл обновления сигналов."""
        while self._running:
            try:
                # Здесь можно добавить логику периодического обновления сигналов
                await asyncio.sleep(self.config["signal_update_interval_seconds"])
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in signal update loop: {e}")
                await asyncio.sleep(60)  # Пауза при ошибке

    def get_statistics(self) -> Dict[str, Any]:
        """Получение общей статистики движка."""
        return {
            "engine_status": "running" if self._running else "stopped",
            "total_symbols": len(self.signals),
            "stats": self.stats,
            "config": self.config,
        }
