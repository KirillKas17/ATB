# -*- coding: utf-8 -*-
"""Liquidity Gravity Monitor for Risk Assessment."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

from domain.market.liquidity_gravity import (
    LiquidityGravityConfig,
    LiquidityGravityModel,
    LiquidityGravityResult,
    OrderBookSnapshot,
)
from domain.value_objects.timestamp import Timestamp


@dataclass
class RiskAssessmentResult:
    """Результат оценки риска на основе гравитации ликвидности."""

    symbol: str
    risk_level: str
    gravity_score: float
    liquidity_score: float
    volatility_score: float
    overall_risk: float
    recommendations: List[str]
    timestamp: Timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "symbol": self.symbol,
            "risk_level": self.risk_level,
            "gravity_score": self.gravity_score,
            "liquidity_score": self.liquidity_score,
            "volatility_score": self.volatility_score,
            "overall_risk": self.overall_risk,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.value,
            "metadata": self.metadata,
        }


@dataclass
class MonitorConfig:
    """Конфигурация мониторинга гравитации ликвидности."""

    # Параметры мониторинга
    update_interval: float = 30.0  # секунды
    max_history_size: int = 1000
    alert_threshold: float = 0.8
    critical_threshold: float = 0.95

    # Параметры гравитации
    gravitational_constant: float = 1e-6
    min_volume_threshold: float = 0.001
    max_price_distance: float = 0.1

    # Параметры оценки риска
    gravity_weight: float = 0.4
    liquidity_weight: float = 0.3
    volatility_weight: float = 0.3

    # Логирование
    enable_detailed_logging: bool = True
    log_alerts: bool = True


class LiquidityGravityMonitor:
    """Монитор гравитации ликвидности для оценки рисков."""

    def __init__(self, config: Optional[MonitorConfig] = None):
        self.config = config or MonitorConfig()

        # Инициализация модели гравитации
        gravity_config = LiquidityGravityConfig(
            gravitational_constant=self.config.gravitational_constant,
            min_volume_threshold=self.config.min_volume_threshold,
            max_price_distance=self.config.max_price_distance,
        )
        self.gravity_model = LiquidityGravityModel(gravity_config)

        # Состояние мониторинга
        self.is_running = False
        self.monitored_symbols: List[str] = []
        self.gravity_history: Dict[str, List[LiquidityGravityResult]] = {}
        self.risk_history: Dict[str, List[RiskAssessmentResult]] = {}

        # Статистика
        self.stats: Dict[str, Any] = {
            "total_assessments": 0,
            "high_risk_detections": 0,
            "critical_risk_detections": 0,
            "last_assessment_time": None,
            "start_time": datetime.now(),
        }

        logger.info(f"LiquidityGravityMonitor initialized with config: {self.config}")

    def add_symbol(self, symbol: str) -> None:
        """Добавление символа для мониторинга."""
        if symbol not in self.monitored_symbols:
            self.monitored_symbols.append(symbol)
            self.gravity_history[symbol] = []
            self.risk_history[symbol] = []
            logger.info(f"Added symbol for monitoring: {symbol}")

    def remove_symbol(self, symbol: str) -> None:
        """Удаление символа из мониторинга."""
        if symbol in self.monitored_symbols:
            self.monitored_symbols.remove(symbol)
            self.gravity_history.pop(symbol, None)
            self.risk_history.pop(symbol, None)
            logger.info(f"Removed symbol from monitoring: {symbol}")

    async def start_monitoring(self) -> None:
        """Запуск мониторинга."""
        if self.is_running:
            logger.warning("Monitoring is already running")
            return

        self.is_running = True
        logger.info("Starting liquidity gravity monitoring")

        try:
            while self.is_running:
                await self._monitoring_cycle()
                await asyncio.sleep(self.config.update_interval)

        except Exception as e:
            logger.error(f"Error in monitoring cycle: {e}")
            self.is_running = False
            raise

    def stop_monitoring(self) -> None:
        """Остановка мониторинга."""
        self.is_running = False
        logger.info("Stopped liquidity gravity monitoring")

    async def _monitoring_cycle(self) -> None:
        """Основной цикл мониторинга."""
        try:
            for symbol in self.monitored_symbols:
                # Получаем данные ордербука (заглушка)
                order_book = await self._get_order_book_snapshot(symbol)

                if order_book:
                    # Анализируем гравитацию ликвидности
                    gravity_result = self.gravity_model.analyze_liquidity_gravity(
                        order_book
                    )

                    # Оцениваем риски
                    risk_result = self._assess_risk(symbol, gravity_result, order_book)

                    # Сохраняем результаты
                    self._save_results(symbol, gravity_result, risk_result)

                    # Проверяем алерты
                    await self._check_alerts(symbol, risk_result)

        except Exception as e:
            logger.error(f"Error in monitoring cycle: {e}")

    async def _get_order_book_snapshot(
        self, symbol: str
    ) -> Optional[OrderBookSnapshot]:
        """Получение снимка ордербука (заглушка)."""
        try:
            # В реальной системе здесь был бы запрос к бирже
            # Пока используем заглушку для тестирования
            import random

            # Симулируем данные ордербука
            mid_price = 50000.0 + random.uniform(-1000, 1000)
            spread = mid_price * 0.001  # 0.1% спред

            bids = [
                (mid_price - spread / 2 - i * 10, random.uniform(0.1, 10.0))
                for i in range(10)
            ]
            asks = [
                (mid_price + spread / 2 + i * 10, random.uniform(0.1, 10.0))
                for i in range(10)
            ]

            return OrderBookSnapshot(
                bids=bids, asks=asks, timestamp=datetime.now(), symbol=symbol
            )

        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {e}")
            return None

    def _assess_risk(
        self,
        symbol: str,
        gravity_result: LiquidityGravityResult,
        order_book: OrderBookSnapshot,
    ) -> RiskAssessmentResult:
        """Оценка риска на основе гравитации ликвидности."""
        try:
            # Оценка риска на основе гравитации ликвидности
            gravity_score = gravity_result.total_gravity
            liquidity_score = gravity_result.liquidity_score
            volatility_score = gravity_result.volatility_score

            # Общий риск
            overall_risk = (
                gravity_score * self.config.gravity_weight
                + (1.0 - liquidity_score) * self.config.liquidity_weight
                + volatility_score * self.config.volatility_weight
            )

            # Определение уровня риска
            if overall_risk >= self.config.critical_threshold:
                risk_level = "critical"
            elif overall_risk >= self.config.alert_threshold:
                risk_level = "high"
            elif overall_risk >= 0.5:
                risk_level = "medium"
            else:
                risk_level = "low"

            # Генерация рекомендаций
            recommendations = self._generate_recommendations(
                risk_level, gravity_score, liquidity_score, volatility_score
            )

            result = RiskAssessmentResult(
                symbol=symbol,
                risk_level=risk_level,
                gravity_score=gravity_score,
                liquidity_score=liquidity_score,
                volatility_score=volatility_score,
                overall_risk=overall_risk,
                recommendations=recommendations,
                timestamp=Timestamp(datetime.now()),
                metadata={
                    "gravity_result": gravity_result.to_dict(),
                    "order_book_stats": {
                        "total_volume": order_book.get_bid_volume() + order_book.get_ask_volume(),
                        "spread_pct": order_book.get_spread_percentage(),
                        "bid_count": len(order_book.bids),
                        "ask_count": len(order_book.asks),
                    },
                },
            )

            return result

        except Exception as e:
            logger.error(f"Error assessing risk for {symbol}: {e}")
            # Возвращаем безопасный результат по умолчанию
            return RiskAssessmentResult(
                symbol=symbol,
                risk_level="unknown",
                gravity_score=0.0,
                liquidity_score=0.0,
                volatility_score=0.0,
                overall_risk=0.0,
                recommendations=["Error in risk assessment"],
                timestamp=Timestamp(datetime.now()),
            )

    def _generate_recommendations(
        self,
        risk_level: str,
        gravity_score: float,
        liquidity_score: float,
        volatility_score: float,
    ) -> List[str]:
        """Генерация рекомендаций на основе оценки риска."""
        recommendations = []

        if risk_level == "critical":
            recommendations.extend(
                [
                    "НЕМЕДЛЕННО: Прекратить торговлю",
                    "Закрыть все открытые позиции",
                    "Увеличить стоп-лоссы",
                    "Снизить размеры позиций до минимума",
                ]
            )
        elif risk_level == "high":
            recommendations.extend(
                [
                    "Снизить размеры позиций",
                    "Увеличить стоп-лоссы",
                    "Избегать новых позиций",
                    "Мониторить ситуацию каждые 5 минут",
                ]
            )
        elif risk_level == "medium":
            recommendations.extend(
                [
                    "Соблюдать осторожность",
                    "Использовать меньшие размеры позиций",
                    "Установить стоп-лоссы",
                    "Мониторить изменения",
                ]
            )
        else:  # low
            recommendations.extend(
                [
                    "Нормальные условия торговли",
                    "Можно использовать стандартные размеры позиций",
                    "Стандартные стоп-лоссы",
                ]
            )

        # Специфические рекомендации
        if gravity_score > 0.8:
            recommendations.append(
                "Высокая гравитация ликвидности - возможны резкие движения"
            )

        if liquidity_score < 0.3:
            recommendations.append("Низкая ликвидность - возможны проскальзывания")

        if volatility_score > 0.7:
            recommendations.append(
                "Высокая волатильность - использовать более широкие стоп-лоссы"
            )

        return recommendations

    def _save_results(
        self,
        symbol: str,
        gravity_result: LiquidityGravityResult,
        risk_result: RiskAssessmentResult,
    ) -> None:
        """Сохранение результатов анализа."""
        # Сохраняем результат гравитации
        if symbol not in self.gravity_history:
            self.gravity_history[symbol] = []
        self.gravity_history[symbol].append(gravity_result)

        # Ограничиваем размер истории
        if len(self.gravity_history[symbol]) > self.config.max_history_size:
            self.gravity_history[symbol] = self.gravity_history[symbol][-self.config.max_history_size:]

        # Сохраняем результат оценки риска
        if symbol not in self.risk_history:
            self.risk_history[symbol] = []
        self.risk_history[symbol].append(risk_result)

        # Ограничиваем размер истории
        if len(self.risk_history[symbol]) > self.config.max_history_size:
            self.risk_history[symbol] = self.risk_history[symbol][-self.config.max_history_size:]

        # Обновляем статистику
        self.stats["total_assessments"] = int(self.stats.get("total_assessments", 0)) + 1
        self.stats["last_assessment_time"] = datetime.now()

        if risk_result.risk_level == "high":
            self.stats["high_risk_detections"] = int(self.stats.get("high_risk_detections", 0)) + 1
        elif risk_result.risk_level == "critical":
            self.stats["critical_risk_detections"] = int(self.stats.get("critical_risk_detections", 0)) + 1

        if self.config.enable_detailed_logging:
            logger.info(
                f"Saved results for {symbol}: "
                f"gravity_score={risk_result.gravity_score:.3f}, "
                f"risk_level={risk_result.risk_level}, "
                f"overall_risk={risk_result.overall_risk:.3f}"
            )

    async def _check_alerts(
        self, symbol: str, risk_result: RiskAssessmentResult
    ) -> None:
        """Проверка и отправка алертов."""
        try:
            if not self.config.log_alerts:
                return

            if risk_result.risk_level in ["high", "critical"]:
                logger.warning(
                    f"RISK ALERT for {symbol}: {risk_result.risk_level.upper()} risk "
                    f"(score: {risk_result.overall_risk:.3f})"
                )

                for recommendation in risk_result.recommendations:
                    logger.warning(f"  - {recommendation}")

        except Exception as e:
            logger.error(f"Error checking alerts for {symbol}: {e}")

    def get_latest_risk_assessment(self, symbol: str) -> Optional[RiskAssessmentResult]:
        """Получение последней оценки риска для символа."""
        try:
            if symbol in self.risk_history and self.risk_history[symbol]:
                return self.risk_history[symbol][-1]
            return None
        except Exception as e:
            logger.error(f"Error getting latest risk assessment for {symbol}: {e}")
            return None

    def get_risk_history(
        self, symbol: str, limit: int = 100
    ) -> List[RiskAssessmentResult]:
        """Получение истории оценок риска."""
        try:
            if symbol in self.risk_history:
                return self.risk_history[symbol][-limit:]
            return []
        except Exception as e:
            logger.error(f"Error getting risk history for {symbol}: {e}")
            return []

    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Получение статистики мониторинга."""
        try:
            stats = self.stats.copy()
            start_time = stats.get("start_time")
            if start_time:
                uptime_seconds = (datetime.now() - start_time).total_seconds()
            else:
                uptime_seconds = 0.0

            stats.update(
                {
                    "monitored_symbols": self.monitored_symbols,
                    "is_running": self.is_running,
                    "uptime_seconds": uptime_seconds,
                    "symbols_count": len(self.monitored_symbols),
                }
            )
            return stats
        except Exception as e:
            logger.error(f"Error getting monitoring statistics: {e}")
            return {}
