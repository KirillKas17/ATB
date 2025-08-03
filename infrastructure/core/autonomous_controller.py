# -*- coding: utf-8 -*-
"""
Автономный контроллер - "мозг" системы.
Обеспечивает автоматическое управление всеми компонентами системы.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from domain.types.messaging_types import Event, EventType, EventPriority
from infrastructure.circuit_breaker.breaker import CircuitBreaker
from infrastructure.core.metrics import MetricsCollector
from infrastructure.messaging.event_bus import EventBus
from infrastructure.ml_services.live_adaptation import LiveAdaptation
from infrastructure.ml_services.meta_learning import MetaLearning
from infrastructure.ml_services.regime_discovery import RegimeDiscovery
from infrastructure.agents.local_ai.controller import LocalAIController
from shared.cache import get_cache_manager
from domain.types.messaging_types import EventName, EventType


class AutonomousController:
    """
    Автономный контроллер - "мозг" системы, который:
    - Автоматически настраивает все параметры
    - Адаптируется к изменениям рынка
    - Принимает решения без участия человека
    - Самообучается и эволюционирует
    """

    def __init__(self, event_bus: EventBus, config: Dict[str, Any]):
        self.event_bus = event_bus
        self.config = config
        # Инициализация всех компонентов
        self.metrics = MetricsCollector(event_bus)
        self.cache_manager = get_cache_manager()
        # Исправление: используем правильный тип EventBus
        self.circuit_breaker = CircuitBreaker(event_bus)  # type: ignore
        # ML компоненты
        self.live_adaptation = LiveAdaptation()
        self.meta_learning = MetaLearning()
        self.regime_discovery = RegimeDiscovery(event_bus)
        # ИИ контроллер
        self.ai_controller = LocalAIController()
        # Автономные настройки
        self.auto_config: Dict[str, Any] = {
            "risk_management": {
                "max_daily_loss": 0.02,  # 2% в день
                "max_weekly_loss": 0.05,  # 5% в неделю
                "position_sizing": "kelly",  # или "fixed", "volatility"
                "correlation_threshold": 0.7,
            },
            "strategy_selection": {
                "confidence_threshold": 0.6,
                "min_trades_for_evaluation": 50,
                "evaluation_period": "1d",
            },
            "market_regime": {
                "detection_sensitivity": 0.8,
                "regime_switch_threshold": 0.3,
                "min_regime_duration": "4h",
            },
        }
        # Состояние системы
        self.system_state: Dict[str, Any] = {
            "is_healthy": True,
            "current_regime": "unknown",
            "active_strategies": [],
            "performance_metrics": {},
            "risk_metrics": {},
        }
        # Запуск автономного цикла
        asyncio.create_task(self._start_autonomous_loop())
        # Отправка уведомления о запуске
        startup_event = Event(
            name=EventName("system.startup"),
            type=EventType.SYSTEM_START,
            data={
                "message": "Autonomous controller started",
                "timestamp": time.time(),
                "version": "1.0.0"
            }
        )
        asyncio.create_task(self.event_bus.publish(startup_event))

    async def _start_autonomous_loop(self) -> None:
        """Основной цикл автономного управления"""
        logger.info("Starting autonomous controller loop")
        while True:
            try:
                # 1. Проверка здоровья системы
                await self._check_system_health()
                # 2. Анализ текущего состояния
                await self._analyze_current_state()
                # 3. Автоматическая настройка параметров
                await self._auto_configure_parameters()
                # 4. Адаптация к изменениям рынка
                await self._adapt_to_market_changes()
                # 5. Оптимизация стратегий
                await self._optimize_strategies()
                # 6. Управление рисками
                await self._manage_risks()
                # 7. Обновление метрик
                await self._update_metrics()
                await asyncio.sleep(300)  # Каждые 5 минут
            except Exception as e:
                logger.error(f"Error in autonomous loop: {e}")
                await self._handle_error(e)
                await asyncio.sleep(60)

    async def _check_system_health(self) -> None:
        """Проверка здоровья системы"""
        health_status = await self.health_checker._check_external_services()
        if not health_status["overall_healthy"]:
            logger.warning(f"System health issues detected: {health_status}")
            await self._handle_health_issues(health_status)
        self.system_state["is_healthy"] = health_status["overall_healthy"]

    async def _handle_health_issues(self, health_status: Dict[str, Any]) -> None:
        """Обработка проблем со здоровьем системы"""
        if not health_status.get("exchange_healthy", True):
            # Логируем проблему с биржей
            logger.warning("Exchange health issue detected")
        if not health_status.get("database_healthy", True):
            # Логируем проблему с базой данных
            logger.warning("Database health issue detected")
        # Отправка уведомления - исправление: убираем неправильные параметры Event
        await self.event_bus.publish(
            Event(
                name=EventName("system.health_issue"),
                type=EventType.SYSTEM_ERROR,
                data=health_status
            )
        )

    async def _analyze_current_state(self) -> None:
        """Анализ текущего состояния системы"""
        # Получение метрик от всех компонентов
        performance = await self.metrics.get_performance_metrics()
        risk_metrics = await self.risk_manager.get_risk_metrics()
        portfolio_metrics = await self.portfolio_manager.get_portfolio_metrics()
        market_conditions = await self._get_market_conditions()
        # Обновление состояния системы
        self.system_state.update(
            {
                "performance_metrics": performance,
                "risk_metrics": risk_metrics,
                "portfolio_metrics": portfolio_metrics,
                "market_conditions": market_conditions,
            }
        )
        logger.info(
            f"Current state analyzed: performance="
            f"{performance.get('sharpe_ratio', 0):.3f}, "
            f"risk={risk_metrics.get('portfolio_risk', 0):.3f}"
        )

    async def _get_market_conditions(self) -> Dict[str, Any]:
        """Получение текущих рыночных условий"""
        try:
            # Получение данных от различных источников
            market_data: Dict[str, Any] = {}
            # Данные от ExchangeManager
            if hasattr(self, "exchange_manager") and self.exchange_manager:
                try:
                    # Получение текущих цен
                    for symbol in self.config.get("trading", {}).get("symbols", []):
                        data = await self.exchange_manager.get_market_data(
                            symbol, "1m", limit=1
                        )
                        if data is not None and not data.empty:
                            market_data[symbol] = {
                                "price": data["close"].iloc[-1],
                                "volume": data["volume"].iloc[-1],
                                "timestamp": data.index[-1],
                            }
                except Exception as e:
                    logger.error(f"Error getting market data: {e}")
            # Расчет волатильности
            volatility = self._calculate_market_volatility(market_data)
            # Определение силы тренда
            trend_strength = self._calculate_trend_strength(market_data)
            # Анализ профиля объема
            volume_profile = self._analyze_volume_profile(market_data)
            # Определение режима рынка
            regime = self._detect_market_regime(market_data)
            return {
                "volatility": volatility,
                "trend_strength": trend_strength,
                "volume_profile": volume_profile,
                "regime": regime,
                "market_data": market_data,
            }
        except Exception as e:
            logger.error(f"Error getting market conditions: {e}")
            return {
                "volatility": 0.02,
                "trend_strength": 0.5,
                "volume_profile": "normal",
                "regime": "unknown",
            }

    def _calculate_market_volatility(self, market_data: Dict[str, Any]) -> float:
        """
        Расчет волатильности рынка.
        Args:
            market_data: Данные рынка
        Returns:
            float: Волатильность рынка
        """
        try:
            if "prices" not in market_data or not market_data["prices"]:
                return 0.0
            
            prices = market_data["prices"]
            if len(prices) < 2:
                return 0.0
            
            # Расчет логарифмических доходностей
            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] > 0:
                    returns.append((prices[i] - prices[i-1]) / prices[i-1])
            
            if not returns:
                return 0.0
            
            # Расчет стандартного отклонения доходностей
            import numpy as np
            volatility = float(np.std(returns))
            return volatility
        except Exception as e:
            logger.error(f"Error calculating market volatility: {e}")
            return 0.0

    def _calculate_trend_strength(self, market_data: Dict[str, Any]) -> float:
        """Расчет силы тренда"""
        try:
            if not market_data:
                return 0.5  # Нейтральное значение
            # Упрощенный расчет силы тренда
            prices = []
            for symbol_data in market_data.values():
                if isinstance(symbol_data, dict) and "price" in symbol_data:
                    prices.append(symbol_data["price"])
            if len(prices) < 2:
                return 0.5
            # Простой расчет тренда
            price_change = (prices[-1] - prices[0]) / prices[0]
            trend_strength = abs(price_change)
            return min(1.0, trend_strength)
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.5

    def _analyze_volume_profile(self, market_data: Dict[str, Any]) -> str:
        """Анализ профиля объема"""
        try:
            if not market_data:
                return "normal"
            # Упрощенный анализ объема
            total_volume = 0
            for symbol_data in market_data.values():
                if isinstance(symbol_data, dict) and "volume" in symbol_data:
                    total_volume += symbol_data["volume"]
            # Определение профиля на основе объема
            if total_volume > 1000000:
                return "high"
            elif total_volume < 100000:
                return "low"
            else:
                return "normal"
        except Exception as e:
            logger.error(f"Error analyzing volume profile: {e}")
            return "normal"

    def _detect_market_regime(self, market_data: Dict[str, Any]) -> str:
        """Определение режима рынка"""
        try:
            if not market_data:
                return "unknown"
            # Упрощенное определение режима
            volatility = self._calculate_market_volatility(market_data)
            trend_strength = self._calculate_trend_strength(market_data)
            if volatility > 0.05:
                return "high_volatility"
            elif trend_strength > 0.1:
                return "trending"
            else:
                return "sideways"
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return "unknown"

    async def _auto_configure_parameters(self) -> None:
        """Автоматическая настройка параметров"""
        try:
            # Настройка параметров риска
            await self._adjust_risk_parameters()
            # Настройка параметров стратегий
            await self._adjust_strategy_parameters()
            # Настройка параметров портфеля
            await self._adjust_portfolio_parameters()
        except Exception as e:
            logger.error(f"Error auto-configuring parameters: {e}")

    async def _adjust_risk_parameters(self) -> None:
        """Настройка параметров риска"""
        try:
            # Получение текущих метрик риска
            risk_metrics = self.system_state.get("risk_metrics", {})
            portfolio_risk = risk_metrics.get("portfolio_risk", 0)
            # Адаптация параметров на основе риска
            if portfolio_risk > 0.05:
                self.auto_config["risk_management"]["max_daily_loss"] *= 0.9
            elif portfolio_risk < 0.01:
                self.auto_config["risk_management"]["max_daily_loss"] *= 1.1
            # Ограничения
            self.auto_config["risk_management"]["max_daily_loss"] = max(
                0.01, min(0.05, self.auto_config["risk_management"]["max_daily_loss"])
            )
        except Exception as e:
            logger.error(f"Error adjusting risk parameters: {e}")

    async def _adjust_strategy_parameters(self) -> None:
        """Настройка параметров стратегий"""
        try:
            # Получение метрик производительности
            performance = self.system_state.get("performance_metrics", {})
            sharpe_ratio = performance.get("sharpe_ratio", 0)
            # Адаптация порога уверенности
            if sharpe_ratio < 0.5:
                self.auto_config["strategy_selection"]["confidence_threshold"] *= 1.1
            elif sharpe_ratio > 1.5:
                self.auto_config["strategy_selection"]["confidence_threshold"] *= 0.9
            # Ограничения
            self.auto_config["strategy_selection"]["confidence_threshold"] = max(
                0.3, min(0.9, self.auto_config["strategy_selection"]["confidence_threshold"])
            )
        except Exception as e:
            logger.error(f"Error adjusting strategy parameters: {e}")

    async def _adjust_portfolio_parameters(self) -> None:
        """Настройка параметров портфеля"""
        try:
            # Получение метрик портфеля
            portfolio_metrics = self.system_state.get("portfolio_metrics", {})
            diversification = portfolio_metrics.get("diversification_score", 0.5)
            # Адаптация параметров диверсификации
            if diversification < 0.3:
                # Увеличиваем диверсификацию
                pass
            elif diversification > 0.8:
                # Уменьшаем диверсификацию
                pass
        except Exception as e:
            logger.error(f"Error adjusting portfolio parameters: {e}")

    async def _adapt_to_market_changes(self) -> None:
        """Адаптация к изменениям рынка"""
        try:
            market_conditions = self.system_state.get("market_conditions", {})
            regime = market_conditions.get("regime", "unknown")
            await self._adapt_to_regime(regime)
        except Exception as e:
            logger.error(f"Error adapting to market changes: {e}")

    async def _adapt_to_regime(self, regime: str) -> None:
        """Адаптация к режиму рынка"""
        try:
            if regime == "high_volatility":
                await self._activate_defensive_strategies()
            elif regime == "trending":
                await self._activate_trend_strategies()
            elif regime == "sideways":
                await self._activate_mean_reversion_strategies()
            else:
                # Неизвестный режим - используем сбалансированный подход
                pass
        except Exception as e:
            logger.error(f"Error adapting to regime {regime}: {e}")

    async def _optimize_strategies(self) -> None:
        """Оптимизация стратегий"""
        try:
            # Получение текущих стратегий
            active_strategies = self.system_state.get("active_strategies", [])
            # Анализ производительности
            for strategy in active_strategies:
                if strategy.get("performance", {}).get("sharpe_ratio", 0) < 0.5:
                    await self._disable_strategy(strategy["name"])
            # Активация новых стратегий при необходимости
            if len(active_strategies) < 3:
                await self._activate_new_strategies()
        except Exception as e:
            logger.error(f"Error optimizing strategies: {e}")

    async def _disable_strategy(self, strategy_name: str) -> None:
        """Отключение стратегии"""
        try:
            # Логика отключения стратегии
            logger.info(f"Disabling strategy: {strategy_name}")
            # Обновление списка активных стратегий
            active_strategies = self.system_state.get("active_strategies", [])
            self.system_state["active_strategies"] = [
                s for s in active_strategies if s["name"] != strategy_name
            ]
        except Exception as e:
            logger.error(f"Error disabling strategy {strategy_name}: {e}")

    async def _activate_new_strategies(self) -> None:
        """Активация новых стратегий"""
        try:
            # Логика активации новых стратегий
            logger.info("Activating new strategies")
            # Здесь должна быть логика выбора и активации стратегий
        except Exception as e:
            logger.error(f"Error activating new strategies: {e}")

    async def _activate_defensive_strategies(self) -> None:
        """Активация защитных стратегий"""
        try:
            logger.info("Activating defensive strategies")
            # Логика активации защитных стратегий
        except Exception as e:
            logger.error(f"Error activating defensive strategies: {e}")

    async def _activate_trend_strategies(self) -> None:
        """Активация трендовых стратегий"""
        try:
            logger.info("Activating trend strategies")
            # Логика активации трендовых стратегий
        except Exception as e:
            logger.error(f"Error activating trend strategies: {e}")

    async def _activate_mean_reversion_strategies(self) -> None:
        """Активация стратегий возврата к среднему"""
        try:
            logger.info("Activating mean reversion strategies")
            # Логика активации стратегий возврата к среднему
        except Exception as e:
            logger.error(f"Error activating mean reversion strategies: {e}")

    async def _manage_risks(self) -> None:
        """Управление рисками"""
        try:
            # Получение текущих метрик риска
            risk_metrics = self.system_state.get("risk_metrics", {})
            portfolio_risk = risk_metrics.get("portfolio_risk", 0)
            # Проверка лимитов риска
            max_daily_loss = self.auto_config["risk_management"]["max_daily_loss"]
            if portfolio_risk > max_daily_loss:
                await self._reduce_risk_exposure()
            # Диверсификация портфеля
            await self._diversify_portfolio()
        except Exception as e:
            logger.error(f"Error managing risks: {e}")

    async def _reduce_risk_exposure(self) -> None:
        """Снижение рисков"""
        try:
            logger.info("Reducing risk exposure")
            # Логика снижения рисков
            # Например, закрытие части позиций, увеличение стоп-лоссов
        except Exception as e:
            logger.error(f"Error reducing risk exposure: {e}")

    async def _diversify_portfolio(self) -> None:
        """Диверсификация портфеля"""
        try:
            logger.info("Diversifying portfolio")
            # Логика диверсификации портфеля
            # Например, добавление новых активов, корректировка весов
        except Exception as e:
            logger.error(f"Error diversifying portfolio: {e}")

    async def _update_metrics(self) -> None:
        """Обновление метрик"""
        try:
            # Обновление метрик производительности
            if hasattr(self, 'metrics_collector'):
                await self.metrics_collector.collect_system_metrics()
                await self.metrics_collector.collect_trading_metrics()
                await self.metrics_collector.collect_risk_metrics()
            # Обновление метрик риска
            await self.risk_manager.update_risk_metrics({})
            # Обновление метрик портфеля
            await self.portfolio_manager.get_portfolio_metrics()
            logger.debug("Metrics updated")
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            # Отправка уведомления об ошибке
            error_event = Event(
                name=EventName("system.metrics_error"),
                type=EventType.SYSTEM_ERROR,
                data={
                    "message": f"Error updating metrics: {e}",
                    "timestamp": time.time(),
                    "error_type": "metrics_update"
                }
            )
            await self.event_bus.publish(error_event)

    async def _handle_error(self, error: Exception) -> None:
        """Обработка ошибок"""
        try:
            logger.error(f"Handling error: {error}")
            # Отправка события об ошибке
            error_event = Event(
                name=EventName("system.error"),
                type=EventType.SYSTEM_ERROR,
                data={
                    "error": str(error),
                    "timestamp": time.time(),
                    "error_type": "system_error"
                }
            )
            await self.event_bus.publish(error_event)
            # Активация защитных мер
            await self._activate_defensive_strategies()
        except Exception as e:
            logger.error(f"Error in error handler: {e}")

    async def emergency_stop(self) -> None:
        """Экстренная остановка системы"""
        try:
            logger.warning("Emergency stop initiated")
            # Отправка события об экстренной остановке
            emergency_event = Event(
                name=EventName("system.emergency_stop"),
                type=EventType.SYSTEM_ERROR,
                data={
                    "reason": "emergency_stop",
                    "timestamp": time.time(),
                    "error_type": "emergency_stop"
                }
            )
            await self.event_bus.publish(emergency_event)
            # Отключение всех стратегий
            await self._disable_all_strategies()
            # Закрытие всех позиций
            # Здесь должна быть логика закрытия позиций
            logger.info("Emergency stop completed")
        except Exception as e:
            logger.error(f"Error in emergency stop: {e}")

    async def _disable_all_strategies(self) -> None:
        """Отключение всех стратегий"""
        try:
            logger.info("Disabling all strategies")
            self.system_state["active_strategies"] = []
        except Exception as e:
            logger.error(f"Error disabling all strategies: {e}")

    async def get_system_status(self) -> Dict[str, Any]:
        """Получение статуса системы"""
        return {
            "is_healthy": self.system_state.get("is_healthy", False),
            "current_regime": self.system_state.get("current_regime", "unknown"),
            "active_strategies": len(self.system_state.get("active_strategies", [])),
            "performance_metrics": self.system_state.get("performance_metrics", {}),
            "risk_metrics": self.system_state.get("risk_metrics", {}),
            "auto_config": self.auto_config,
            "uptime": time.time() - getattr(self, "_start_time", time.time()),
        }
