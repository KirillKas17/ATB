"""Модуль для управления рисками в торговой системе.
Этот модуль содержит менеджер рисков для управления всеми аспектами
риск-менеджмента, включая ограничения по риску, корреляционный анализ
и динамическое изменение размера позиции.
"""

import time
from typing import Any, Dict, List

from shared.numpy_utils import np
from loguru import logger

from infrastructure.messaging.event_bus import Event, EventBus, EventPriority, EventName, EventType


class RiskManager:
    """
    Менеджер рисков - управляет всеми аспектами риск-менеджмента.
    Включает:
    - Ограничения по риску на сделку, день, неделю
    - Корреляционный анализ
    - Динамическое изменение размера позиции
    - Приоритизацию сигналов
    """

    def __init__(self, event_bus: EventBus, config: Dict[str, Any]) -> None:
        """Инициализация менеджера рисков.
        Args:
            event_bus: Шина событий для уведомлений
            config: Конфигурация риск-менеджмента
        """
        self.event_bus = event_bus
        self.config = config
        # Риск-лимиты
        self.risk_limits = {
            "max_risk_per_trade": config.get(
                "max_risk_per_trade", 0.02
            ),  # 2% на сделку
            "max_daily_loss": config.get("max_daily_loss", 0.05),  # 5% в день
            "max_weekly_loss": config.get("max_weekly_loss", 0.15),  # 15% в неделю
            "max_portfolio_risk": config.get(
                "max_portfolio_risk", 0.10
            ),  # 10% портфеля
            "max_correlation": config.get(
                "max_correlation", 0.7
            ),  # Максимальная корреляция
        }
        # Текущие метрики риска
        self.current_metrics: Dict[str, Any] = {
            "daily_loss": 0.0,
            "weekly_loss": 0.0,
            "portfolio_risk": 0.0,
            "open_positions": [],
            "correlations": {},
        }
        # История рисков
        self.risk_history: List[Dict[str, Any]] = []

    async def check_trade_risk(self, trade_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Проверка риска для новой сделки.
        Args:
            trade_params: Параметры сделки
        Returns:
            Dict[str, Any]: Оценка риска сделки
        """
        risk_assessment: Dict[str, Any] = {
            "is_allowed": True,
            "risk_score": 0.0,
            "position_size": 0.0,
            "warnings": [],
        }
        # Проверка лимитов
        if not await self._check_risk_limits(trade_params):
            risk_assessment["is_allowed"] = False
            if isinstance(risk_assessment.get("warnings"), list):
                risk_assessment["warnings"].append("Risk limits exceeded")
            else:
                risk_assessment["warnings"] = ["Risk limits exceeded"]
            return risk_assessment
        # Расчет риска сделки
        trade_risk = await self._calculate_trade_risk(trade_params)
        risk_assessment["risk_score"] = trade_risk
        # Расчет размера позиции
        position_size = await self._calculate_position_size(trade_params, trade_risk)
        risk_assessment["position_size"] = position_size
        # Проверка корреляций
        correlation_risk = await self._check_correlation_risk(trade_params)
        if correlation_risk > self.risk_limits["max_correlation"]:
            if isinstance(risk_assessment.get("warnings"), list):
                risk_assessment["warnings"].append(
                    f"High correlation risk: {correlation_risk:.3f}"
                )
            else:
                risk_assessment["warnings"] = [f"High correlation risk: {correlation_risk:.3f}"]
        return risk_assessment

    async def _check_risk_limits(self, trade_params: Dict[str, Any]) -> bool:
        """
        Проверка лимитов риска.
        Args:
            trade_params: Параметры сделки
        Returns:
            bool: True если лимиты не превышены
        """
        # Проверка дневного лимита
        if self.current_metrics["daily_loss"] >= self.risk_limits["max_daily_loss"]:
            logger.warning("Daily loss limit reached")
            return False
        # Проверка недельного лимита
        if self.current_metrics["weekly_loss"] >= self.risk_limits["max_weekly_loss"]:
            logger.warning("Weekly loss limit reached")
            return False
        # Проверка риска портфеля
        if (
            self.current_metrics["portfolio_risk"]
            >= self.risk_limits["max_portfolio_risk"]
        ):
            logger.warning("Portfolio risk limit reached")
            return False
        return True

    async def _calculate_trade_risk(self, trade_params: Dict[str, Any]) -> float:
        """
        Расчет риска сделки.
        Args:
            trade_params: Параметры сделки
        Returns:
            float: Оценка риска сделки
        """
        # Базовый риск на основе волатильности
        volatility = trade_params.get("volatility", 0.02)
        base_risk = volatility * 2  # Умножаем на 2 для консервативности
        # Дополнительные факторы риска
        market_regime = trade_params.get("market_regime", "unknown")
        if market_regime == "high_volatility":
            base_risk *= 1.5
        elif market_regime == "low_volatility":
            base_risk *= 0.8
        # Риск стратегии
        strategy_confidence = trade_params.get("strategy_confidence", 0.5)
        strategy_risk = 1.0 - strategy_confidence
        base_risk *= 1.0 + strategy_risk
        return float(min(base_risk, 0.1))  # Максимум 10% риска на сделку

    async def _calculate_position_size(
        self, trade_params: Dict[str, Any], trade_risk: float
    ) -> float:
        """
        Расчет размера позиции.
        Args:
            trade_params: Параметры сделки
            trade_risk: Оценка риска сделки
        Returns:
            float: Рекомендуемый размер позиции
        """
        # Метод Kelly Criterion
        win_rate = trade_params.get("win_rate", 0.5)
        avg_win = trade_params.get("avg_win", 0.02)
        avg_loss = trade_params.get("avg_loss", 0.01)
        if avg_loss > 0:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Ограничиваем 25%
        else:
            kelly_fraction = 0.01  # Минимальный размер
        # Корректировка на основе риска
        risk_adjusted_size = kelly_fraction * (1.0 - trade_risk)
        return float(max(risk_adjusted_size, 0.001))  # Минимум 0.1%

    async def _check_correlation_risk(self, trade_params: Dict[str, Any]) -> float:
        """
        Проверка корреляционного риска.
        Args:
            trade_params: Параметры сделки
        Returns:
            float: Оценка корреляционного риска
        """
        symbol = trade_params.get("symbol", "")
        # Получение корреляций с существующими позициями
        correlations = self.current_metrics.get("correlations", {})
        if isinstance(correlations, dict) and symbol in correlations:
            return float(correlations[symbol])
        return 0.0

    async def update_risk_metrics(self, trade_result: Dict[str, Any]) -> None:
        """
        Обновление метрик риска после сделки.
        Args:
            trade_result: Результат сделки
        """
        # Обновление дневного убытка
        if trade_result.get("pnl", 0) < 0:
            self.current_metrics["daily_loss"] += abs(trade_result["pnl"])
        # Обновление недельного убытка
        if trade_result.get("pnl", 0) < 0:
            self.current_metrics["weekly_loss"] += abs(trade_result["pnl"])
        # Обновление риска портфеля
        await self._update_portfolio_risk()
        # Добавление в историю
        self.risk_history.append(
            {
                "timestamp": trade_result.get("timestamp"),
                "trade_id": trade_result.get("trade_id"),
                "pnl": trade_result.get("pnl", 0),
                "risk_metrics": self.current_metrics.copy(),
            }
        )
        # Отправка события
        await self.event_bus.publish(
            Event(
                name="risk.metrics.update",
                type="risk",
                data=self.current_metrics,
                priority=EventPriority.NORMAL,
            )
        )

    async def _update_portfolio_risk(self) -> None:
        """Продвинутый расчет риска портфеля."""
        try:
            # Получение данных о позициях
            positions = self.current_metrics.get("open_positions", [])
            if not isinstance(positions, list) or not positions:
                self.current_metrics["portfolio_risk"] = 0.0
                return
            # Расчет VaR (Value at Risk)
            pnl_history = [pos.get("unrealized_pnl", 0) for pos in positions if isinstance(pos, dict)]
            if len(pnl_history) > 1:
                # Параметрический VaR
                mean_pnl = np.mean(pnl_history)
                std_pnl = np.std(pnl_history)
                var_95 = mean_pnl - 1.645 * std_pnl  # 95% доверительный интервал
                # Conditional VaR (Expected Shortfall)
                negative_pnls = [pnl for pnl in pnl_history if pnl < 0]
                if negative_pnls:
                    cvar_95 = np.mean(negative_pnls)
                else:
                    cvar_95 = 0.0
                # Максимальная просадка
                cumulative_pnl = np.cumsum(pnl_history)
                running_max = np.maximum.accumulate(cumulative_pnl)
                drawdown = (cumulative_pnl - running_max) / running_max
                max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
                # Общий риск портфеля (взвешенная комбинация метрик)
                portfolio_risk = (
                    0.4 * abs(var_95)  # VaR
                    + 0.3 * abs(cvar_95)  # CVaR
                    + 0.2 * abs(max_drawdown)  # Максимальная просадка
                    + 0.1 * np.std(pnl_history)  # Волатильность
                )
                self.current_metrics["portfolio_risk"] = min(portfolio_risk, 1.0)
                self.current_metrics["var_95"] = var_95
                self.current_metrics["cvar_95"] = cvar_95
                self.current_metrics["max_drawdown"] = max_drawdown
            else:
                # Простая метрика для одной позиции
                total_exposure = sum([pos.get("exposure", 0) for pos in positions if isinstance(pos, dict)])
                self.current_metrics["portfolio_risk"] = min(total_exposure, 1.0)
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            # Fallback к простой метрике
            positions = self.current_metrics.get("open_positions", [])
            if isinstance(positions, list):
                total_exposure = sum(
                    [
                        pos.get("exposure", 0)
                        for pos in positions if isinstance(pos, dict)
                    ]
                )
                self.current_metrics["portfolio_risk"] = min(total_exposure, 1.0)
            else:
                self.current_metrics["portfolio_risk"] = 0.0

    async def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Получение текущих метрик риска.
        Returns:
            Dict[str, Any]: Текущие метрики риска
        """
        return self.current_metrics.copy()

    async def update_risk_limits(self, new_limits: Dict[str, float]) -> None:
        """
        Обновление лимитов риска.
        Args:
            new_limits: Новые лимиты риска
        """
        self.risk_limits.update(new_limits)
        logger.info(f"Updated risk limits: {new_limits}")
        # Отправка события
        await self.event_bus.publish(
            Event(
                name="risk.limits.update",
                type="risk",
                data=new_limits,
                priority=EventPriority.HIGH,
            )
        )

    async def close_all_positions(self) -> None:
        """Закрытие всех позиций при превышении лимитов."""
        logger.warning("Closing all positions due to risk limits")
        try:
            # Интеграция с position manager через event bus
            await self.event_bus.publish(
                Event(
                    name="risk.close_all_positions",
                    type="risk",
                    data={
                        "reason": "Risk limits exceeded",
                        "risk_metrics": self.current_metrics,
                        "timestamp": time.time(),
                    },
                    priority=EventPriority.CRITICAL,
                )
            )
            # Отправка команды на закрытие позиций
            await self.event_bus.publish(
                Event(
                    name="risk.close_all_positions.command",
                    type="risk",
                    data={
                        "reason": "Risk management",
                        "risk_level": "critical",
                        "portfolio_risk": self.current_metrics["portfolio_risk"],
                    },
                    priority=EventPriority.CRITICAL,
                )
            )
            # Сброс метрик риска
            self.current_metrics["daily_loss"] = 0.0
            self.current_metrics["weekly_loss"] = 0.0
            self.current_metrics["portfolio_risk"] = 0.0
            self.current_metrics["open_positions"] = []
            logger.info("Risk management: All positions closed and metrics reset")
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")

    async def adjust_position_sizes(self) -> None:
        """Динамическое изменение размера позиций на основе риска."""
        try:
            # Анализ текущих позиций
            positions = self.current_metrics.get("open_positions", [])
            if isinstance(positions, list):
                for position in positions:
                    if isinstance(position, dict):
                        # Проверка необходимости корректировки
                        if await self._need_position_adjustment(position):
                            await self._adjust_position(position)
            # Проверка общего риска портфеля
            portfolio_risk = self.current_metrics.get("portfolio_risk", 0.0)
            max_portfolio_risk = self.risk_limits.get("max_portfolio_risk", 1.0)
            if isinstance(portfolio_risk, (int, float)) and isinstance(max_portfolio_risk, (int, float)):
                if portfolio_risk > max_portfolio_risk * 0.8:
                    await self._reduce_portfolio_risk()
        except Exception as e:
            logger.error(f"Error adjusting position sizes: {e}")

    async def _need_position_adjustment(self, position: Dict[str, Any]) -> bool:
        """
        Продвинутая проверка необходимости корректировки позиции.
        Args:
            position: Данные позиции
        Returns:
            bool: True если нужна корректировка
        """
        try:
            # Проверка убыточности
            unrealized_pnl = position.get("unrealized_pnl", 0)
            position_size = position.get("size", 0)
            # Если позиция убыточна более чем на 50% от размера
            if unrealized_pnl < -(position_size * 0.5):
                return True
            # Проверка времени в позиции
            entry_time = position.get("entry_time", 0)
            current_time = time.time()
            time_in_position = current_time - entry_time
            # Если позиция открыта более 24 часов и убыточна
            if time_in_position > 86400 and unrealized_pnl < 0:
                return True
            # Проверка волатильности
            volatility = position.get("volatility", 0)
            if volatility > 0.05:  # Высокая волатильность
                return True
            return False
        except Exception as e:
            logger.error(f"Error checking position adjustment need: {e}")
            return False

    async def _adjust_position(self, position: Dict[str, Any]) -> None:
        """
        Корректировка позиции.
        Args:
            position: Данные позиции
        """
        try:
            symbol = position.get("symbol", "")
            current_size = position.get("size", 0)
            # Уменьшение размера позиции на 25%
            new_size = current_size * 0.75
            # Отправка команды на корректировку
            await self.event_bus.publish(
                Event(
                    name="risk.position.adjust",
                    type="risk",
                    data={
                        "symbol": symbol,
                        "new_size": new_size,
                        "reason": "Risk management adjustment",
                        "original_size": current_size,
                        "adjustment_factor": 0.75,
                    },
                    priority=EventPriority.HIGH,
                )
            )
            logger.info(
                f"Position adjustment requested for {symbol}: "
                f"{current_size} -> {new_size}"
            )
        except Exception as e:
            logger.error(f"Error adjusting position: {e}")

    async def _reduce_portfolio_risk(self) -> None:
        """Снижение общего риска портфеля."""
        try:
            # Сортировка позиций по убытку
            positions = sorted(
                self.current_metrics["open_positions"],
                key=lambda x: x.get("unrealized_pnl", 0),
            )
            # Закрытие 25% самых убыточных позиций
            positions_to_close = positions[: max(1, len(positions) // 4)]
            for position in positions_to_close:
                symbol = position.get("symbol", "")
                await self.event_bus.publish(
                    Event(
                        name="risk.position.close",
                        type="risk",
                        data={
                            "symbol": symbol,
                            "reason": "Portfolio risk reduction",
                            "risk_level": "high",
                        },
                        priority=EventPriority.HIGH,
                    )
                )
            logger.info(
                f"Portfolio risk reduction: Closing {len(positions_to_close)} "
                f"positions"
            )
        except Exception as e:
            logger.error(f"Error reducing portfolio risk: {e}")

    async def calculate_position_risk(
        self, symbol: str, size: float, price: float
    ) -> Dict[str, Any]:
        """
        Расчет риска позиции.
        Args:
            symbol: Торговая пара
            size: Размер позиции
            price: Цена
        Returns:
            Dict[str, Any]: Оценка риска позиции
        """
        try:
            # Базовый расчет риска
            exposure = size * price
            volatility = self.current_metrics.get("correlations", {}).get(symbol, 0.02)
            # VaR расчет
            var_95 = exposure * volatility * 1.645
            # Максимальный убыток
            max_loss = exposure * 0.1  # 10% от экспозиции
            return {
                "symbol": symbol,
                "exposure": exposure,
                "var_95": var_95,
                "max_loss": max_loss,
                "volatility": volatility,
                "risk_score": min(var_95 / exposure, 1.0),
            }
        except Exception as e:
            logger.error(f"Error calculating position risk: {e}")
            return {
                "symbol": symbol,
                "exposure": 0,
                "var_95": 0,
                "max_loss": 0,
                "volatility": 0,
                "risk_score": 0,
            }

    async def update_position_data(self, position_data: Dict[str, Any]) -> None:
        """
        Обновление данных о позициях.
        Args:
            position_data: Данные о позициях
        """
        try:
            # Обновление списка открытых позиций
            if "open_positions" in position_data:
                self.current_metrics["open_positions"] = position_data["open_positions"]
            # Обновление корреляций
            if "correlations" in position_data:
                correlations = self.current_metrics.get("correlations", {})
                if isinstance(correlations, dict):
                    correlations.update(position_data["correlations"])
            # Пересчет риска портфеля
            await self._update_portfolio_risk()
            logger.info("Position data updated")
        except Exception as e:
            logger.error(f"Error updating position data: {e}")

    async def get_risk_report(self) -> Dict[str, Any]:
        """
        Получение отчета о рисках.
        Returns:
            Dict[str, Any]: Отчет о рисках
        """
        try:
            # Генерация алертов
            alerts = self._generate_risk_alerts()
            # Расчет дополнительных метрик
            positions = self.current_metrics.get("open_positions", [])
            if isinstance(positions, list):
                total_exposure = sum(
                    [
                        pos.get("exposure", 0)
                        for pos in positions if isinstance(pos, dict)
                    ]
                )
                # Статистика по позициям
                position_stats = {
                    "total_positions": len(positions),
                    "profitable_positions": sum(
                        1
                        for pos in positions if isinstance(pos, dict)
                        if pos.get("unrealized_pnl", 0) > 0
                    ),
                    "losing_positions": sum(
                        1
                        for pos in positions if isinstance(pos, dict)
                        if pos.get("unrealized_pnl", 0) < 0
                    ),
                }
            else:
                total_exposure = 0
                position_stats = {
                    "total_positions": 0,
                    "profitable_positions": 0,
                    "losing_positions": 0,
                }
            return {
                "timestamp": time.time(),
                "risk_metrics": self.current_metrics,
                "risk_limits": self.risk_limits,
                "total_exposure": total_exposure,
                "position_stats": position_stats,
                "alerts": alerts,
                "risk_status": self.get_risk_status(),
            }
        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            return {}

    def _generate_risk_alerts(self) -> List[Dict[str, Any]]:
        """
        Генерация алертов о рисках.
        Returns:
            List[Dict[str, Any]]: Список алертов
        """
        alerts = []
        # Проверка дневного лимита
        daily_loss = self.current_metrics.get("daily_loss", 0.0)
        max_daily_loss = self.risk_limits.get("max_daily_loss", 1.0)
        if isinstance(daily_loss, (int, float)) and isinstance(max_daily_loss, (int, float)):
            if daily_loss > max_daily_loss * 0.8:
                alerts.append(
                    {
                        "type": "warning",
                        "message": "Daily loss approaching limit",
                        "value": daily_loss,
                        "limit": max_daily_loss,
                    }
                )
        # Проверка риска портфеля
        portfolio_risk = self.current_metrics.get("portfolio_risk", 0.0)
        max_portfolio_risk = self.risk_limits.get("max_portfolio_risk", 1.0)
        if isinstance(portfolio_risk, (int, float)) and isinstance(max_portfolio_risk, (int, float)):
            if portfolio_risk > max_portfolio_risk * 0.9:
                alerts.append(
                    {
                        "type": "critical",
                        "message": "Portfolio risk near limit",
                        "value": portfolio_risk,
                        "limit": max_portfolio_risk,
                    }
                )
        # Проверка количества позиций
        positions = self.current_metrics.get("open_positions", [])
        if isinstance(positions, list) and len(positions) > 10:
            alerts.append(
                {
                    "type": "info",
                    "message": "High number of open positions",
                    "value": len(positions),
                    "recommendation": "Consider reducing position count",
                }
            )
        return alerts

    async def reset_daily_metrics(self) -> None:
        """Сброс дневных метрик."""
        self.current_metrics["daily_loss"] = 0.0
        logger.info("Daily risk metrics reset")

    async def reset_weekly_metrics(self) -> None:
        """Сброс недельных метрик."""
        self.current_metrics["weekly_loss"] = 0.0
        logger.info("Weekly risk metrics reset")

    def get_risk_status(self) -> Dict[str, Any]:
        """
        Получение статуса риска.
        Returns:
            Dict[str, Any]: Статус риска
        """
        try:
            # Определение общего статуса риска
            risk_level = "low"
            portfolio_risk = self.current_metrics.get("portfolio_risk", 0.0)
            if isinstance(portfolio_risk, (int, float)):
                if portfolio_risk > 0.7:
                    risk_level = "critical"
                elif portfolio_risk > 0.5:
                    risk_level = "high"
                elif portfolio_risk > 0.3:
                    risk_level = "medium"
            positions = self.current_metrics.get("open_positions", [])
            position_count = len(positions) if isinstance(positions, list) else 0
            return {
                "risk_level": risk_level,
                "portfolio_risk": portfolio_risk,
                "daily_loss": self.current_metrics.get("daily_loss", 0.0),
                "weekly_loss": self.current_metrics.get("weekly_loss", 0.0),
                "position_count": position_count,
                "is_safe": risk_level in ["low", "medium"],
            }
        except Exception as e:
            logger.error(f"Error getting risk status: {e}")
            return {
                "risk_level": "unknown",
                "portfolio_risk": 0.0,
                "daily_loss": 0.0,
                "weekly_loss": 0.0,
                "position_count": 0,
                "is_safe": False,
            }
