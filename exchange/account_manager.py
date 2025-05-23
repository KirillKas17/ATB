import asyncio
from asyncio import Lock
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional

import backoff
import numpy as np
from loguru import logger

from .bybit_client import BybitClient
from .order_manager import OrderManager


@dataclass
class AccountMetrics:
    """Метрики аккаунта"""

    equity: float  # Общий капитал
    free_margin: float  # Свободная маржа
    used_margin: float  # Использованная маржа
    unrealized_pnl: float  # Нереализованная прибыль/убыток
    leverage_usage: float  # Использование плеча
    total_positions: int  # Количество открытых позиций
    total_orders: int  # Количество активных ордеров
    margin_ratio: float  # Соотношение маржи
    risk_score: float  # Оценка риска (0-1)
    daily_pnl: float  # Прибыль за день
    weekly_pnl: float  # Прибыль за неделю
    monthly_pnl: float  # Прибыль за месяц


@dataclass
class RiskConfig:
    """Конфигурация рисков"""

    max_leverage_usage: float = 0.8  # Максимальное использование плеча
    min_free_margin: float = 0.2  # Минимальная свободная маржа
    max_positions: int = 10  # Максимальное количество позиций
    max_orders_per_position: int = 3  # Максимальное количество ордеров на позицию
    max_daily_loss: float = 0.05  # Максимальный дневной убыток
    max_position_size: float = 0.1  # Максимальный размер позиции от капитала
    min_margin_ratio: float = 0.1  # Минимальное соотношение маржи
    max_drawdown: float = 0.2  # Максимальная просадка
    risk_free_rate: float = 0.02  # Безрисковая ставка
    position_timeout: int = 3600  # Таймаут позиции в секундах


class AccountManager:
    def __init__(
        self,
        client: BybitClient,
        order_manager: OrderManager,
        risk_config: Optional[RiskConfig] = None,
    ):
        """Инициализация менеджера аккаунта"""
        self.client = client
        self.order_manager = order_manager
        self.risk_config = risk_config or RiskConfig()

        # Кэш и состояние
        self.metrics_cache = None
        self.last_update = None
        self.cache_ttl = 60  # секунд
        self.lock = Lock()

        # Мониторинг
        self.monitor_task = None
        self.risk_alerts = set()
        self.position_history = defaultdict(list)
        self.pnl_history = []

        # Метрики производительности
        self.performance_metrics = {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    @backoff.on_exception(backoff.expo, (ConnectionError, TimeoutError), max_tries=3)
    async def start(self):
        """Запуск менеджера с повторными попытками"""
        try:
            self.monitor_task = asyncio.create_task(self._monitor_account())
            logger.info("Account manager started")
        except Exception as e:
            logger.error(f"Error starting account manager: {str(e)}")
            raise

    async def stop(self):
        """Корректная остановка менеджера"""
        try:
            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass

            # Сохранение истории
            await self._save_history()
            logger.info("Account manager stopped")
        except Exception as e:
            logger.error(f"Error stopping account manager: {str(e)}")
            raise

    @lru_cache(maxsize=100)
    async def get_metrics(self) -> AccountMetrics:
        """Получение метрик аккаунта с кэшированием"""
        async with self.lock:
            try:
                # Проверка кэша
                if (
                    self.metrics_cache
                    and self.last_update
                    and (datetime.now() - self.last_update).total_seconds()
                    < self.cache_ttl
                ):
                    return self.metrics_cache

                # Получение баланса
                balance = await self.client.get_balance()

                # Расчет базовых метрик
                equity = sum(info["total"] for info in balance.values())
                used_margin = sum(info["used"] for info in balance.values())
                free_margin = equity - used_margin

                # Расчет PnL
                unrealized_pnl = await self._calculate_unrealized_pnl()
                daily_pnl = await self._calculate_period_pnl(timedelta(days=1))
                weekly_pnl = await self._calculate_period_pnl(timedelta(weeks=1))
                monthly_pnl = await self._calculate_period_pnl(timedelta(days=30))

                # Расчет рисков
                leverage_usage = used_margin / equity if equity > 0 else 0
                margin_ratio = (
                    free_margin / used_margin if used_margin > 0 else float("inf")
                )
                risk_score = await self._calculate_risk_score()

                # Подсчет позиций и ордеров
                total_positions = len(self.order_manager.active_orders)
                total_orders = sum(
                    1
                    for order in self.order_manager.active_orders.values()
                    if order.type != "position"
                )

                # Создание метрик
                metrics = AccountMetrics(
                    equity=equity,
                    free_margin=free_margin,
                    used_margin=used_margin,
                    unrealized_pnl=unrealized_pnl,
                    leverage_usage=leverage_usage,
                    total_positions=total_positions,
                    total_orders=total_orders,
                    margin_ratio=margin_ratio,
                    risk_score=risk_score,
                    daily_pnl=daily_pnl,
                    weekly_pnl=weekly_pnl,
                    monthly_pnl=monthly_pnl,
                )

                # Обновление кэша
                self.metrics_cache = metrics
                self.last_update = datetime.now()

                return metrics

            except Exception as e:
                logger.error(f"Error getting account metrics: {str(e)}")
                raise

    async def get_available_margin(self, symbol: str) -> float:
        """Расчет доступной маржи с учетом рисков"""
        try:
            metrics = await self.get_metrics()

            # Проверка ограничений
            if metrics.leverage_usage >= self.risk_config.max_leverage_usage:
                return 0.0

            if metrics.free_margin / metrics.equity < self.risk_config.min_free_margin:
                return 0.0

            if metrics.total_positions >= self.risk_config.max_positions:
                return 0.0

            if metrics.margin_ratio < self.risk_config.min_margin_ratio:
                return 0.0

            # Расчет доступной маржи с учетом рисков
            available_margin = (
                metrics.free_margin
                * (1 - metrics.leverage_usage / self.risk_config.max_leverage_usage)
                * (1 - metrics.risk_score)
            )

            return max(0.0, available_margin)

        except Exception as e:
            logger.error(f"Error calculating available margin: {str(e)}")
            raise

    async def can_open_position(
        self, symbol: str, amount: float, leverage: float
    ) -> bool:
        """Проверка возможности открытия позиции с расширенными проверками"""
        try:
            # Получение текущей цены
            ticker = await self.client.get_ticker(symbol)
            price = float(ticker["last"])

            # Расчет требуемой маржи
            required_margin = (amount * price) / leverage

            # Проверка доступной маржи
            available_margin = await self.get_available_margin(symbol)

            # Проверка размера позиции
            metrics = await self.get_metrics()
            max_position_size = metrics.equity * self.risk_config.max_position_size
            if amount * price > max_position_size:
                return False

            # Проверка дневного убытка
            if metrics.daily_pnl < -metrics.equity * self.risk_config.max_daily_loss:
                return False

            # Проверка просадки
            if self.performance_metrics["max_drawdown"] > self.risk_config.max_drawdown:
                return False

            return required_margin <= available_margin

        except Exception as e:
            logger.error(f"Error checking position possibility: {str(e)}")
            raise

    async def _monitor_account(self):
        """Мониторинг аккаунта с расширенной функциональностью"""
        try:
            while True:
                try:
                    # Обновление метрик
                    await self.get_metrics()

                    # Проверка ограничений
                    await self._check_risk_limits()

                    # Обновление метрик производительности
                    await self._update_performance_metrics()

                    # Проверка таймаутов позиций
                    await self._check_position_timeouts()

                    # Очистка старых данных
                    await self._cleanup_old_data()

                except Exception as e:
                    logger.error(f"Error in account monitor: {str(e)}")

                await asyncio.sleep(1)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in account monitor: {str(e)}")
            raise

    async def _calculate_unrealized_pnl(self) -> float:
        """Расчет нереализованной прибыли с учетом комиссий"""
        try:
            total_pnl = 0.0

            for order in self.order_manager.active_orders.values():
                if order.type == "position":
                    # Получение текущей цены
                    ticker = await self.client.get_ticker(order.symbol)
                    current_price = float(ticker["last"])

                    # Расчет PnL с учетом комиссий
                    if order.side == "buy":
                        pnl = (current_price - order.price) * order.amount
                    else:
                        pnl = (order.price - current_price) * order.amount

                    # Учет комиссий
                    pnl -= order.amount * current_price * 0.00075  # 0.075% комиссия

                    total_pnl += pnl

            return total_pnl

        except Exception as e:
            logger.error(f"Error calculating unrealized PnL: {str(e)}")
            raise

    async def _calculate_period_pnl(self, period: timedelta) -> float:
        """Расчет прибыли за период"""
        try:
            start_time = datetime.now() - period
            return sum(
                pnl for timestamp, pnl in self.pnl_history if timestamp >= start_time
            )
        except Exception as e:
            logger.error(f"Error calculating period PnL: {str(e)}")
            raise

    async def _calculate_risk_score(self) -> float:
        """Расчет оценки риска (0-1)"""
        try:
            metrics = await self.get_metrics()

            # Факторы риска
            leverage_risk = metrics.leverage_usage / self.risk_config.max_leverage_usage
            margin_risk = 1 - (metrics.free_margin / metrics.equity)
            position_risk = metrics.total_positions / self.risk_config.max_positions
            drawdown_risk = (
                self.performance_metrics["max_drawdown"] / self.risk_config.max_drawdown
            )

            # Взвешенная оценка
            risk_score = (
                leverage_risk * 0.3
                + margin_risk * 0.3
                + position_risk * 0.2
                + drawdown_risk * 0.2
            )

            return min(1.0, max(0.0, risk_score))

        except Exception as e:
            logger.error(f"Error calculating risk score: {str(e)}")
            raise

    async def _update_performance_metrics(self):
        """Обновление метрик производительности"""
        try:
            if not self.pnl_history:
                return

            # Расчет доходности
            returns = [pnl for _, pnl in self.pnl_history]
            avg_return = sum(returns) / len(returns)

            # Волатильность
            volatility = np.std(returns) if len(returns) > 1 else 0

            # Sharpe Ratio
            if volatility > 0:
                self.performance_metrics["sharpe_ratio"] = (
                    avg_return - self.risk_config.risk_free_rate
                ) / volatility

            # Sortino Ratio
            downside_returns = [r for r in returns if r < 0]
            if downside_returns:
                downside_std = np.std(downside_returns)
                if downside_std > 0:
                    self.performance_metrics["sortino_ratio"] = (
                        avg_return - self.risk_config.risk_free_rate
                    ) / downside_std

            # Win Rate
            winning_trades = sum(1 for r in returns if r > 0)
            self.performance_metrics["win_rate"] = winning_trades / len(returns)

            # Profit Factor
            gross_profit = sum(r for r in returns if r > 0)
            gross_loss = abs(sum(r for r in returns if r < 0))
            self.performance_metrics["profit_factor"] = (
                gross_profit / gross_loss if gross_loss > 0 else float("inf")
            )

            # Maximum Drawdown
            cumulative_returns = np.cumsum(returns)
            max_drawdown = 0
            peak = cumulative_returns[0]

            for value in cumulative_returns:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)

            self.performance_metrics["max_drawdown"] = max_drawdown

        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
            raise

    async def _check_position_timeouts(self):
        """Проверка таймаутов позиций"""
        try:
            current_time = datetime.now()

            for order in self.order_manager.active_orders.values():
                if order.type == "position":
                    position_time = (current_time - order.timestamp).total_seconds()
                    if position_time > self.risk_config.position_timeout:
                        logger.warning(f"Position timeout for {order.symbol}")
                        # Здесь можно добавить логику закрытия позиции

        except Exception as e:
            logger.error(f"Error checking position timeouts: {str(e)}")
            raise

    async def _cleanup_old_data(self):
        """Очистка старых данных"""
        try:
            current_time = datetime.now()

            # Очистка истории PnL
            self.pnl_history = [
                (timestamp, pnl)
                for timestamp, pnl in self.pnl_history
                if (current_time - timestamp).days < 30
            ]

            # Очистка истории позиций
            for symbol in self.position_history:
                self.position_history[symbol] = [
                    pos
                    for pos in self.position_history[symbol]
                    if (current_time - pos["timestamp"]).days < 30
                ]

        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
            raise

    async def _save_history(self):
        """Сохранение истории"""
        try:
            # Здесь можно добавить логику сохранения в базу данных
            pass
        except Exception as e:
            logger.error(f"Error saving history: {str(e)}")
            raise

    async def _check_risk_limits(self):
        """Проверка ограничений риска"""
        try:
            metrics = await self.get_metrics()

            # Проверка использования плеча
            if metrics.leverage_usage > self.risk_config.max_leverage_usage:
                logger.warning("Leverage usage exceeds limit")
                # TODO: Implement risk reduction logic

            # Проверка свободной маржи
            if metrics.free_margin / metrics.equity < self.risk_config.min_free_margin:
                logger.warning("Free margin below minimum")
                # TODO: Implement margin increase logic

            # Проверка количества позиций
            if metrics.total_positions > self.risk_config.max_positions:
                logger.warning("Number of positions exceeds limit")
                # TODO: Implement position reduction logic

        except Exception as e:
            logger.error(f"Error checking risk limits: {str(e)}")
            raise
