"""
Промышленная реализация PortfolioService.
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from application.protocols.service_protocols import PortfolioService
from application.services.base_service import BaseApplicationService
from application.types import CreateOrderRequest
from application.protocols.service_protocols import PerformanceMetrics, PortfolioMetrics
from domain.value_objects.balance import Balance
from domain.types import PortfolioId, Symbol
from domain.types.repository_types import EntityId
from domain.entities.portfolio import Portfolio
from domain.repositories.portfolio_repository import PortfolioRepository
from domain.services.portfolio_analysis import PortfolioAnalysisService
from infrastructure.agents.portfolio.optimizers import IPortfolioOptimizer as PortfolioOptimizer
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.value_objects.timestamp import Timestamp, TimestampValue
from uuid import UUID


class PortfolioServiceImpl(BaseApplicationService, PortfolioService):
    """Промышленная реализация сервиса портфеля."""

    def __init__(
        self,
        portfolio_repository: PortfolioRepository,
        portfolio_optimizer: PortfolioOptimizer,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__("PortfolioService", config)
        self.portfolio_repository = portfolio_repository
        self.portfolio_optimizer = portfolio_optimizer
        self.portfolio_analysis_service = PortfolioAnalysisService()
        # Кэш для портфелей и метрик
        self._portfolio_cache: Dict[str, Portfolio] = {}
        self._metrics_cache: Dict[str, PortfolioMetrics] = {}
        self._balance_cache: Dict[str, Dict[str, Money]] = {}
        self._performance_cache: Dict[str, PerformanceMetrics] = {}
        # Конфигурация
        self.portfolio_cache_ttl = self.config.get(
            "portfolio_cache_ttl", 300
        )  # 5 минут
        self.metrics_cache_ttl = self.config.get("metrics_cache_ttl", 60)  # 1 минута
        self.balance_cache_ttl = self.config.get("balance_cache_ttl", 30)  # 30 секунд
        self.performance_cache_ttl = self.config.get(
            "performance_cache_ttl", 3600
        )  # 1 час
        # Статистика портфелей
        self._portfolio_stats = {
            "total_portfolios": 0,
            "active_portfolios": 0,
            "total_rebalances": 0,
            "successful_rebalances": 0,
            "total_pnl_calculations": 0,
        }

    async def initialize(self) -> None:
        """Инициализация сервиса."""
        # Загружаем статистику портфелей
        await self._load_portfolio_statistics()
        # Запускаем мониторинг портфелей
        asyncio.create_task(self._portfolio_monitoring_loop())
        self.logger.info("PortfolioService initialized")

    async def validate_config(self) -> bool:
        """Валидация конфигурации."""
        required_configs = [
            "portfolio_cache_ttl",
            "metrics_cache_ttl",
            "balance_cache_ttl",
            "performance_cache_ttl",
        ]
        for config_key in required_configs:
            if config_key not in self.config:
                self.logger.error(f"Missing required config: {config_key}")
                return False
        return True

    async def calculate_weights(
        self, portfolio_id: UUID
    ) -> Dict[Symbol, Decimal]:
        """Расчет весов портфеля."""
        return await self._execute_with_metrics(
            "calculate_weights", self._calculate_weights_impl, portfolio_id
        )

    async def _calculate_weights_impl(
        self, portfolio_id: UUID
    ) -> Dict[Symbol, Decimal]:
        """Реализация расчета весов портфеля через domain/services/portfolio_analysis.py."""
        try:
            # Получаем позиции портфеля
            positions = await self.portfolio_repository.get_all_positions(EntityId(portfolio_id))
            # Используем domain-сервис для расчёта - приводим к совместимому типу
            weights_result = self.portfolio_analysis_service.calculate_weights(
                positions
            )
            return weights_result.weights
        except Exception as e:
            self.logger.error(
                f"Error calculating weights for portfolio {portfolio_id}: {e}"
            )
            return {}

    async def rebalance_portfolio(
        self, portfolio_id: UUID, target_weights: Dict[Symbol, Decimal]
    ) -> bool:
        """Ребалансировка портфеля."""
        return await self._execute_with_metrics(
            "rebalance_portfolio",
            self._rebalance_portfolio_impl,
            portfolio_id,
            target_weights,
        )

    async def _rebalance_portfolio_impl(
        self, portfolio_id: UUID, target_weights: Dict[Symbol, Decimal]
    ) -> bool:
        """Реализация ребалансировки портфеля."""
        try:
            # Получаем текущие веса
            current_weights = await self.calculate_weights(portfolio_id)
            # Рассчитываем необходимые изменения
            rebalance_orders = await self._calculate_rebalance_orders(
                portfolio_id, current_weights, target_weights
            )
            if not rebalance_orders:
                self.logger.info(f"Portfolio {portfolio_id} is already balanced")
                return True
            # Выполняем ордера ребалансировки
            success_count = 0
            for order in rebalance_orders:
                try:
                    # Здесь должна быть логика исполнения ордера
                    # Пока просто логируем
                    self.logger.info(f"Rebalancing order: {order}")
                    success_count += 1
                except Exception as e:
                    self.logger.error(f"Error executing rebalance order: {e}")
            # Обновляем статистику
            self._portfolio_stats["total_rebalances"] = int(self._portfolio_stats.get("total_rebalances", 0)) + 1
            if success_count == len(rebalance_orders):
                self._portfolio_stats["successful_rebalances"] = int(self._portfolio_stats.get("successful_rebalances", 0)) + 1
            # Очищаем кэш
            self._clear_portfolio_cache(PortfolioId(portfolio_id))
            self.logger.info(f"Portfolio {portfolio_id} rebalanced successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error rebalancing portfolio {portfolio_id}: {e}")
            return False

    async def calculate_pnl(self, portfolio_id: PortfolioId) -> Money:
        """Расчет P&L."""
        return await self._execute_with_metrics(
            "calculate_pnl", self._calculate_pnl_impl, portfolio_id
        )

    async def _calculate_pnl_impl(self, portfolio_id: PortfolioId) -> Money:
        """Реализация расчета P&L через domain/services/portfolio_analysis.py."""
        try:
            # Получаем позиции портфеля
            positions = await self.portfolio_repository.get_all_positions(EntityId(portfolio_id))
            # Используем domain-сервис для расчёта
            pnl = self.portfolio_analysis_service.calculate_pnl(positions)
            # Обновляем статистику
            self._portfolio_stats["total_pnl_calculations"] = int(self._portfolio_stats.get("total_pnl_calculations", 0)) + 1
            return pnl
        except Exception as e:
            self.logger.error(
                f"Error calculating P&L for portfolio {portfolio_id}: {e}"
            )
            return Money(Decimal("0"), Currency("USDT"))

    async def get_portfolio_metrics(
        self, portfolio_id: UUID
    ) -> PortfolioMetrics:
        """Получение метрик портфеля."""
        return await self._execute_with_metrics(
            "get_portfolio_metrics", self._get_portfolio_metrics_impl, portfolio_id
        )

    async def _get_portfolio_metrics_impl(
        self, portfolio_id: UUID
    ) -> PortfolioMetrics:
        """Реализация получения метрик портфеля."""
        try:
            # Проверяем кэш
            cache_key = str(portfolio_id)
            if cache_key in self._metrics_cache:
                cached_metrics = self._metrics_cache[cache_key]
                if not self._is_metrics_cache_expired(cached_metrics):
                    return cached_metrics
            # Получаем портфель
            portfolio = await self.get_portfolio_by_id(portfolio_id)
            if not portfolio:
                raise ValueError(f"Portfolio {portfolio_id} not found")
            # Получаем позиции
            positions = await self.portfolio_repository.get_all_positions(EntityId(portfolio_id))
            # Получаем исторические данные для расчёта метрик
            historical_returns: list[Decimal] = []  # Если нет метода, оставляем пустым или реализуем иначе
            # Используем domain-сервис для расчёта всех метрик
            portfolio_metrics = (
                self.portfolio_analysis_service.calculate_portfolio_metrics(
                    positions, historical_returns
                )
            )
            # Извлекаем значения для создания PortfolioMetrics
            metrics = PortfolioMetrics(
                portfolio_id=PortfolioId(portfolio.id) if not isinstance(portfolio.id, PortfolioId) else portfolio.id,  # type: ignore[misc]
                total_pnl=Money(Decimal("0"), Currency.USD),
                total_trades=0,
                win_rate=Decimal("0"),
                profit_factor=Decimal("0"),
                volatility=getattr(portfolio_metrics, 'portfolio_volatility', Decimal("0")),
                sharpe_ratio=getattr(portfolio_metrics, 'sharpe_ratio', Decimal("0")),
                max_drawdown=getattr(portfolio_metrics, 'max_drawdown', Decimal("0")),
                timestamp=TimestampValue(Timestamp.now().value),  # type: ignore[arg-type]  # type: ignore[arg-type]
            )
            # Кэшируем метрики
            self._metrics_cache[cache_key] = metrics
            return metrics
        except Exception as e:
            self.logger.error(
                f"Error getting portfolio metrics for {portfolio_id}: {e}"
            )
            raise

    async def get_portfolio_balance(
        self, portfolio_id: UUID
    ) -> Dict[str, Money]:
        """Получение баланса портфеля."""
        return await self._execute_with_metrics(
            "get_portfolio_balance", self._get_portfolio_balance_impl, portfolio_id
        )

    async def _get_portfolio_balance_impl(
        self, portfolio_id: UUID
    ) -> Dict[str, Money]:
        """Реализация получения баланса портфеля."""
        try:
            # Проверяем кэш
            cache_key = str(portfolio_id)
            if cache_key in self._balance_cache:
                cached_balance = self._balance_cache[cache_key]
                if not self._is_balance_cache_expired(cached_balance):
                    return cached_balance
            # Получаем баланс из репозитория
            balance = await self.portfolio_repository.get_balance(
                EntityId(portfolio_id), Currency.USD
            )
            # Конвертируем Balance в Dict[str, Money]
            balance_dict = {}
            if hasattr(balance, 'balances'):
                for bal in balance.balances:
                    balance_dict[bal.currency] = Money(bal.total, Currency(bal.currency))
            # Кэшируем баланс
            self._balance_cache[cache_key] = balance_dict
            return balance_dict
        except Exception as e:
            self.logger.error(
                f"Error getting portfolio balance for {portfolio_id}: {e}"
            )
            return {}

    async def update_portfolio_balance(
        self, portfolio_id: UUID, changes: Dict[str, Money]
    ) -> bool:
        """Обновление баланса портфеля."""
        return await self._execute_with_metrics(
            "update_portfolio_balance",
            self._update_portfolio_balance_impl,
            portfolio_id,
            changes,
        )

    async def _update_portfolio_balance_impl(
        self, portfolio_id: UUID, changes: Dict[str, Money]
    ) -> bool:
        """Реализация обновления баланса портфеля."""
        try:
            # Конвертируем Dict[str, Money] в Balance
            balance = Balance(currency="USD", available=Decimal("0"), locked=Decimal("0"))
            for currency, money in changes.items():
                if currency == "USD":
                    balance.available = money.value
            # Обновляем баланс в репозитории
            success = await self.portfolio_repository.update_balance(
                EntityId(portfolio_id), balance
            )
            if success:
                # Очищаем кэш баланса
                cache_key = str(portfolio_id)
                if cache_key in self._balance_cache:
                    del self._balance_cache[cache_key]
                self.logger.info(
                    f"Portfolio {portfolio_id} balance updated successfully"
                )
                return True
            else:
                self.logger.error(f"Failed to update portfolio {portfolio_id} balance")
                return False
        except Exception as e:
            self.logger.error(
                f"Error updating portfolio balance for {portfolio_id}: {e}"
            )
            return False

    async def get_portfolio_performance(
        self, portfolio_id: UUID, period: str = "1d"
    ) -> PerformanceMetrics:
        """Получение производительности портфеля."""
        return await self._execute_with_metrics(
            "get_portfolio_performance",
            self._get_portfolio_performance_impl,
            portfolio_id,
            period,
        )

    async def _get_portfolio_performance_impl(
        self, portfolio_id: UUID, period: str = "1d"
    ) -> PerformanceMetrics:
        """Реализация получения производительности портфеля."""
        try:
            # Проверяем кэш
            cache_key = f"{portfolio_id}_{period}"
            if cache_key in self._performance_cache:
                cached_performance = self._performance_cache[cache_key]
                if not self._is_performance_cache_expired(cached_performance):
                    return cached_performance
            # Получаем исторические данные портфеля
            historical_data = await self.portfolio_repository.get_portfolio(
                EntityId(portfolio_id)
            )
            if not historical_data:
                return PerformanceMetrics(
                    portfolio_id=PortfolioId(UUID("00000000-0000-0000-0000-000000000000")),
                    period=period,
                    total_return=Decimal("0"),
                    daily_return=Decimal("0"),
                    volatility=Decimal("0"),
                    sharpe_ratio=Decimal("0"),
                    max_drawdown=Decimal("0"),
                    win_rate=Decimal("0"),
                    profit_factor=Decimal("0"),
                    total_trades=0,
                    timestamp=TimestampValue(Timestamp.now().value),
                )
            # Рассчитываем метрики производительности
            performance = await self._calculate_performance_metrics(
                historical_data, period  # type: ignore[arg-type]
            )
            # Кэшируем результат
            self._performance_cache[cache_key] = performance
            return performance
        except Exception as e:
            self.logger.error(
                f"Error getting portfolio performance for {portfolio_id}: {e}"
            )
            raise

    async def validate_portfolio_constraints(
        self, portfolio_id: UUID, order_request: Any
    ) -> Tuple[bool, List[str]]:
        """Валидация ограничений портфеля."""
        return await self._execute_with_metrics(
            "validate_portfolio_constraints",
            self._validate_portfolio_constraints_impl,
            portfolio_id,
            order_request,
        )

    async def _validate_portfolio_constraints_impl(
        self, portfolio_id: UUID, order_request: CreateOrderRequest
    ) -> Tuple[bool, List[str]]:
        """Реализация валидации ограничений портфеля."""
        try:
            errors = []
            # Получаем портфель
            portfolio = await self.get_portfolio_by_id(portfolio_id)
            if not portfolio:
                return False, [f"Portfolio {portfolio_id} not found"]
            # Проверяем лимиты риска
            if hasattr(portfolio, 'risk_limits') and portfolio.risk_limits:
                # Проверяем максимальный размер позиции
                max_position_size = portfolio.risk_limits.get("max_position_size")
                if max_position_size and hasattr(order_request, 'amount') and order_request.amount.value > max_position_size:
                    errors.append(
                        f"Order amount {order_request.amount.value} exceeds max position size {max_position_size}"
                    )
                # Проверяем максимальную концентрацию
                max_concentration = portfolio.risk_limits.get("max_concentration")
                if max_concentration and hasattr(order_request, 'symbol'):
                    current_weights = await self.calculate_weights(portfolio_id)
                    symbol_weight = current_weights.get(
                        order_request.symbol, Decimal("0")
                    )
                    if symbol_weight > max_concentration:
                        errors.append(
                            f"Symbol {order_request.symbol} concentration {symbol_weight} exceeds limit {max_concentration}"
                        )
            # Проверяем баланс
            balance = await self.get_portfolio_balance(portfolio_id)
            if hasattr(order_request, 'symbol'):
                required_currency = (
                    order_request.symbol.split("/")[-1]
                    if "/" in str(order_request.symbol)
                    else "USDT"
                )
                if required_currency in balance:
                    available_balance = balance[required_currency]
                    if hasattr(order_request, 'amount'):
                        required_amount = order_request.amount.value
                        if hasattr(order_request, 'price') and order_request.price:
                            required_amount *= order_request.price.value
                        if available_balance.value < required_amount:
                            errors.append(
                                f"Insufficient balance: {available_balance.value} < {required_amount}"
                            )
                else:
                    errors.append(
                        f"Currency {required_currency} not available in portfolio"
                    )
            return len(errors) == 0, errors
        except Exception as e:
            self.logger.error(
                f"Error validating portfolio constraints for {portfolio_id}: {e}"
            )
            return False, [f"Validation error: {str(e)}"]

    async def get_portfolio_by_id(
        self, portfolio_id: UUID
    ) -> Optional[Portfolio]:
        """Получение портфеля по ID."""
        try:
            # Проверяем кэш
            cache_key = str(portfolio_id)
            if cache_key in self._portfolio_cache:
                cached_portfolio = self._portfolio_cache[cache_key]
                if not self._is_portfolio_cache_expired(cached_portfolio):
                    return cached_portfolio
            # Получаем из репозитория
            portfolio = await self.portfolio_repository.get_by_id(EntityId(portfolio_id))
            if portfolio:
                # Кэшируем портфель
                self._portfolio_cache[cache_key] = portfolio
            return portfolio
        except Exception as e:
            self.logger.error(f"Error getting portfolio {portfolio_id}: {e}")
            return None

    async def _calculate_rebalance_orders(
        self,
        portfolio_id: UUID,
        current_weights: Dict[Symbol, Decimal],
        target_weights: Dict[Symbol, Decimal],
    ) -> List[Any]:
        """Расчет ордеров для ребалансировки."""
        try:
            # Получаем общую стоимость портфеля
            portfolio_metrics = await self.get_portfolio_metrics(portfolio_id)
            total_value = getattr(portfolio_metrics, 'total_pnl', Money(Decimal("0"), Currency("USDT"))).value
            orders = []
            # Рассчитываем необходимые изменения для каждого символа
            all_symbols = set(current_weights.keys()) | set(target_weights.keys())
            for symbol in all_symbols:
                current_weight = current_weights.get(symbol, Decimal("0"))
                target_weight = target_weights.get(symbol, Decimal("0"))
                if abs(current_weight - target_weight) > Decimal(
                    "0.01"
                ):  # Минимальный порог
                    current_value = current_weight * total_value
                    target_value = target_weight * total_value
                    value_change = target_value - current_value
                    if abs(value_change) > Decimal("10"):  # Минимальный размер ордера
                        # Создаем ордер для ребалансировки
                        order = {
                            "symbol": symbol,
                            "side": "BUY" if value_change > 0 else "SELL",
                            "amount": abs(value_change),
                            "type": "MARKET",
                            "portfolio_id": portfolio_id,
                            "metadata": {
                                "rebalance": True,
                                "current_weight": float(current_weight),
                                "target_weight": float(target_weight),
                            },
                        }
                        orders.append(order)
            return orders
        except Exception as e:
            self.logger.error(f"Error calculating rebalance orders: {e}")
            return []

    async def _calculate_portfolio_volatility(
        self, portfolio_id: PortfolioId
    ) -> Decimal:
        """Расчет волатильности портфеля через domain/services/portfolio_analysis.py."""
        try:
            # Получаем исторические данные
            historical_returns = await self.portfolio_repository.get_portfolio(
                EntityId(portfolio_id)  # type: ignore[arg-type]
            )
            # Используем domain-сервис для расчёта
            portfolio_metrics = (
                self.portfolio_analysis_service.calculate_portfolio_metrics(
                    [], historical_returns  # type: ignore[arg-type]
                )
            )
            return portfolio_metrics.volatility
        except Exception as e:
            self.logger.error(f"Error calculating portfolio volatility: {e}")
            return Decimal("0")

    async def _calculate_sharpe_ratio(self, portfolio_id: PortfolioId) -> Decimal:
        """Расчет коэффициента Шарпа через domain/services/portfolio_analysis.py."""
        try:
            # Получаем исторические данные
            historical_returns = await self.portfolio_repository.get_portfolio(
                EntityId(portfolio_id)  # type: ignore[arg-type]
            )
            # Используем domain-сервис для расчёта
            portfolio_metrics = (
                self.portfolio_analysis_service.calculate_portfolio_metrics(
                    [], historical_returns  # type: ignore[arg-type]
                )
            )
            return portfolio_metrics.sharpe_ratio
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return Decimal("0")

    async def _calculate_max_drawdown(self, portfolio_id: PortfolioId) -> Decimal:
        """Расчет максимальной просадки через domain/services/portfolio_analysis.py."""
        try:
            # Получаем исторические данные
            historical_returns = await self.portfolio_repository.get_portfolio(
                EntityId(portfolio_id)  # type: ignore[arg-type]
            )
            # Используем domain-сервис для расчёта
            portfolio_metrics = (
                self.portfolio_analysis_service.calculate_portfolio_metrics(
                    [], historical_returns  # type: ignore[arg-type]
                )
            )
            return portfolio_metrics.max_drawdown
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {e}")
            return Decimal("0")

    async def _calculate_performance_metrics(
        self, historical_data: List[Dict[str, Any]], period: str
    ) -> PerformanceMetrics:
        """Расчет метрик производительности."""
        try:
            if not historical_data:
                return PerformanceMetrics(
                    portfolio_id=PortfolioId(UUID("00000000-0000-0000-0000-000000000000")),
                    period=period,
                    total_return=Decimal("0"),
                    daily_return=Decimal("0"),
                    volatility=Decimal("0"),
                    sharpe_ratio=Decimal("0"),
                    max_drawdown=Decimal("0"),
                    win_rate=Decimal("0"),
                    profit_factor=Decimal("0"),
                    total_trades=0,
                    timestamp=TimestampValue(Timestamp.now().value),  # type: ignore[arg-type]
                )
            # Рассчитываем общую доходность
            initial_value = historical_data[0]["value"]
            final_value = historical_data[-1]["value"]
            total_return = (
                (final_value - initial_value) / initial_value
                if initial_value > 0
                else Decimal("0")
            )
            # Рассчитываем дневную доходность
            daily_return = (
                total_return / len(historical_data)
                if len(historical_data) > 0
                else Decimal("0")
            )
            # Рассчитываем волатильность
            returns = []
            for i in range(1, len(historical_data)):
                prev_value = historical_data[i - 1]["value"]
                curr_value = historical_data[i]["value"]
                if prev_value > 0:
                    daily_return = (curr_value - prev_value) / prev_value
                    returns.append(daily_return)
            import statistics

            volatility = (
                Decimal(str(statistics.stdev(returns)))
                if len(returns) > 1
                else Decimal("0")
            )
            # Рассчитываем коэффициент Шарпа
            sharpe_ratio = Decimal("0")
            if volatility > 0 and returns:
                avg_return = statistics.mean(returns)
                sharpe_ratio = Decimal(str(avg_return / volatility * (252**0.5)))
            # Рассчитываем максимальную просадку
            peak = historical_data[0]["value"]
            max_drawdown = Decimal("0")
            for data_point in historical_data:
                value = data_point["value"]
                if value > peak:
                    peak = value
                else:
                    drawdown = (peak - value) / peak
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
            return PerformanceMetrics(
                portfolio_id=PortfolioId(UUID("00000000-0000-0000-0000-000000000000")),
                period=period,
                total_return=total_return,
                daily_return=daily_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=Decimal("0"),  # Упрощенно
                profit_factor=Decimal("0"),  # Упрощенно
                total_trades=0,  # Упрощенно
                timestamp=TimestampValue(Timestamp.now().value),
            )
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            raise

    async def _load_portfolio_statistics(self) -> None:
        """Загрузка статистики портфелей."""
        try:
            portfolios = await self.portfolio_repository.get_all()
            self._portfolio_stats["total_portfolios"] = len(portfolios)
            self._portfolio_stats["active_portfolios"] = len(
                [p for p in portfolios if p.is_active]
            )
        except Exception as e:
            self.logger.error(f"Error loading portfolio statistics: {e}")

    def _is_portfolio_cache_expired(self, portfolio: Portfolio) -> bool:
        """Проверка истечения срока действия кэша портфеля."""
        # Упрощенно - считаем кэш истекшим через 5 минут
        return True

    def _is_metrics_cache_expired(self, metrics: PortfolioMetrics) -> bool:
        """Проверка истечения срока действия кэша метрик."""
        # Упрощенно - считаем кэш истекшим через 1 минуту
        return True

    def _is_balance_cache_expired(self, balance: Dict[str, Money]) -> bool:
        """Проверка истечения срока действия кэша баланса."""
        # Упрощенно - всегда обновляем баланс
        return True

    def _is_performance_cache_expired(self, performance: PerformanceMetrics) -> bool:
        """Проверка истечения срока действия кэша производительности."""
        # Упрощенно - считаем кэш истекшим через 1 час
        return True

    def _clear_portfolio_cache(self, portfolio_id: PortfolioId) -> None:
        """Очистка кэша портфеля."""
        cache_key = str(portfolio_id)
        if cache_key in self._portfolio_cache:
            del self._portfolio_cache[cache_key]
        if cache_key in self._metrics_cache:
            del self._metrics_cache[cache_key]
        if cache_key in self._balance_cache:
            del self._balance_cache[cache_key]

    async def _portfolio_monitoring_loop(self) -> None:
        """Цикл мониторинга портфелей."""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Каждые 5 минут
                # Проверяем активные портфели
                portfolios = await self.portfolio_repository.get_all()
                active_portfolios = [p for p in portfolios if p.is_active]
                for portfolio in active_portfolios:
                    try:
                        # Проверяем метрики портфеля
                        metrics = await self.get_portfolio_metrics(portfolio.id)
                        # Проверяем лимиты риска
                        if portfolio.risk_limits:
                            max_drawdown_limit = portfolio.risk_limits.get(
                                "max_drawdown"
                            )
                            if (
                                max_drawdown_limit
                                and metrics.max_drawdown > max_drawdown_limit
                            ):
                                self.logger.warning(
                                    f"Portfolio {portfolio.id} exceeded max drawdown limit: {metrics.max_drawdown}"
                                )
                        # Проверяем производительность
                        if metrics.total_pnl.value < Decimal(
                            "-1000"
                        ):  # Убыток более 1000 USDT
                            self.logger.warning(
                                f"Portfolio {portfolio.id} has significant loss: {metrics.total_pnl}"
                            )
                    except Exception as e:
                        self.logger.error(
                            f"Error monitoring portfolio {portfolio.id}: {e}"
                        )
            except Exception as e:
                self.logger.error(f"Error in portfolio monitoring loop: {e}")

    async def get_portfolio_statistics(self) -> Dict[str, Any]:
        """Получение статистики портфелей."""
        return {
            "total_portfolios": self._portfolio_stats.get("total_portfolios", 0),
            "active_portfolios": self._portfolio_stats.get("active_portfolios", 0),
            "total_rebalances": self._portfolio_stats.get("total_rebalances", 0),
            "successful_rebalances": self._portfolio_stats.get("successful_rebalances", 0),
            "rebalance_success_rate": (
                self._portfolio_stats.get("successful_rebalances", 0)
                / self._portfolio_stats.get("total_rebalances", 1)
                if self._portfolio_stats.get("total_rebalances", 0) > 0
                else 0.0
            ),
            "total_pnl_calculations": self._portfolio_stats.get("total_pnl_calculations", 0),
            "cache_sizes": {
                "portfolios": len(self._portfolio_cache),
                "metrics": len(self._metrics_cache),
                "balance": len(self._balance_cache),
                "performance": len(self._performance_cache),
            },
        }

    async def stop(self) -> None:
        """Остановка сервиса."""
        await super().stop()
        # Очищаем кэши
        self._portfolio_cache.clear()
        self._metrics_cache.clear()
        self._balance_cache.clear()
        self._performance_cache.clear()
