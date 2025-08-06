"""Модуль продвинутого управления портфелем с оптимизацией и ребалансировкой."""

import asyncio
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from loguru import logger
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from shared.numpy_utils import np
import pandas as pd
import time

from infrastructure.messaging.event_bus import Event, EventBus, EventPriority
from infrastructure.messaging.event_bus import EventName, EventType

warnings.filterwarnings("ignore")


@dataclass
class PortfolioOptimizationResult:
    """Результат оптимизации портфеля."""

    optimal_weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    efficient_frontier: List[Tuple[float, float]]
    risk_metrics: Dict[str, float]
    optimization_method: str
    timestamp: float


@dataclass
class RebalancingAction:
    """Действие ребалансировки."""

    symbol: str
    current_weight: float
    target_weight: float
    action: str  # 'buy' or 'sell'
    size_change: float
    estimated_cost: float
    priority: str  # 'high', 'medium', 'low'


class PortfolioManager:
    """
    Продвинутый менеджер портфеля с:
    - Оптимизацией по Марковицу
    - Black-Litterman моделью
    - Динамической ребалансировкой
    - Управлением рисками
    - Корреляционным анализом.
    """

    def __init__(self, event_bus: EventBus, config: Dict[str, Any]) -> None:
        """
        Инициализация менеджера портфеля.
        Args:
            event_bus: Шина событий для коммуникации
            config: Конфигурация портфеля
        """
        self.event_bus = event_bus
        self.portfolio_config = config
        # Портфель
        self.portfolio: Dict[str, Any] = {
            "positions": {},
            "weights": {},
            "total_value": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "cash": config.get("initial_cash", 10000.0),
        }
        # Корреляционная матрица
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        # История оптимизаций
        self.optimization_history: List[PortfolioOptimizationResult] = []
        # Конфигурация оптимизации
        self.optimization_config: Dict[str, Any] = {
            "method": "markowitz",  # 'markowitz', 'black_litterman', 'risk_parity'
            "risk_free_rate": 0.02,
            "target_return": None,
            "max_iterations": 1000,
            "tolerance": 1e-6,
            "rebalance_threshold": 0.05,
            "transaction_costs": 0.001,  # 0.1% за сделку
            "slippage": 0.0005,  # 0.05% проскальзывание
            "max_position_size": 0.3,
            "min_position_size": 0.01,
            "diversification_target": 0.7,
        }
        # Обновление конфигурации
        self.optimization_config.update(config.get("optimization", {}))
        # Исторические данные
        self.price_history: Dict[str, pd.Series] = {}
        self.returns_history: Dict[str, pd.Series] = {}
        # Статистика
        self.rebalancing_stats: Dict[str, Any] = {
            "total_rebalances": 0,
            "successful_rebalances": 0,
            "total_cost": 0.0,
            "last_rebalance": None,
        }
        logger.info("Advanced Portfolio Manager initialized")

    async def add_position(self, symbol: str, size: float, price: float) -> None:
        """
        Добавление позиции в портфель.
        Args:
            symbol: Символ актива
            size: Размер позиции
            price: Цена актива
        """
        if symbol in self.portfolio["positions"]:
            # Обновление существующей позиции
            position = self.portfolio["positions"][symbol]
            total_size = position["size"] + size
            avg_price = (
                position["size"] * position["avg_price"] + size * price
            ) / total_size
            position["size"] = total_size
            position["avg_price"] = avg_price
            position["current_price"] = price
            position["unrealized_pnl"] = (price - avg_price) * total_size
        else:
            # Новая позиция
            self.portfolio["positions"][symbol] = {
                "size": size,
                "avg_price": price,
                "current_price": price,
                "unrealized_pnl": 0.0,
                "entry_time": asyncio.get_event_loop().time(),
            }
        # Обновление весов
        await self._update_weights()
        # Проверка лимитов
        await self._check_position_limits(symbol)
        logger.info(f"Added position: {symbol}, size: {size}, price: {price}")

    async def remove_position(self, symbol: str, size: float, price: float) -> None:
        """
        Удаление позиции из портфеля.
        Args:
            symbol: Символ актива
            size: Размер позиции
            price: Цена актива
        """
        if symbol not in self.portfolio["positions"]:
            logger.warning(f"Position {symbol} not found in portfolio")
            return
        position = self.portfolio["positions"][symbol]
        if size >= position["size"]:
            # Полное закрытие позиции
            realized_pnl = (price - position["avg_price"]) * position["size"]
            self.portfolio["realized_pnl"] += realized_pnl
            del self.portfolio["positions"][symbol]
        else:
            # Частичное закрытие
            realized_pnl = (price - position["avg_price"]) * size
            self.portfolio["realized_pnl"] += realized_pnl
            new_size = position["size"] - size
            self.portfolio["positions"][symbol]["size"] = new_size
            self.portfolio["positions"][symbol]["unrealized_pnl"] = (
                price - position["avg_price"]
            ) * new_size
        # Обновление весов
        await self._update_weights()
        logger.info(f"Removed position: {symbol}, size: {size}, price: {price}")

    async def _update_weights(self) -> None:
        """Обновление весов портфеля."""
        total_value = 0.0
        # Расчет общей стоимости
        for symbol, position in self.portfolio["positions"].items():
            position_value = position["size"] * position["current_price"]
            total_value += position_value
        # Добавляем наличные
        total_value += self.portfolio["cash"]
        self.portfolio["total_value"] = total_value
        # Обновление весов
        if total_value > 0:
            for symbol, position in self.portfolio["positions"].items():
                position_value = position["size"] * position["current_price"]
                self.portfolio["weights"][symbol] = position_value / total_value
            # Вес наличных
            self.portfolio["weights"]["CASH"] = self.portfolio["cash"] / total_value
        else:
            self.portfolio["weights"] = {}

    async def _check_position_limits(self, symbol: str) -> None:
        """
        Проверка лимитов позиций.
        Args:
            symbol: Символ актива
        """
        weight = self.portfolio["weights"].get(symbol, 0)
        if weight > self.optimization_config["max_position_size"]:
            logger.warning(f"Position {symbol} exceeds max size limit: {weight:.3f}")
            # Отправка события
            await self.event_bus.publish(
                Event(
                    name=EventName("position_limit_exceeded"),
                    type=EventType.RISK_ALERT,
                    data={
                        "symbol": symbol,
                        "weight": weight,
                        "limit": self.optimization_config["max_position_size"],
                    },
                    priority=EventPriority.HIGH,
                )
            )

    async def get_portfolio_metrics(self) -> Dict[str, Any]:
        """
        Получение метрик портфеля.
        Returns:
            Dict[str, Any]: Словарь с метриками портфеля
        """
        # Расчет общей нереализованной P&L
        total_unrealized = sum(
            [pos["unrealized_pnl"] for pos in self.portfolio["positions"].values()]
        )
        self.portfolio["unrealized_pnl"] = total_unrealized
        # Расчет доходности
        total_pnl = self.portfolio["realized_pnl"] + total_unrealized
        total_return = (
            total_pnl / self.portfolio["total_value"]
            if self.portfolio["total_value"] > 0
            else 0
        )
        # Диверсификация
        diversification_score = self._calculate_diversification_score()
        # Риск-метрики
        risk_metrics = await self._calculate_risk_metrics()
        return {
            "total_value": self.portfolio["total_value"],
            "unrealized_pnl": total_unrealized,
            "realized_pnl": self.portfolio["realized_pnl"],
            "total_return": total_return,
            "diversification_score": diversification_score,
            "position_count": len(self.portfolio["positions"]),
            "weights": self.portfolio["weights"],
            "risk_metrics": risk_metrics,
            "cash": self.portfolio["cash"],
        }

    def _calculate_diversification_score(self) -> float:
        """
        Расчет показателя диверсификации портфеля.
        Returns:
            float: Показатель диверсификации (0-1)
        """
        try:
            if not self.portfolio["weights"]:
                return 0.0
            weights = np.array(list(self.portfolio["weights"].values()))
            # Исключаем наличные из расчета диверсификации
            if "CASH" in self.portfolio["weights"]:
                cash_weight = self.portfolio["weights"]["CASH"]
                weights = weights[:-1]  # Убираем наличные
                # Если весь портфель в наличных
                if cash_weight >= 0.99:
                    return 0.0
            # Нормализация весов
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                return 0.0
            # Индекс Херфиндаля-Хиршмана (HHI)
            hhi = np.sum(weights**2)
            # Преобразование в показатель диверсификации (1 - HHI)
            diversification = 1 - hhi
            return float(diversification)
        except Exception as e:
            logger.error(f"Error calculating diversification score: {e}")
            return 0.0

    async def _calculate_risk_metrics(self) -> Dict[str, float]:
        """
        Расчет метрик риска портфеля.
        Returns:
            Dict[str, float]: Словарь с метриками риска
        """
        try:
            if not self.returns_history:
                return {}
            
            # Подготовка данных
            portfolio_returns = pd.DataFrame(self.returns_history)
            weights = np.array(list(self.portfolio["weights"].values()))
            
            if len(weights) == 0:
                return {}
            # Нормализация весов
            weights = weights / weights.sum()
            # Взвешенные доходности - исправляем numpy.dot
            if hasattr(portfolio_returns, 'to_numpy'):
                portfolio_array = portfolio_returns.to_numpy()
            else:
                portfolio_array = np.asarray(portfolio_returns)
            weighted_returns = np.dot(portfolio_array, weights)
            
            # Метрики риска
            volatility = float(np.std(weighted_returns) * np.sqrt(252))
            var_95 = float(np.percentile(weighted_returns, 5))
            cvar_95 = float(np.mean(weighted_returns[weighted_returns <= var_95]))
            max_drawdown = self._calculate_max_drawdown(pd.Series(weighted_returns))
            
            # Исправляем skew, kurtosis через scipy
            from scipy import stats
            skewness = float(stats.skew(weighted_returns))
            kurtosis = float(stats.kurtosis(weighted_returns))
            
            return {
                "volatility": volatility,
                "var_95": var_95,
                "cvar_95": cvar_95,
                "max_drawdown": max_drawdown,
                "skewness": skewness,
                "kurtosis": kurtosis,
            }
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Расчет максимальной просадки.
        Args:
            returns: Временной ряд доходностей
        Returns:
            float: Максимальная просадка
        """
        try:
            if len(returns) < 2:
                return 0.0
            # Кумулятивные доходности - исправляем cumprod
            returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
            cumulative = np.cumprod(1 + returns_array)
            # Бегущий максимум
            running_max = np.maximum.accumulate(cumulative)
            # Просадка
            drawdown = (cumulative - running_max) / running_max
            # Максимальная просадка
            max_drawdown = np.min(drawdown)
            return float(max_drawdown)
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0

    async def get_correlations(self) -> Dict[str, float]:
        """
        Получение корреляций между активами.
        Returns:
            Dict[str, float]: Словарь с корреляциями
        """
        # Возвращаем только flat dict[str, float]
        return {k: float(v) for k, v in self.correlation_matrix.items()}

    async def update_correlations(self, price_data: Dict[str, pd.Series]) -> None:
        """
        Обновление корреляционной матрицы.
        Args:
            price_data: Словарь с ценовыми данными по активам
        """
        if len(price_data) < 2:
            return
        try:
            # Расчет доходностей
            returns_data: Dict[str, pd.Series] = {}
            for symbol, prices in price_data.items():
                returns_data[symbol] = prices.pct_change().dropna()
            # Создание DataFrame доходностей
            returns_df = pd.DataFrame(returns_data)
            # Расчет корреляционной матрицы - исправляем to_numpy
            if hasattr(returns_df, 'corr'):
                correlation_matrix = returns_df.corr()
                if hasattr(correlation_matrix, 'to_numpy'):
                    correlation_array = correlation_matrix.to_numpy()
                else:
                    correlation_array = np.asarray(correlation_matrix)
            else:
                if hasattr(returns_df, 'to_numpy'):
                    returns_array = returns_df.to_numpy()
                else:
                    returns_array = np.asarray(returns_df)
                correlation_array = np.corrcoef(returns_array, rowvar=False)
            
            # Сохранение корреляций
            symbols = list(returns_df.columns)
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i != j:
                        correlation_key = f"{symbol1}_{symbol2}"
                        self.correlation_matrix[correlation_key] = float(correlation_array[i, j])  # type: ignore[assignment]
            # Обновление исторических данных
            self.price_history.update(price_data)
            self.returns_history.update(returns_data)  # Исправление: используем update вместо присваивания
            logger.info(f"Updated correlations for {len(price_data)} assets")
        except Exception as e:
            logger.error(f"Error updating correlations: {e}")

    async def optimize_portfolio(
        self, method: Optional[str] = None
    ) -> PortfolioOptimizationResult:
        """
        Продвинутая оптимизация портфеля.
        Args:
            method: Метод оптимизации
        Returns:
            PortfolioOptimizationResult: Результат оптимизации
        """
        if not self.returns_history:
            logger.warning("No historical data for portfolio optimization")
            return PortfolioOptimizationResult(
                optimal_weights={},
                expected_return=0.0,
                expected_risk=0.0,
                sharpe_ratio=0.0,
                efficient_frontier=[],
                risk_metrics={},
                optimization_method="none",
                timestamp=asyncio.get_event_loop().time(),
            )

        try:
            # Выбор метода оптимизации
            if method == "markowitz":
                return await self._optimize_markowitz()
            elif method == "black_litterman":
                return await self._optimize_black_litterman()
            elif method == "risk_parity":
                return await self._optimize_risk_parity()
            else:
                # По умолчанию используем Markowitz
                return await self._optimize_markowitz()
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            return PortfolioOptimizationResult(
                optimal_weights={},
                expected_return=0.0,
                expected_risk=0.0,
                sharpe_ratio=0.0,
                efficient_frontier=[],
                risk_metrics={},
                optimization_method="error",
                timestamp=asyncio.get_event_loop().time(),
            )

    async def _optimize_markowitz(self) -> PortfolioOptimizationResult:
        """
        Оптимизация по методу Марковица.
        Returns:
            PortfolioOptimizationResult: Результат оптимизации
        """
        try:
            # Подготовка данных
            returns_df = pd.DataFrame(self.returns_history)
            
            # Исправляем mean и cov для DataFrame
            if hasattr(returns_df, 'mean'):
                expected_returns = returns_df.mean() * 252
            else:
                returns_array = returns_df.to_numpy() if hasattr(returns_df, 'to_numpy') else np.asarray(returns_df)
                expected_returns = pd.Series(np.mean(returns_array, axis=0) * 252, index=returns_df.columns)
            
            if hasattr(returns_df, 'cov'):
                cov_matrix = returns_df.cov() * 252
            else:
                returns_array = returns_df.to_numpy() if hasattr(returns_df, 'to_numpy') else np.asarray(returns_df)
                cov_matrix = pd.DataFrame(np.cov(returns_array, rowvar=False) * 252, 
                                        index=returns_df.columns, columns=returns_df.columns)

            # Функция для минимизации (отрицательный коэффициент Шарпа)
            def objective(weights):
                portfolio_return = np.sum(weights * expected_returns.to_numpy())
                if hasattr(cov_matrix, 'to_numpy'):
                    cov_array = cov_matrix.to_numpy()
                else:
                    cov_array = np.asarray(cov_matrix)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_array, weights)))
                if portfolio_vol == 0:
                    return 0
                return -(portfolio_return - 0.02) / portfolio_vol  # 2% безрисковая ставка

            # Ограничения
            n_assets = len(expected_returns)
            constraints = [
                {"type": "eq", "fun": lambda x: np.sum(x) - 1}  # Сумма весов = 1
            ]
            bounds = [(0, 0.3) for _ in range(n_assets)]  # Максимум 30% на актив

            # Начальные веса (равные)
            initial_weights = np.array([1/n_assets] * n_assets)

            # Оптимизация - исправляем вызов minimize
            from scipy.optimize import minimize
            result = minimize(
                objective,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000}
            )

            if result.success:
                optimal_weights = dict(zip(expected_returns.index, result.x))
                portfolio_return = np.sum(result.x * expected_returns.to_numpy())
                if hasattr(cov_matrix, 'to_numpy'):
                    cov_array = cov_matrix.to_numpy()
                else:
                    cov_array = np.asarray(cov_matrix)
                portfolio_vol = np.sqrt(np.dot(result.x.T, np.dot(cov_array, result.x)))
                sharpe_ratio = (portfolio_return - 0.02) / portfolio_vol if portfolio_vol > 0 else 0

                # Расчет эффективной границы
                efficient_frontier = await self._calculate_efficient_frontier(
                    expected_returns, cov_matrix
                )

                # Исправляем risk_metrics
                if hasattr(returns_df, 'to_numpy'):
                    returns_array = returns_df.to_numpy()
                else:
                    returns_array = np.asarray(returns_df)
                portfolio_returns_array = np.dot(returns_array, result.x)
                var_95 = np.percentile(portfolio_returns_array, 5)
                cvar_95 = np.mean(portfolio_returns_array[portfolio_returns_array <= var_95])

                return PortfolioOptimizationResult(
                    optimal_weights=optimal_weights,
                    expected_return=float(portfolio_return),
                    expected_risk=float(portfolio_vol),
                    sharpe_ratio=float(sharpe_ratio),
                    efficient_frontier=efficient_frontier,
                    risk_metrics={
                        "var_95": float(var_95),
                        "cvar_95": float(cvar_95),
                    },
                    optimization_method="markowitz",
                    timestamp=asyncio.get_event_loop().time(),
                )
            else:
                logger.warning("Portfolio optimization failed")
                return PortfolioOptimizationResult(
                    optimal_weights={},
                    expected_return=0.0,
                    expected_risk=0.0,
                    sharpe_ratio=0.0,
                    efficient_frontier=[],
                    risk_metrics={},
                    optimization_method="markowitz_failed",
                    timestamp=asyncio.get_event_loop().time(),
                )
        except Exception as e:
            logger.error(f"Error in Markowitz optimization: {e}")
            return PortfolioOptimizationResult(
                optimal_weights={},
                expected_return=0.0,
                expected_risk=0.0,
                sharpe_ratio=0.0,
                efficient_frontier=[],
                risk_metrics={},
                optimization_method="markowitz_error",
                timestamp=asyncio.get_event_loop().time(),
            )

    async def _optimize_black_litterman(self) -> PortfolioOptimizationResult:
        """
        Оптимизация по методу Black-Litterman.
        Returns:
            PortfolioOptimizationResult: Результат оптимизации
        """
        try:
            # Подготовка данных
            returns_df = pd.DataFrame(self.returns_history)
            market_caps = {symbol: 1.0 for symbol in returns_df.columns}  # Упрощенно
            market_weights = np.array(list(market_caps.values()))
            market_weights = market_weights / market_weights.sum()

            # Рыночные ожидаемые доходности (CAPM)
            risk_free_rate = 0.02
            market_return = 0.08  # Предполагаемая рыночная доходность
            betas = {}
            for symbol in returns_df.columns:
                if hasattr(returns_df, 'mean'):
                    market_returns = returns_df.mean(axis=1)
                else:
                    returns_array = returns_df.to_numpy() if hasattr(returns_df, 'to_numpy') else np.asarray(returns_df)
                market_returns = pd.Series(np.mean(returns_array, axis=1))
                asset_returns = returns_df[symbol]
                asset_array = asset_returns.to_numpy() if hasattr(asset_returns, 'to_numpy') else np.asarray(asset_returns)
                market_array = market_returns.to_numpy() if hasattr(market_returns, 'to_numpy') else np.asarray(market_returns)
                beta = np.cov(asset_array, market_array)[0, 1] / np.var(market_array)
                betas[symbol] = beta

            # Ожидаемые доходности Black-Litterman
            bl_returns = {}
            for symbol in returns_df.columns:
                bl_returns[symbol] = risk_free_rate + betas[symbol] * (market_return - risk_free_rate)

            # Ковариационная матрица
            if hasattr(returns_df, 'cov'):
                cov_matrix = returns_df.cov() * 252
            else:
                cov_matrix = pd.DataFrame(np.cov(returns_df.to_numpy(), rowvar=False) * 252,  # type: ignore[attr-defined]
                                        index=returns_df.columns, columns=returns_df.columns)

            # Функция для минимизации
            def objective(weights):
                portfolio_return = sum(weights[i] * list(bl_returns.values())[i] for i in range(len(weights)))
                if hasattr(cov_matrix, 'to_numpy'):
                    cov_array = cov_matrix.to_numpy()
                else:
                    cov_array = np.asarray(cov_matrix)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_array, weights)))
                if portfolio_vol == 0:
                    return 0
                return -(portfolio_return - risk_free_rate) / portfolio_vol

            # Ограничения
            n_assets = len(bl_returns)
            constraints = [
                {"type": "eq", "fun": lambda x: np.sum(x) - 1}
            ]
            bounds = [(0, 0.3) for _ in range(n_assets)]

            # Начальные веса
            initial_weights = np.array([1/n_assets] * n_assets)

            # Оптимизация - исправляем вызов minimize
            from scipy.optimize import minimize
            result = minimize(
                objective,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000}
            )

            if result.success:
                optimal_weights = dict(zip(bl_returns.keys(), result.x))
                portfolio_return = sum(result.x[i] * list(bl_returns.values())[i] for i in range(len(result.x)))
                if hasattr(cov_matrix, 'to_numpy'):
                    cov_array = cov_matrix.to_numpy()
                else:
                    cov_array = np.asarray(cov_matrix)
                portfolio_vol = np.sqrt(np.dot(result.x.T, np.dot(cov_array, result.x)))
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0

                return PortfolioOptimizationResult(
                    optimal_weights=optimal_weights,
                    expected_return=float(portfolio_return),
                    expected_risk=float(portfolio_vol),
                    sharpe_ratio=float(sharpe_ratio),
                    efficient_frontier=[],
                    risk_metrics={},
                    optimization_method="black_litterman",
                    timestamp=asyncio.get_event_loop().time(),
                )
            else:
                return PortfolioOptimizationResult(
                    optimal_weights={},
                    expected_return=0.0,
                    expected_risk=0.0,
                    sharpe_ratio=0.0,
                    efficient_frontier=[],
                    risk_metrics={},
                    optimization_method="black_litterman_failed",
                    timestamp=asyncio.get_event_loop().time(),
                )
        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {e}")
            return PortfolioOptimizationResult(
                optimal_weights={},
                expected_return=0.0,
                expected_risk=0.0,
                sharpe_ratio=0.0,
                efficient_frontier=[],
                risk_metrics={},
                optimization_method="black_litterman_error",
                timestamp=asyncio.get_event_loop().time(),
            )

    async def _optimize_risk_parity(self) -> PortfolioOptimizationResult:
        """
        Оптимизация по методу Risk Parity.
        Returns:
            PortfolioOptimizationResult: Результат оптимизации
        """
        try:
            # Подготовка данных
            returns_df = pd.DataFrame(self.returns_history)
            
            # Ковариационная матрица
            if hasattr(returns_df, 'cov'):
                cov_matrix = returns_df.cov() * 252
            else:
                if hasattr(returns_df, 'to_numpy'):
                    returns_array = returns_df.to_numpy()
                elif hasattr(returns_df, 'values'):
                    returns_array = returns_df.values
                else:
                    returns_array = np.array(returns_df)
                cov_matrix = pd.DataFrame(np.cov(returns_array, rowvar=False) * 252,
                                        index=returns_df.columns, columns=returns_df.columns)

            # Функция для минимизации (равное распределение риска)
            def objective(weights):
                if hasattr(cov_matrix, 'to_numpy'):
                    cov_array = cov_matrix.to_numpy()
                else:
                    cov_array = np.asarray(cov_matrix)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_array, weights)))
                if portfolio_vol == 0:
                    return 0
                
                # Риск каждого актива
                asset_risks = []
                for i in range(len(weights)):
                    asset_risk = weights[i] * np.sqrt(cov_array[i, i])
                    asset_risks.append(asset_risk)
                
                # Дисперсия рисков (должна быть минимальной)
                risk_variance = np.var(asset_risks)
                return risk_variance

            # Ограничения
            n_assets = len(returns_df.columns)
            constraints = [
                {"type": "eq", "fun": lambda x: np.sum(x) - 1}
            ]
            bounds = [(0, 0.5) for _ in range(n_assets)]

            # Начальные веса
            initial_weights = np.array([1/n_assets] * n_assets)

            # Оптимизация - исправляем вызов minimize
            from scipy.optimize import minimize
            result = minimize(
                objective,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000}
            )

            if result.success:
                optimal_weights = dict(zip(returns_df.columns, result.x))
                
                # Расчет ожидаемой доходности
                if hasattr(returns_df, 'mean'):
                    expected_returns = returns_df.mean() * 252
                else:
                    expected_returns = pd.Series(np.mean(returns_df.values, axis=0) * 252, index=returns_df.columns)  # type: ignore[attr-defined]
                
                portfolio_return = np.sum(result.x * expected_returns.to_numpy())
                if hasattr(cov_matrix, 'to_numpy'):
                    cov_array = cov_matrix.to_numpy()
                else:
                    cov_array = np.asarray(cov_matrix)
                portfolio_vol = np.sqrt(np.dot(result.x.T, np.dot(cov_array, result.x)))
                sharpe_ratio = (portfolio_return - 0.02) / portfolio_vol if portfolio_vol > 0 else 0

                return PortfolioOptimizationResult(
                    optimal_weights=optimal_weights,
                    expected_return=float(portfolio_return),
                    expected_risk=float(portfolio_vol),
                    sharpe_ratio=float(sharpe_ratio),
                    efficient_frontier=[],
                    risk_metrics={},
                    optimization_method="risk_parity",
                    timestamp=asyncio.get_event_loop().time(),
                )
            else:
                return PortfolioOptimizationResult(
                    optimal_weights={},
                    expected_return=0.0,
                    expected_risk=0.0,
                    sharpe_ratio=0.0,
                    efficient_frontier=[],
                    risk_metrics={},
                    optimization_method="risk_parity_failed",
                    timestamp=asyncio.get_event_loop().time(),
                )
        except Exception as e:
            logger.error(f"Error in Risk Parity optimization: {e}")
            return PortfolioOptimizationResult(
                optimal_weights={},
                expected_return=0.0,
                expected_risk=0.0,
                sharpe_ratio=0.0,
                efficient_frontier=[],
                risk_metrics={},
                optimization_method="risk_parity_error",
                timestamp=asyncio.get_event_loop().time(),
            )

    async def _calculate_efficient_frontier(
        self, expected_returns: pd.Series, cov_matrix: pd.DataFrame
    ) -> List[Tuple[float, float]]:
        """
        Расчет эффективной границы.
        Args:
            expected_returns: Ожидаемые доходности
            cov_matrix: Ковариационная матрица
        Returns:
            List[Tuple[float, float]]: Точки эффективной границы (риск, доходность)
        """
        try:
            efficient_frontier = []
            n_assets = len(expected_returns)
            
            # Генерируем различные уровни риска
            for target_vol in np.linspace(0.05, 0.30, 20):
                def objective(weights):
                    if hasattr(cov_matrix, 'to_numpy'):
                        cov_array = cov_matrix.to_numpy()
                    else:
                        cov_array = np.asarray(cov_matrix)
                    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_array, weights)))
                    return -np.sum(weights * expected_returns.to_numpy())  # Минимизируем отрицательную доходность

                def constraint_vol(weights):
                    if hasattr(cov_matrix, 'to_numpy'):
                        cov_array = cov_matrix.to_numpy()
                    else:
                        cov_array = np.asarray(cov_matrix)
                    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_array, weights)))
                    return portfolio_vol - target_vol

                constraints = [
                    {"type": "eq", "fun": lambda x: np.sum(x) - 1},
                    {"type": "eq", "fun": constraint_vol}
                ]
                bounds = [(0, 0.3) for _ in range(n_assets)]
                initial_weights = np.array([1/n_assets] * n_assets)

                # Оптимизация - исправляем вызов minimize
                from scipy.optimize import minimize
                result = minimize(
                    objective,
                    initial_weights,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": 1000}
                )

                if result.success:
                    portfolio_return = np.sum(result.x * expected_returns.to_numpy())
                    if hasattr(cov_matrix, 'to_numpy'):
                        cov_array = cov_matrix.to_numpy()
                    else:
                        cov_array = np.asarray(cov_matrix)
                    portfolio_vol = np.sqrt(np.dot(result.x.T, np.dot(cov_array, result.x)))
                    efficient_frontier.append((float(portfolio_vol), float(portfolio_return)))

            return efficient_frontier
        except Exception as e:
            logger.error(f"Error calculating efficient frontier: {e}")
            return []

    async def rebalance_portfolio(
        self, target_weights: Dict[str, float]
    ) -> List[RebalancingAction]:
        """
        Ребалансировка портфеля.
        Args:
            target_weights: Целевые веса активов
        Returns:
            List[RebalancingAction]: Список действий для ребалансировки
        """
        try:
            actions = []
            current_weights = self.portfolio["weights"].copy()

            for symbol, target_weight in target_weights.items():
                current_weight = current_weights.get(symbol, 0.0)
                weight_diff = target_weight - current_weight

                if abs(weight_diff) > 0.01:  # Минимальный порог для ребалансировки
                    # Расчет размера позиции
                    portfolio_value = self.portfolio["total_value"]
                    size_change = weight_diff * portfolio_value

                    # Определение действия
                    if size_change > 0:
                        action = "buy"
                        priority = "high" if abs(weight_diff) > 0.1 else "medium"
                    else:
                        action = "sell"
                        priority = "high" if abs(weight_diff) > 0.1 else "medium"

                    # Оценка стоимости
                    estimated_cost = abs(size_change) * 0.001  # 0.1% комиссия

                    rebalancing_action = RebalancingAction(
                        symbol=symbol,
                        current_weight=current_weight,
                        target_weight=target_weight,
                        action=action,
                        size_change=abs(size_change),
                        estimated_cost=estimated_cost,
                        priority=priority,
                    )
                    actions.append(rebalancing_action)

            # Сортировка по приоритету
            priority_order = {"high": 0, "medium": 1, "low": 2}
            actions.sort(key=lambda x: priority_order.get(x.priority, 3))

            return actions
        except Exception as e:
            logger.error(f"Error in portfolio rebalancing: {e}")
            return []

    async def _execute_rebalancing_trade(self, action: RebalancingAction) -> bool:
        """
        Выполнение торговой операции для ребалансировки.
        Args:
            action: Действие ребалансировки
        Returns:
            bool: Успешность выполнения
        """
        try:
            # Здесь должна быть интеграция с торговой системой
            logger.info(
                f"Executing rebalancing trade: {action.action} {action.symbol} "
                f"size: {action.size_change:.2f}"
            )

            # Симуляция выполнения
            if action.action == "buy":
                await self.add_position(action.symbol, action.size_change, 100.0)  # Упрощенно
            else:
                await self.remove_position(action.symbol, action.size_change, 100.0)

            return True
        except Exception as e:
            logger.error(f"Error executing rebalancing trade: {e}")
            return False

    async def adjust_position_size(self, symbol: str, action: str) -> None:
        """
        Корректировка размера позиции.
        Args:
            symbol: Символ актива
            action: Действие ('increase' или 'decrease')
        """
        try:
            current_position = self.portfolio["positions"].get(symbol, {})
            current_size = current_position.get("size", 0)

            if action == "increase":
                new_size = current_size * 1.1  # Увеличиваем на 10%
                await self.add_position(symbol, new_size - current_size, 100.0)
            elif action == "decrease":
                new_size = current_size * 0.9  # Уменьшаем на 10%
                await self.remove_position(symbol, current_size - new_size, 100.0)

            logger.info(f"Adjusted position size for {symbol}: {action}")
        except Exception as e:
            logger.error(f"Error adjusting position size: {e}")

    async def _rebalance_position(self, symbol: str, target_weight: float) -> None:
        """
        Ребалансировка конкретной позиции.
        Args:
            symbol: Символ актива
            target_weight: Целевой вес
        """
        try:
            current_weight = self.portfolio["weights"].get(symbol, 0)
            weight_diff = target_weight - current_weight

            if abs(weight_diff) > 0.01:
                portfolio_value = self.portfolio["total_value"]
                size_change = weight_diff * portfolio_value

                if size_change > 0:
                    await self.add_position(symbol, size_change, 100.0)
                else:
                    await self.remove_position(symbol, abs(size_change), 100.0)

                logger.info(f"Rebalanced position {symbol}: {current_weight:.3f} -> {target_weight:.3f}")
        except Exception as e:
            logger.error(f"Error rebalancing position {symbol}: {e}")

    async def get_portfolio_status(self) -> Dict[str, Any]:
        """
        Получение статуса портфеля.
        Returns:
            Dict[str, Any]: Статус портфеля
        """
        try:
            metrics = await self.get_portfolio_metrics()
            
            status = {
                "portfolio_value": self.portfolio["total_value"],
                "cash": self.portfolio["cash"],
                "total_pnl": metrics["unrealized_pnl"] + metrics["realized_pnl"],
                "unrealized_pnl": metrics["unrealized_pnl"],
                "realized_pnl": metrics["realized_pnl"],
                "total_return": metrics["total_return"],
                "diversification_score": metrics["diversification_score"],
                "position_count": metrics["position_count"],
                "risk_metrics": metrics["risk_metrics"],
                "positions": self.portfolio["positions"],
                "weights": self.portfolio["weights"],
                "last_update": time.time(),
            }
            
            return status
        except Exception as e:
            logger.error(f"Error getting portfolio status: {e}")
            return {}
