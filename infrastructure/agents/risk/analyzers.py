"""
Калькуляторы рисков для risk agent.
"""

from shared.numpy_utils import np, ArrayLike
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from .types import RiskConfig, RiskLevel, RiskLimits, RiskMetrics




class IRiskCalculator(ABC):
    """Интерфейс калькулятора рисков."""

    @abstractmethod
    def calculate_portfolio_risk(self, portfolio_data: Dict[str, Any]) -> RiskMetrics:
        """Расчет риска портфеля."""
        pass

    @abstractmethod
    def calculate_position_risk(self, position_data: Dict[str, Any]) -> RiskMetrics:
        """Расчет риска позиции."""
        pass

    @abstractmethod
    def calculate_var(
        self, returns: ArrayLike, confidence_level: float = 0.95
    ) -> float:
        """Расчет Value at Risk."""
        pass

    @abstractmethod
    def calculate_max_drawdown(self, equity_curve: ArrayLike) -> float:
        """Расчет максимальной просадки."""
        pass


class DefaultRiskCalculator(IRiskCalculator):
    """Реализация калькулятора рисков по умолчанию."""

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()

    def calculate_portfolio_risk(self, portfolio_data: Dict[str, Any]) -> RiskMetrics:
        """Расчет риска портфеля."""
        try:
            # Извлекаем данные
            positions = portfolio_data.get("positions", [])
            market_data = portfolio_data.get("market_data", {})
            # Расчет метрик риска
            total_value = sum(pos.get("value", 0) for pos in positions)
            total_pnl = sum(pos.get("unrealized_pnl", 0) for pos in positions)
            # Расчет волатильности
            returns = []
            for pos in positions:
                if "returns" in pos:
                    returns.extend(pos["returns"])
            volatility = np.std(returns) if returns else 0.0  # type: ignore[attr-defined]
            # Расчет VaR
            var_95 = self.calculate_var(np.array(returns), 0.95) if returns else 0.0
            # Расчет максимальной просадки
            equity_curve = portfolio_data.get("equity_curve", [])
            max_dd = (
                self.calculate_max_drawdown(np.array(equity_curve))
                if equity_curve
                else 0.0
            )
            # Определение уровня риска
            risk_level = self._determine_risk_level(volatility, var_95, max_dd)
            return RiskMetrics(
                value=var_95,
                details={
                    "volatility": volatility,
                    "var_95": var_95,
                    "max_drawdown": max_dd,
                    "total_value": total_value,
                    "total_pnl": total_pnl,
                    "risk_level": risk_level,
                    "position_count": len(positions),
                },
            )
        except Exception as e:
            return RiskMetrics(value=0.0, details={"error": str(e)})

    def calculate_position_risk(self, position_data: Dict[str, Any]) -> RiskMetrics:
        """Расчет риска позиции."""
        try:
            # Извлекаем данные позиции
            size = position_data.get("size", 0)
            entry_price = position_data.get("entry_price", 0)
            current_price = position_data.get("current_price", 0)
            volatility = position_data.get("volatility", 0)
            # Расчет P&L
            pnl = (current_price - entry_price) * size
            # Расчет риска позиции
            position_risk = abs(size) * volatility * np.sqrt(1 / 252)  # Дневной риск
            # Определение уровня риска
            risk_level = self._determine_position_risk_level(position_risk, pnl)
            return RiskMetrics(
                value=position_risk,
                details={
                    "pnl": pnl,
                    "size": size,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "volatility": volatility,
                    "risk_level": risk_level,
                },
            )
        except Exception as e:
            return RiskMetrics(value=0.0, details={"error": str(e)})

    def calculate_var(
        self, returns: ArrayLike, confidence_level: float = 0.95
    ) -> float:
        """Расчет Value at Risk."""
        try:
            if len(returns) == 0:
                return 0.0
            # Используем исторический VaR
            var_percentile = (1 - confidence_level) * 100
            var_value = np.percentile(returns, var_percentile)
            return float(abs(var_value))
        except Exception:
            return 0.0

    def calculate_max_drawdown(self, equity_curve: ArrayLike) -> float:
        """Расчет максимальной просадки."""
        try:
            if len(equity_curve) == 0:
                return 0.0
            # Расчет кумулятивных максимумов
            peak = np.maximum.accumulate(equity_curve)
            # Расчет просадки
            drawdown = (equity_curve - peak) / peak
            # Максимальная просадка
            min_drawdown = np.min(drawdown)
            max_dd = abs(float(min_drawdown))
            return max_dd
        except Exception:
            return 0.0

    def _determine_risk_level(
        self, volatility: float, var: float, max_dd: float
    ) -> RiskLevel:
        """Определение уровня риска портфеля."""
        # Простая логика определения уровня риска
        if volatility > 0.05 or var > 0.1 or max_dd > 0.15:
            return RiskLevel.HIGH
        elif volatility > 0.02 or var > 0.05 or max_dd > 0.08:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _determine_position_risk_level(
        self, position_risk: float, pnl: float
    ) -> RiskLevel:
        """Определение уровня риска позиции."""
        # Логика для позиций
        if position_risk > 0.1 or pnl < -0.05:
            return RiskLevel.HIGH
        elif position_risk > 0.05 or pnl < -0.02:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW


class RiskMetricsCalculator:
    """Расширенный калькулятор метрик риска."""

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.base_calculator = DefaultRiskCalculator(config)

    def calculate_sharpe_ratio(
        self, returns: ArrayLike, risk_free_rate: float = 0.02
    ) -> float:
        """Расчет коэффициента Шарпа."""
        try:
            if len(returns) == 0:
                return 0.0
            excess_returns = (
                returns - risk_free_rate / 252
            )  # Дневная безрисковая ставка
            excess_mean = float(np.mean(excess_returns))
            excess_std = float(np.std(excess_returns))
            sharpe = (
                excess_mean / excess_std
                if excess_std > 0
                else 0.0
            )
            return float(sharpe * float(np.sqrt(252)))  # Годовой коэффициент Шарпа
        except Exception:
            return 0.0

    def calculate_sortino_ratio(
        self, returns: ArrayLike, risk_free_rate: float = 0.02
    ) -> float:
        """Расчет коэффициента Сортино."""
        try:
            if len(returns) == 0:
                return 0.0
            excess_returns = returns - risk_free_rate / 252
            downside_returns = excess_returns[excess_returns < 0]
            if len(downside_returns) == 0:
                return 0.0
            downside_deviation = float(np.std(downside_returns))
            excess_mean = float(np.mean(excess_returns))
            sortino = (
                excess_mean / downside_deviation
                if downside_deviation > 0
                else 0.0
            )
            return float(sortino * float(np.sqrt(252)))
        except Exception:
            return 0.0

    def calculate_calmar_ratio(self, returns: ArrayLike, max_dd: float) -> float:
        """Расчет коэффициента Кальмара."""
        try:
            if len(returns) == 0 or max_dd == 0:
                return 0.0
            annual_return = float(np.mean(returns)) * 252
            calmar = annual_return / max_dd
            return float(calmar)
        except Exception:
            return 0.0

    def calculate_beta(
        self, portfolio_returns: ArrayLike, market_returns: ArrayLike
    ) -> float:
        """Расчет беты портфеля."""
        try:
            if len(portfolio_returns) == 0 or len(market_returns) == 0:
                return 1.0
            # Убеждаемся, что массивы одинаковой длины
            min_length = min(len(portfolio_returns), len(market_returns))
            portfolio_returns = portfolio_returns[:min_length]
            market_returns = market_returns[:min_length]
            # Расчет ковариации и дисперсии
            covariance = np.cov(portfolio_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            beta = covariance / market_variance if market_variance > 0 else 1.0
            return float(beta)
        except Exception:
            return 1.0

    def calculate_correlation_matrix(
        self, returns_data: Dict[str, ArrayLike]
    ) -> Dict[str, Dict[str, float]]:
        """Расчет корреляционной матрицы."""
        try:
            symbols = list(returns_data.keys())
            correlation_matrix: Dict[str, Dict[str, float]] = {}
            for i, symbol1 in enumerate(symbols):
                correlation_matrix[symbol1] = {}
                for j, symbol2 in enumerate(symbols):
                    if i == j:
                        correlation_matrix[symbol1][symbol2] = 1.0
                    else:
                        returns1 = returns_data[symbol1]
                        returns2 = returns_data[symbol2]
                        # Убеждаемся, что массивы одинаковой длины
                        min_length = min(len(returns1), len(returns2))
                        if min_length > 0:
                            corr = np.corrcoef(
                                returns1[:min_length], returns2[:min_length]
                            )[0, 1]
                            correlation_matrix[symbol1][symbol2] = (
                                corr if not np.isnan(corr) else 0.0
                            )
                        else:
                            correlation_matrix[symbol1][symbol2] = 0.0
            return correlation_matrix
        except Exception:
            return {}

    def calculate_portfolio_risk_metrics(
        self, portfolio_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Расчет полного набора метрик риска портфеля."""
        try:
            # Базовые метрики
            base_metrics = self.base_calculator.calculate_portfolio_risk(portfolio_data)
            # Дополнительные метрики
            returns = portfolio_data.get("returns", [])
            market_returns = portfolio_data.get("market_returns", [])
            metrics = {
                "base_metrics": base_metrics.details,
                "sharpe_ratio": self.calculate_sharpe_ratio(np.array(returns)),
                "sortino_ratio": self.calculate_sortino_ratio(np.array(returns)),
                "calmar_ratio": self.calculate_calmar_ratio(
                    np.array(returns), base_metrics.details.get("max_drawdown", 0) if base_metrics.details else 0
                ),
                "beta": self.calculate_beta(
                    np.array(returns), np.array(market_returns)
                ),
                "volatility": np.std(returns) if returns else 0.0,
                "annualized_return": np.mean(returns) * 252 if returns else 0.0,
                "win_rate": self._calculate_win_rate(returns),
                "profit_factor": self._calculate_profit_factor(returns),
                "max_consecutive_losses": self._calculate_max_consecutive_losses(
                    returns
                ),
            }
            return metrics
        except Exception as e:
            return {"error": str(e)}

    def _calculate_win_rate(self, returns: List[float]) -> float:
        """Расчет процента прибыльных сделок."""
        try:
            if not returns:
                return 0.0
            winning_trades = sum(1 for r in returns if r > 0)
            return winning_trades / len(returns)
        except Exception:
            return 0.0

    def _calculate_profit_factor(self, returns: List[float]) -> float:
        """Расчет фактора прибыли."""
        try:
            if not returns:
                return 0.0
            gross_profit = sum(r for r in returns if r > 0)
            gross_loss = abs(sum(r for r in returns if r < 0))
            return gross_profit / gross_loss if gross_loss > 0 else 0.0
        except Exception:
            return 0.0

    def _calculate_max_consecutive_losses(self, returns: List[float]) -> int:
        """Расчет максимального количества последовательных убытков."""
        try:
            if not returns:
                return 0
            max_consecutive = 0
            current_consecutive = 0
            for r in returns:
                if r < 0:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
            return max_consecutive
        except Exception:
            return 0
