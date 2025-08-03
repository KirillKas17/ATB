"""
Сервис анализа рисков.
Предоставляет методы для расчета различных метрик риска портфеля.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from scipy.optimize import minimize

from domain.entities.risk_metrics import RiskMetrics
from domain.exceptions.base_exceptions import RiskAnalysisError
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.value_objects.price import Price


class RiskMetricType(Enum):
    """Типы метрик риска."""

    VAR = "value_at_risk"
    CVAR = "conditional_value_at_risk"
    SHARPE = "sharpe_ratio"
    SORTINO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    CORRELATION = "correlation"


@dataclass
class PositionRisk:
    """Риск отдельной позиции."""

    position_id: str
    symbol: str
    var_95: Money
    cvar_95: Money
    beta: Decimal
    volatility: Decimal
    correlation_with_portfolio: Decimal
    marginal_contribution: Decimal
    component_var: Decimal
    stress_test_loss: Money
    liquidity_score: Decimal
    concentration_impact: Decimal


@dataclass
class RiskLimits:
    """Лимиты риска."""

    max_portfolio_var: Money
    max_position_var: Money
    max_drawdown_limit: Decimal
    max_concentration: Decimal
    min_sharpe_ratio: Decimal
    max_correlation: Decimal
    max_leverage: Decimal
    min_liquidity_score: Decimal


@runtime_checkable
class RiskAnalysisProtocol(Protocol):
    """Протокол для анализа рисков."""

    def calculate_portfolio_risk(
        self, returns: pd.Series, weights: Optional[np.ndarray] = None
    ) -> RiskMetrics:
        """Рассчитать риск портфеля."""
        ...

    def calculate_position_risk(
        self, position_returns: pd.Series, portfolio_returns: pd.Series
    ) -> PositionRisk:
        """Рассчитать риск позиции."""
        ...

    def validate_risk_limits(
        self, metrics: RiskMetrics, limits: RiskLimits
    ) -> Tuple[bool, List[str]]:
        """Проверить лимиты риска."""
        ...

    def stress_test(
        self, portfolio_data: pd.DataFrame, scenarios: Dict[str, Dict[str, float]]
    ) -> Dict[str, RiskMetrics]:
        """Стресс-тестирование."""
        ...

    def optimize_portfolio(
        self, returns: pd.DataFrame, risk_free_rate: float = 0.02
    ) -> Tuple[np.ndarray, RiskMetrics]:
        """Оптимизация портфеля."""
        ...


class RiskAnalysisService(RiskAnalysisProtocol):
    """Базовый сервис анализа рисков."""

    def __init__(self, risk_free_rate: float = 0.02, confidence_level: float = 0.95):
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level

    def calculate_portfolio_risk(
        self, returns: pd.Series, weights: Optional[np.ndarray] = None
    ) -> RiskMetrics:
        """Рассчитать риск портфеля."""
        raise NotImplementedError("Subclasses must implement calculate_portfolio_risk")

    def calculate_position_risk(
        self, position_returns: pd.Series, portfolio_returns: pd.Series
    ) -> PositionRisk:
        """Рассчитать риск позиции."""
        raise NotImplementedError("Subclasses must implement calculate_position_risk")

    def validate_risk_limits(
        self, metrics: RiskMetrics, limits: RiskLimits
    ) -> Tuple[bool, List[str]]:
        """Проверить лимиты риска."""
        raise NotImplementedError("Subclasses must implement validate_risk_limits")

    def stress_test(
        self, portfolio_data: pd.DataFrame, scenarios: Dict[str, Dict[str, float]]
    ) -> Dict[str, RiskMetrics]:
        """Стресс-тестирование."""
        raise NotImplementedError("Subclasses must implement stress_test")

    def optimize_portfolio(
        self, returns: pd.DataFrame, risk_free_rate: float = 0.02
    ) -> Tuple[np.ndarray, RiskMetrics]:
        """Оптимизация портфеля."""
        raise NotImplementedError("Subclasses must implement optimize_portfolio")


class DefaultRiskAnalysisService(RiskAnalysisProtocol):
    """Реализация по умолчанию сервиса анализа рисков."""

    def __init__(self, risk_free_rate: float = 0.02, confidence_level: float = 0.95):
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level

    def calculate_portfolio_risk(
        self, returns: pd.Series, weights: Optional[np.ndarray] = None
    ) -> RiskMetrics:
        """
        Рассчитать комплексные метрики риска портфеля.
        
        Args:
            returns: Временной ряд доходностей портфеля
            weights: Веса активов (если None, предполагается равновзвешенный портфель)
        Returns:
            RiskMetrics: Комплексные метрики риска
        """
        if returns.empty:
            return RiskMetrics()
        # Базовые метрики
        volatility = self._calculate_volatility(returns)
        # Расчет VaR и CVaR
        returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
        returns_array = returns_array.astype(float)
        var_95 = np.percentile(returns_array, 5)  # 5-й процентиль для 95% VaR
        var_99 = np.percentile(returns_array, 1)  # 1-й процентиль для 99% VaR
        
        # Расчет CVaR
        returns_filtered_95 = returns_array[returns_array <= var_95]
        returns_filtered_99 = returns_array[returns_array <= var_99]
        cvar_95 = np.mean(returns_filtered_95) if len(returns_filtered_95) > 0 else 0
        cvar_99 = np.mean(returns_filtered_99) if len(returns_filtered_99) > 0 else 0
        
        # Расчет максимальной просадки
        cumulative_returns = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown: float = np.min(drawdown)
        # Метрики доходности
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio(returns, Decimal(str(max_drawdown)))
        # Бета и альфа (если есть бенчмарк)
        beta = self._calculate_beta(returns)
        alpha = self._calculate_alpha(returns, beta)
        # Информационная эффективность
        information_ratio = self._calculate_information_ratio(returns)
        treynor_ratio = self._calculate_treynor_ratio(returns, beta)
        jensen_alpha = self._calculate_jensen_alpha(returns, beta)
        # Корреляции и концентрация
        avg_correlation = self._calculate_avg_correlation(returns)
        concentration_risk = (
            self._calculate_concentration_risk(weights)
            if weights is not None
            else Decimal("0")
        )
        liquidity_risk = self._calculate_liquidity_risk(returns)
        tail_risk = self._calculate_tail_risk(returns)
        # Стресс-тестирование
        stress_test_results = self._run_stress_tests(returns)
        # Декомпозиция риска
        risk_decomposition = self._decompose_risk(returns, weights)
        # Расчет ожидаемой доходности
        expected_return = Decimal(str(np.mean(returns_array)))
        
        # Создаем RiskMetrics с правильными типами
        risk_metrics = RiskMetrics(
            var_95=Money(Decimal(str(var_95)), Currency.USD),
            var_99=Money(Decimal(str(var_99)), Currency.USD),
            volatility=Decimal(str(volatility)),
            sharpe_ratio=Decimal(str(sharpe_ratio)),
            max_drawdown=Decimal(str(max_drawdown)),
            beta=Decimal(str(beta)),
            alpha=Decimal(str(alpha)),
            correlation=Decimal(str(avg_correlation)),
            metadata={
                "alpha": str(alpha),
                "information_ratio": str(information_ratio),
                "treynor_ratio": str(treynor_ratio),
                "jensen_alpha": str(jensen_alpha),
                "tail_risk": str(tail_risk),
                "stress_test_results": str(stress_test_results),
                "risk_decomposition": str(risk_decomposition),
                "expected_return": str(expected_return),
            }
        )
        return risk_metrics

    def calculate_position_risk(
        self, position_returns: pd.Series, portfolio_returns: pd.Series
    ) -> PositionRisk:
        """Рассчитать риск отдельной позиции."""
        if position_returns.empty or portfolio_returns.empty:
            return PositionRisk(
                position_id="",
                symbol="",
                var_95=Money(Decimal("0"), Currency.USD),
                cvar_95=Money(Decimal("0"), Currency.USD),
                beta=Decimal("0"),
                volatility=Decimal("0"),
                correlation_with_portfolio=Decimal("0"),
                marginal_contribution=Decimal("0"),
                component_var=Decimal("0"),
                stress_test_loss=Money(Decimal("0"), Currency.USD),
                liquidity_score=Decimal("0"),
                concentration_impact=Decimal("0"),
            )
        # Базовые метрики позиции
        position_var = self._calculate_var(position_returns, self.confidence_level)
        position_cvar = self._calculate_cvar(position_returns, self.confidence_level)
        position_volatility = self._calculate_volatility(position_returns)
        # Корреляция с портфелем
        correlation = self._calculate_correlation(position_returns, portfolio_returns)
        # Бета позиции
        beta = self._calculate_beta(position_returns, portfolio_returns)
        # Маржинальный вклад в риск
        marginal_contribution = self._calculate_marginal_contribution(
            position_returns, portfolio_returns
        )
        # Компонентный VaR
        component_var = self._calculate_component_var(
            position_returns, portfolio_returns
        )
        # Стресс-тест позиции
        stress_loss = self._calculate_position_stress_loss(position_returns)
        # Ликвидность
        liquidity_score = self._calculate_liquidity_score(position_returns)
        # Влияние на концентрацию
        concentration_impact = self._calculate_concentration_impact(
            position_returns, portfolio_returns
        )
        return PositionRisk(
            position_id="",
            symbol="",
            var_95=Money(position_var, Currency.USD),
            cvar_95=Money(position_cvar, Currency.USD),
            beta=beta,
            volatility=position_volatility,
            correlation_with_portfolio=correlation,
            marginal_contribution=marginal_contribution,
            component_var=component_var,
            stress_test_loss=Money(stress_loss, Currency.USD),
            liquidity_score=liquidity_score,
            concentration_impact=concentration_impact,
        )

    def validate_risk_limits(
        self, metrics: RiskMetrics, limits: RiskLimits
    ) -> Tuple[bool, List[str]]:
        """Проверить соблюдение лимитов риска."""
        violations = []
        # Проверка VaR
        if metrics.var_95 > limits.max_portfolio_var:
            violations.append(
                f"Portfolio VaR {metrics.var_95} exceeds limit {limits.max_portfolio_var}"
            )
        # Проверка максимальной просадки
        if metrics.max_drawdown > limits.max_drawdown_limit:
            violations.append(
                f"Max drawdown {metrics.max_drawdown} exceeds limit {limits.max_drawdown_limit}"
            )
        # Проверка концентрации
        if metrics.correlation > limits.max_concentration:
            violations.append(
                f"Correlation {metrics.correlation} exceeds limit {limits.max_concentration}"
            )
        # Проверка коэффициента Шарпа
        if metrics.sharpe_ratio < limits.min_sharpe_ratio:
            violations.append(
                f"Sharpe ratio {metrics.sharpe_ratio} below limit {limits.min_sharpe_ratio}"
            )
        # Проверка корреляции
        if metrics.correlation > limits.max_correlation:
            violations.append(
                f"Correlation {metrics.correlation} exceeds limit {limits.max_correlation}"
            )
        # Проверка ликвидности
        if metrics.correlation < limits.min_liquidity_score:
            violations.append(
                f"Liquidity ratio {metrics.correlation} below limit {limits.min_liquidity_score}"
            )
        return len(violations) == 0, violations

    def stress_test(
        self, portfolio_data: pd.DataFrame, scenarios: Dict[str, Dict[str, float]]
    ) -> Dict[str, RiskMetrics]:
        """Стресс-тестирование портфеля."""
        results: Dict[str, RiskMetrics] = {}
        for scenario_name, scenario_params in scenarios.items():
            # Применяем стресс-сценарий
            stressed_data = self._apply_stress_scenario(portfolio_data, scenario_params)
            # Рассчитываем доходности
            stressed_returns = stressed_data.pct_change().dropna()
            # Если stressed_returns - DataFrame, берем первую колонку или среднее
            if isinstance(stressed_returns, pd.DataFrame):
                squeezed = stressed_returns.squeeze()
                if isinstance(squeezed, pd.Series):
                    stressed_returns_series = squeezed
                else:
                    stressed_returns_series = pd.Series([squeezed])
            else:
                stressed_returns_series = pd.Series(stressed_returns) if not isinstance(stressed_returns, pd.Series) else stressed_returns
            stressed_metrics = self.calculate_portfolio_risk(stressed_returns_series)  # type: ignore[unreachable]
            results[scenario_name] = stressed_metrics
        return results

    def optimize_portfolio(
        self, returns: pd.DataFrame, risk_free_rate: float = 0.02
    ) -> Tuple[np.ndarray, RiskMetrics]:
        """
        Оптимизировать портфель по методу Марковица.
        """
        if returns.empty:
            return np.array([]), RiskMetrics()
        # Рассчитываем ковариационную матрицу
        returns_values = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
        cov_matrix = np.cov(returns_values, rowvar=False)
        
        # Рассчитываем средние доходности
        mean_returns = np.mean(returns_values, axis=0)

        # Функция для максимизации коэффициента Шарпа
        def negative_sharpe_ratio(weights: np.ndarray) -> float:
            portfolio_return: float = np.sum(mean_returns * weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if portfolio_vol == 0:
                return 0.0
            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
            return float(-sharpe)

        # Ограничения
        n_assets = len(returns.columns)
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        bounds = [(0.0, 1.0) for _ in range(n_assets)]
        initial_weights = np.array([1 / n_assets] * n_assets)
        result = minimize(
            fun=negative_sharpe_ratio,
            x0=initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"disp": False}
        )
        if result.success:
            optimal_weights = result.x
            portfolio_returns = returns_values.dot(optimal_weights)
            optimal_metrics = self.calculate_portfolio_risk(
                pd.Series(portfolio_returns), optimal_weights
            )
            return optimal_weights, optimal_metrics
        else:
            raise RiskAnalysisError(f"Portfolio optimization failed: {result.message}")

    def _calculate_volatility(self, returns: pd.Series) -> Decimal:
        """Рассчитать волатильность."""
        returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
        std_value = np.std(returns_array)
        return Decimal(str(std_value * np.sqrt(252)))  # Годовая волатильность

    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> Decimal:
        """Рассчитать Value at Risk."""
        returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
        returns_array = returns_array.astype(float)
        return Decimal(str(np.percentile(returns_array, (1 - confidence_level) * 100)))

    def _calculate_cvar(self, returns: pd.Series, confidence_level: float) -> Decimal:
        """Рассчитать Conditional Value at Risk."""
        returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
        returns_array = returns_array.astype(float)
        var = np.percentile(returns_array, (1 - confidence_level) * 100)
        returns_filtered = returns_array[returns_array <= var]
        cvar_value = np.mean(returns_filtered) if len(returns_filtered) > 0 else 0
        return Decimal(str(cvar_value))

    def _calculate_max_drawdown(self, returns: pd.Series) -> Decimal:
        """Рассчитать максимальную просадку."""
        returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
        cumulative = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return Decimal(str(np.min(drawdown)))

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> Decimal:
        """Рассчитать коэффициент Шарпа."""
        returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
        excess_returns = returns_array - self.risk_free_rate / 252
        std = np.std(returns_array)
        mean_return = np.mean(excess_returns)
        sharpe_ratio = float(mean_return / std * np.sqrt(252)) if std > 0 else 0.0
        return Decimal(str(sharpe_ratio))

    def _calculate_sortino_ratio(self, returns: pd.Series) -> Decimal:
        """Рассчитать коэффициент Сортино."""
        returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
        excess_returns = returns_array - self.risk_free_rate / 252
        downside_returns = returns_array[returns_array < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        mean_return = np.mean(excess_returns)
        return Decimal(
            str(mean_return / downside_std * np.sqrt(252))
            if downside_std > 0 else "0"
        )

    def _calculate_calmar_ratio(
        self, returns: pd.Series, max_drawdown: Decimal
    ) -> Decimal:
        """Рассчитать коэффициент Кальмара."""
        returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
        annual_return = np.mean(returns_array) * 252
        max_dd = float(max_drawdown)
        if max_dd == 0:
            return Decimal("0")
        return Decimal(str(annual_return / abs(max_dd)))

    def _calculate_beta(
        self, returns: pd.Series, market_returns: Optional[pd.Series] = None
    ) -> Decimal:
        """Рассчитать бета."""
        returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
        if market_returns is None:
            market_mean = np.mean(returns_array)
            market_returns_array = np.full_like(returns_array, market_mean)
        else:
            market_returns_array = market_returns.to_numpy() if hasattr(market_returns, 'to_numpy') else np.asarray(market_returns)
        
        covariance = np.cov(returns_array, market_returns_array)[0, 1]
        market_variance = np.var(market_returns_array)
        return Decimal(
            str(covariance / market_variance) if market_variance > 0 else "0"
        )

    def _calculate_alpha(self, returns: pd.Series, beta: Decimal) -> Decimal:
        """Рассчитать альфа."""
        returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
        market_return = np.mean(returns_array)
        actual_return = np.mean(returns_array)
        risk_free_return = self.risk_free_rate / 252
        beta_float = float(beta)
        expected_return = risk_free_return + beta_float * (
            market_return - risk_free_return
        )
        return Decimal(str(actual_return - expected_return))

    def _calculate_information_ratio(self, returns: pd.Series) -> Decimal:
        """Рассчитать информационный коэффициент."""
        returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
        benchmark_return = np.mean(returns_array)
        active_return = returns_array - benchmark_return
        tracking_error = np.std(active_return)
        active_mean = np.mean(active_return)
        return Decimal(
            str(active_mean / tracking_error) if tracking_error > 0 else "0"
        )

    def _calculate_treynor_ratio(self, returns: pd.Series, beta: Decimal) -> Decimal:
        """Рассчитать коэффициент Трейнора."""
        returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
        excess_return = np.mean(returns_array) - self.risk_free_rate / 252
        beta_float = float(beta)
        return Decimal(str(excess_return / beta_float) if beta_float != 0 else "0")

    def _calculate_jensen_alpha(self, returns: pd.Series, beta: Decimal) -> Decimal:
        """Рассчитать альфа Дженсена."""
        return self._calculate_alpha(returns, beta)

    def _calculate_correlation(
        self, returns1: pd.Series, returns2: pd.Series
    ) -> Decimal:
        """Рассчитать корреляцию."""
        returns1_array = returns1.to_numpy() if hasattr(returns1, 'to_numpy') else np.asarray(returns1)
        returns2_array = returns2.to_numpy() if hasattr(returns2, 'to_numpy') else np.asarray(returns2)
        corr = np.corrcoef(returns1_array, returns2_array)[0, 1]
        return Decimal(str(corr))

    def _calculate_avg_correlation(self, returns: pd.Series) -> Decimal:
        """Рассчитать среднюю корреляцию."""
        # Упрощенно - возвращаем 0 для одномерного ряда
        return Decimal("0")

    def _calculate_concentration_risk(self, weights: Optional[np.ndarray]) -> Decimal:
        """Рассчитать риск концентрации."""
        if weights is None:
            return Decimal("0")
        # Индекс Херфиндаля-Хиршмана
        hhi: float = np.sum(weights**2)
        return Decimal(str(hhi))

    def _calculate_liquidity_risk(self, returns: pd.Series) -> Decimal:
        """Рассчитать риск ликвидности."""
        # Упрощенно - используем волатильность как прокси
        return self._calculate_volatility(returns)

    def _calculate_tail_risk(self, returns: pd.Series) -> Decimal:
        """Рассчитать риск хвостового распределения."""
        returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
        # VaR на уровне 1%
        var_01 = np.percentile(returns_array, 1)
        # CVaR на уровне 1%
        cvar_01 = np.mean(returns_array[returns_array <= var_01])
        return Decimal(str(abs(cvar_01)))

    def _calculate_marginal_contribution(
        self, position_returns: pd.Series, portfolio_returns: pd.Series
    ) -> Decimal:
        """Рассчитать маржинальный вклад в риск."""
        # Упрощенно - используем корреляцию
        return self._calculate_correlation(position_returns, portfolio_returns)

    def _calculate_component_var(
        self, position_returns: pd.Series, portfolio_returns: pd.Series
    ) -> Decimal:
        """Рассчитать компонентный VaR."""
        # Упрощенно - используем корреляцию и VaR позиции
        correlation = self._calculate_correlation(position_returns, portfolio_returns)
        position_var = self._calculate_var(position_returns, self.confidence_level)
        correlation_float = float(correlation)
        position_var_float = float(position_var)
        return Decimal(str(correlation_float * position_var_float))

    def _calculate_position_stress_loss(self, position_returns: pd.Series) -> Decimal:
        """Рассчитать потери позиции при стрессе."""
        # Упрощенно - используем 99-й перцентиль
        return self._calculate_var(position_returns, 0.99)

    def _calculate_liquidity_score(self, returns: pd.Series) -> Decimal:
        """Рассчитать оценку ликвидности."""
        # Упрощенно - используем обратную волатильность
        volatility = self._calculate_volatility(returns)
        volatility_float = float(volatility)
        return Decimal(str(1 / volatility_float) if volatility_float > 0 else "0")

    def _calculate_concentration_impact(
        self, position_returns: pd.Series, portfolio_returns: pd.Series
    ) -> Decimal:
        """Рассчитать влияние на концентрацию."""
        # Упрощенно - используем долю в портфеле
        return Decimal("0.1")  # Предполагаем 10%

    def _run_stress_tests(self, returns: pd.Series) -> Dict[str, Decimal]:
        """Выполнить базовые стресс-тесты."""
        returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
        scenarios = {
            "market_crash": -0.2,
            "high_volatility": 2.0,
            "correlation_breakdown": 0.5,
        }
        results = {}
        for scenario, multiplier in scenarios.items():
            if scenario == "market_crash":
                stressed_returns = returns_array * multiplier
            elif scenario == "high_volatility":
                stressed_returns = returns_array * multiplier
            else:
                stressed_returns = returns_array
            var = self._calculate_var(pd.Series(stressed_returns), self.confidence_level)
            results[scenario] = var
        return results

    def _decompose_risk(
        self, returns: pd.Series, weights: Optional[np.ndarray]
    ) -> Dict[str, Decimal]:
        """Декомпозировать риск на факторы."""
        # Упрощенно - возвращаем базовые метрики
        return {
            "volatility_risk": self._calculate_volatility(returns),
            "tail_risk": self._calculate_tail_risk(returns),
            "liquidity_risk": self._calculate_liquidity_risk(returns),
        }

    def _apply_stress_scenario(
        self, data: pd.DataFrame, scenario_params: Dict[str, float]
    ) -> pd.DataFrame:
        """Применить стрессовый сценарий к данным."""
        # Упрощенно - умножаем данные на множители
        stressed_data = data.copy()
        for param, multiplier in scenario_params.items():
            if param in stressed_data.columns:
                stressed_data[param] = stressed_data[param] * multiplier
        return stressed_data


class AdvancedRiskAnalysisService(DefaultRiskAnalysisService):
    """Продвинутый сервис анализа рисков с ML-компонентами."""

    def __init__(self, risk_free_rate: float = 0.02, confidence_level: float = 0.95):
        super().__init__(risk_free_rate, confidence_level)
        self._ml_models: Dict[str, Any] = {}
        self._regime_detector = None

    def detect_regime_change(self, returns: pd.Series) -> Dict[str, Any]:
        """Обнаружить изменение рыночного режима."""
        # Реализация обнаружения смены режима
        volatility = self._calculate_volatility(returns)
        returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
        mean_return = np.mean(returns_array)
        # Простая логика определения режима
        if volatility > 0.3:
            regime = "high_volatility"
        elif mean_return > 0.001:
            regime = "bull_market"
        elif mean_return < -0.001:
            regime = "bear_market"
        else:
            regime = "sideways"
        return {
            "regime": regime,
            "confidence": 0.8,
            "volatility": volatility,
            "mean_return": mean_return,
        }

    def forecast_risk(self, returns: pd.Series, horizon: int = 30) -> RiskMetrics:
        """Спрогнозировать риск на будущий период."""
        # Упрощенная реализация прогнозирования риска
        current_metrics = self.calculate_portfolio_risk(returns)
        # Простая экстраполяция
        forecasted_volatility = current_metrics.volatility * Decimal(
            "1.1"
        )  # +10%
        forecasted_var = Money(
            current_metrics.var_95.amount * Decimal("1.15"), Currency.USD
        )  # +15%
        return RiskMetrics(
            var_95=forecasted_var,
            var_99=current_metrics.var_99,
            volatility=forecasted_volatility,
            sharpe_ratio=current_metrics.sharpe_ratio,
            sortino_ratio=current_metrics.sortino_ratio,
            max_drawdown=current_metrics.max_drawdown,
            beta=current_metrics.beta,
            alpha=current_metrics.alpha,
            skewness=current_metrics.skewness,
            kurtosis=current_metrics.kurtosis,
            correlation=current_metrics.correlation,
            metadata=current_metrics.metadata,
        )


# Фабрика для создания сервисов анализа рисков
def create_risk_analysis_service(
    service_type: str = "default", **kwargs: Any
) -> RiskAnalysisProtocol:
    """Создать сервис анализа рисков."""
    if service_type == "advanced":
        return AdvancedRiskAnalysisService(**kwargs)
    else:
        return DefaultRiskAnalysisService(**kwargs)
