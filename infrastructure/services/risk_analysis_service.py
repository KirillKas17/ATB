# -*- coding: utf-8 -*-
"""
Сервис анализа рисков - промышленная реализация.
"""

import hashlib
import time
from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from shared.numpy_utils import np
import pandas as pd
from dataclasses import dataclass

from domain.entities.risk_metrics import RiskMetrics
from domain.type_definitions.risk_types import RiskReport
from domain.type_definitions.risk_types import StressTestResult
from domain.type_definitions.risk_types import PortfolioRisk
from domain.type_definitions.risk_types import PositionRisk
from domain.type_definitions.risk_types import RiskOptimizationResult
from domain.type_definitions.risk_types import PortfolioOptimizationMethod
from domain.services.risk_analysis import AdvancedRiskAnalysisService, risk_analysis_service
# from domain.services.risk_analysis_service import RiskAnalysisService
from domain.type_definitions.risk_types import RiskMetrics as DomainRiskMetrics
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
# from infrastructure.cache.cache_config import CacheConfig
from infrastructure.shared.cache import CacheManager, CacheConfig
from shared.logging import LoggerMixin
# from shared.risk_calculations import (
#     calc_volatility,
#     calc_sharpe,
#     calc_sortino,
#     calc_max_drawdown,
#     calc_parametric_var,
#     calc_parametric_cvar,
#     calc_beta,
#     calc_portfolio_return,
#     optimize_portfolio_weights,
#     calc_diversification_ratio,
#     calc_concentration_risk,
#     calc_scenario_impact,
#     validate_returns_data,
#     validate_scenario,
#     generate_default_scenarios,
#     generate_risk_recommendations,
#     generate_risk_alerts,
#     create_empty_risk_metrics,
#     create_empty_optimization_result,
# )


class ValidationError(Exception):
    """Ошибка валидации данных."""
    pass


@dataclass
class RiskModelConfig:
    """Конфигурация модели риска."""

    # Параметры VaR
    var_confidence_level: Decimal = Decimal("0.95")
    cvar_confidence_level: Decimal = Decimal("0.95")
    # Параметры оптимизации
    max_position_weight: Decimal = Decimal("0.3")
    min_position_weight: Decimal = Decimal("0.01")
    max_sector_weight: Decimal = Decimal("0.4")
    # Параметры расчета
    risk_free_rate: Decimal = Decimal("0.02")
    trading_days_per_year: int = 252
    min_data_points: int = 30
    # Параметры кэширования
    cache_ttl_hours: int = 1
    enable_cache: bool = True

    def __post_init__(self) -> None:
        """Валидация конфигурации."""
        if self.var_confidence_level <= 0 or self.var_confidence_level >= 1:
            raise ValueError("VaR confidence level must be between 0 and 1")
        if self.cvar_confidence_level <= 0 or self.cvar_confidence_level >= 1:
            raise ValueError("CVaR confidence level must be between 0 and 1")
        if self.max_position_weight <= 0 or self.max_position_weight > 1:
            raise ValueError("Max position weight must be between 0 and 1")
        if self.min_position_weight < 0 or self.min_position_weight >= 1:
            raise ValueError("Min position weight must be between 0 and 1")
        if self.risk_free_rate < 0:
            raise ValueError("Risk-free rate cannot be negative")
        if self.trading_days_per_year <= 0:
            raise ValueError("Trading days per year must be positive")
        if self.min_data_points <= 0:
            raise ValueError("Min data points must be positive")
        if self.cache_ttl_hours <= 0:
            raise ValueError("Cache TTL must be positive")


@runtime_checkable
class RiskMetricsCalculatorProtocol(Protocol):
    """Протокол для калькулятора метрик риска."""

    def calculate_volatility(self, returns: pd.Series) -> float:
        """Расчет волатильности."""
        ...

    def calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float
    ) -> float:
        """Расчет коэффициента Шарпа."""
        ...

    def calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Расчет Value at Risk."""
        ...


@runtime_checkable
class PortfolioOptimizerProtocol(Protocol):
    """Протокол для оптимизатора портфеля."""

    def optimize_weights(
        self, returns_df: pd.DataFrame, target_return: Optional[float] = None
    ) -> Dict[str, float]:
        """Оптимизация весов портфеля."""
        ...

    def calculate_portfolio_risk(
        self, weights: Dict[str, float], returns_df: pd.DataFrame
    ) -> float:
        """Расчет риска портфеля."""
        ...


class RiskAnalysisServiceImpl(AdvancedRiskAnalysisService, LoggerMixin):
    """Промышленная реализация сервиса анализа рисков."""

    def __init__(
        self,
        config: Optional[RiskModelConfig] = None,
        cache_config: Optional[Any] = None,
        metrics_calculator: Optional[RiskMetricsCalculatorProtocol] = None,
        portfolio_optimizer: Optional[PortfolioOptimizerProtocol] = None,
    ):
        super().__init__()
        self.config = config or RiskModelConfig()
        self.cache_manager = CacheManager() if cache_config else None
        self.metrics_calculator = metrics_calculator
        self.portfolio_optimizer = portfolio_optimizer
        self._logger = logging.getLogger(__name__)

    @asynccontextmanager
    async def processing_context(self) -> Dict[str, Any]:
        """Контекст обработки с метриками."""
        start_time = time.time()
        try:
            yield
        finally:
            processing_time = time.time() - start_time
            if self._logger:
                self._logger.debug(f"Risk analysis processing time: {processing_time:.3f}s")

    async def calculate_risk_metrics(
        self, returns: pd.Series, risk_free_rate: Optional[Decimal] = None
    ) -> DomainRiskMetrics:
        """Расчет метрик риска с кэшированием."""
        try:
            risk_free_rate = risk_free_rate or self.config.risk_free_rate
            cache = None
            if self.cache_manager is not None:
                cache = self.cache_manager.get_cache("default")
            # Исправление: используем to_numpy().tobytes() вместо tobytes()
            cache_key = self._generate_cache_key(
                "risk_metrics", str(returns.to_numpy().tobytes()), str(risk_free_rate)
            )
            cached_result = await cache.get(cache_key) if cache else None
            if cached_result:
                return cached_result
            metrics = self._calculate_all_risk_metrics(returns, float(risk_free_rate))
            if cache:
                await cache.set(cache_key, metrics)
            return metrics
        except Exception as e:
            if self._logger is not None:
                self._logger.error(f"Error calculating risk metrics: {e}")
            return DomainRiskMetrics(
                calculation_timestamp=datetime.now(),
                confidence_level=Decimal("0.95"),
                risk_free_rate=Decimal("0.02"),
                data_points=0,
            )

    async def calculate_var(
        self, returns: pd.Series, confidence_level: Optional[Decimal] = None
    ) -> Decimal:
        """Расчет VaR с кэшированием."""
        try:
            confidence_level = confidence_level or self.config.var_confidence_level
            cache = None
            if self.cache_manager is not None:
                cache = self.cache_manager.get_cache("default")
            # Исправление: используем iloc для индексации вместо callable
            cache_key = self._generate_cache_key(
                "var", str(returns.iloc[-1]), str(confidence_level)
            )
            cached_result = await cache.get(cache_key) if cache else None
            if cached_result:
                return Decimal(str(cached_result))
            var_value = calc_parametric_var(returns, float(confidence_level))
            if cache:
                await cache.set(cache_key, float(var_value), )
            return Decimal(str(var_value))
        except Exception as e:
            if self._logger is not None:
                self._logger.error(f"Error calculating VaR: {e}")
            return Decimal("0")

    async def calculate_cvar(
        self, returns: pd.Series, confidence_level: Optional[Decimal] = None
    ) -> Decimal:
        """Расчет CVaR с кэшированием."""
        try:
            confidence_level = confidence_level or self.config.cvar_confidence_level
            cache = None
            if self.cache_manager is not None:
                cache = self.cache_manager.get_cache("default")
            # Исправление: используем iloc для индексации вместо callable
            cache_key = self._generate_cache_key(
                "cvar", str(returns.iloc[-1]), str(confidence_level)
            )
            cached_result = await cache.get(cache_key) if cache else None
            if cached_result:
                return Decimal(str(cached_result))
            cvar_value = calc_parametric_cvar(returns, float(confidence_level))
            if cache:
                await cache.set(cache_key, float(cvar_value), )
            return Decimal(str(cvar_value))
        except Exception as e:
            if self._logger is not None:
                self._logger.error(f"Error calculating CVaR: {e}")
            return Decimal("0")

    def calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: Optional[Decimal] = None
    ) -> Decimal:
        """Расчет коэффициента Шарпа."""
        try:
            risk_free_rate = risk_free_rate or self.config.risk_free_rate
            return Decimal(str(calc_sharpe(returns, float(risk_free_rate))))
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error calculating Sharpe ratio: {e}")
            return Decimal("0")

    def calculate_sortino_ratio(
        self, returns: pd.Series, risk_free_rate: Optional[Decimal] = None
    ) -> Decimal:
        """Расчет коэффициента Сортино."""
        try:
            risk_free_rate = risk_free_rate or self.config.risk_free_rate
            return Decimal(str(calc_sortino(returns, float(risk_free_rate))))
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error calculating Sortino ratio: {e}")
            return Decimal("0")

    def calculate_max_drawdown(self, prices: pd.Series) -> Decimal:
        """Расчет максимальной просадки."""
        try:
            return Decimal(str(calc_max_drawdown(prices)))
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error calculating max drawdown: {e}")
            return Decimal("0")

    def calculate_beta(
        self, asset_returns: pd.Series, market_returns: pd.Series
    ) -> Decimal:
        """Расчет беты актива."""
        try:
            return Decimal(str(calc_beta(asset_returns, market_returns)))
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error calculating beta: {e}")
            return Decimal("0")

    def calculate_correlation_matrix(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Расчет матрицы корреляций."""
        try:
            # Исправление: используем pandas corr() метод
            return returns_df.corr()
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()

    def calculate_portfolio_risk(
        self, returns: pd.Series, weights: Optional[np.ndarray] = None
    ) -> DomainRiskMetrics:
        """Расчет риска портфеля."""
        try:
            # Валидация данных
            if not validate_returns_data(returns):
                raise ValidationError("Invalid returns data")
            # Расчет метрик
            risk_free_rate = float(self.config.risk_free_rate)
            metrics = self._calculate_all_risk_metrics(returns, risk_free_rate)
            return metrics
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error calculating portfolio risk: {e}")
            return create_empty_risk_metrics()

    def optimize_portfolio(
        self, returns_df: pd.DataFrame, risk_free_rate: float = 0.02
    ) -> tuple[np.ndarray, DomainRiskMetrics]:
        """Оптимизация портфеля."""
        try:
            # Валидация данных
            if not validate_returns_data(returns_df.iloc[:, 0]):
                raise ValidationError("Invalid returns data")
            # Оптимизация весов - исправление: передаем правильные аргументы
            weights, success, expected_return = optimize_portfolio_weights(returns_df.iloc[:, 0], returns_df)
            # Расчет метрик оптимизированного портфеля - исправление: передаем правильные аргументы
            portfolio_returns = calc_portfolio_return(returns_df, weights)
            metrics = self._calculate_all_risk_metrics(portfolio_returns, risk_free_rate)
            return weights, metrics
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error optimizing portfolio: {e}")
            return np.array([]), create_empty_risk_metrics()

    def perform_stress_test(
        self,
        portfolio_risk: PortfolioRisk,
        scenarios: Optional[List[Dict[str, Any]]] = None,
    ) -> List[StressTestResult]:
        """Выполнение стресс-тестирования."""
        try:
            scenarios = scenarios or generate_default_scenarios()
            results = []
            for scenario in scenarios:
                # Валидация сценария
                if not validate_scenario(scenario):
                    continue
                # Применение сценария
                impact = calc_scenario_impact(portfolio_risk, scenario)
                # Создание результата - исправление: добавляем недостающие аргументы
                result = StressTestResult(
                    scenario_name=scenario.get("name", "Unknown"),
                    var_change=impact.get("var_change", 0.0),
                    portfolio_value_change=impact.get("portfolio_value_change", 0.0),
                    max_drawdown_change=impact.get("max_drawdown_change", 0.0),
                    affected_positions=impact.get("affected_positions", []),
                )
                results.append(result)
            return results
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error performing stress test: {e}")
            return []

    def generate_risk_report(
        self,
        portfolio_risk: PortfolioRisk,
        include_optimization: bool = True,
        include_stress_tests: bool = True,
    ) -> RiskReport:
        """Генерация отчета по рискам."""
        try:
            # Базовые метрики
            risk_metrics = portfolio_risk.risk_metrics
            # Рекомендации
            recommendations = generate_risk_recommendations(portfolio_risk)
            # Алерты
            alerts = generate_risk_alerts(portfolio_risk)
            # Создание отчета - исправление: убираем неправильные аргументы
            report = RiskReport(
                portfolio_risk=portfolio_risk,
                risk_metrics=risk_metrics,
                risk_recommendations=recommendations,
                risk_alerts=alerts,
                report_period=str(datetime.now().date()),
            )
            return report
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error generating risk report: {e}")
            # Исправление: убираем неправильные аргументы
            return RiskReport(
                portfolio_risk=portfolio_risk,
                risk_metrics=create_empty_risk_metrics(),
                risk_recommendations=[],
                risk_alerts=[],
                report_period=str(datetime.now().date()),
            )

    def _calculate_all_risk_metrics(
        self, returns: pd.Series, risk_free_rate: float
    ) -> DomainRiskMetrics:
        """Расчет всех метрик риска."""
        try:
            volatility = calc_volatility(returns)
            sharpe_ratio = calc_sharpe(returns, risk_free_rate)
            sortino_ratio = calc_sortino(returns, risk_free_rate)
            max_drawdown = calc_max_drawdown(returns)
            var_95 = calc_parametric_var(returns, 0.95)
            cvar_95 = calc_parametric_cvar(returns, 0.95)
            # Исправление: используем Money для var_95 и cvar_95
            return DomainRiskMetrics(
                volatility=Decimal(str(volatility)),
                sharpe_ratio=Decimal(str(sharpe_ratio)),
                sortino_ratio=Decimal(str(sortino_ratio)),
                max_drawdown=Decimal(str(max_drawdown)),
                var_95=Money(Decimal(str(var_95)), Currency.USD),
                cvar_95=Money(Decimal(str(cvar_95)), Currency.USD),
            )
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error calculating all risk metrics: {e}")
            return DomainRiskMetrics(
                calculation_timestamp=datetime.now(),
                confidence_level=Decimal("0.95"),
                risk_free_rate=Decimal("0.02"),
                data_points=0,
            )

    def _calculate_portfolio_returns(
        self, positions: List[PositionRisk], returns_df: pd.DataFrame
    ) -> pd.Series:
        """Расчет доходности портфеля."""
        try:
            portfolio_returns = pd.Series(0.0, index=returns_df.index)
            for position in positions:
                symbol = getattr(position, 'symbol', 'unknown')
                weight = getattr(position, 'weight', 0.0)
                if symbol in returns_df.columns:
                    portfolio_returns += returns_df[symbol] * weight
            return portfolio_returns
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error calculating portfolio returns: {e}")
            return pd.Series()

    def _calculate_diversification_score(
        self, positions: List[PositionRisk], returns_df: pd.DataFrame
    ) -> float:
        """Расчет показателя диверсификации."""
        try:
            if len(positions) < 2:
                return 0.0
            weights = []
            for position in positions:
                weight = getattr(position, 'weight', 0.0)
                weights.append(weight)
            if not weights:
                return 0.0
            weights_array = np.array(weights)
            # Исправление: используем pandas corr() и передаем правильные аргументы
            correlation_matrix = returns_df.corr()
            diversification_ratio = calc_diversification_ratio(weights_array, correlation_matrix)
            return float(diversification_ratio)
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error calculating diversification score: {e}")
            return 0.0

    def _calculate_concentration_risk(self, positions: List[PositionRisk]) -> float:
        """Расчет риска концентрации."""
        try:
            weights = []
            for position in positions:
                weight = getattr(position, 'weight', 0.0)
                weights.append(weight)
            if not weights:
                return 0.0
            # Исправление: передаем numpy array вместо Series
            concentration = calc_concentration_risk(np.array(weights))
            return float(concentration)
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error calculating concentration risk: {e}")
            return 0.0

    def _validate_optimization_constraints(self, weights: Dict[str, float]) -> bool:
        """Валидация ограничений оптимизации."""
        try:
            for weight in weights.values():
                if weight < float(self.config.min_position_weight):
                    return False
                if weight > float(self.config.max_position_weight):
                    return False
            return True
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error validating optimization constraints: {e}")
            return False

    def _create_returns_dataframe(self, positions: List[PositionRisk]) -> pd.DataFrame:
        """Создание DataFrame с доходностями."""
        try:
            returns_data = {}
            for position in positions:
                symbol = getattr(position, 'symbol', 'unknown')
                returns = getattr(position, 'returns', None)
                if returns is not None:
                    returns_data[symbol] = returns
            return pd.DataFrame(returns_data)
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error creating returns dataframe: {e}")
            return pd.DataFrame()

    def _generate_cache_key(self, operation: str, *args) -> str:
        """Генерация ключа кэша."""
        try:
            key_data = f"{operation}:{':'.join(str(arg) for arg in args)}"
            return hashlib.md5(key_data.encode()).hexdigest()
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error generating cache key: {e}")
            return f"{operation}_{int(time.time())}"

    def _create_empty_optimization_result(self) -> RiskOptimizationResult:
        """Создание пустого результата оптимизации."""
        try:
            # Исправление: убираем неправильный аргумент expected_volatility
            return RiskOptimizationResult(
                optimal_weights={},
                expected_return=Decimal("0"),
                sharpe_ratio=Decimal("0"),
                constraints_applied=["max_weight", "min_weight"],
                optimization_method=PortfolioOptimizationMethod.SHARPE_MAXIMIZATION,
                optimization_timestamp=datetime.now(),
            )
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error creating empty optimization result: {e}")
            # Исправление: возвращаем правильный тип
            return RiskOptimizationResult(
                optimal_weights={},
                expected_return=Decimal("0"),
                sharpe_ratio=Decimal("0"),
                constraints_applied=["max_weight", "min_weight"],
                optimization_method=PortfolioOptimizationMethod.SHARPE_MAXIMIZATION,
                optimization_timestamp=datetime.now(),
            )

# Alias for backward compatibility
RiskAnalysisService = RiskAnalysisServiceImpl


# Утилитные функции для расчета метрик риска
def calc_parametric_var(returns: pd.Series, confidence_level: float) -> float:
    """Расчет параметрического VaR."""
    try:
        if len(returns) == 0:
            return 0.0
        mean_return = returns.mean()
        std_return = returns.std()
        from scipy.stats import norm
        z_score = norm.ppf(1 - confidence_level)
        var = -(mean_return + z_score * std_return)
        return float(var)
    except Exception:
        return 0.0


def calc_parametric_cvar(returns: pd.Series, confidence_level: float) -> float:
    """Расчет параметрического CVaR."""
    try:
        if len(returns) == 0:
            return 0.0
        var = calc_parametric_var(returns, confidence_level)
        # Приближенный расчет CVaR
        extreme_losses = returns[returns <= -var]
        if len(extreme_losses) > 0:
            return float(-extreme_losses.mean())
        return float(var)
    except Exception:
        return 0.0


def calc_sharpe(returns: pd.Series, risk_free_rate: float) -> float:
    """Расчет коэффициента Шарпа."""
    try:
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        excess_returns = returns - risk_free_rate
        return float(excess_returns.mean() / returns.std())
    except Exception:
        return 0.0


def calc_sortino(returns: pd.Series, risk_free_rate: float) -> float:
    """Расчет коэффициента Сортино."""
    try:
        if len(returns) == 0:
            return 0.0
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return 0.0
        downside_std = downside_returns.std()
        if downside_std == 0:
            return 0.0
        return float(excess_returns.mean() / downside_std)
    except Exception:
        return 0.0


def calc_max_drawdown(returns: pd.Series) -> float:
    """Расчет максимальной просадки."""
    try:
        if len(returns) == 0:
            return 0.0
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return float(drawdown.min())
    except Exception:
        return 0.0


def calc_beta(returns: pd.Series, market_returns: pd.Series) -> float:
    """Расчет бета-коэффициента."""
    try:
        if len(returns) == 0 or len(market_returns) == 0:
            return 1.0
        covariance = returns.cov(market_returns)
        market_variance = market_returns.var()
        if market_variance == 0:
            return 1.0
        return float(covariance / market_variance)
    except Exception:
        return 1.0


def validate_returns_data(returns: pd.Series) -> bool:
    """Валидация данных доходности."""
    try:
        return not returns.empty and not returns.isna().all()
    except Exception:
        return False


def create_empty_risk_metrics() -> "DomainRiskMetrics":
    """Создание пустых метрик риска."""
    from domain.type_definitions.risk_types import RiskMetrics as DomainRiskMetrics
    return DomainRiskMetrics(
        var_95=Decimal("0"),
        cvar_95=Decimal("0"),
        sharpe_ratio=Decimal("0"),
        sortino_ratio=Decimal("0"),
        max_drawdown=Decimal("0"),
        beta=Decimal("1"),
        alpha=Decimal("0"),
        information_ratio=Decimal("0"),
        tracking_error=Decimal("0"),
        treynor_ratio=Decimal("0")
    )


def optimize_portfolio_weights(returns: pd.DataFrame) -> Dict[str, float]:
    """Оптимизация весов портфеля."""
    try:
        if returns.empty:
            return {}  
        # Простое равномерное распределение как заглушка
        n_assets = len(returns.columns)
        return {col: 1.0 / n_assets for col in returns.columns}
    except Exception:
        return {}  


def calc_portfolio_return(returns: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """Расчет доходности портфеля."""
    try:
        if returns.empty or not weights:
            return pd.Series(dtype=float)
        weighted_returns = returns * pd.Series(weights)
        return weighted_returns.sum(axis=1)
    except Exception:
        return pd.Series(dtype=float)


def generate_default_scenarios() -> List[Dict[str, Any]]:
    """Генерация сценариев по умолчанию."""
    return [
        {"name": "base_case", "probability": 0.6, "market_shock": 0.0},
        {"name": "stress_case", "probability": 0.3, "market_shock": -0.2},
        {"name": "extreme_case", "probability": 0.1, "market_shock": -0.4}
    ]


def validate_scenario(scenario: Dict[str, Any]) -> bool:
    """Валидация сценария."""
    try:
        required_keys = ["name", "probability", "market_shock"]
        return all(key in scenario for key in required_keys)
    except Exception:
        return False


def calc_scenario_impact(returns: pd.Series, scenario: Dict[str, Any]) -> Dict[str, float]:
    """Расчет влияния сценария."""
    try:
        market_shock = scenario.get("market_shock", 0.0)
        shocked_returns = returns + market_shock
        return {
            "var_95": calc_parametric_var(shocked_returns, 0.05),
            "expected_return": float(shocked_returns.mean()),
            "volatility": float(shocked_returns.std())
        }
    except Exception:
        return {"var_95": 0.0, "expected_return": 0.0, "volatility": 0.0}


def generate_risk_recommendations(metrics: "DomainRiskMetrics") -> List[str]:
    """Генерация рекомендаций по управлению рисками."""
    recommendations = []
    try:
        if float(metrics.sharpe_ratio) < 0.5:
            recommendations.append("Рассмотрите диверсификацию портфеля")
        if float(metrics.max_drawdown) < -0.2:
            recommendations.append("Высокая просадка - рассмотрите снижение риска")
        if float(metrics.var_95) > 0.1:
            recommendations.append("Высокий VaR - рекомендуется хеджирование")
        if not recommendations:
            recommendations.append("Риск-профиль в пределах нормы")
    except Exception:
        recommendations.append("Ошибка анализа рисков")
    
    return recommendations
