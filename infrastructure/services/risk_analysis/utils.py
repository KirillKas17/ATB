"""
Утилиты для риск-анализа.
Содержит вспомогательные функции для валидации, преобразований,
кэширования и общих утилит риск-анализа.
"""

import pandas as pd
from shared.numpy_utils import np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

from domain.type_definitions.risk_types import PortfolioRisk, PositionRisk, RiskMetrics
from domain.value_objects import Currency, Money

__all__ = [
    "validate_returns_data",
    "validate_market_data",
    "extract_returns_from_market_data",
    "create_empty_risk_metrics",
    "create_empty_portfolio_risk",
    "create_empty_optimization_result",
    "convert_to_decimal",
    "convert_to_money",
    "clean_cache",
    "validate_portfolio_data",
]


def validate_returns_data(returns: pd.Series, min_points: int = 30) -> bool:
    """Валидация данных доходностей."""
    if returns is None or returns.empty:
        return False
    if len(returns) < min_points:
        return False
    if returns.isna().all().all():
        return False
    # Проверяем на бесконечные значения
    if np.isinf(returns.to_numpy()).any():
        return False
    return True


def validate_market_data(market_data: pd.DataFrame) -> bool:
    """Валидация рыночных данных."""
    if market_data is None or market_data.empty:
        return False
    required_columns = ["open", "high", "low", "close", "volume"]
    for col in required_columns:
        if col not in market_data.columns:
            return False
    # Проверяем на отрицательные цены
    price_columns = ["open", "high", "low", "close"]
    for col in price_columns:
        if (market_data[col] <= 0).any():
            return False
    # Проверяем логику high >= low
    if (market_data["high"] < market_data["low"]).any():
        return False
    return True


def validate_portfolio_data(positions: List[PositionRisk]) -> bool:
    """Валидация данных портфеля."""
    if positions is None:
        return False
    if len(positions) == 0:
        return False
    for position in positions:
        if not isinstance(position, PositionRisk):
            return False
        if position.market_value.value <= 0:
            return False
    return True


def extract_returns_from_market_data(market_data: pd.DataFrame) -> pd.DataFrame:
    """Извлечение доходностей из рыночных данных."""
    if not validate_market_data(market_data):
        return pd.DataFrame()
    if "close" in market_data.columns:
        returns = market_data["close"].pct_change().dropna()
        return pd.DataFrame({"returns": returns})
    return pd.DataFrame()


def create_empty_risk_metrics() -> RiskMetrics:
    """Создание пустых метрик риска."""
    return RiskMetrics(
        calculation_timestamp=datetime.now(),
        confidence_level=Decimal("0.95"),
        risk_free_rate=Decimal("0.02"),
        data_points=0,
    )


def create_empty_portfolio_risk() -> PortfolioRisk:
    """Создание пустого риска портфеля."""
    return PortfolioRisk(
        total_value=Money(Decimal("0"), Currency.USD),
        total_risk=Decimal("0"),
        risk_metrics=create_empty_risk_metrics(),
        position_risks=[],
        correlation_matrix=pd.DataFrame(),
        risk_decomposition={},
        calculation_timestamp=datetime.now(),
        portfolio_id="empty",
        risk_model_version="2.0",
    )


def create_empty_optimization_result() -> Dict[str, Any]:
    """Создание пустого результата оптимизации."""
    return {
        "optimal_weights": {},
        "expected_return": Decimal("0"),
        "expected_risk": Decimal("0"),
        "sharpe_ratio": Decimal("0"),
        "optimization_method": "sharpe_maximization",
        "efficient_frontier": pd.DataFrame(),
        "risk_contribution": {},
        "return_contribution": {},
        "rebalancing_recommendations": [],
        "optimization_timestamp": datetime.now(),
        "constraints_applied": [],
        "optimization_time_seconds": 0.0,
        "convergence_status": "failed",
    }


def convert_to_decimal(value: Union[float, int, str, Decimal]) -> Decimal:
    """Конвертация значения в Decimal."""
    if isinstance(value, Decimal):
        return value
    elif isinstance(value, (int, float)):
        return Decimal(str(value))
    elif isinstance(value, str):
        return Decimal(value)
    else:
        raise ValueError(f"Cannot convert {type(value)} to Decimal")


def convert_to_money(
    value: Union[float, int, str, Decimal], currency: Currency = Currency.USD
) -> Money:
    """Конвертация значения в Money."""
    decimal_value = convert_to_decimal(value)
    return Money(decimal_value, currency)


class RiskAnalysisCache:
    """Кэш для риск-анализа."""

    def __init__(self, ttl_hours: int = 1):
        self.cache: Dict[str, Any] = {}
        self.timestamps: Dict[str, datetime] = {}
        self.ttl = timedelta(hours=ttl_hours)

    def get(self, key: str) -> Optional[Any]:
        """Получение значения из кэша."""
        if key not in self.cache:
            return None
        if datetime.now() - self.timestamps[key] > self.ttl:
            del self.cache[key]
            del self.timestamps[key]
            return None
        return self.cache[key]

    def set(self, key: str, value: Any) -> None:
        """Установка значения в кэш."""
        self.cache[key] = value
        self.timestamps[key] = datetime.now()

    def clear(self) -> None:
        """Очистка кэша."""
        self.cache.clear()
        self.timestamps.clear()

    def clean_expired(self) -> None:
        """Очистка истёкших записей."""
        current_time = datetime.now()
        expired_keys = [
            key
            for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]
            del self.timestamps[key]


def clean_cache(cache: RiskAnalysisCache) -> None:
    """Очистка кэша."""
    cache.clean_expired()


def calculate_portfolio_returns(
    positions: List[PositionRisk], returns_df: pd.DataFrame
) -> pd.Series:
    """Расчёт доходностей портфеля."""
    if not positions or returns_df.empty:
        return pd.Series()
    # Упрощённо: средняя доходность всех активов
    return returns_df.mean(axis=1)


def calculate_risk_decomposition(
    weights: np.ndarray, volatilities: np.ndarray, correlation_matrix: pd.DataFrame
) -> Dict[str, Decimal]:
    """Расчёт декомпозиции риска."""
    return {
        f"asset_{i}": Decimal(str(weight * vol))
        for i, (weight, vol) in enumerate(zip(weights, volatilities))
    }


def calculate_liquidity_risk(positions: List[PositionRisk]) -> float:
    """Расчёт риска ликвидности."""
    # Упрощённо: возвращаем 0
    return 0.0
