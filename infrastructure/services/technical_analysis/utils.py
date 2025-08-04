"""
Утилиты для технического анализа.
Содержит вспомогательные функции для валидации, преобразований,
кэширования и общих утилит технического анализа.
"""

import pandas as pd
from shared.numpy_utils import np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

from domain.type_definitions.technical_types import (
    TechnicalAnalysisReport,
    TechnicalIndicatorResult,
    MarketStructureResult,
    MarketStructure,
    TrendStrength,
    VolumeProfileResult,
)

__all__ = [
    "validate_ohlcv_data",
    "validate_indicator_data",
    "extract_ohlcv_components",
    "create_empty_technical_result",
    "create_empty_indicator_result",
    "convert_to_decimal",
    "convert_to_datetime",
    "clean_cache",
    "validate_market_data",
    "calculate_returns",
    "normalize_data",
]


def validate_ohlcv_data(data: pd.DataFrame) -> bool:
    """Валидация OHLCV данных."""
    if data is None or data.empty:
        return False
    required_columns = ["open", "high", "low", "close"]
    if not all(col in data.columns for col in required_columns):
        return False
    # Проверяем логику данных
    if (data["high"] < data["low"]).any():
        return False
    if (data["high"] < data["open"]).any() or (data["high"] < data["close"]).any():
        return False
    if (data["low"] > data["open"]).any() or (data["low"] > data["close"]).any():
        return False
    # Проверяем на отрицательные цены
    price_columns = ["open", "high", "low", "close"]
    for col in price_columns:
        if (data[col] <= 0).any():
            return False
    return True


def validate_indicator_data(data: pd.Series, min_points: int = 20) -> bool:
    """Валидация данных индикатора."""
    if data is None or data.empty:
        return False
    if len(data) < min_points:
        return False
    if data.isna().all():
        return False
    # Проверяем на бесконечные значения
    if np.isinf(data.values).any():
        return False
    return True


def validate_market_data(market_data: pd.DataFrame) -> bool:
    """Валидация рыночных данных."""
    return validate_ohlcv_data(market_data)


def extract_ohlcv_components(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """Извлечение компонентов OHLCV из данных."""
    if not validate_ohlcv_data(data):
        return {}
    components = {}
    for col in ["open", "high", "low", "close"]:
        if col in data.columns:
            components[col] = data[col]
    if "volume" in data.columns:
        components["volume"] = data["volume"]
    return components


def create_empty_technical_result() -> TechnicalAnalysisReport:
    """Создание пустого результата технического анализа."""
    empty_indicator_result = create_empty_indicator_result()
    empty_market_structure = MarketStructureResult(
        structure=MarketStructure.SIDEWAYS,
        trend_strength=TrendStrength.WEAK,
        volatility=Decimal("0"),
        adx=Decimal("0"),
        rsi=Decimal("0"),
        confidence=Decimal("0"),
    )
    return TechnicalAnalysisReport(
        indicator_results=empty_indicator_result,
        signals=[],
        market_structure=empty_market_structure,
    )


def create_empty_indicator_result() -> TechnicalIndicatorResult:
    """Создание пустого результата индикатора."""
    empty_market_structure = MarketStructureResult(
        structure=MarketStructure.SIDEWAYS,
        trend_strength=TrendStrength.WEAK,
        volatility=Decimal("0"),
        adx=Decimal("0"),
        rsi=Decimal("0"),
        confidence=Decimal("0"),
    )
    empty_volume_profile = VolumeProfileResult(
        poc=0.0,
        value_area_high=0.0,
        value_area_low=0.0,
        volume_by_price={},
        histogram=[],
        price_levels=[],
    )
    return TechnicalIndicatorResult(
        indicators={},
        market_structure=empty_market_structure,
        volume_profile=empty_volume_profile,
        support_levels=[],
        resistance_levels=[],
    )


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


def convert_to_datetime(value: Union[str, datetime, pd.Timestamp]) -> datetime:
    """Конвертация значения в datetime."""
    if isinstance(value, datetime):
        return value
    elif isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    else:
        # Пытаемся конвертировать как строку
        try:
            return pd.to_datetime(value).to_pydatetime()
        except (ValueError, TypeError):
            raise ValueError(f"Cannot convert {type(value)} to datetime")


class TechnicalAnalysisCache:
    """Кэш для технического анализа."""

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


def clean_cache(cache: TechnicalAnalysisCache) -> None:
    """Очистка кэша."""
    cache.clean_expired()


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Расчёт доходностей."""
    return prices.pct_change().dropna()


def normalize_data(data: pd.Series, method: str = "minmax") -> pd.Series:
    """Нормализация данных."""
    if data.empty:
        return data
    if method == "minmax":
        min_val = data.min()
        max_val = data.max()
        if max_val > min_val:
            return (data - min_val) / (max_val - min_val)
        else:
            return data
    elif method == "zscore":
        mean_val = data.mean()
        std_val = data.std()
        if std_val > 0:
            return (data - mean_val) / std_val
        else:
            return data
    else:
        return data


def calculate_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Расчёт корреляционной матрицы."""
    if returns_df.empty:
        return pd.DataFrame()
    # Удаляем строки с NaN
    clean_df = returns_df.dropna()
    if len(clean_df.index) < 20:
        return pd.DataFrame()
    # Рассчитываем корреляцию
    correlation_matrix = clean_df.corr()
    # Заполняем NaN значения нулями
    correlation_matrix = correlation_matrix.fillna(0)
    return correlation_matrix


def detect_outliers(
    data: pd.Series, method: str = "iqr", threshold: float = 1.5
) -> pd.Series:
    """Обнаружение выбросов."""
    if data.empty:
        return pd.Series()
    if method == "iqr":
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outliers = pd.Series(
            (data < lower_bound) | (data > upper_bound), index=data.index
        )
    elif method == "zscore":
        z_scores = np.abs((data.to_numpy() - data.mean()) / data.std())
        outliers = z_scores > threshold
    else:
        outliers = pd.Series(False, index=data.index)
    return outliers


def smooth_data(data: pd.Series, method: str = "sma", window: int = 5) -> pd.Series:
    """Сглаживание данных."""
    if data.empty:
        return data
    if method == "sma":
        return data.rolling(window=window).mean()
    elif method == "ema":
        return data.ewm(span=window).mean()
    elif method == "median":
        return data.rolling(window=window).median()
    else:
        return data


def calculate_volatility(
    data: pd.Series, window: int = 20, annualization_factor: float = 252.0
) -> pd.Series:
    """Расчёт волатильности."""
    if data.empty:
        return pd.Series()
    returns = calculate_returns(data)
    volatility = returns.rolling(window=window).std() * np.sqrt(annualization_factor)
    return volatility


def calculate_drawdown(data: pd.Series) -> pd.Series:
    """Расчёт просадки."""
    if data.empty:
        return pd.Series()
    rolling_max = data.expanding().max()
    drawdown = (data - rolling_max) / rolling_max
    return drawdown


def calculate_max_drawdown(data: pd.Series) -> float:
    """Расчёт максимальной просадки."""
    drawdown = calculate_drawdown(data)
    return float(drawdown.min())


def calculate_sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252
) -> float:
    """Расчёт коэффициента Шарпа."""
    if returns.empty:
        return 0.0
    excess_returns = returns - risk_free_rate / periods_per_year
    return float(excess_returns.mean() / returns.std() * np.sqrt(periods_per_year))


def calculate_sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252
) -> float:
    """Расчёт коэффициента Сортино."""
    if returns.empty:
        return 0.0
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = returns[returns.lt(0)]
    if len(downside_returns) == 0:
        return float("inf")
    downside_deviation = downside_returns.std()
    if downside_deviation == 0:
        return float("inf")
    return float(excess_returns.mean() / downside_deviation * np.sqrt(periods_per_year))


def calculate_calmar_ratio(
    returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252
) -> float:
    """Расчёт коэффициента Кальмара."""
    if returns.empty:
        return 0.0
    max_drawdown = calculate_max_drawdown(returns.cumsum())
    if max_drawdown == 0:
        return 0.0
    annual_return = returns.mean() * periods_per_year
    return float(annual_return / abs(max_drawdown))


def calculate_information_ratio(
    returns: pd.Series, benchmark_returns: pd.Series
) -> float:
    """Расчёт информационного коэффициента."""
    if returns.empty or benchmark_returns.empty:
        return 0.0
    # Приводим к одинаковой длине
    common_index = returns.index.intersection(benchmark_returns.index)
    if len(common_index) < 20:
        return 0.0
    returns_aligned: pd.Series = returns.loc[common_index]
    benchmark_aligned: pd.Series = benchmark_returns.loc[common_index]
    excess_returns = returns_aligned - benchmark_aligned
    tracking_error = excess_returns.std()
    if tracking_error == 0:
        return 0.0
    return float(excess_returns.mean() / tracking_error)


def calculate_beta(returns: pd.Series, market_returns: pd.Series) -> float:
    """Расчёт беты."""
    if returns.empty or market_returns.empty:
        return 1.0
    # Приводим к одинаковой длине
    common_index = returns.index.intersection(market_returns.index)
    if len(common_index) < 20:
        return 1.0
    returns_aligned: pd.Series = returns.loc[common_index]
    market_aligned: pd.Series = market_returns.loc[common_index]
    covariance = np.cov(returns_aligned, market_aligned)[0, 1]
    market_variance = np.var(market_aligned)
    if market_variance == 0:
        return 1.0
    return float(covariance / market_variance)


def calculate_alpha(
    returns: pd.Series, market_returns: pd.Series, risk_free_rate: float = 0.02
) -> float:
    """Расчёт альфы."""
    if returns.empty or market_returns.empty:
        return 0.0
    beta = calculate_beta(returns, market_returns)
    # Приводим к одинаковой длине
    common_index = returns.index.intersection(market_returns.index)
    if len(common_index) < 20:
        return 0.0
    returns_aligned: pd.Series = returns.loc[common_index]
    market_aligned: pd.Series = market_returns.loc[common_index]
    excess_returns = returns_aligned - risk_free_rate / 252
    excess_market_returns = market_aligned - risk_free_rate / 252
    alpha = excess_returns.mean() - beta * excess_market_returns.mean()
    return float(alpha * 252)  # Годовая альфа
