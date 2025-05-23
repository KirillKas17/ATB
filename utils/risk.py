from typing import Dict, Optional

import numpy as np
from loguru import logger

from utils.logger import setup_logger

logger = setup_logger(__name__)


def calculate_position_size(
    account_balance: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss: float,
    max_position_size: Optional[float] = None,
) -> float:
    """
    Расчет размера позиции на основе риска

    Args:
        account_balance: Баланс счета
        risk_per_trade: Риск на сделку (в процентах)
        entry_price: Цена входа
        stop_loss: Цена стоп-лосса
        max_position_size: Максимальный размер позиции

    Returns:
        Размер позиции в единицах актива
    """
    try:
        # Расчет риска в деньгах
        risk_amount = account_balance * (risk_per_trade / 100)

        # Расчет размера позиции
        price_risk = abs(entry_price - stop_loss)
        if price_risk == 0:
            logger.warning("Нулевой риск цены, используем минимальный размер")
            return 0.01

        position_size = risk_amount / price_risk

        # Проверка максимального размера
        if max_position_size is not None:
            position_size = min(position_size, max_position_size)

        return position_size

    except Exception as e:
        logger.error(f"Ошибка расчета размера позиции: {str(e)}")
        raise


def calculate_risk_metrics(
    returns: np.ndarray, benchmark_returns: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Расчет метрик риска

    Args:
        returns: Массив доходностей
        benchmark_returns: Массив доходностей бенчмарка

    Returns:
        Dict с метриками риска
    """
    try:
        metrics = {
            "volatility": float(np.std(returns) * np.sqrt(252)),
            "sharpe_ratio": float(np.mean(returns) / np.std(returns) * np.sqrt(252)),
            "max_drawdown": float(np.min(np.cumsum(returns))),
            "var_95": float(np.percentile(returns, 5)),
        }

        if benchmark_returns is not None:
            metrics["beta"] = float(
                np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
            )
            metrics["alpha"] = float(
                np.mean(returns) - metrics["beta"] * np.mean(benchmark_returns)
            )

        return metrics

    except Exception as e:
        logger.error(f"Ошибка расчета метрик риска: {str(e)}")
        raise


def calculate_stop_loss(
    entry_price: float,
    atr: float,
    risk_multiplier: float = 2.0,
    direction: str = "long",
) -> float:
    """
    Расчет стоп-лосса на основе ATR

    Args:
        entry_price: Цена входа
        atr: Average True Range
        risk_multiplier: Множитель риска
        direction: Направление позиции ('long' или 'short')

    Returns:
        Цена стоп-лосса
    """
    try:
        stop_distance = atr * risk_multiplier

        if direction.lower() == "long":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    except Exception as e:
        logger.error(f"Ошибка расчета стоп-лосса: {str(e)}")
        raise
