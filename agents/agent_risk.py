import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.types import Signal
from utils.indicators import calculate_atr, calculate_volatility
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class RiskMetrics:
    """Класс для хранения метрик риска"""

    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    max_drawdown: float
    kelly_criterion: float
    volatility: float
    exposure_level: float
    confidence_score: float


class IRiskCalculator(ABC):
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> RiskMetrics:
        pass


class DefaultRiskCalculator(IRiskCalculator):
    def calculate(self, data: pd.DataFrame) -> RiskMetrics:
        # ... современный расчёт VaR, ES, drawdown, ML ...
        return RiskMetrics(0, 0, 0, 0, 0, 0, 0)


class RiskAgent:
    """
    Агент управления рисками: расчёт аллокаций, стоп-лосс/тейк-профит, плеча, метрик риска, stress-тесты.
    TODO: Вынести расчёт метрик риска в отдельный модуль/класс (SRP).
    """

    config: Dict[str, Any]
    risk_metrics: Dict[str, RiskMetrics]
    portfolio_metrics: RiskMetrics
    calculator: IRiskCalculator

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Инициализация агента управления рисками.
        :param config: словарь с параметрами риск-менеджмента
        """
        self.config = config or {
            "max_position_size": 0.1,
            "max_portfolio_risk": 0.02,
            "var_confidence": 0.95,
            "max_leverage": 5.0,
            "min_leverage": 1.0,
            "kelly_threshold": 0.5,
            "drawdown_threshold": 0.1,
            "volatility_lookback": 20,
            "equity_lookback": 100,
        }
        self.risk_metrics: Dict[str, RiskMetrics] = {}
        self.portfolio_metrics = RiskMetrics(
            var_95=0.0,
            var_99=0.0,
            max_drawdown=0.0,
            kelly_criterion=0.0,
            volatility=0.0,
            exposure_level=0.0,
            confidence_score=0.0,
        )
        self.calculator = DefaultRiskCalculator()

    def calculate_risk_allocation(
        self,
        market_data: Dict[str, pd.DataFrame],
        strategy_confidence: Dict[str, float],
        current_positions: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Расчет оптимального распределения риска по активам.
        :param market_data: OHLCV данные по активам
        :param strategy_confidence: уверенность в стратегии по каждому активу
        :param current_positions: текущие позиции
        :return: оптимальные размеры позиций
        """
        try:
            allocations: Dict[str, float] = {}
            total_risk_budget = self.config["max_portfolio_risk"]
            for symbol, data in market_data.items():
                if not self._validate_ohlcv(data):
                    logger.error(f"Некорректные данные для {symbol}")
                    continue
                self._update_risk_metrics(symbol, data)
            self._calculate_portfolio_risk(current_positions)
            for symbol, metrics in self.risk_metrics.items():
                confidence = strategy_confidence.get(symbol, 0.5)
                risk_score = self._calculate_risk_score(metrics, confidence)
                allocations[symbol] = risk_score * total_risk_budget
            total_allocation = sum(allocations.values())
            if total_allocation > 0:
                allocations = {k: v / total_allocation for k, v in allocations.items()}
            logger.info(f"Calculated risk allocations: {allocations}")
            return allocations
        except Exception as e:
            logger.error(f"Error in risk allocation calculation: {str(e)}")
            return {symbol: 0.0 for symbol in market_data.keys()}

    def get_stop_loss_take_profit(
        self, entry_price: float, strategy_id: str, market_data: pd.DataFrame
    ) -> Tuple[float, float]:
        """
        Расчет уровней стоп-лосс и тейк-профит.
        :param entry_price: цена входа
        :param strategy_id: идентификатор стратегии
        :param market_data: OHLCV данные
        :return: (stop_loss, take_profit)
        """
        try:
            metrics = self.risk_metrics.get(strategy_id)
            if not metrics or not self._validate_ohlcv(market_data):
                return entry_price * 0.95, entry_price * 1.05
            atr = calculate_atr(
                high=pd.Series(market_data['high']),
                low=pd.Series(market_data['low']),
                close=pd.Series(market_data['close'])
            )
            atr_value = float(atr.iloc[-1]) if not atr.empty else 0.0
            stop_distance = atr_value * 2
            take_distance = atr_value * 3
            var_adjustment = metrics.var_95 / entry_price if entry_price != 0 else 0.05
            stop_loss = entry_price * (1 - max(stop_distance, var_adjustment))
            take_profit = entry_price * (1 + take_distance)
            logger.info(
                f"Calculated SL/TP for {strategy_id}: SL={stop_loss:.2f}, TP={take_profit:.2f}"
            )
            return float(stop_loss), float(take_profit)
        except Exception as e:
            logger.error(f"Error calculating SL/TP: {str(e)}")
            return entry_price * 0.95, entry_price * 1.05

    def get_leverage_score(
        self, symbol: str, strategy_confidence: float, market_data: pd.DataFrame
    ) -> float:
        """
        Расчет оптимального плеча для позиции.
        :param symbol: торговая пара
        :param strategy_confidence: уверенность в стратегии
        :param market_data: OHLCV данные
        :return: оптимальное плечо
        """
        try:
            metrics = self.risk_metrics.get(symbol)
            if not metrics:
                return self.config["min_leverage"]
            volatility_factor = (
                1 / (metrics.volatility + 0.01) if metrics.volatility is not None else 1.0
            )
            confidence_factor = strategy_confidence
            kelly_factor = min(metrics.kelly_criterion, 1.0) if metrics.kelly_criterion > 0 else 0.5
            leverage = (
                self.config["min_leverage"]
                + (self.config["max_leverage"] - self.config["min_leverage"])
                * volatility_factor
                * confidence_factor
                * kelly_factor
            )
            leverage = min(max(leverage, self.config["min_leverage"]), self.config["max_leverage"])
            logger.info(f"Calculated leverage for {symbol}: {leverage:.2f}")
            return leverage
        except Exception as e:
            logger.error(f"Error calculating leverage: {str(e)}")
            return self.config["min_leverage"]

    def _update_risk_metrics(self, symbol: str, data: pd.DataFrame) -> None:
        """Обновление метрик риска для актива"""
        try:
            if not self._validate_ohlcv(data):
                logger.error(f"Некорректные данные для {symbol}")
                return
            returns = data["close"].pct_change().dropna()
            if returns.empty:
                logger.error(f"Нет данных для расчёта returns по {symbol}")
                return
            var_95 = float(np.percentile(returns, 5))
            var_99 = float(np.percentile(returns, 1))
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = float(abs(drawdowns.min()))
            volatility = float(returns.std() * np.sqrt(252))
            win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0.0
            avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0.0
            avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0.0
            kelly = float(
                (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win if avg_win != 0 else 0
            )
            self.risk_metrics[symbol] = RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                max_drawdown=max_drawdown,
                kelly_criterion=kelly,
                volatility=volatility,
                exposure_level=0.0,
                confidence_score=0.0,
            )
        except Exception as e:
            logger.error(f"Error updating risk metrics for {symbol}: {str(e)}")

    def _calculate_portfolio_risk(self, positions: Dict[str, float]) -> None:
        """Расчет общего риска портфеля"""
        try:
            portfolio_var = 0.0
            portfolio_volatility = 0.0
            for symbol, position in positions.items():
                metrics = self.risk_metrics.get(symbol)
                if metrics:
                    portfolio_var += position * metrics.var_95
                    portfolio_volatility += position * metrics.volatility
            self.portfolio_metrics.var_95 = portfolio_var
            self.portfolio_metrics.volatility = portfolio_volatility
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {str(e)}")

    def _calculate_risk_score(self, metrics: RiskMetrics, confidence: float) -> float:
        """Расчет общего скор риска для актива"""
        try:
            var_score = 1 - min(metrics.var_95 / 0.1, 1.0)
            drawdown_score = 1 - min(metrics.max_drawdown / self.config["drawdown_threshold"], 1.0)
            volatility_score = 1 - min(metrics.volatility / 0.5, 1.0)
            risk_score = var_score * 0.4 + drawdown_score * 0.3 + volatility_score * 0.3
            return risk_score * confidence
        except Exception as e:
            logger.error(f"Error calculating risk score: {str(e)}")
            return 0.0

    def get_risk_metrics(self, symbol: str) -> Optional[RiskMetrics]:
        """Получение метрик риска для актива"""
        return self.risk_metrics.get(symbol)

    def get_portfolio_metrics(self) -> RiskMetrics:
        """Получить метрики риска портфеля."""
        return self.portfolio_metrics

    async def get_signals(self) -> List[Signal]:
        """Получение сигналов управления рисками"""
        try:
            signals = []
            for symbol, metrics in self.risk_metrics.items():
                if metrics.max_drawdown > self.config["drawdown_threshold"]:
                    signals.append(
                        Signal(
                            pair=symbol,
                            action="reduce_position",
                            price=0.0,
                            size=0.0
                        )
                    )
            return signals
        except Exception as e:
            logger.error(f"Error getting risk signals: {str(e)}")
            return []

    def _validate_ohlcv(self, df: pd.DataFrame) -> bool:
        """Проверка наличия необходимых колонок в OHLCV-данных."""
        required = {"open", "high", "low", "close", "volume"}
        return isinstance(df, pd.DataFrame) and required.issubset(df.columns)

    def calculate_position_size(self, price: float, risk_per_trade: float, low: float, close: float) -> float:
        """Расчет размера позиции."""
        try:
            # Используем фиксированный баланс для примера
            account_balance = 10000.0  # Заменить на реальный баланс
            position_size = (account_balance * risk_per_trade) / price
            return float(position_size)
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0

    def create_signal(self, pair: str, action: str, price: float, size: float) -> Dict[str, Any]:
        """Создание сигнала."""
        try:
            return {
                "pair": pair,
                "action": action,
                "price": price,
                "size": size,
                "timestamp": datetime.now(),
                "source": "risk_agent"
            }
        except Exception as e:
            logger.error(f"Error creating signal: {str(e)}")
            return {}
