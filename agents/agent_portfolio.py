import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import talib

from agents.agent_market_regime import MarketRegime, MarketRegimeAgent
from core.correlation_chain import CorrelationChain
from utils.indicators import calculate_atr, calculate_volatility
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class AssetMetrics:
    """Data class to store asset-specific metrics"""

    symbol: str
    expected_return: float
    risk_score: float
    liquidity_score: float
    trend_strength: float
    volume_score: float
    correlation_with_btc: float
    current_weight: float
    target_weight: float


class IPortfolioOptimizer(ABC):
    @abstractmethod
    def optimize(
        self, metrics: Dict[str, AssetMetrics], constraints: Dict[str, Any]
    ) -> Dict[str, float]:
        pass


class MeanVarianceOptimizer(IPortfolioOptimizer):
    def optimize(
        self, metrics: Dict[str, AssetMetrics], constraints: Dict[str, Any]
    ) -> Dict[str, float]:
        # ... реализация mean-variance оптимизации ...
        return {k: 1 / len(metrics) for k in metrics}


class PortfolioCacheService:
    def __init__(self):
        self.metrics: Dict[str, AssetMetrics] = {}
        self.state: Dict[str, Any] = {}

    def clear(self):
        self.metrics.clear()
        self.state.clear()


class PortfolioCorrelationChain(CorrelationChain):
    def __init__(self):
        super().__init__()
        self.correlations: Dict[str, Dict[str, float]] = {}
        
    def update_correlations(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """Обновление корреляций между активами."""
        try:
            for symbol1, data1 in market_data.items():
                if symbol1 not in self.correlations:
                    self.correlations[symbol1] = {}
                for symbol2, data2 in market_data.items():
                    if symbol1 != symbol2:
                        corr = data1['close'].corr(data2['close'])
                        self.correlations[symbol1][symbol2] = float(corr)
        except Exception as e:
            logger.error(f"Error updating correlations: {str(e)}")
            
    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Получение корреляции между двумя активами."""
        try:
            return self.correlations.get(symbol1, {}).get(symbol2, 0.0)
        except Exception as e:
            logger.error(f"Error getting correlation: {str(e)}")
            return 0.0


class PortfolioAgent:
    """
    Агент управления портфелем: оптимизация, ребалансировка, симуляция, stress-тесты, асинхронность.
    """

    config: Dict[str, Any]
    optimizer: IPortfolioOptimizer
    cache: PortfolioCacheService
    market_regime_agent: MarketRegimeAgent
    correlation_chain: PortfolioCorrelationChain

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Инициализация агента управления портфелем.
        :param config: словарь параметров
        """
        self.config = config or {
            "max_position_size": 0.3,
            "min_position_size": 0.05,
            "rebalance_threshold": 0.1,
            "risk_free_rate": 0.02,
            "max_correlation": 0.7,
            "btc_dominance_threshold": 0.4,
            "volatility_lookback": 20,
            "correlation_lookback": 60,
        }
        self.optimizer = MeanVarianceOptimizer()
        self.cache = PortfolioCacheService()
        self.market_regime_agent = MarketRegimeAgent()
        self.correlation_chain = PortfolioCorrelationChain()

    async def update_portfolio_weights(
        self,
        market_data: Dict[str, pd.DataFrame],
        risk_data: Dict[str, Dict],
        backtest_results: Dict[str, Dict],
    ) -> Dict[str, float]:
        """
        Обновить веса портфеля на основе рыночных условий и метрик риска.
        :param market_data: OHLCV-данные по активам
        :param risk_data: метрики риска по активам
        :param backtest_results: результаты бэктеста по активам
        :return: словарь новых весов
        """
        try:
            btc_data = market_data.get("BTC/USDT", None)
            if btc_data is not None:
                regime, confidence = self.market_regime_agent.detect_regime(btc_data)
            else:
                regime = MarketRegime.SIDEWAYS
                confidence = 0.0
            self._update_asset_metrics(market_data, risk_data, backtest_results)
            self._calculate_btc_dominance(market_data)
            self.correlation_chain.update_correlations(market_data)
            weights = self._calculate_optimal_weights(regime, confidence)
            weights = self._apply_portfolio_constraints(weights)
            self._update_portfolio_state(weights, market_data)
            logger.info(f"Updated portfolio weights: {weights}")
            return weights
        except Exception as e:
            logger.error(f"Error updating portfolio weights: {str(e)}")
            return self.cache.state["current_weights"]

    def _update_asset_metrics(
        self,
        market_data: Dict[str, pd.DataFrame],
        risk_data: Dict[str, Dict],
        backtest_results: Dict[str, Dict],
    ):
        """Update metrics for each asset"""
        for symbol, data in market_data.items():
            if symbol not in self.cache.metrics:
                self.cache.metrics[symbol] = AssetMetrics(
                    symbol=symbol,
                    expected_return=0.0,
                    risk_score=0.0,
                    liquidity_score=0.0,
                    trend_strength=0.0,
                    volume_score=0.0,
                    correlation_with_btc=0.0,
                    current_weight=0.0,
                    target_weight=0.0,
                )

            metrics = self.cache.metrics[symbol]

            # Update metrics
            metrics.expected_return = backtest_results[symbol].get("expected_return", 0.0)
            metrics.risk_score = risk_data[symbol].get("risk_score", 1.0)
            metrics.liquidity_score = self._calculate_liquidity_score(data)
            metrics.trend_strength = self._calculate_trend_strength(data)
            metrics.volume_score = self._calculate_volume_score(data)
            metrics.correlation_with_btc = self.correlation_chain.get_correlation(
                symbol, "BTC/USDT"
            )

    def _calculate_liquidity_score(self, data: pd.DataFrame) -> float:
        """Calculate liquidity score based on volume and spread"""
        avg_volume = data["volume"].mean()
        avg_spread = (data["high"] - data["low"]).mean() / data["close"].mean()

        volume_score = min(1.0, avg_volume / 1e6)  # Normalize to 1M volume
        spread_score = 1.0 - min(1.0, avg_spread / 0.01)  # Normalize to 1% spread

        return (volume_score + spread_score) / 2

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength using ADX"""
        try:
            high = data["high"].values
            low = data["low"].values
            close = data["close"].values
            # Используем numpy для расчета ADX
            tr1 = np.abs(high[1:] - low[1:])
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr = np.mean(tr)
            return min(1.0, float(atr) / 100.0)
        except Exception as e:
            logger.error(f"Error calculating trend strength: {str(e)}")
            return 0.0

    def _calculate_volume_score(self, data: pd.DataFrame) -> float:
        """Calculate volume score based on recent volume trends"""
        recent_volume = data["volume"][-20:].mean()
        historical_volume = data["volume"][-100:].mean()

        return min(1.0, recent_volume / historical_volume)

    def _calculate_btc_dominance(self, market_data: Dict[str, pd.DataFrame]):
        """Calculate BTC dominance in the portfolio"""
        btc_data = market_data.get("BTC/USDT", None)
        if btc_data is None:
            self.cache.state["btc_dominance"] = 0.0
            return

        btc_market_cap = btc_data["close"][-1] * btc_data["volume"][-1]
        total_market_cap = sum(
            data["close"][-1] * data["volume"][-1] for data in market_data.values()
        )

        self.cache.state["btc_dominance"] = btc_market_cap / total_market_cap

    def _calculate_optimal_weights(
        self, regime: MarketRegime, confidence: float
    ) -> Dict[str, float]:
        """Calculate optimal weights based on market regime and asset metrics"""
        weights = {}
        total_score = 0.0

        for symbol, metrics in self.cache.metrics.items():
            # Calculate base score
            base_score = (
                metrics.expected_return * 0.3
                + (1 - metrics.risk_score) * 0.2
                + metrics.liquidity_score * 0.15
                + metrics.trend_strength * 0.15
                + metrics.volume_score * 0.1
                + (1 - abs(metrics.correlation_with_btc)) * 0.1
            )

            # Adjust score based on market regime
            regime_multiplier = self._get_regime_multiplier(regime, metrics)
            final_score = base_score * regime_multiplier * confidence

            weights[symbol] = final_score
            total_score += final_score

        # Normalize weights
        if total_score > 0:
            weights = {k: v / total_score for k, v in weights.items()}

        return weights

    def _get_regime_multiplier(self, regime: MarketRegime, metrics: AssetMetrics) -> float:
        """Get multiplier based on market regime and asset characteristics"""
        multipliers = {
            MarketRegime.TREND: 1.0 + metrics.trend_strength * 0.5,
            MarketRegime.SIDEWAYS: 1.0 + (1 - metrics.trend_strength) * 0.5,
            MarketRegime.REVERSAL: 1.0 + metrics.volume_score * 0.5,
            MarketRegime.MANIPULATION: 1.0 + metrics.liquidity_score * 0.5,
            MarketRegime.VOLATILITY: 1.0 + (1 - metrics.risk_score) * 0.5,
            MarketRegime.ANOMALY: 0.5,  # Reduce exposure during anomalies
        }

        return multipliers.get(regime, 1.0)

    def _apply_portfolio_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply portfolio constraints to weights"""
        # Apply position size limits
        weights = {
            k: max(min(v, self.config["max_position_size"]), self.config["min_position_size"])
            for k, v in weights.items()
        }

        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def _update_portfolio_state(
        self, weights: Dict[str, float], market_data: Dict[str, pd.DataFrame]
    ):
        """Update portfolio state with new weights and metrics"""
        self.cache.state["current_weights"] = weights
        self.cache.state["target_weights"] = weights

        # Calculate portfolio metrics
        returns = []
        for symbol, data in market_data.items():
            if symbol in weights:
                returns.append(data["close"].pct_change() * weights[symbol])

        if returns:
            portfolio_returns = pd.concat(returns, axis=1).sum(axis=1)
            self.cache.state["volatility"] = portfolio_returns.std() * np.sqrt(252)
            self.cache.state["sharpe_ratio"] = (
                portfolio_returns.mean() * 252 - self.config["risk_free_rate"]
            ) / self.cache.state["volatility"]

    def suggest_trades(self) -> List[Dict]:
        """
        Suggest trades to rebalance the portfolio.

        Returns:
            List[Dict]: List of suggested trades with symbol, side, and size
        """
        trades = []

        for symbol, metrics in self.cache.metrics.items():
            current_weight = metrics.current_weight
            target_weight = metrics.target_weight

            if abs(current_weight - target_weight) > self.config["rebalance_threshold"]:
                trades.append(
                    {
                        "symbol": symbol,
                        "side": "buy" if target_weight > current_weight else "sell",
                        "size": abs(target_weight - current_weight),
                    }
                )

        return trades

    def get_portfolio_state(self) -> Dict:
        """
        Get the current state of the portfolio.

        Returns:
            Dict: Portfolio state including weights, metrics, and performance
        """
        return self.cache.state

    def update_correlations(self, pair: str, data: pd.DataFrame) -> None:
        """Обновление корреляций."""
        try:
            market_data = {pair: data}
            self.correlation_chain.update_correlations(market_data)
        except Exception as e:
            logger.error(f"Error updating correlations: {str(e)}")

    def get_correlation(self, pair1: str, pair2: str) -> float:
        """Получение корреляции между парами."""
        try:
            return self.correlation_chain.get_correlation(pair1, pair2)
        except Exception as e:
            logger.error(f"Error getting correlation: {str(e)}")
            return 0.0
