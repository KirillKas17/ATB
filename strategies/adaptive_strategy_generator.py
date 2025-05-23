from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from meta_learning import MetaLearner

from agents.market_regime import MarketRegimeAgent
from ml.decision_reasoner import DecisionReasoner
from ml.meta_learning import MetaLearning
from ml.pattern_discovery import PatternDiscovery
from strategies.base_strategy import BaseStrategy
from strategies.manipulation_strategies import (
    manipulation_strategy_fake_breakout,
    manipulation_strategy_stop_hunt,
)
from strategies.reversal_strategies import (
    reversal_strategy_fibo_pinbar,
    reversal_strategy_rsi_divergence,
)
from strategies.sideways_strategies import (
    sideways_strategy_bb_rsi,
    sideways_strategy_stoch_obv,
)
from strategies.trend_strategies import (
    trend_strategy_ema_macd,
    trend_strategy_price_action,
)
from strategies.volatility_strategies import (
    volatility_strategy_atr_breakout,
    volatility_strategy_ema_keltner,
)
from utils.feature_engineering import generate_features
from utils.indicators import calculate_atr, calculate_rsi, calculate_volatility


class AdaptiveStrategyGenerator(BaseStrategy):
    """Генератор адаптивных стратегий"""

    def __init__(
        self,
        market_regime_agent: MarketRegimeAgent,
        meta_learner: MetaLearner,
        backtest_results: Dict[str, Dict],
        base_strategies: List[Callable],
        meta_learning_rate: float = 0.01,
        adaptation_threshold: float = 0.7,
        log_dir: str = "logs/adaptive",
    ):
        """
        Инициализация генератора.

        Args:
            market_regime_agent: Агент определения рыночного режима
            meta_learner: Мета-обучающая модель
            backtest_results: Результаты бэктеста стратегий
            base_strategies: Список базовых стратегий
            meta_learning_rate: Скорость мета-обучения
            adaptation_threshold: Порог адаптации
            log_dir: Директория для логов
        """
        super().__init__({"log_dir": log_dir})
        self.market_regime_agent = market_regime_agent
        self.meta_learner = meta_learner
        self.backtest_results = backtest_results or {}
        self.base_strategies = base_strategies or []
        self.meta_learning_rate = meta_learning_rate
        self.adaptation_threshold = adaptation_threshold
        self.strategy_weights = {
            strategy.__name__: 1.0 for strategy in self.base_strategies
        }

        # Маппинг режимов на стратегии
        self.regime_strategies = {
            "trend": [trend_strategy_ema_macd, trend_strategy_price_action],
            "sideways": [sideways_strategy_bb_rsi, sideways_strategy_stoch_obv],
            "reversal": [
                reversal_strategy_rsi_divergence,
                reversal_strategy_fibo_pinbar,
            ],
            "volatility": [
                volatility_strategy_atr_breakout,
                volatility_strategy_ema_keltner,
            ],
            "manipulation": [
                manipulation_strategy_stop_hunt,
                manipulation_strategy_fake_breakout,
            ],
        }

        self.pattern_discovery = PatternDiscovery()
        self.decision_reasoner = DecisionReasoner()
        self.meta_learning = MetaLearning()
        self.strategy_cache = {}

    def generate_adaptive_strategy(
        self, symbol: str, timeframe: str, regime: str
    ) -> Dict[str, Any]:
        """
        Генерация адаптивной стратегии.

        Args:
            symbol: Торговый инструмент
            timeframe: Таймфрейм
            regime: Рыночный режим

        Returns:
            Dict[str, Any]: Конфигурация стратегии
        """
        try:
            # Получение базовых стратегий для режима
            base_strategies = self.regime_strategies.get(regime, [])
            if not base_strategies:
                raise ValueError(f"No strategies found for regime: {regime}")

            # Анализ результатов бэктеста
            best_strategy = self._select_best_strategy(
                base_strategies, symbol, timeframe
            )
            if best_strategy is None:
                raise ValueError("No best strategy found for regime")

            # Получение мета-предсказаний
            meta_predictions = (
                self.meta_learner.predict(
                    symbol=symbol, timeframe=timeframe, regime=regime
                )
                or {}
            )

            # Модификация параметров
            modified_params = self._modify_strategy_params(
                strategy=best_strategy, meta_predictions=meta_predictions, regime=regime
            )

            # Расчет confidence score
            confidence = self._calculate_confidence_score(
                strategy=best_strategy, meta_predictions=meta_predictions, regime=regime
            )

            return {
                "strategy": best_strategy.__name__,
                "parameters": modified_params,
                "confidence": confidence,
                "regime": regime,
                "timeframe": timeframe,
            }

        except Exception as e:
            logger.error(f"Error generating adaptive strategy: {str(e)}")
            return {
                "strategy": None,
                "parameters": {"window_size": 50, "entry_threshold": 0.7},
                "confidence": 0.0,
                "regime": regime,
                "timeframe": timeframe,
            }

    def _select_best_strategy(
        self, strategies: List[Callable], symbol: str, timeframe: str
    ) -> Optional[Callable]:
        """Выбор лучшей стратегии на основе бэктеста"""
        if not strategies:
            return None

        best_strategy = None
        best_score = -np.inf

        for strategy in strategies:
            strategy_name = strategy.__name__
            if strategy_name in self.backtest_results:
                results = self.backtest_results[strategy_name]
                score = self._calculate_strategy_score(results)

                if score > best_score:
                    best_score = score
                    best_strategy = strategy

        return best_strategy if best_score > self.adaptation_threshold else None

    def _calculate_strategy_score(self, results: Dict) -> float:
        """Расчет оценки стратегии"""
        if not results:
            return 0.0

        # Комбинированная оценка на основе метрик
        return (
            results.get("win_rate", 0) * 0.3
            + results.get("profit_factor", 0) * 0.3
            + results.get("sharpe_ratio", 0) * 0.2
            + results.get("max_drawdown", 0) * 0.2
        )

    def _modify_strategy_params(
        self, strategy: Callable, meta_predictions: Dict[str, float], regime: str
    ) -> Dict[str, Any]:
        """Модификация параметров стратегии с дефолтами"""
        if not strategy:
            return {"window_size": 50, "entry_threshold": 0.7}

        base_params = self._get_base_params(strategy, regime)

        # Модификация на основе мета-предсказаний
        modified_params = {}
        for param, value in base_params.items():
            pred = meta_predictions.get(param)
            if pred is not None:
                modified_params[param] = self._adapt_parameter(
                    param=param, base_value=value, prediction=pred, regime=regime
                )
            else:
                modified_params[param] = value

        # Гарантируем дефолты
        if "window_size" not in modified_params:
            modified_params["window_size"] = 50
        if "entry_threshold" not in modified_params:
            modified_params["entry_threshold"] = 0.7
        return modified_params

    def _get_base_params(self, strategy: Callable, regime: str) -> Dict[str, Any]:
        """Получение базовых параметров стратегии с дефолтами для ключевых параметров"""
        base_params = {
            "trend": {
                "ema_fast": 20,
                "ema_medium": 50,
                "ema_slow": 200,
                "adx_threshold": 25.0,
                "atr_period": 14,
                "risk_reward": 2.0,
                "window_size": 50,
                "entry_threshold": 0.7,
            },
            "sideways": {
                "bb_period": 20,
                "bb_std": 2.0,
                "rsi_period": 14,
                "stoch_k": 14,
                "stoch_d": 3,
                "obv_threshold": 1.5,
                "window_size": 50,
                "entry_threshold": 0.7,
            },
            "reversal": {
                "rsi_period": 14,
                "ma_period": 20,
                "envelope_percent": 2.0,
                "volume_threshold": 2.0,
                "window_size": 50,
                "entry_threshold": 0.7,
            },
            "volatility": {
                "atr_period": 14,
                "keltner_period": 20,
                "ema_fast": 10,
                "ema_slow": 20,
                "volume_threshold": 2.0,
                "window_size": 50,
                "entry_threshold": 0.7,
            },
            "manipulation": {
                "volume_delta_threshold": 1.5,
                "fractal_period": 5,
                "vwap_period": 20,
                "imbalance_threshold": 0.7,
                "window_size": 50,
                "entry_threshold": 0.7,
            },
        }
        # Гарантируем, что всегда есть дефолтные значения
        params = base_params.get(regime, {})
        if "window_size" not in params:
            params["window_size"] = 50
        if "entry_threshold" not in params:
            params["entry_threshold"] = 0.7
        return params

    def _adapt_parameter(
        self, param: str, base_value: Any, prediction: float, regime: str
    ) -> Any:
        """Адаптация параметра на основе предсказания"""
        # Коэффициенты адаптации для разных параметров
        adaptation_factors = {
            "ema_fast": 0.2,
            "ema_medium": 0.15,
            "ema_slow": 0.1,
            "atr_period": 0.1,
            "risk_reward": 0.3,
            "bb_std": 0.2,
            "rsi_period": 0.1,
            "volume_threshold": 0.3,
        }

        factor = adaptation_factors.get(param, 0.1)

        if isinstance(base_value, (int, float)):
            # Адаптация числовых параметров
            return base_value * (1 + (prediction - 0.5) * factor)
        else:
            return base_value

    def _calculate_confidence_score(
        self, strategy: Callable, meta_predictions: Dict[str, float], regime: str
    ) -> float:
        """Расчет confidence score"""
        if not strategy:
            return 0.0

        # Веса для разных компонентов
        weights = {
            "backtest_score": 0.4,
            "meta_prediction": 0.3,
            "regime_strength": 0.3,
        }

        # Оценка на основе бэктеста
        backtest_score = self._calculate_strategy_score(
            self.backtest_results.get(strategy.__name__, {})
        )

        # Оценка на основе мета-предсказаний
        meta_score = (
            np.mean(list(meta_predictions.values())) if meta_predictions else 0.0
        )

        # Оценка силы режима
        regime_strength = (
            self.market_regime_agent.get_regime_strength(regime)
            if self.market_regime_agent
            else 0.0
        )

        # Комбинированная оценка
        confidence = (
            backtest_score * weights["backtest_score"]
            + meta_score * weights["meta_prediction"]
            + regime_strength * weights["regime_strength"]
        )

        return min(max(confidence, 0.0), 1.0)  # Нормализация в [0, 1]

    def generate_hybrid_strategy(
        self, pair: str, timeframe: str, regime: str
    ) -> Dict[str, Any]:
        """Generate hybrid strategy combining ML and rule-based signals.

        Args:
            pair: Trading pair symbol
            timeframe: Trading timeframe
            regime: Market regime

        Returns:
            Dictionary with strategy configuration
        """
        try:
            # Get ML signals
            ml_signals = self._get_ml_signals(pair, timeframe)

            # Get rule-based signals
            rule_signals = self._get_rule_signals(pair, timeframe, regime)

            # Combine signals
            hybrid_strategy = self._combine_signals(ml_signals, rule_signals, regime)

            # Cache strategy
            self.strategy_cache[pair] = hybrid_strategy

            return hybrid_strategy

        except Exception as e:
            logger.error(f"Error generating hybrid strategy for {pair}: {str(e)}")
            return {}

    def _get_ml_signals(self, pair: str, timeframe: str) -> Dict[str, Any]:
        """Получение ML-сигналов с использованием унифицированных признаков"""
        try:
            market_data = self._get_market_data(pair, timeframe)
            if market_data.empty:
                return {}

            indicators = self._calculate_indicators(market_data)
            regime = (
                self.market_regime_agent.get_regime(pair, timeframe)
                if self.market_regime_agent
                else "unknown"
            )
            features = generate_features(market_data, indicators, pair, regime)
            return {"features": features}
        except Exception as e:
            logger.error(f"Error getting ML signals: {str(e)}")
            return {}

    def _get_rule_signals(
        self, pair: str, timeframe: str, regime: str
    ) -> Dict[str, Any]:
        """Get rule-based signals for the current regime.

        Args:
            pair: Trading pair symbol
            timeframe: Trading timeframe
            regime: Market regime

        Returns:
            Dictionary with rule-based signals
        """
        try:
            # Get base strategy for regime
            base_strategy = (
                self.meta_learning.get_strategy_for_regime(regime)
                if self.meta_learning
                else None
            )
            if not base_strategy:
                return {}

            # Get market data
            market_data = self._get_market_data(pair, timeframe)
            if market_data.empty:
                return {}

            # Apply strategy rules
            signals = self._apply_strategy_rules(base_strategy, market_data)
            return signals

        except Exception as e:
            logger.error(f"Error getting rule signals for {pair}: {str(e)}")
            return {}

    def _combine_signals(
        self, ml_signals: Dict[str, Any], rule_signals: Dict[str, Any], regime: str
    ) -> Dict[str, Any]:
        """Combine ML and rule-based signals.

        Args:
            ml_signals: ML-based signals
            rule_signals: Rule-based signals
            regime: Market regime

        Returns:
            Combined strategy configuration
        """
        try:
            if not ml_signals and not rule_signals:
                return {}

            # Calculate weights based on regime
            weights = self._get_regime_weights(regime)

            # Combine signals
            combined = {
                "entry_signal": self._combine_entry_signals(
                    ml_signals.get("direction"),
                    rule_signals.get("entry_signal"),
                    weights,
                ),
                "exit_signal": self._combine_exit_signals(
                    ml_signals.get("confidence"),
                    rule_signals.get("exit_signal"),
                    weights,
                ),
                "stop_loss": self._combine_stop_loss(
                    ml_signals.get("features"), rule_signals.get("stop_loss"), weights
                ),
                "take_profit": self._combine_take_profit(
                    ml_signals.get("features"), rule_signals.get("take_profit"), weights
                ),
                "position_size": self._calculate_position_size(
                    ml_signals.get("confidence"),
                    rule_signals.get("position_size"),
                    weights,
                ),
            }

            return combined

        except Exception as e:
            logger.error(f"Error combining signals: {str(e)}")
            return {}

    def _get_regime_weights(self, regime: str) -> Dict[str, float]:
        """Get signal weights based on market regime.

        Args:
            regime: Market regime

        Returns:
            Dictionary with weights for ML and rule signals
        """
        weights = {
            "trend": {"ml": 0.7, "rule": 0.3},
            "volatility": {"ml": 0.4, "rule": 0.6},
            "sideways": {"ml": 0.3, "rule": 0.7},
            "manipulation": {"ml": 0.2, "rule": 0.8},
        }
        return weights.get(regime, {"ml": 0.5, "rule": 0.5})

    def _combine_entry_signals(
        self,
        ml_direction: Optional[str],
        rule_signal: Optional[str],
        weights: Dict[str, float],
    ) -> str:
        """Combine entry signals from ML and rules.

        Args:
            ml_direction: ML prediction direction
            rule_signal: Rule-based entry signal
            weights: Signal weights

        Returns:
            Combined entry signal
        """
        try:
            if not ml_direction and not rule_signal:
                return "hold"

            if not ml_direction:
                return rule_signal or "hold"

            if not rule_signal:
                return ml_direction

            # If signals agree, use stronger signal
            if ml_direction == rule_signal:
                return ml_direction

            # If signals disagree, use weighted combination
            return f"({ml_direction} & {rule_signal})"

        except Exception as e:
            logger.error(f"Error combining entry signals: {str(e)}")
            return rule_signal or "hold"

    def _combine_exit_signals(
        self,
        ml_confidence: Optional[float],
        rule_signal: Optional[str],
        weights: Dict[str, float],
    ) -> str:
        """Combine exit signals from ML and rules.

        Args:
            ml_confidence: ML prediction confidence
            rule_signal: Rule-based exit signal
            weights: Signal weights

        Returns:
            Combined exit signal
        """
        try:
            if not rule_signal:
                return "ml_confidence < 0.5"

            # Add ML confidence check to rule signal
            return f"({rule_signal}) | (ml_confidence < 0.5)"

        except Exception as e:
            logger.error(f"Error combining exit signals: {str(e)}")
            return rule_signal or "hold"

    def _combine_stop_loss(
        self,
        ml_features: Optional[pd.DataFrame],
        rule_stop: Optional[str],
        weights: Dict[str, float],
    ) -> str:
        """Combine stop loss levels from ML and rules.

        Args:
            ml_features: ML feature DataFrame
            rule_stop: Rule-based stop loss
            weights: Signal weights

        Returns:
            Combined stop loss configuration
        """
        try:
            if not rule_stop:
                return "atr_trailing"

            # Use ATR-based stop loss with rule-based adjustments
            return f"atr_trailing & {rule_stop}"

        except Exception as e:
            logger.error(f"Error combining stop loss: {str(e)}")
            return rule_stop or "atr_trailing"

    def _combine_take_profit(
        self,
        ml_features: Optional[pd.DataFrame],
        rule_tp: Optional[str],
        weights: Dict[str, float],
    ) -> str:
        """Combine take profit levels from ML and rules.

        Args:
            ml_features: ML feature DataFrame
            rule_tp: Rule-based take profit
            weights: Signal weights

        Returns:
            Combined take profit configuration
        """
        try:
            if not rule_tp:
                return "risk_reward_ratio: 2.0"

            # Use risk-reward ratio with rule-based adjustments
            return f"risk_reward_ratio: 2.0 & {rule_tp}"

        except Exception as e:
            logger.error(f"Error combining take profit: {str(e)}")
            return rule_tp or "risk_reward_ratio: 2.0"

    def _calculate_position_size(
        self,
        ml_confidence: Optional[float],
        rule_size: Optional[float],
        weights: Dict[str, float],
    ) -> float:
        """Calculate position size based on ML confidence and rules.

        Args:
            ml_confidence: ML prediction confidence
            rule_size: Rule-based position size
            weights: Signal weights

        Returns:
            Combined position size
        """
        try:
            if not rule_size:
                return 1.0

            # Weight the position size by ML confidence
            return rule_size * (0.5 + (ml_confidence or 0.0) * 0.5)

        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return rule_size or 1.0

    def _get_market_data(self, pair: str, timeframe: str) -> pd.DataFrame:
        """Получение расширенных рыночных данных.

        Args:
            pair: Торговая пара
            timeframe: Таймфрейм

        Returns:
            DataFrame с расширенными рыночными данными
        """
        try:
            # Получение базовых данных
            market_data = (
                self.market_data.get_market_data(pair, timeframe)
                if hasattr(self, "market_data")
                else None
            )
            if market_data is None:
                return pd.DataFrame()

            order_book = (
                self.market_data.get_order_book(pair)
                if hasattr(self, "market_data")
                else None
            )

            # Расчет расширенных метрик
            df = pd.DataFrame(market_data)
            if df.empty:
                return df

            # Волатильность и ATR
            df["atr"] = calculate_atr(df, 14)
            df["volatility"] = df["close"].pct_change().rolling(20).std()

            # Объемные профили
            df["volume_delta"] = df["volume"].diff()
            df["volume_ma"] = df["volume"].rolling(20).mean()
            df["volume_ratio"] = df["volume"] / df["volume_ma"]

            # Структура рынка
            df["high_low_range"] = df["high"] - df["low"]
            df["close_open_range"] = df["close"] - df["open"]
            df["body_ratio"] = abs(df["close_open_range"]) / df["high_low_range"]

            # Импульс и моментум
            df["momentum"] = df["close"].pct_change(10)
            df["rsi"] = calculate_rsi(df["close"], 14)

            # Тренд и тренд-индикаторы
            df["ema_fast"] = calculate_ema(df["close"], 20)
            df["ema_medium"] = calculate_ema(df["close"], 50)
            df["ema_slow"] = calculate_ema(df["close"], 200)
            df["trend_strength"] = abs(df["ema_fast"] - df["ema_slow"]) / df["ema_slow"]

            # Ликвидность и спред
            if order_book:
                df["spread"] = (
                    order_book["asks"][0] - order_book["bids"][0]
                ) / order_book["asks"][0]
                df["liquidity_ratio"] = (
                    order_book["bids_volume"].sum() / order_book["asks_volume"].sum()
                )

            # Паттерны и формации
            df["doji"] = (
                abs(df["close"] - df["open"]) <= 0.1 * df["high_low_range"]
            ).astype(int)
            df["hammer"] = (
                (df["close"] > df["open"])
                & (df["low"] < df["open"] - 2 * (df["open"] - df["close"]))
            ).astype(int)

            # Экстремумы и уровни
            df["local_high"] = df["high"].rolling(20).max()
            df["local_low"] = df["low"].rolling(20).min()
            df["price_position"] = (df["close"] - df["local_low"]) / (
                df["local_high"] - df["local_low"]
            )

            # Временные метрики
            df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
            df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

            # Нормализация
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[
                numeric_cols
            ].std()
            # Обработка NaN
            df = df.fillna(0)
            return df

        except Exception as e:
            logger.error(f"Error getting market data for {pair}: {str(e)}")
            return pd.DataFrame()

    def _apply_strategy_rules(
        self, strategy: Dict[str, Any], market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Применение расширенных правил стратегии.

        Args:
            strategy: Конфигурация стратегии
            market_data: Рыночные данные

        Returns:
            Словарь с сигналами и метриками
        """
        try:
            if market_data.empty:
                return {}

            signals = {
                "entry": None,
                "exit": None,
                "stop_loss": None,
                "take_profit": None,
                "position_size": None,
                "confidence": 0.0,
                "risk_metrics": {},
                "market_context": {},
            }

            # Анализ тренда
            trend_strength = market_data["trend_strength"].iloc[-1]
            trend_direction = (
                "up"
                if market_data["ema_fast"].iloc[-1] > market_data["ema_slow"].iloc[-1]
                else "down"
            )

            # Анализ волатильности
            volatility = market_data["volatility"].iloc[-1]
            atr = market_data["atr"].iloc[-1]

            # Анализ объема
            volume_trend = market_data["volume_ratio"].iloc[-1]
            liquidity = market_data.get("liquidity_ratio", pd.Series([1.0])).iloc[-1]

            # Анализ импульса
            momentum = market_data["momentum"].iloc[-1]
            rsi = market_data["rsi"].iloc[-1]

            # Анализ паттернов
            doji = market_data["doji"].iloc[-1]
            hammer = market_data["hammer"].iloc[-1]

            # Анализ уровней
            price_position = market_data["price_position"].iloc[-1]

            # Расчет сигналов входа
            entry_conditions = []
            if trend_strength > 0.5:
                if trend_direction == "up" and rsi < 30:
                    entry_conditions.append("trend_reversal_long")
                elif trend_direction == "down" and rsi > 70:
                    entry_conditions.append("trend_reversal_short")

            if volume_trend > 1.5 and liquidity > 1.2:
                if momentum > 0 and price_position < 0.3:
                    entry_conditions.append("volume_breakout_long")
                elif momentum < 0 and price_position > 0.7:
                    entry_conditions.append("volume_breakout_short")

            if doji or hammer:
                if price_position < 0.2:
                    entry_conditions.append("pattern_reversal_long")
                elif price_position > 0.8:
                    entry_conditions.append("pattern_reversal_short")

            # Расчет сигналов выхода
            exit_conditions = []
            if trend_strength < 0.3:
                exit_conditions.append("trend_weakening")
            if volume_trend < 0.5:
                exit_conditions.append("volume_drying")
            if abs(momentum) > 0.05:
                exit_conditions.append("momentum_reversal")

            # Выбор наиболее сильного entry_condition
            if entry_conditions:
                # Выбор по максимальному тренду или объему
                best_entry = max(
                    entry_conditions,
                    key=lambda cond: abs(trend_strength) + abs(volume_trend),
                )
                atr_multiplier = 2.0 if volatility > 0.02 else 1.5
                risk_reward = 2.0 if trend_strength > 0.7 else 1.5
                current_price = market_data["close"].iloc[-1]
                stop_loss = (
                    current_price - (atr * atr_multiplier)
                    if "long" in best_entry
                    else current_price + (atr * atr_multiplier)
                )
                take_profit = (
                    current_price + (abs(current_price - stop_loss) * risk_reward)
                    if "long" in best_entry
                    else current_price - (abs(current_price - stop_loss) * risk_reward)
                )
                signals.update(
                    {
                        "entry": best_entry,
                        "exit": exit_conditions[0] if exit_conditions else None,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "position_size": min(1.0, 1.0 / volatility),
                        "confidence": min(0.9, trend_strength * (1 + volume_trend) / 2),
                        "risk_metrics": {
                            "volatility": volatility,
                            "liquidity": liquidity,
                            "trend_strength": trend_strength,
                        },
                        "market_context": {
                            "trend": trend_direction,
                            "momentum": momentum,
                            "volume_trend": volume_trend,
                            "price_position": price_position,
                        },
                    }
                )

            return signals

        except Exception as e:
            logger.error(f"Error applying strategy rules: {str(e)}")
            return {}

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Расчет индикаторов"""
        if data.empty:
            return data

        # ATR
        data["atr"] = calculate_atr(data)

        # RSI
        data["rsi"] = calculate_rsi(data["close"])

        # Волатильность
        data["volatility"] = calculate_volatility(data["close"])

        return data

    def update_strategy_weights(self, performance: Dict[str, float]) -> None:
        """
        Обновление весов стратегий.

        Args:
            performance: Словарь с производительностью стратегий
        """
        try:
            if not performance:
                return

            total_performance = float(sum(performance.values()))
            if total_performance == 0:
                return

            for strategy_name, perf in performance.items():
                if strategy_name in self.strategy_weights:
                    self.strategy_weights[strategy_name] = float(
                        perf / total_performance
                    )

        except Exception as e:
            logger.error(f"Error updating strategy weights: {str(e)}")

    def get_best_strategy(self, data: pd.DataFrame) -> Optional[Callable]:
        """
        Получение лучшей стратегии для текущих рыночных условий.

        Args:
            data: Данные для анализа

        Returns:
            Optional[Callable]: Лучшая стратегия или None
        """
        try:
            if data.empty:
                return None

            # Генерируем гибридную стратегию
            hybrid = self.generate_hybrid_strategy(data)
            if hybrid:
                return hybrid

            # Если гибридная стратегия не сгенерирована, выбираем лучшую базовую
            return self._select_best_strategy(self.base_strategies, data)

        except Exception as e:
            logger.error(f"Error getting best strategy: {str(e)}")
            return None
