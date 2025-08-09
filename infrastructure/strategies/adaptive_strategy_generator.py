from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from shared.numpy_utils import np
import pandas as pd
from loguru import logger
from decimal import Decimal

from domain.services.technical_analysis import DefaultTechnicalAnalysisService
from domain.type_definitions.strategy_types import (
    MarketContext,
    MarketRegime,
    Signal as DomainSignal,
    StrategyAnalysis,
    StrategyDirection,
    StrategyMetrics,
    StrategyType,
)
from infrastructure.strategies.base_strategy import Signal
from infrastructure.ml_services.advanced_price_predictor import MetaLearner
from infrastructure.ml_services.decision_reasoner import DecisionReasoner
from infrastructure.ml_services.pattern_discovery import PatternDiscovery
from infrastructure.strategies.base_strategy import BaseStrategy

# Импорты стратегий (заглушки для совместимости)
def reversal_strategy_fibo_pinbar(data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    """Заглушка для стратегии разворота"""
    return {"signal": "hold", "confidence": 0.5}

def sideways_strategy_bb_rsi(data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    """Заглушка для боковой стратегии"""
    return {"signal": "hold", "confidence": 0.5}

def trend_strategy_ema_macd(data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    """Заглушка для трендовой стратегии"""
    return {"signal": "hold", "confidence": 0.5}

def volatility_strategy_atr_breakout(data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    """Заглушка для волатильной стратегии"""
    return {"signal": "hold", "confidence": 0.5}


@dataclass
class AdaptationConfig:
    """Конфигурация адаптации"""

    # Параметры адаптации
    adaptation_threshold: float = 0.7
    learning_rate: float = 0.01
    memory_size: int = 1000
    adaptation_frequency: int = 100
    regime_detection_sensitivity: float = 0.8
    # Параметры управления рисками
    risk_per_trade: float = 0.02
    max_position_size: float = 0.2
    # Общие параметры
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=lambda: ["1h"])
    log_dir: str = "logs/adaptive"


class AdaptiveStrategyGenerator(BaseStrategy):
    """Генератор адаптивных стратегий"""

    def __init__(
        self,
        market_regime_agent: Any,  # Заглушка для MarketRegimeAgent
        meta_learner: MetaLearner,
        backtest_results: Dict[str, Dict],
        base_strategies: List[Callable],
        config: Optional[Union[Dict[str, Any], AdaptationConfig]] = None,
    ) -> None:
        # Преобразуем конфигурацию в объект AdaptationConfig
        if isinstance(config, AdaptationConfig):
            config_obj = config
        elif isinstance(config, dict):
            config_obj = AdaptationConfig(**config)
        else:
            config_obj = AdaptationConfig()
        
        super().__init__(config_obj.__dict__)
        
        self.config: dict[str, Any] = config_obj.__dict__
        self._config_obj: AdaptationConfig = config_obj
        self.market_regime_agent = market_regime_agent
        self.meta_learner = meta_learner
        self.backtest_results = backtest_results or {}
        self.base_strategies = base_strategies or []
        self.strategy_weights: Dict[str, float] = {
            strategy.__name__: 1.0 for strategy in self.base_strategies
        }
        # Инициализация технического анализа
        self.technical_analysis = DefaultTechnicalAnalysisService()
        # Маппинг режимов на стратегии
        self.regime_strategies: Dict[str, List[Callable]] = {
            "trend": [trend_strategy_ema_macd],
            "sideways": [sideways_strategy_bb_rsi],
            "reversal": [reversal_strategy_fibo_pinbar],
            "volatility": [volatility_strategy_atr_breakout],
        }
        self.pattern_discovery = PatternDiscovery()
        self.decision_reasoner = DecisionReasoner()
        self.strategy_cache: Dict[str, Any] = {}

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            # Валидация данных
            is_valid, error_msg = self.validate_data(data)
            if not is_valid:
                raise ValueError(f"Invalid data: {error_msg}")
            # Определение рыночного режима
            market_regime = self._detect_market_regime(data)
            # Анализ рыночного контекста
            market_context = self._analyze_market_context(data, market_regime)
            # Получение ML предсказаний
            ml_predictions = self._get_ml_predictions(data)
            # Анализ паттернов
            patterns_result = self.pattern_discovery.discover_patterns(data)
            # discover_patterns возвращает None, поэтому используем пустой список
            patterns: List[Dict[str, Any]] = [] if patterns_result is None else patterns_result
            # Генерация адаптивных сигналов
            signals = self._generate_adaptive_signals(
                data, market_regime, ml_predictions
            )
            # Расчет метрик
            metrics = self._calculate_adaptive_metrics(
                data, market_regime, ml_predictions
            )
            # Оценка риска
            risk_assessment = self._assess_adaptive_risk(
                data, market_context, ml_predictions
            )
            # Рекомендации
            recommendations = self._generate_adaptive_recommendations(
                data, market_regime, ml_predictions, patterns
            )
            # Исправление: правильное преобразование в dict
            indicators_data = self._calculate_indicators(data)
            indicators_dict: Dict[str, pd.Series] = {}
            if hasattr(indicators_data, 'to_dict'):
                indicators_dict = indicators_data.to_dict()
            elif isinstance(indicators_data, dict):
                indicators_dict = indicators_data
            else:
                # Fallback: создаем пустой словарь
                indicators_dict = {}
            
            result = StrategyAnalysis(
                strategy_id=f"adaptive_{id(self)}",
                timestamp=datetime.now(),
                market_data=data,
                indicators=indicators_dict,
                signals=signals,  # Теперь signals уже правильного типа
                metrics=metrics,
                market_regime=market_regime,
                confidence=self._calculate_adaptive_confidence(
                    data, market_regime, ml_predictions
                ),
                risk_assessment=risk_assessment,
                recommendations=recommendations,
                metadata={
                    "market_context": market_context,
                    "ml_predictions": ml_predictions,
                    "patterns": patterns,
                    "adaptation_config": self._config_obj.__dict__,
                },
                strategy_type=StrategyType.ADAPTIVE,
                direction=StrategyDirection.HOLD,
                entry_price=0.0,
                exit_price=0.0,
                stop_loss=0.0,
                take_profit=0.0,
            )
            return result.__dict__
        except Exception as e:
            logger.error(f"Error in adaptive analysis: {str(e)}")
            raise

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        try:
            # Определение рыночного режима
            market_regime = self._detect_market_regime(data)
            # Получение ML предсказаний
            ml_predictions = self._get_ml_predictions(data)
            # Выбор лучшей стратегии для режима
            best_strategy = self._select_best_strategy_for_regime(market_regime, data)
            if not best_strategy:
                return Signal(
                    id="no_strategy_signal",
                    symbol="BTC/USDT",
                    signal_type="hold",
                    confidence=Decimal("0.5"),
                    price=Decimal("50000.0"),
                    amount=Decimal("0.0"),
                    created_at=datetime.now()
                )
            # Генерация базового сигнала
            base_signal = self._generate_base_signal(best_strategy, data)
            if not base_signal:
                return Signal(
                    id="no_base_signal",
                    symbol="BTC/USDT",
                    signal_type="hold",
                    confidence=Decimal("0.5"),
                    price=Decimal("50000.0"),
                    amount=Decimal("0.0"),
                    created_at=datetime.now()
                )
            # Адаптация сигнала
            adapted_signal = self._adapt_signal(base_signal, ml_predictions, market_regime)
            # Проверка условий адаптации
            if not self._check_adaptation_conditions(adapted_signal, market_regime):
                return Signal(
                    id="adaptation_failed_signal",
                    symbol="BTC/USDT",
                    signal_type="hold",
                    confidence=Decimal("0.5"),
                    price=Decimal("50000.0"),
                    amount=Decimal("0.0"),
                    created_at=datetime.now()
                )
            return adapted_signal
        except Exception as e:
            logger.error(f"Error generating adaptive signal: {str(e)}")
            return Signal(
                id="error_signal",
                symbol="BTC/USDT",
                signal_type="hold",
                confidence=Decimal("0.5"),
                price=Decimal("50000.0"),
                amount=Decimal("0.0"),
                created_at=datetime.now()
            )

    def _detect_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """Определение рыночного режима с использованием агента"""
        try:
            if self.market_regime_agent:
                regime_prediction = self.market_regime_agent.predict_regime(data)
                return MarketRegime(regime_prediction)
            else:
                # Fallback логика
                return self._fallback_regime_detection(data)
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return MarketRegime.SIDEWAYS

    def _fallback_regime_detection(self, data: pd.DataFrame) -> MarketRegime:
        """Резервное определение режима"""
        try:
            # Безопасная индексация DataFrame
            try:
                if len(data) >= 20 and "close" in data.columns:
                    close_prices = data.iloc[-20:]["close"]
                else:
                    return MarketRegime.SIDEWAYS
            except (AttributeError, IndexError, KeyError):
                return MarketRegime.SIDEWAYS
            
            # Безопасный анализ тренда
            if len(close_prices) >= 2:
                try:
                    trend = (float(close_prices.iloc[-1]) - float(close_prices.iloc[0])) / float(close_prices.iloc[0])
                    volatility = close_prices.pct_change().std()
                    
                    if abs(trend) > 0.05:  # 5% тренд
                        return MarketRegime.TRENDING_UP if trend > 0 else MarketRegime.TRENDING_DOWN
                    elif volatility > 0.03:  # 3% волатильность
                        return MarketRegime.VOLATILE
                    else:
                        return MarketRegime.SIDEWAYS
                except (IndexError, TypeError, ZeroDivisionError):
                    return MarketRegime.SIDEWAYS
            else:
                return MarketRegime.SIDEWAYS
        except Exception as e:
            logger.error(f"Error in fallback regime detection: {str(e)}")
            return MarketRegime.SIDEWAYS

    def _analyze_market_context(
        self, data: pd.DataFrame, regime: MarketRegime
    ) -> MarketContext:
        """Анализ рыночного контекста"""
        try:
            # Безопасная индексация DataFrame
            current_price = 0.0
            volume = 0.0
            
            try:
                if len(data) > 0 and "close" in data.columns and "volume" in data.columns:
                    current_price = float(data.iloc[-1]["close"])
                    volume = float(data.iloc[-1]["volume"])
            except (IndexError, KeyError, TypeError):
                pass
            
            return MarketContext(
                regime=regime,
                volatility=0.0,
                trend_strength=0.0,
                volume_profile={},
                liquidity_conditions={},
                market_sentiment=0.0,
                correlation_matrix=pd.DataFrame(),
                timestamp=datetime.now(),
            )
        except Exception as e:
            logger.error(f"Error analyzing market context: {str(e)}")
            return MarketContext(
                regime=regime,
                volatility=0.0,
                trend_strength=0.0,
                volume_profile={},
                liquidity_conditions={},
                market_sentiment=0.0,
                correlation_matrix=pd.DataFrame(),
                timestamp=datetime.now(),
            )

    def _get_ml_predictions(self, data: pd.DataFrame) -> Dict[str, float]:
        """Получение ML предсказаний"""
        try:
            if self.meta_learner:
                features = self._extract_features(data)
                predictions = self.meta_learner.predict(features)
                return predictions
            else:
                return {}
        except Exception as e:
            logger.error(f"Error getting ML predictions: {str(e)}")
            return {}

    def _extract_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Извлечение признаков для ML"""
        features: Dict[str, float] = {}
        try:
            # Безопасное извлечение признаков
            if len(data) == 0:
                return features
                
            # RSI
            try:
                rsi_result = self.technical_analysis.calculate_rsi(data["close"], 14)
                if hasattr(rsi_result, 'iloc') and len(rsi_result) > 0:
                    features["rsi"] = float(rsi_result.iloc[-1])
                else:
                    features["rsi"] = 50.0
            except (AttributeError, IndexError, KeyError):
                features["rsi"] = 50.0
            
            # MACD
            try:
                macd_result = self.technical_analysis.calculate_macd(data["close"])
                if hasattr(macd_result, 'iloc') and len(macd_result) > 0:
                    features["macd"] = float(macd_result.iloc[-1])
                else:
                    features["macd"] = 0.0
            except (AttributeError, IndexError, KeyError):
                features["macd"] = 0.0
            
            # Bollinger Bands
            try:
                bb_result = self.technical_analysis.calculate_bollinger_bands(data["close"])
                if hasattr(bb_result, 'iloc') and len(bb_result) > 0:
                    features["bb_position"] = float(bb_result.iloc[-1])
                else:
                    features["bb_position"] = 0.5
            except (AttributeError, IndexError, KeyError):
                features["bb_position"] = 0.5
            
            # Волатильность
            try:
                if len(data) > 20 and "close" in data.columns:
                    pct_change = data["close"].pct_change()
                    if len(pct_change) > 20:
                        rolling_std = pct_change.rolling(20).std()
                        if len(rolling_std) > 0:
                            features["volatility"] = float(rolling_std.iloc[-1])
                        else:
                            features["volatility"] = 0.0
                    else:
                        features["volatility"] = 0.0
                else:
                    features["volatility"] = 0.0
            except (AttributeError, IndexError):
                features["volatility"] = 0.0
            
            # Моментум
            try:
                if len(data) > 10 and "close" in data.columns:
                    pct_change_10 = data["close"].pct_change(10)
                    if len(pct_change_10) > 0:
                        features["momentum"] = float(pct_change_10.iloc[-1])
                    else:
                        features["momentum"] = 0.0
                else:
                    features["momentum"] = 0.0
            except (AttributeError, IndexError):
                features["momentum"] = 0.0
            
            # Объем
            try:
                if len(data) > 20 and "volume" in data.columns:
                    current_volume = data["volume"].iloc[-1]
                    avg_volume = data["volume"].rolling(20).mean().iloc[-1]
                    if avg_volume > 0:
                        features["volume_ratio"] = float(current_volume / avg_volume)
                    else:
                        features["volume_ratio"] = 1.0
                else:
                    features["volume_ratio"] = 1.0
            except (AttributeError, IndexError, ZeroDivisionError):
                features["volume_ratio"] = 1.0
            
            # EMA
            try:
                ema_20_result = self.technical_analysis.calculate_ema(data["close"], 20)
                ema_50_result = self.technical_analysis.calculate_ema(data["close"], 50)
                
                if (hasattr(ema_20_result, 'iloc') and hasattr(ema_50_result, 'iloc') and 
                    len(ema_20_result) > 0 and len(ema_50_result) > 0):
                    ema_20_val = ema_20_result.iloc[-1]
                    ema_50_val = ema_50_result.iloc[-1]
                    if ema_50_val > 0:
                        features["trend_strength"] = float(abs(ema_20_val - ema_50_val) / ema_50_val)
                    else:
                        features["trend_strength"] = 0.0
                else:
                    features["trend_strength"] = 0.0
            except (AttributeError, IndexError, ZeroDivisionError):
                features["trend_strength"] = 0.0
            # Удаляем дублирующий код, так как он уже обработан выше
            return features
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return {}

    def _generate_adaptive_signals(
        self, data: pd.DataFrame, regime: MarketRegime, ml_predictions: Dict[str, float]
    ) -> List[Signal]:
        """Генерация адаптивных сигналов"""
        signals: List[Signal] = []
        try:
            # Выбор стратегии для режима
            strategy = self._select_best_strategy_for_regime(regime, data)
            if not strategy:
                return signals
            # Генерация базового сигнала
            base_signal = self._generate_base_signal(strategy, data)
            if base_signal:
                # Адаптация сигнала
                adapted_signal = self._adapt_signal(base_signal, ml_predictions, regime)
                if adapted_signal:
                    signals.append(adapted_signal)
            return signals
        except Exception as e:
            logger.error(f"Error generating adaptive signals: {str(e)}")
            return signals

    def _select_best_strategy_for_regime(
        self, regime: MarketRegime, data: pd.DataFrame
    ) -> Optional[Callable]:
        """Выбор лучшей стратегии для режима"""
        try:
            regime_strategies = self.regime_strategies.get(regime.value, [])
            if not regime_strategies:
                return None
            best_strategy = None
            best_score = -np.inf
            for strategy in regime_strategies:
                if strategy is not None:
                    strategy_name = strategy.__name__
                    if strategy_name in self.backtest_results:
                        score = self._calculate_strategy_score(
                            self.backtest_results[strategy_name]
                        )
                        if score > best_score:
                            best_score = score
                            best_strategy = strategy
            return (
                best_strategy
                if best_score > self._config_obj.adaptation_threshold
                else None
            )
        except Exception as e:
            logger.error(f"Error selecting best strategy: {str(e)}")
            return None

    def _generate_base_signal(
        self, strategy: Callable, data: pd.DataFrame
    ) -> Optional[Signal]:
        """Генерация базового сигнала от стратегии"""
        try:
            # Вызов стратегии
            result = strategy(data)
            if not result:
                return None
            # Преобразование результата в Signal
            return self._convert_to_signal(result, data)
        except Exception as e:
            logger.error(f"Error generating base signal: {str(e)}")
            return None

    def _convert_to_signal(
        self, result: Dict[str, Any], data: pd.DataFrame
    ) -> Optional[Signal]:
        """Преобразование результата стратегии в Signal"""
        try:
            if not result or "direction" not in result:
                return None
            direction = StrategyDirection(result["direction"])
            entry_price = float(result.get("entry_price", data["close"].iloc[-1]))  # type: ignore[index]
            return Signal(
                direction=direction,
                entry_price=entry_price,
                stop_loss=result.get("stop_loss"),
                take_profit=result.get("take_profit"),
                confidence=result.get("confidence", 0.5),
                # strategy_type=StrategyType.ADAPTIVE,  # type: ignore[call-arg]
                # market_regime=MarketRegime.SIDEWAYS,  # Will be updated  # type: ignore[call-arg]
                # risk_score=result.get("risk_score", 0.5),  # type: ignore[call-arg]
                # expected_return=result.get("expected_return", 0.0),  # type: ignore[call-arg]
            )
        except Exception as e:
            logger.error(f"Error converting to signal: {str(e)}")
            return None

    def _adapt_signal(
        self,
        base_signal: Signal,
        ml_predictions: Dict[str, float],
        regime: MarketRegime,
    ) -> Signal:
        """Адаптация сигнала на основе ML предсказаний"""
        try:
            # Адаптация направления
            if "direction_confidence" in ml_predictions:
                ml_confidence = ml_predictions["direction_confidence"]
                base_signal.confidence = (base_signal.confidence + ml_confidence) / 2
            # Адаптация размера позиции
            if "position_size" in ml_predictions:
                # Исправление: position_size не является атрибутом Signal
                pass
            # Адаптация стоп-лосса
            if "stop_loss_adjustment" in ml_predictions:
                adjustment = ml_predictions["stop_loss_adjustment"]
                if base_signal.stop_loss:
                    base_signal.stop_loss *= 1 + adjustment
            # Адаптация тейк-профита
            if "take_profit_adjustment" in ml_predictions:
                adjustment = ml_predictions["take_profit_adjustment"]
                if base_signal.take_profit:
                    base_signal.take_profit *= 1 + adjustment
            # Обновление рыночного режима
            # base_signal.market_regime = regime  # type: ignore[attr-defined]
            return base_signal
        except Exception as e:
            logger.error(f"Error adapting signal: {str(e)}")
            return base_signal

    def _check_adaptation_conditions(
        self, signal: Signal, regime: MarketRegime
    ) -> bool:
        """Проверка условий адаптации"""
        try:
            # Проверка уверенности
            if signal.confidence < self._config_obj.adaptation_threshold:
                return False
            # Проверка риска
            if hasattr(signal, 'risk_score') and signal.risk_score > 0.8:  # type: ignore[attr-defined]
                return False
            # Проверка ожидаемой доходности
            if hasattr(signal, 'expected_return') and signal.expected_return < 0.01:  # type: ignore[attr-defined]
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking adaptation conditions: {str(e)}")
            return False

    def _calculate_adaptive_metrics(
        self, data: pd.DataFrame, regime: MarketRegime, ml_predictions: Dict[str, float]
    ) -> StrategyMetrics:
        """Расчет адаптивных метрик"""
        try:
            # Базовые метрики
            volatility = data["close"].pct_change().rolling(20).std().iloc[-1]
            # Адаптивные метрики
            adaptation_score = self._calculate_adaptation_score(
                data, regime, ml_predictions
            )
            ml_confidence = ml_predictions.get("confidence", 0.5)
            return StrategyMetrics(
                volatility=volatility,
                additional={
                    "adaptation_score": adaptation_score,
                    "ml_confidence": ml_confidence,
                    "regime_stability": self._calculate_regime_stability(data, regime),
                },
            )
        except Exception as e:
            logger.error(f"Error calculating adaptive metrics: {str(e)}")
            return StrategyMetrics()

    def _calculate_adaptation_score(
        self, data: pd.DataFrame, regime: MarketRegime, ml_predictions: Dict[str, float]
    ) -> float:
        """Расчет оценки адаптации"""
        try:
            # Факторы адаптации
            regime_consistency = (
                1.0 if regime == self._detect_market_regime(data) else 0.5
            )
            ml_confidence = ml_predictions.get("confidence", 0.5)
            # Оценка качества данных
            data_quality = 0.9
            try:
                if hasattr(data, 'isnull') and hasattr(data.isnull(), 'to_numpy'):
                    data_quality = 0.9 if not np.any(data.isnull().to_numpy()) else 0.6  # type: ignore[attr-defined]
            except (AttributeError, TypeError):
                data_quality = 0.9
            adaptation_score = (regime_consistency + ml_confidence + data_quality) / 3
            return max(0.0, min(1.0, adaptation_score))
        except Exception as e:
            logger.error(f"Error calculating adaptation score: {str(e)}")
            return 0.5

    def _calculate_regime_stability(
        self, data: pd.DataFrame, regime: MarketRegime
    ) -> float:
        """Расчет стабильности режима"""
        try:
            # Анализ стабильности режима на исторических данных
            # Исправление: используем правильные методы DataFrame
            if hasattr(data, 'tail'):
                recent_data = data.tail(50)
            else:
                recent_data = data[-50:] if len(data) >= 50 else data
            
            regime_changes = 0
            for i in range(1, len(recent_data)):
                if hasattr(recent_data, 'iloc'):
                    current_regime = self._detect_market_regime(recent_data.iloc[:i+1])
                else:
                    current_regime = self._detect_market_regime(recent_data[:i+1])
                if current_regime != regime:
                    regime_changes += 1
            stability = 1.0 - (regime_changes / len(recent_data))
            return max(0.0, min(1.0, stability))
        except Exception as e:
            logger.error(f"Error calculating regime stability: {str(e)}")
            return 0.5

    def _assess_adaptive_risk(
        self,
        data: pd.DataFrame,
        market_context: MarketContext,
        ml_predictions: Dict[str, float],
    ) -> Dict[str, float]:
        """Оценка адаптивного риска"""
        try:
            risk_assessment = {}
            # Рыночный риск
            risk_assessment["market_risk"] = float(
                data["close"].pct_change().rolling(20).std().iloc[-1] * 10
            )
            # Риск ML модели
            risk_assessment["ml_risk"] = 1.0 - ml_predictions.get("confidence", 0.5)
            # Риск адаптации
            risk_assessment["adaptation_risk"] = 1.0 - self._calculate_adaptation_score(
                data, MarketRegime.SIDEWAYS, ml_predictions  # Исправление: передаем MarketRegime
            )
            # Риск режима
            risk_assessment["regime_risk"] = 1.0 - self._calculate_regime_stability(
                data, MarketRegime.SIDEWAYS  # Исправление: передаем MarketRegime
            )
            # Общий риск
            risk_assessment["total_risk"] = (
                risk_assessment["market_risk"] * 0.3
                + risk_assessment["ml_risk"] * 0.3
                + risk_assessment["adaptation_risk"] * 0.2
                + risk_assessment["regime_risk"] * 0.2
            )
            return risk_assessment
        except Exception as e:
            logger.error(f"Error assessing adaptive risk: {str(e)}")
            return {"total_risk": 0.5}

    def _calculate_adaptive_confidence(
        self, data: pd.DataFrame, regime: MarketRegime, ml_predictions: Dict[str, float]
    ) -> float:
        """Расчет адаптивной уверенности"""
        try:
            # Факторы уверенности
            ml_confidence = ml_predictions.get("confidence", 0.5)
            regime_stability = self._calculate_regime_stability(data, regime)
            # Исправление: используем правильные методы DataFrame
            if hasattr(data, 'isnull') and hasattr(data.isnull(), 'values'):
                data_quality = 0.9 if not np.any(data.isnull().values) else 0.6  # type: ignore[attr-defined]
            else:
                data_quality = 0.9  # Fallback
            # Взвешенная уверенность
            confidence = (
                ml_confidence * 0.5 + regime_stability * 0.3 + data_quality * 0.2
            )
            return max(0.1, min(1.0, confidence))
        except Exception as e:
            logger.error(f"Error calculating adaptive confidence: {str(e)}")
            return 0.5

    def _generate_adaptive_recommendations(
        self,
        data: pd.DataFrame,
        regime: MarketRegime,
        ml_predictions: Dict[str, float],
        patterns: List[Dict[str, Any]],
    ) -> List[str]:
        """Генерация адаптивных рекомендаций"""
        recommendations = []
        try:
            # Рекомендации по режиму
            if regime == MarketRegime.TRENDING_UP:
                recommendations.append(
                    "Рынок в восходящем тренде - используйте трендовые стратегии"
                )
            elif regime == MarketRegime.TRENDING_DOWN:
                recommendations.append(
                    "Рынок в нисходящем тренде - используйте стратегии коротких позиций"
                )
            elif regime == MarketRegime.SIDEWAYS:
                recommendations.append(
                    "Рынок в боковом движении - используйте range trading стратегии"
                )
            elif regime == MarketRegime.VOLATILE:
                recommendations.append(
                    "Высокая волатильность - используйте стратегии с широкими стопами"
                )
            # Рекомендации по ML
            ml_confidence = ml_predictions.get("confidence", 0.5)
            if ml_confidence > 0.8:
                recommendations.append(
                    "Высокая уверенность ML модели - можно увеличить размер позиции"
                )
            elif ml_confidence < 0.3:
                recommendations.append(
                    "Низкая уверенность ML модели - используйте консервативные настройки"
                )
            # Рекомендации по паттернам
            if patterns:
                recommendations.append(
                    f"Обнаружено {len(patterns)} паттернов - учитывайте в торговле"
                )
            return recommendations
        except Exception as e:
            logger.error(f"Error generating adaptive recommendations: {str(e)}")
            return ["Ошибка в генерации рекомендаций"]

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
            try:
                # Исправление: правильный вызов predict для MetaLearner
                if hasattr(self.meta_learner, 'predict'):
                    meta_predictions = self.meta_learner.predict(
                        symbol=symbol, timeframe=timeframe, regime=regime
                    ) or {}
                else:
                    meta_predictions = {}
            except Exception as e:
                logger.error(f"Error getting meta predictions: {str(e)}")
                meta_predictions = {}
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
        return best_strategy if best_score > self._config_obj.adaptation_threshold else None

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
        if strategy is None:
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
        """Расчет confidence score для стратегии"""
        try:
            # Базовый confidence
            base_confidence = 0.5
            # Корректировка на основе мета-предсказаний
            if "confidence" in meta_predictions:
                base_confidence = meta_predictions["confidence"]
            # Корректировка на основе режима
            regime_confidence = {
                "trend": 0.7,
                "sideways": 0.6,
                "reversal": 0.5,
                "volatility": 0.4,
            }.get(regime, 0.5)
            # Финальный confidence
            final_confidence = (base_confidence + regime_confidence) / 2
            return float(max(0.0, min(1.0, final_confidence)))  # Исправление: явное приведение к float
        except Exception as e:
            logger.error(f"Error calculating confidence score: {str(e)}")
            return 0.5

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
            features = self._extract_features(market_data)
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
                self.meta_learner.get_strategy_for_regime(regime)
                if self.meta_learner
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
        """Получение рыночных данных с расширенными индикаторами"""
        try:
            # Заглушка для получения данных
            df = pd.DataFrame({
                "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="1H"),
                "open": np.random.randn(100).cumsum() + 100,
                "high": np.random.randn(100).cumsum() + 102,
                "low": np.random.randn(100).cumsum() + 98,
                "close": np.random.randn(100).cumsum() + 100,
                "volume": np.random.randint(1000, 10000, 100),
            })
            # Технические индикаторы
            df["ema_fast"] = self.technical_analysis.calculate_ema(df["close"], 12)
            df["ema_slow"] = self.technical_analysis.calculate_ema(df["close"], 26)
            df["rsi"] = self.technical_analysis.calculate_rsi(df["close"], 14)
            df["atr"] = self.technical_analysis.calculate_atr(df, 14)
            # Дополнительные метрики
            df["trend_strength"] = abs(df["ema_fast"] - df["ema_slow"]) / df["ema_slow"]
            df["volatility"] = df["close"].pct_change().rolling(20).std()
            df["momentum"] = df["close"].pct_change(10)
            df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
            df["high_low_range"] = df["high"] - df["low"]
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
            # Исправление: правильная обработка timestamp
            if hasattr(df["timestamp"], 'dt'):
                df["hour"] = df["timestamp"].dt.hour
                df["day_of_week"] = df["timestamp"].dt.dayofweek
            else:
                # Fallback для других типов timestamp
                timestamp_series = pd.to_datetime(df["timestamp"])
                if hasattr(timestamp_series, 'dt'):
                    df["hour"] = timestamp_series.dt.hour
                    df["day_of_week"] = timestamp_series.dt.dayofweek
                else:
                    # Fallback если нет атрибута dt
                    df["hour"] = 12  # По умолчанию
                    df["day_of_week"] = 0  # По умолчанию
            
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
            # Нормализация
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[
                numeric_cols
            ].std()
            # Обработка NaN
            try:
                if hasattr(df, 'fillna'):
                    df = df.fillna(0)  # type: ignore[attr-defined]
                else:
                    # Fallback для обработки бесконечных значений
                    df = df.fillna(0)  # type: ignore[attr-defined]
            except (AttributeError, TypeError):
                # Если fillna недоступен, используем numpy
                df = df.replace([np.inf, -np.inf], np.nan)  # type: ignore[attr-defined]
                df = df.fillna(0)  # type: ignore[attr-defined]
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
            signals: Dict[str, Any] = {
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
            trend_strength = market_data["trend_strength"].iloc[-1]  # type: ignore[index]
            trend_direction = (
                "up"
                if market_data["ema_fast"].iloc[-1] > market_data["ema_slow"].iloc[-1]  # type: ignore[index]
                else "down"
            )
            # Анализ волатильности
            volatility = market_data["volatility"].iloc[-1]  # type: ignore[index]
            atr = market_data["atr"].iloc[-1]  # type: ignore[index]
            # Анализ объема
            volume_trend = market_data["volume_ratio"].iloc[-1]  # type: ignore[index]
            liquidity = market_data.get("liquidity_ratio", pd.Series([1.0])).iloc[-1]
            # Анализ импульса
            momentum = market_data["momentum"].iloc[-1]  # type: ignore[index]
            rsi = market_data["rsi"].iloc[-1]  # type: ignore[index]
            # Анализ паттернов
            doji = market_data["doji"].iloc[-1]  # type: ignore[index]
            hammer = market_data["hammer"].iloc[-1]  # type: ignore[index]
            # Анализ уровней
            price_position = market_data["price_position"].iloc[-1]  # type: ignore[index]
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
                current_price = market_data["close"].iloc[-1]  # type: ignore[index]
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
                            "volatility": float(volatility),
                            "liquidity": float(liquidity),
                            "trend_strength": float(trend_strength),
                        },
                        "market_context": {
                            "trend": str(trend_direction),
                            "momentum": float(momentum),
                            "volume_trend": float(volume_trend),
                            "price_position": float(price_position),
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
        # ATR - исправление: передаем DataFrame
        data["atr"] = self.technical_analysis.calculate_atr(data, 14)
        # RSI
        data["rsi"] = self.technical_analysis.calculate_rsi(data["close"], 14)
        # Волатильность
        data["volatility"] = data["close"].pct_change().rolling(20).std()
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
            # Генерируем гибридную стратегию - исправление: добавляем недостающие параметры
            hybrid = self.generate_hybrid_strategy("BTCUSDT", "1h", "trend")
            if hybrid:
                return lambda x: hybrid  # Возвращаем callable
            # Если гибридная стратегия не сгенерирована, выбираем лучшую базовую
            return self._select_best_strategy(self.base_strategies, "BTCUSDT", "1h")
        except Exception as e:
            logger.error(f"Error getting best strategy: {str(e)}")
            return None
