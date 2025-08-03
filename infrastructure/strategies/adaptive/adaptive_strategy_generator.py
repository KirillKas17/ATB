"""
Упрощенный генератор адаптивных стратегий
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from loguru import logger

from domain.types.strategy_types import (
    MarketRegime,
    StrategyAnalysis,
    StrategyDirection,
    StrategyMetrics,
    StrategyType,
)
from infrastructure.strategies.base_strategy import BaseStrategy, Signal

from .market_regime_detector import MarketRegimeDetector
from .ml_signal_processor import MLSignalProcessor
from .strategy_selector import StrategySelector


@dataclass
class AdaptationConfig:
    """Конфигурация адаптации"""

    # Параметры адаптации
    adaptation_threshold: float = 0.1
    learning_rate: float = 0.01
    max_iterations: int = 100
    convergence_tolerance: float = 1e-6
    
    # Параметры ML
    feature_window: int = 50
    prediction_horizon: int = 10
    confidence_threshold: float = 0.7
    
    # Параметры риск-менеджмента
    max_position_size: float = 0.1
    stop_loss_threshold: float = 0.02
    take_profit_threshold: float = 0.05
    
    # Параметры мониторинга
    performance_window: int = 100
    rebalancing_frequency: int = 24  # часы


class AdaptiveStrategyGenerator(BaseStrategy):
    """
    Генератор адаптивных торговых стратегий.
    
    Особенности:
    - Адаптация к изменениям рыночного режима
    - Машинное обучение для прогнозирования
    - Динамический выбор стратегий
    - Управление рисками в реальном времени
    """

    def __init__(self, config: AdaptationConfig):
        super().__init__()
        self.adaptation_config = config
        self.regime_detector = MarketRegimeDetector()
        self.ml_processor = MLSignalProcessor()
        self.strategy_selector = StrategySelector(backtest_results={})
        
        # Состояние адаптации
        self.current_regime = MarketRegime.TRENDING_UP
        self.performance_history: List[float] = []
        self.adaptation_history: List[Dict[str, Any]] = []
        self.last_adaptation = datetime.now()
        
        # Кэш для оптимизации
        self._signal_cache: Dict[str, Signal] = {}
        self._analysis_cache: Dict[str, Any] = {}
        
        logger.info("AdaptiveStrategyGenerator initialized")

    def analyze(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ рыночных данных для адаптации стратегии.
        
        Args:
            market_data: DataFrame с рыночными данными
            
        Returns:
            Словарь с результатами анализа
        """
        try:
            # Определение текущего рыночного режима
            regime = self.regime_detector.detect_regime(market_data)
            self.current_regime = regime
            
            # Анализ производительности
            performance_metrics = self._calculate_performance_metrics(market_data)
            
            # Анализ волатильности
            volatility_analysis = self._analyze_volatility(market_data)
            
            # Анализ трендов
            trend_analysis = self._analyze_trends(market_data)
            
            # Анализ объемов
            volume_analysis = self._analyze_volumes(market_data)
            
            # ML-анализ
            ml_analysis = self.ml_processor.analyze(market_data) if hasattr(self.ml_processor, 'analyze') else {}
            
            analysis_result = {
                "regime": regime.value,
                "performance": performance_metrics,
                "volatility": volatility_analysis,
                "trend": trend_analysis,
                "volume": volume_analysis,
                "ml_predictions": ml_analysis,
                "timestamp": datetime.now().isoformat(),
                "adaptation_needed": self._check_adaptation_needed(performance_metrics),
            }
            
            # Кэширование результата
            cache_key = f"analysis_{market_data.index[-1]}"
            self._analysis_cache[cache_key] = analysis_result
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in market analysis: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "adaptation_needed": False,
            }

    def generate_signal(self, market_data: pd.DataFrame) -> Optional[Signal]:
        """
        Генерация торгового сигнала на основе адаптивного анализа.
        
        Args:
            market_data: DataFrame с рыночными данными
            
        Returns:
            Торговый сигнал или None
        """
        try:
            # Проверка кэша
            cache_key = f"signal_{market_data.index[-1]}"
            if cache_key in self._signal_cache:
                return self._signal_cache[cache_key]
            
            # Анализ рынка
            analysis = self.analyze(market_data)
            
            # Проверка необходимости адаптации
            if analysis.get("adaptation_needed", False):
                self._adapt_strategy(analysis)
            
            # Генерация сигнала с учетом текущего режима
            signal = self._generate_adaptive_signal(market_data, analysis)
            
            # Применение риск-менеджмента
            signal = self._apply_risk_management(signal, market_data)
            
            # Кэширование сигнала
            if signal:
                self._signal_cache[cache_key] = signal
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return None

    def _calculate_performance_metrics(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Расчет метрик производительности."""
        try:
            if len(market_data) < 2:
                return {"sharpe_ratio": 0.0, "max_drawdown": 0.0, "win_rate": 0.0}
            
            # Расчет доходности
            returns = market_data['close'].pct_change().dropna()
            
            # Коэффициент Шарпа
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0.0
            
            # Максимальная просадка
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            
            # Процент прибыльных сделок (упрощенная оценка)
            win_rate = (returns > 0).mean()
            
            # Безопасное получение total_return
            total_return = 0.0
            try:
                if len(cumulative_returns) > 0:
                    total_return = float(cumulative_returns.iloc[-1] - 1)
            except (IndexError, TypeError):
                total_return = 0.0
            
            return {
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "win_rate": float(win_rate),
                "volatility": float(returns.std()),
                "total_return": total_return,
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {"sharpe_ratio": 0.0, "max_drawdown": 0.0, "win_rate": 0.0}

    def _analyze_volatility(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Анализ волатильности."""
        try:
            returns = market_data['close'].pct_change().dropna()
            
            # Историческая волатильность
            historical_vol = 0.0
            try:
                if len(returns) >= 20:
                    rolling_std = returns.rolling(window=20).std()
                    if len(rolling_std) > 0:
                        historical_vol = rolling_std.iloc[-1]
            except (IndexError, TypeError):
                historical_vol = 0.0
            
            # Реализованная волатильность
            realized_vol = returns.std()
            
            # Волатильность волатильности
            vol_of_vol = returns.rolling(window=10).std().std()
            
            return {
                "historical_volatility": float(historical_vol),
                "realized_volatility": float(realized_vol),
                "volatility_of_volatility": float(vol_of_vol),
                "volatility_regime": "high" if historical_vol > 0.02 else "low",
                "volatility_score": float(historical_vol * 100),
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volatility: {str(e)}")
            return {"historical_volatility": 0.0, "realized_volatility": 0.0}

    def _analyze_trends(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Анализ трендов."""
        try:
            # Простая скользящая средняя
            sma_short = market_data['close'].rolling(window=10).mean()
            sma_long = market_data['close'].rolling(window=50).mean()
            
            # Текущий тренд
            current_trend = "neutral"
            trend_strength = 0.0
            try:
                if len(sma_short) > 0 and len(sma_long) > 0:
                    current_trend = "up" if sma_short.iloc[-1] > sma_long.iloc[-1] else "down"
                    if sma_long.iloc[-1] > 0:
                        trend_strength = abs(sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
            except (IndexError, TypeError, ZeroDivisionError):
                current_trend = "neutral"
                trend_strength = 0.0
            
            # RSI
            current_rsi = 50.0
            try:
                delta = market_data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                if len(rsi) > 0:
                    current_rsi = rsi.iloc[-1]
            except (IndexError, TypeError, ZeroDivisionError):
                current_rsi = 50.0
            
            # Безопасное получение SMA значений
            sma_short_val = 0.0
            sma_long_val = 0.0
            try:
                if len(sma_short) > 0:
                    sma_short_val = float(sma_short.iloc[-1])
                if len(sma_long) > 0:
                    sma_long_val = float(sma_long.iloc[-1])
            except (IndexError, TypeError):
                pass
            
            return {
                "trend_direction": current_trend,
                "trend_strength": float(trend_strength),
                "rsi": float(current_rsi),
                "sma_short": sma_short_val,
                "sma_long": sma_long_val,
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")
            return {"trend_direction": "neutral", "trend_strength": 0.0}

    def _analyze_volumes(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Анализ объемов."""
        try:
            if 'volume' not in market_data.columns:
                return {"volume_trend": 0.0, "volume_sma": 0.0}
            
            # Тренд объема
            volume_trend = 0.0
            volume_sma_val = 0.0
            current_volume_val = 0.0
            volume_price_correlation = 0.0
            
            try:
                volume_sma = market_data['volume'].rolling(window=20).mean()
                if len(market_data) > 0:
                    current_volume_val = float(market_data['volume'].iloc[-1])
                if len(volume_sma) > 0:
                    volume_sma_val = float(volume_sma.iloc[-1])
                    if volume_sma_val > 0:
                        volume_trend = (current_volume_val - volume_sma_val) / volume_sma_val
                
                # Объем относительно цены
                price_change = market_data['close'].pct_change()
                volume_price_correlation = float(price_change.corr(market_data['volume']))
            except (IndexError, TypeError, ZeroDivisionError):
                pass
            
            return {
                "volume_trend": volume_trend,
                "volume_sma": volume_sma_val,
                "volume_price_correlation": volume_price_correlation,
                "current_volume": current_volume_val,
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volumes: {str(e)}")
            return {"volume_trend": 0.0, "volume_sma": 0.0}

    def _check_adaptation_needed(self, performance_metrics: Dict[str, float]) -> bool:
        """Проверка необходимости адаптации стратегии."""
        try:
            # Проверяем производительность
            sharpe_ratio = performance_metrics.get("sharpe_ratio", 0.0)
            max_drawdown = performance_metrics.get("max_drawdown", 0.0)
            win_rate = performance_metrics.get("win_rate", 0.0)
            
            # Условия для адаптации
            needs_adaptation = (
                sharpe_ratio < self.adaptation_config.adaptation_threshold or
                max_drawdown > 0.1 or  # 10% просадка
                win_rate < 0.4  # Менее 40% прибыльных сделок
            )
            
            # Проверяем время с последней адаптации
            time_since_adaptation = (datetime.now() - self.last_adaptation).total_seconds() / 3600
            if time_since_adaptation < self.adaptation_config.rebalancing_frequency:
                needs_adaptation = False
            
            return needs_adaptation
            
        except Exception as e:
            logger.error(f"Error checking adaptation need: {str(e)}")
            return False

    def _adapt_strategy(self, analysis: Dict[str, Any]) -> None:
        """Адаптация стратегии на основе анализа."""
        try:
            logger.info("Starting strategy adaptation")
            
            # Обновление параметров на основе режима
            regime_str = analysis.get("regime", "trending_up")
            try:
                regime = MarketRegime(regime_str)
            except Exception:
                regime = MarketRegime.TRENDING_UP
            
            if regime == MarketRegime.TRENDING_UP:
                self._adapt_for_trending_market(analysis)
            elif regime == MarketRegime.SIDEWAYS:
                self._adapt_for_ranging_market(analysis)
            elif regime == MarketRegime.VOLATILE:
                self._adapt_for_volatile_market(analysis)
            
            # Обновление истории адаптации
            self.adaptation_history.append({
                "timestamp": datetime.now().isoformat(),
                "regime": regime.value,
                "analysis": analysis,
            })
            
            self.last_adaptation = datetime.now()
            logger.info("Strategy adaptation completed")
            
        except Exception as e:
            logger.error(f"Error adapting strategy: {str(e)}")

    def _adapt_for_trending_market(self, analysis: Dict[str, Any]) -> None:
        """Адаптация для трендового рынка."""
        trend_analysis = analysis.get("trend", {})
        trend_direction = trend_analysis.get("trend_direction", "neutral")
        
        if trend_direction == "up":
            # Увеличиваем позиции на покупку
            self.adaptation_config.max_position_size *= 1.1
            self.adaptation_config.take_profit_threshold *= 1.05
        else:
            # Увеличиваем позиции на продажу
            self.adaptation_config.max_position_size *= 1.1
            self.adaptation_config.stop_loss_threshold *= 0.95

    def _adapt_for_ranging_market(self, analysis: Dict[str, Any]) -> None:
        """Адаптация для бокового рынка."""
        # Уменьшаем размер позиций
        self.adaptation_config.max_position_size *= 0.8
        # Уменьшаем стоп-лосс и тейк-профит
        self.adaptation_config.stop_loss_threshold *= 0.9
        self.adaptation_config.take_profit_threshold *= 0.9

    def _adapt_for_volatile_market(self, analysis: Dict[str, Any]) -> None:
        """Адаптация для волатильного рынка."""
        volatility_analysis = analysis.get("volatility", {})
        historical_vol = volatility_analysis.get("historical_volatility", 0.0)
        
        if historical_vol > 0.03:  # Высокая волатильность
            # Значительно уменьшаем размер позиций
            self.adaptation_config.max_position_size *= 0.5
            # Увеличиваем стоп-лосс
            self.adaptation_config.stop_loss_threshold *= 1.5

    def _generate_adaptive_signal(self, market_data: pd.DataFrame, analysis: Dict[str, Any]) -> Optional[Signal]:
        """Генерация адаптивного сигнала."""
        try:
            # Получение ML-прогноза
            ml_predictions = analysis.get("ml_predictions", {})
            prediction_confidence = ml_predictions.get("confidence", 0.0)
            prediction_direction = ml_predictions.get("direction", "neutral")
            
            # Анализ тренда
            trend_analysis = analysis.get("trend", {})
            trend_direction = trend_analysis.get("trend_direction", "neutral")
            trend_strength = trend_analysis.get("trend_strength", 0.0)
            
            # Анализ объема
            volume_analysis = analysis.get("volume", {})
            volume_trend = volume_analysis.get("volume_trend", 0.0)
            
            # Генерация сигнала на основе всех факторов
            signal_strength = 0.0
            signal_direction = StrategyDirection.HOLD
            
            # Фактор ML-прогноза
            if prediction_confidence > self.adaptation_config.confidence_threshold:
                if prediction_direction == "up":
                    signal_strength += 0.3
                elif prediction_direction == "down":
                    signal_strength -= 0.3
            
            # Фактор тренда
            if trend_strength > 0.01:  # Значимый тренд
                if trend_direction == "up":
                    signal_strength += 0.2
                elif trend_direction == "down":
                    signal_strength -= 0.2
            
            # Фактор объема
            if volume_trend > 0.1:  # Увеличение объема
                if signal_strength > 0:
                    signal_strength += 0.1
                elif signal_strength < 0:
                    signal_strength -= 0.1
            
            # Определение направления сигнала
            if signal_strength > 0.2:
                signal_direction = StrategyDirection.LONG
            elif signal_strength < -0.2:
                signal_direction = StrategyDirection.SHORT
            else:
                signal_direction = StrategyDirection.HOLD
            
            # Создание сигнала
            if signal_direction != StrategyDirection.HOLD:
                current_price = 0.0
                try:
                    if len(market_data) > 0 and 'close' in market_data.columns:
                        current_price = float(market_data['close'].iloc[-1])
                except (IndexError, KeyError, TypeError):
                    return None
                
                signal = Signal(
                    direction=signal_direction,
                    entry_price=current_price,
                    confidence=min(abs(signal_strength), 1.0),
                    timestamp=datetime.now(),
                    metadata={
                        "regime": analysis.get("regime", "unknown"),
                        "trend_strength": trend_strength,
                        "volume_trend": volume_trend,
                        "ml_confidence": prediction_confidence,
                        "adaptation_config": {
                            "max_position_size": self.adaptation_config.max_position_size,
                            "stop_loss": self.adaptation_config.stop_loss_threshold,
                            "take_profit": self.adaptation_config.take_profit_threshold,
                        }
                    }
                )
                
                return signal  # type: ignore
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating adaptive signal: {str(e)}")
            return None

    def _apply_risk_management(self, signal: Optional[Signal], market_data: pd.DataFrame) -> Optional[Signal]:
        """Применение риск-менеджмента к сигналу."""
        if not signal:
            return None
        
        try:
            # Проверка максимального размера позиции
            if signal.confidence > self.adaptation_config.max_position_size:
                signal.confidence = self.adaptation_config.max_position_size
            
            # Добавление стоп-лосса и тейк-профита
            current_price = 0.0
            try:
                if len(market_data) > 0 and 'close' in market_data.columns:
                    current_price = float(market_data['close'].iloc[-1])
            except (IndexError, KeyError, TypeError):
                return None
            
            if signal.direction == StrategyDirection.LONG:
                stop_loss = current_price * (1 - self.adaptation_config.stop_loss_threshold)
                take_profit = current_price * (1 + self.adaptation_config.take_profit_threshold)
            elif signal.direction == StrategyDirection.SHORT:
                stop_loss = current_price * (1 + self.adaptation_config.stop_loss_threshold)
                take_profit = current_price * (1 - self.adaptation_config.take_profit_threshold)
            else:
                stop_loss = None
                take_profit = None
            
            # Обновление метаданных сигнала
            signal.metadata.update({
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward_ratio": abs(take_profit - current_price) / abs(stop_loss - current_price) if stop_loss and take_profit else None,
            })
            
            return signal
            
        except Exception as e:
            logger.error(f"Error applying risk management: {str(e)}")
            return signal

    def get_performance_summary(self) -> Dict[str, Any]:
        """Получение сводки производительности."""
        try:
            if not self.performance_history:
                return {"message": "No performance data available"}
            
            recent_performance = self.performance_history[-self.adaptation_config.performance_window:]
            
            return {
                "total_signals": len(self.performance_history),
                "recent_performance": recent_performance,
                "average_performance": sum(recent_performance) / len(recent_performance),
                "adaptation_count": len(self.adaptation_history),
                "current_regime": self.current_regime.value,
                "last_adaptation": self.last_adaptation.isoformat(),
                "config": {
                    "max_position_size": self.adaptation_config.max_position_size,
                    "stop_loss_threshold": self.adaptation_config.stop_loss_threshold,
                    "take_profit_threshold": self.adaptation_config.take_profit_threshold,
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {str(e)}")
            return {"error": str(e)}

    def reset_adaptation(self) -> None:
        """Сброс адаптации к начальным параметрам."""
        try:
            self.adaptation_config = AdaptationConfig()
            self.adaptation_history.clear()
            self.performance_history.clear()
            self._signal_cache.clear()
            self._analysis_cache.clear()
            self.last_adaptation = datetime.now()
            
            logger.info("Adaptation reset to initial parameters")
            
        except Exception as e:
            logger.error(f"Error resetting adaptation: {str(e)}")
