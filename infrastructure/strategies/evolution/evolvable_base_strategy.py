"""
Упрощенная эволюционная базовая стратегия
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    from torch_mock import torch
    torch.nn = torch.nn
    nn = torch.nn
    TORCH_AVAILABLE = False
from loguru import logger

from domain.type_definitions.strategy_types import (
    MarketRegime,
    Signal as DomainSignal,
    StrategyAnalysis,
    StrategyDirection,
    StrategyMetrics,
    StrategyType,
)
from infrastructure.strategies.base_strategy import BaseStrategy

from .evolution_manager import EvolutionManager
from .performance_tracker import PerformanceTracker


@dataclass
class EvolutionConfig:
    """Конфигурация эволюционной стратегии"""

    # Параметры ML модели
    input_dim: int = 20
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    # Параметры эволюции
    evolution_enabled: bool = True
    adaptation_rate: float = 0.1
    # Параметры управления рисками
    risk_per_trade: float = 0.02
    max_position_size: float = 0.2
    # Общие параметры
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=lambda: ["1h"])
    log_dir: str = "logs"


class StrategyMLModel(nn.Module):
    """ML модель для оптимизации стратегии"""

    def __init__(self, input_dim: int = 20, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # buy, sell, hold
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EvolvableBaseStrategy(BaseStrategy):
    """Эволюционная базовая стратегия с ML-оптимизацией"""

    def __init__(self, config: Optional[Union[Dict[str, Any], EvolutionConfig]] = None):
        # Подготовка конфигурации для базового класса
        if isinstance(config, EvolutionConfig):
            config_dict = {
                "input_dim": config.input_dim,
                "hidden_dim": config.hidden_dim,
                "learning_rate": config.learning_rate,
                "evolution_enabled": config.evolution_enabled,
                "adaptation_rate": config.adaptation_rate,
                "risk_per_trade": config.risk_per_trade,
                "max_position_size": config.max_position_size,
                "symbols": config.symbols,
                "timeframes": config.timeframes,
                "log_dir": config.log_dir,
            }
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = {}
        
        super().__init__(config_dict)
        
        # Устанавливаем конфигурацию - исправление: убираем дублирование атрибутов
        if isinstance(config, EvolutionConfig):
            self._evolution_config = config
        elif isinstance(config, dict):
            self._evolution_config = EvolutionConfig(**config)
        else:
            self._evolution_config = EvolutionConfig()
            
        # ML модель
        self.model = StrategyMLModel(self._evolution_config.input_dim, self._evolution_config.hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self._evolution_config.learning_rate)
        # Компоненты эволюции
        self.evolution_manager = EvolutionManager()
        self.performance_tracker = PerformanceTracker()
        # Состояние стратегии
        self.performance = 0.0
        self.confidence = 0.0
        self.training_data: List[Any] = []
        logger.info("EvolvableBaseStrategy initialized")

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ рыночных данных с эволюционным подходом.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Dict[str, Any]: Результат анализа
        """
        try:
            # Валидация данных
            is_valid, error_msg = self.validate_data(data)
            if not is_valid:
                raise ValueError(f"Invalid data: {error_msg}")
            # Извлечение признаков
            features = self._extract_features(data)
            # Получение предсказаний
            ml_predictions = self._get_predictions(features)
            # Определение рыночного режима
            market_regime = self._detect_regime(data)
            # Генерация сигналов
            signals = self._generate_signals(data, ml_predictions, market_regime)
            # Расчет метрик
            metrics = self._calculate_metrics(data, ml_predictions)
            # Оценка риска
            risk_assessment = self._assess_risk(data, ml_predictions, market_regime)
            # Рекомендации
            recommendations = self._generate_recommendations(
                data, ml_predictions, market_regime
            )
            return {
                "strategy_id": f"evolvable_{id(self)}",
                "timestamp": datetime.now(),  # Исправление: используем datetime.now()
                "market_data": data,
                "indicators": self._calculate_indicators(data),
                "signals": signals,
                "metrics": metrics,
                "market_regime": market_regime,
                "confidence": self._calculate_confidence(data, ml_predictions),
                "risk_assessment": risk_assessment,
                "recommendations": recommendations,
                "metadata": {
                    "ml_predictions": ml_predictions,
                    "performance": self.performance,
                    "training_samples": len(self.training_data),
                },
            }
        except Exception as e:
            logger.error(f"Error in evolutionary analysis: {str(e)}")
            raise

    def generate_signal(self, data: pd.DataFrame) -> Optional[DomainSignal]:
        """
        Генерация эволюционного торгового сигнала.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Optional[DomainSignal]: Торговый сигнал или None
        """
        try:
            # Извлечение признаков
            features = self._extract_features(data)
            # Получение предсказаний
            ml_predictions = self._get_predictions(features)
            # Определение рыночного режима
            market_regime = self._detect_regime(data)
            # Генерация базового сигнала
            base_signal = self._generate_base_signal(data, ml_predictions)
            if not base_signal:
                return None
            # Эволюционная адаптация
            evolved_signal = self._evolve_signal(
                base_signal, ml_predictions, market_regime
            )
            # Проверка условий
            if not self._check_conditions(evolved_signal, ml_predictions):
                return None
            return evolved_signal
        except Exception as e:
            logger.error(f"Error generating evolutionary signal: {str(e)}")
            return None

    def _extract_features(self, data: pd.DataFrame) -> List[float]:
        """Извлечение признаков"""
        try:
            features = []
            # Технические индикаторы
            features.append(
                float(data["close"].pct_change().rolling(20).std().iloc[-1])
            )
            features.append(float(data["close"].pct_change(10).iloc[-1]))
            features.append(
                float(
                    data["volume"].iloc[-1] / data["volume"].rolling(20).mean().iloc[-1]
                )
            )
            # Трендовые метрики
            ema_20 = data["close"].ewm(span=20).mean()
            ema_50 = data["close"].ewm(span=50).mean()
            features.append(
                float(abs(ema_20.iloc[-1] - ema_50.iloc[-1]) / ema_50.iloc[-1])
            )
            # Дополнительные метрики
            for i in range(16):
                features.append(0.0)
            return features[:20]
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return [0.0] * 20

    def _get_predictions(self, features: List[float]) -> Dict[str, Any]:
        """Получение предсказаний ML-модели"""
        try:
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                output = self.model(x)
                probs = torch.softmax(output, dim=1).squeeze().numpy()
            buy_prob, sell_prob, hold_prob = probs[0], probs[1], probs[2]
            if buy_prob > sell_prob and buy_prob > hold_prob:
                direction = "buy"
                confidence = buy_prob
            elif sell_prob > buy_prob and sell_prob > hold_prob:
                direction = "sell"
                confidence = sell_prob
            else:
                direction = "hold"
                confidence = hold_prob
            return {
                "direction": direction,
                "confidence": float(confidence),
                "buy_probability": float(buy_prob),
                "sell_probability": float(sell_prob),
                "hold_probability": float(hold_prob),
            }
        except Exception as e:
            logger.error(f"Error getting predictions: {str(e)}")
            return {
                "direction": "hold",
                "confidence": 0.5,
                "buy_probability": 0.33,
                "sell_probability": 0.33,
                "hold_probability": 0.34,
            }

    def _detect_regime(self, data: pd.DataFrame) -> MarketRegime:
        """Определение рыночного режима"""
        try:
            ema_20 = data["close"].ewm(span=20).mean()
            ema_50 = data["close"].ewm(span=50).mean()
            trend_strength = abs(ema_20.iloc[-1] - ema_50.iloc[-1]) / ema_50.iloc[-1]
            volatility = data["close"].pct_change().rolling(20).std().iloc[-1]
            if trend_strength > 0.05:
                return (
                    MarketRegime.TRENDING_UP
                    if ema_20.iloc[-1] > ema_50.iloc[-1]
                    else MarketRegime.TRENDING_DOWN
                )
            elif volatility > 0.03:
                return MarketRegime.VOLATILE
            else:
                return MarketRegime.SIDEWAYS
        except Exception as e:
            logger.error(f"Error detecting regime: {str(e)}")
            return MarketRegime.SIDEWAYS

    def _generate_signals(
        self, data: pd.DataFrame, ml_predictions: Dict[str, Any], regime: MarketRegime
    ) -> List[DomainSignal]:
        """Генерация сигналов"""
        signals = []
        try:
            base_signal = self._generate_base_signal(data, ml_predictions)
            if base_signal:
                evolved_signal = self._evolve_signal(
                    base_signal, ml_predictions, regime
                )
                if evolved_signal:
                    signals.append(evolved_signal)
            return signals
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return signals

    def _generate_base_signal(
        self, data: pd.DataFrame, ml_predictions: Dict[str, Any]
    ) -> Optional[DomainSignal]:
        """Генерация базового сигнала"""
        try:
            direction = ml_predictions.get("direction", "hold")
            confidence = ml_predictions.get("confidence", 0.5)
            if direction == "hold" or confidence < 0.6:
                return None
            close = data["close"].iloc[-1]
            if direction == "buy":
                signal_direction = StrategyDirection.LONG
            elif direction == "sell":
                signal_direction = StrategyDirection.SHORT
            else:
                return None
            return DomainSignal(
                direction=signal_direction,
                entry_price=close,
                confidence=confidence,
                strategy_type=StrategyType.EVOLVABLE,
                market_regime=MarketRegime.SIDEWAYS,
                risk_score=1.0 - confidence,
                expected_return=0.02,
            )
        except Exception as e:
            logger.error(f"Error generating base signal: {str(e)}")
            return None

    def _evolve_signal(
        self,
        base_signal: DomainSignal,
        ml_predictions: Dict[str, Any],
        regime: MarketRegime,
    ) -> DomainSignal:
        """Эволюционная адаптация сигнала"""
        try:
            evolution_factor = self.performance * 0.3 + self.confidence * 0.7
            base_signal.confidence = min(
                1.0, base_signal.confidence * (1 + evolution_factor)
            )
            base_signal.market_regime = regime
            base_signal.expected_return = ml_predictions.get("confidence", 0.5) * 0.05
            return base_signal
        except Exception as e:
            logger.error(f"Error evolving signal: {str(e)}")
            return base_signal

    def _check_conditions(
        self, signal: DomainSignal, ml_predictions: Dict[str, Any]
    ) -> bool:
        """Проверка условий"""
        try:
            if signal.confidence < 0.6:
                return False
            if self.performance < 0.5:
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking conditions: {str(e)}")
            return False

    def _calculate_metrics(
        self, data: pd.DataFrame, ml_predictions: Dict[str, Any]
    ) -> StrategyMetrics:
        """Расчет метрик"""
        try:
            volatility = data["close"].pct_change().rolling(20).std().iloc[-1]
            evolution_score = self._calculate_evolution_score(data, ml_predictions)
            return StrategyMetrics(
                volatility=volatility,
                additional={
                    "evolution_score": evolution_score,
                    "model_confidence": ml_predictions.get("confidence", 0.5),
                    "model_performance": self.performance,
                },
            )
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return StrategyMetrics()

    def _calculate_evolution_score(
        self, data: pd.DataFrame, ml_predictions: Dict[str, Any]
    ) -> float:
        """Расчет оценки эволюции"""
        try:
            model_performance = self.performance
            model_confidence = ml_predictions.get("confidence", 0.5)
            # Исправление: используем безопасные методы DataFrame
            try:
                if hasattr(data.isnull().any(), 'any'):
                    data_quality = 0.9 if not data.isnull().any().any() else 0.6  # type: ignore[attr-defined]
                else:
                    data_quality = 0.9
            except (AttributeError, TypeError):
                data_quality = 0.9  # Fallback
            evolution_score = (
                model_performance * 0.4 + model_confidence * 0.3 + data_quality * 0.3
            )
            return max(0.0, min(1.0, evolution_score))
        except Exception as e:
            logger.error(f"Error calculating evolution score: {str(e)}")
            return 0.5

    def _assess_risk(
        self, data: pd.DataFrame, ml_predictions: Dict[str, Any], regime: MarketRegime
    ) -> Dict[str, Any]:
        """Оценка риска"""
        try:
            risk_assessment = {}
            # Безопасный расчет market_risk
            try:
                if len(data) > 20 and "close" in data.columns:
                    pct_change = data["close"].pct_change()
                    if len(pct_change) > 20:
                        rolling_std = pct_change.rolling(20).std()
                        if len(rolling_std) > 0:
                            risk_assessment["market_risk"] = float(rolling_std.iloc[-1] * 10)
                        else:
                            risk_assessment["market_risk"] = 0.5
                    else:
                        risk_assessment["market_risk"] = 0.5
                else:
                    risk_assessment["market_risk"] = 0.5
            except (AttributeError, IndexError, TypeError):
                risk_assessment["market_risk"] = 0.5
                
            risk_assessment["model_risk"] = 1.0 - ml_predictions.get("confidence", 0.5)
            risk_assessment["evolution_risk"] = 1.0 - self._calculate_evolution_score(
                data, ml_predictions
            )
            risk_assessment["total_risk"] = (
                risk_assessment["market_risk"] * 0.4
                + risk_assessment["model_risk"] * 0.3
                + risk_assessment["evolution_risk"] * 0.3
            )
            return risk_assessment
        except Exception as e:
            logger.error(f"Error assessing risk: {str(e)}")
            return {"total_risk": 0.5}

    def _calculate_confidence(
        self, data: pd.DataFrame, ml_predictions: Dict[str, Any]
    ) -> float:
        """Расчет уверенности"""
        try:
            model_confidence = ml_predictions.get("confidence", 0.5)
            evolution_confidence = self.performance
            # Исправление: используем безопасные методы DataFrame
            try:
                if hasattr(data.isnull().any(), 'any'):
                    data_quality = 0.9 if not data.isnull().any().any() else 0.6  # type: ignore[attr-defined]
                else:
                    data_quality = 0.9
            except (AttributeError, TypeError):
                data_quality = 0.9  # Fallback
            confidence = (
                model_confidence * 0.5 + evolution_confidence * 0.3 + data_quality * 0.2
            )
            return max(0.1, min(1.0, confidence))
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5

    def _generate_recommendations(
        self, data: pd.DataFrame, ml_predictions: Dict[str, Any], regime: MarketRegime
    ) -> List[str]:
        """Генерация рекомендаций"""
        recommendations = []
        try:
            model_confidence = ml_predictions.get("confidence", 0.5)
            if model_confidence > 0.8:
                recommendations.append("Высокая уверенность ML модели")
            elif model_confidence < 0.3:
                recommendations.append("Низкая уверенность ML модели")
            if self.performance > 0.8:
                recommendations.append("Высокая производительность эволюционной модели")
            elif self.performance < 0.3:
                recommendations.append(
                    "Низкая производительность - требуется переобучение"
                )
            return recommendations
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return ["Ошибка в генерации рекомендаций"]

    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Расчет индикаторов"""
        try:
            return {
                "close": data["close"],
                "volume": data["volume"],
                "volatility": data["close"].pct_change().rolling(20).std(),
            }
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return {}

    def save_state(self) -> Any:
        """Сохранение состояния - исправление: соответствие сигнатуре базового класса"""
        try:
            state = {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "performance": self.performance,
                "confidence": self.confidence,
                "training_data": self.training_data,
            }
            return state
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return None

    def load_state(self, path: str) -> bool:
        """Загрузка состояния"""
        try:
            if os.path.exists(path):
                state = torch.load(path)
                self.model.load_state_dict(state["model_state"])
                self.optimizer.load_state_dict(state["optimizer_state"])
                self.performance = state["performance"]
                self.confidence = state["confidence"]
                self.training_data = state["training_data"]
                logger.info(f"Strategy state loaded from {path}")
                return True
        except Exception as e:
            logger.error(f"Error loading state: {e}")
        return False
