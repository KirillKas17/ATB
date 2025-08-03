"""
Эволюционная базовая стратегия
Расширяет базовую стратегию эволюционными возможностями
"""

import os
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from decimal import Decimal

from domain.types.strategy_types import (
    EvolutionMetrics,
    MarketRegime,
    Signal as DomainSignal,
    StrategyAnalysis,
    StrategyDirection,
    StrategyMetrics,
    StrategyType,
)
from infrastructure.core.evolution_manager import (
    EvolvableComponent,
    register_for_evolution,
)
from infrastructure.strategies.base_strategy import BaseStrategy, Signal


@dataclass
class EvolvableStrategyConfig:
    """Конфигурация эволюционной стратегии"""
    
    # Базовые параметры стратегии
    min_signals: int = 3
    max_signals: int = 10
    confidence_threshold: float = 0.6
    # Эволюционные параметры
    learning_rate: float = 1e-3
    adaptation_rate: float = 0.01
    evolution_threshold: float = 0.5
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


class RLTradingAgent(nn.Module):
    """Reinforcement Learning агент для генерации стратегий"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EvolvableBaseStrategy(BaseStrategy, EvolvableComponent):
    """
    Эволюционная базовая стратегия
    Объединяет классическую логику стратегии с ML-оптимизацией
    """

    def __init__(self, config: Optional[Union[Dict[str, Any], EvolvableStrategyConfig]] = None):
        if isinstance(config, EvolvableStrategyConfig):
            config_obj = config
        elif isinstance(config, dict):
            config_obj = EvolvableStrategyConfig(**config)
        else:
            config_obj = EvolvableStrategyConfig()
        BaseStrategy.__init__(self, config_obj.__dict__)
        EvolvableComponent.__init__(self, "EvolvableBaseStrategy")
        self.config: dict[str, Any] = config_obj.__dict__
        self._config_obj: EvolvableStrategyConfig = config_obj
        self.model = StrategyMLModel()
        # Исправление: добавляем проверку на существование learning_rate
        if 'learning_rate' in self.config:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.performance = 0.0
        self.confidence = 0.0
        self.training_data: List[Any] = []
        self.signal_history: List[Any] = []
        self.logger = logger
        self.symbol = "BTCUSDT"  # Добавляем атрибут symbol
        self.name = "EvolvableBaseStrategy"  # Добавляем атрибут name
        register_for_evolution(self)
        logger.info("EvolvableBaseStrategy initialized")

    def analyze(self, data: pd.DataFrame) -> dict[str, Any]:
        try:
            is_valid, error_msg = self.validate_data(data)
            if not is_valid:
                raise ValueError(f"Invalid data: {error_msg}")
            features = self._extract_strategy_features(data)
            ml_predictions = self._get_ml_predictions(features)
            market_regime = self._detect_market_regime(data)
            signals = self._generate_evolutionary_signals(
                data, ml_predictions, market_regime
            )
            metrics = self._calculate_evolutionary_metrics(data, ml_predictions)
            risk_assessment = self._assess_evolutionary_risk(
                data, ml_predictions, market_regime
            )
            recommendations = self._generate_evolutionary_recommendations(
                data, ml_predictions, market_regime
            )
            result = StrategyAnalysis(
                strategy_id=f"evolvable_{id(self)}",
                timestamp=datetime.now(),
                market_data=data,
                indicators=self._calculate_indicators(data),
                signals=signals,
                metrics=metrics,
                market_regime=market_regime,
                confidence=self._calculate_evolutionary_confidence(
                    data, ml_predictions
                ),
                risk_assessment=risk_assessment,
                recommendations=recommendations,
                metadata={
                    "ml_predictions": ml_predictions,
                    "evolution_metrics": self._get_evolution_metrics(),
                    "model_performance": self.performance,
                },
            )
            return result.__dict__
        except Exception as e:
            logger.error(f"Error in evolutionary analysis: {str(e)}")
            raise

    def generate_signal(self, market_data: pd.DataFrame) -> Signal:
        """Генерация торгового сигнала."""
        try:
            # Исправление: используем правильный тип возвращаемого значения
            if market_data.empty:
                return Signal(
                    direction="hold",  # type: ignore[call-arg]
                    trading_pair="BTC/USDT",  # type: ignore[call-arg]
                    signal_type="hold",  # type: ignore[call-arg]
                    confidence=0.0,  # type: ignore[call-arg]
                    strength=0.0,  # type: ignore[call-arg]
                    metadata={"id": "empty_signal", "price": "0.0", "amount": "0.0"}
                )
            
            # Анализ данных
            features = self._extract_strategy_features(market_data)
            if features is None:
                return Signal(
                    direction="hold",  # type: ignore[call-arg]
                    trading_pair="BTC/USDT",  # type: ignore[call-arg]
                    signal_type="hold",  # type: ignore[call-arg]
                    confidence=0.0,  # type: ignore[call-arg]
                    strength=0.0,  # type: ignore[call-arg]
                    metadata={"id": "no_features_signal", "price": "0.0", "amount": "0.0"}
                )
            
            # Предсказание
            prediction = self._get_ml_predictions(features)
            if prediction is None:
                return Signal(
                    direction="hold",  # type: ignore[call-arg]
                    trading_pair="BTC/USDT",  # type: ignore[call-arg]
                    signal_type="hold",  # type: ignore[call-arg]
                    confidence=0.0,  # type: ignore[call-arg]
                    strength=0.0,  # type: ignore[call-arg]
                    metadata={"id": "no_prediction_signal", "price": "0.0", "amount": "0.0"}
                )
            
            # Создание сигнала - исправляем конструктор
            direction = StrategyDirection.LONG if prediction.get("direction", "hold") == "buy" else StrategyDirection.SHORT
            entry_price = float(market_data["close"].iloc[-1]) if not market_data.empty else 50000.0
            confidence = float(prediction.get("confidence", 0.5))
            signal = Signal(
                direction=direction,
                entry_price=entry_price,
                confidence=confidence,
                timestamp=datetime.now(),
                metadata={
                    "id": "evolvable_signal",
                    "price": str(entry_price),
                    "amount": "0.1",
                    "strategy": self.name,
                    "prediction": prediction
                }
            )
            
            return signal
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return Signal(
                direction="hold",  # type: ignore[call-arg]
                trading_pair="BTC/USDT",  # type: ignore[call-arg]
                signal_type="hold",  # type: ignore[call-arg]
                confidence=0.0,  # type: ignore[call-arg]
                strength=0.0,  # type: ignore[call-arg]
                metadata={"error": str(e)}
            )

    def _extract_strategy_features(self, market_data: pd.DataFrame) -> List[float]:
        """Извлечение признаков для ML модели"""
        try:
            features = []
            # Технические индикаторы
            features.append(
                float(market_data["close"].pct_change().rolling(20).std().iloc[-1])
            )  # Волатильность
            features.append(
                float(market_data["close"].pct_change(10).iloc[-1])
            )  # Моментум
            features.append(
                float(
                    market_data["volume"].iloc[-1]
                    / market_data["volume"].rolling(20).mean().iloc[-1]
                )
            )  # Объем
            # Трендовые метрики
            ema_20 = market_data["close"].ewm(span=20).mean()
            ema_50 = market_data["close"].ewm(span=50).mean()
            features.append(
                float(abs(ema_20.iloc[-1] - ema_50.iloc[-1]) / ema_50.iloc[-1])
            )  # Сила тренда
            # Ценовые метрики
            features.append(
                float(
                    (market_data["high"].iloc[-1] - market_data["low"].iloc[-1])
                    / market_data["close"].iloc[-1]
                )
            )  # Диапазон
            features.append(
                float(
                    market_data["close"].iloc[-1]
                    / market_data["close"].rolling(20).mean().iloc[-1]
                )
            )  # Отклонение от MA
            # Дополнительные метрики (заполняем до 20 признаков)
            for i in range(14):
                features.append(0.0)
            return features[:20]  # Ограничиваем до 20 признаков
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return [0.0] * 20

    def _get_ml_predictions(self, features: List[float]) -> Dict[str, Union[str, float]]:
        """Получение предсказаний от ML модели"""
        try:
            if len(features) == 0:
                return {}
            # Преобразование в тензор
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            # Получение предсказаний
            with torch.no_grad():
                predictions = self.model(features_tensor)
                probabilities = torch.softmax(predictions, dim=1)
            # Интерпретация результатов
            buy_prob = float(probabilities[0, 0].item())
            sell_prob = float(probabilities[0, 1].item())
            hold_prob = float(probabilities[0, 2].item())
            # Определение направления
            max_prob = max(buy_prob, sell_prob, hold_prob)
            if max_prob == buy_prob:
                direction = "buy"
                confidence = buy_prob
            elif max_prob == sell_prob:
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
                "model_uncertainty": float(1.0 - max_prob),
            }
        except Exception as e:
            logger.error(f"Error getting ML predictions: {str(e)}")
            return {
                "direction": "hold",
                "confidence": 0.5,
                "buy_probability": 0.33,
                "sell_probability": 0.33,
                "hold_probability": 0.34,
                "model_uncertainty": 0.66,
            }

    def _detect_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """Определение рыночного режима"""
        try:
            # Анализ тренда
            ema_20 = data["close"].ewm(span=20).mean()
            ema_50 = data["close"].ewm(span=50).mean()
            trend_strength = abs(ema_20.iloc[-1] - ema_50.iloc[-1]) / ema_50.iloc[-1]
            # Анализ волатильности
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
            logger.error(f"Error detecting market regime: {str(e)}")
            return MarketRegime.SIDEWAYS

    def _generate_evolutionary_signals(
        self, data: pd.DataFrame, ml_predictions: Dict[str, Union[str, float]], regime: MarketRegime
    ) -> List[DomainSignal]:
        """Генерация эволюционных сигналов"""
        signals = []
        try:
            # Генерация базового сигнала
            base_signal = self._generate_base_signal(data, ml_predictions)
            if base_signal:
                # Эволюционная адаптация
                evolved_signal = self._evolve_signal(
                    base_signal, ml_predictions, regime
                )
                if evolved_signal:
                    signals.append(evolved_signal)
            return signals
        except Exception as e:
            logger.error(f"Error generating evolutionary signals: {str(e)}")
            return signals

    def _generate_base_signal(
        self, data: pd.DataFrame, ml_predictions: Dict[str, Union[str, float]]
    ) -> Optional[DomainSignal]:
        """Генерация базового сигнала"""
        try:
            direction = ml_predictions.get("direction", "hold")
            confidence = ml_predictions.get("confidence", 0.5)
            # Исправляем обращение к config
            config_threshold = getattr(self.config, 'confidence_threshold', 0.6)
            if direction == "hold" or confidence < config_threshold:
                return None
            close = data["close"].iloc[-1]
            # Определение направления
            if direction == "buy":
                signal_direction = StrategyDirection.LONG
            elif direction == "sell":
                signal_direction = StrategyDirection.SHORT
            else:
                return None
            return DomainSignal(
                direction=signal_direction,
                entry_price=close,
                confidence=float(confidence),
                strategy_type=StrategyType.EVOLVABLE,
                market_regime=MarketRegime.SIDEWAYS,  # Будет обновлено
                risk_score=1.0 - confidence,
                expected_return=0.02,
            )
        except Exception as e:
            logger.error(f"Error generating base signal: {str(e)}")
            return None

    def _evolve_signal(
        self,
        base_signal: DomainSignal,
        ml_predictions: Dict[str, Union[str, float]],
        regime: MarketRegime,
    ) -> DomainSignal:
        """Эволюционная адаптация сигнала"""
        try:
            # Адаптация уверенности на основе эволюционных метрик
            evolution_factor = self.performance * 0.3 + self.confidence * 0.7
            base_signal.confidence = min(
                1.0, base_signal.confidence * (1 + evolution_factor)
            )
            # Адаптация размера позиции
            if ml_predictions.get("confidence", 0.5) > 0.8:
                base_signal.position_size = 1.0
            else:
                base_signal.position_size = 0.5
            # Обновление рыночного режима
            base_signal.market_regime = regime
            # Обновление ожидаемой доходности
            base_signal.expected_return = ml_predictions.get("confidence", 0.5) * 0.05
            return base_signal
        except Exception as e:
            logger.error(f"Error evolving signal: {str(e)}")
            return base_signal

    def _check_evolution_conditions(
        self, signal: DomainSignal, ml_predictions: Dict[str, Union[str, float]]
    ) -> bool:
        """Проверка эволюционных условий"""
        try:
            # Проверка уверенности - исправляем обращение к config
            config_threshold = getattr(self.config, 'confidence_threshold', 0.6)
            if signal.confidence < config_threshold:
                return False
            # Проверка производительности модели - исправляем обращение к config
            config_evolution_threshold = getattr(self.config, 'evolution_threshold', 0.5)
            if self.performance < config_evolution_threshold:
                return False
            # Проверка неопределенности модели
            if ml_predictions.get("model_uncertainty", 1.0) > 0.7:
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking evolution conditions: {str(e)}")
            return False

    def _calculate_evolutionary_metrics(
        self, data: pd.DataFrame, ml_predictions: Dict[str, Union[str, float]]
    ) -> StrategyMetrics:
        """Расчет эволюционных метрик"""
        try:
            # Базовые метрики
            volatility = data["close"].pct_change().rolling(20).std().iloc[-1]
            # Эволюционные метрики
            evolution_score = self._calculate_evolution_score(data, ml_predictions)
            model_confidence = ml_predictions.get("confidence", 0.5)
            return StrategyMetrics(
                volatility=volatility,
                additional={
                    "evolution_score": evolution_score,
                    "model_confidence": model_confidence,
                    "model_performance": self.performance,
                    "training_samples": len(self.training_data),
                },
            )
        except Exception as e:
            logger.error(f"Error calculating evolutionary metrics: {str(e)}")
            return StrategyMetrics()

    def _calculate_evolution_score(
        self, data: pd.DataFrame, ml_predictions: Dict[str, Union[str, float]]
    ) -> float:
        """Расчет оценки эволюции"""
        try:
            # Факторы эволюции
            model_performance = self.performance
            model_confidence = ml_predictions.get("confidence", 0.5)
            # Исправляем DataFrame.any() на правильный синтаксис
            if hasattr(data, 'isnull'):
                if hasattr(data.isnull(), 'to_numpy'):
                    data_quality = 1.0 if not data.empty and not data.isnull().to_numpy().any() else 0.7  # type: ignore[attr-defined]
                elif hasattr(data.isnull(), 'values'):
                    data_quality = 1.0 if not data.empty and not data.isnull().values.any() else 0.7  # type: ignore[attr-defined]
                else:
                    data_quality = 1.0 if not data.empty else 0.7
            else:
                data_quality = 1.0 if not data.empty else 0.7
            training_samples = min(1.0, len(self.training_data) / 1000)  # Нормализация
            evolution_score = (
                model_performance * 0.4
                + model_confidence * 0.3
                + data_quality * 0.2
                + training_samples * 0.1
            )
            return max(0.0, min(1.0, evolution_score))
        except Exception as e:
            logger.error(f"Error calculating evolution score: {str(e)}")
            return 0.5

    def _assess_evolutionary_risk(
        self, data: pd.DataFrame, ml_predictions: Dict[str, Union[str, float]], regime: MarketRegime
    ) -> Dict[str, float]:
        """Оценка эволюционного риска"""
        try:
            risk_assessment = {}
            # Рыночный риск
            risk_assessment["market_risk"] = (
                data["close"].pct_change().rolling(20).std().iloc[-1] * 10
            )
            # Риск модели
            risk_assessment["model_risk"] = 1.0 - ml_predictions.get("confidence", 0.5)
            # Риск эволюции
            risk_assessment["evolution_risk"] = 1.0 - self.performance
            # Общий риск
            risk_assessment["total_risk"] = (
                risk_assessment["market_risk"] * 0.4
                + risk_assessment["model_risk"] * 0.4
                + risk_assessment["evolution_risk"] * 0.2
            )
            return risk_assessment
        except Exception as e:
            logger.error(f"Error assessing evolutionary risk: {str(e)}")
            return {
                "market_risk": 0.5,
                "model_risk": 0.5,
                "evolution_risk": 0.5,
                "total_risk": 0.5,
            }

    def _calculate_evolutionary_confidence(
        self, data: pd.DataFrame, ml_predictions: Dict[str, Union[str, float]]
    ) -> float:
        """Расчет эволюционной уверенности"""
        try:
            # Базовая уверенность модели
            base_confidence = ml_predictions.get("confidence", 0.5)
            # Корректировка на основе производительности
            performance_factor = self.performance
            # Корректировка на основе качества данных
            if hasattr(data, 'isnull'):
                if hasattr(data.isnull(), 'to_numpy'):
                    data_quality = 1.0 if not data.empty and not data.isnull().to_numpy().any() else 0.7  # type: ignore[attr-defined]
                elif hasattr(data.isnull(), 'values'):
                    data_quality = 1.0 if not data.empty and not data.isnull().values.any() else 0.7  # type: ignore[attr-defined]
                else:
                    data_quality = 1.0 if not data.empty else 0.7
            else:
                data_quality = 1.0 if not data.empty else 0.7
            # Финальная уверенность
            final_confidence = (
                base_confidence * 0.6
                + performance_factor * 0.3
                + data_quality * 0.1
            )
            return max(0.0, min(1.0, final_confidence))
        except Exception as e:
            logger.error(f"Error calculating evolutionary confidence: {str(e)}")
            return 0.5

    def _generate_evolutionary_recommendations(
        self, data: pd.DataFrame, ml_predictions: Dict[str, Union[str, float]], regime: MarketRegime
    ) -> List[str]:
        """Генерация эволюционных рекомендаций"""
        recommendations = []
        try:
            # Рекомендации по модели
            model_confidence = ml_predictions.get("confidence", 0.5)
            if model_confidence > 0.8:
                recommendations.append(
                    "Высокая уверенность ML модели - можно увеличить размер позиции"
                )
            elif model_confidence < 0.3:
                recommendations.append(
                    "Низкая уверенность ML модели - используйте консервативные настройки"
                )
            # Рекомендации по эволюции
            if self.performance > 0.8:
                recommendations.append(
                    "Высокая производительность эволюционной модели - стратегия оптимизирована"
                )
            elif self.performance < 0.3:
                recommendations.append(
                    "Низкая производительность - требуется переобучение модели"
                )
            # Рекомендации по данным
            if len(self.training_data) < 1000:
                recommendations.append(
                    "Недостаточно данных для обучения - собирайте больше исторических данных"
                )
            return recommendations
        except Exception as e:
            logger.error(f"Error generating evolutionary recommendations: {str(e)}")
            return ["Ошибка в генерации рекомендаций"]

    def _get_evolution_metrics(self) -> Dict[str, float]:
        """Получение метрик эволюции"""
        try:
            return {
                "evolution_score": float(self.performance),
                "adaptation_rate": float(getattr(self.config, 'adaptation_rate', 0.01)),
                "learning_rate": float(getattr(self.config, 'learning_rate', 1e-3)),
                "performance": float(self.performance),
                "confidence": float(self.confidence),
            }
        except Exception as e:
            logger.error(f"Error getting evolution metrics: {str(e)}")
            return {
                "evolution_score": 0.5,
                "adaptation_rate": 0.01,
                "learning_rate": 1e-3,
                "performance": 0.5,
                "confidence": 0.5,
            }

    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Расчет индикаторов"""
        try:
            return {
                "close": data["close"],
                "volume": data["volume"],
                "volatility": data["close"].pct_change().rolling(20).std(),
                "momentum": data["close"].pct_change(10),
            }
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return {}

    async def adapt(self, data: Any) -> bool:
        """Адаптация стратегии к новым данным"""
        try:
            # Адаптация параметров на основе новых данных
            if isinstance(data, pd.DataFrame):
                features = self._extract_strategy_features(data)
                ml_predictions = self._get_ml_predictions(features)
                # Обновление параметров
                self.performance = self._calculate_evolution_score(data, ml_predictions)
                self.confidence = ml_predictions.get("confidence", 0.5)
            return True
        except Exception as e:
            logger.error(f"Error adapting strategy: {str(e)}")
            return False

    async def learn(self, data: Any) -> bool:
        """Обучение стратегии на новых данных"""
        try:
            # Обучение ML модели на новых данных
            if isinstance(data, pd.DataFrame):
                features = self._extract_strategy_features(data)
                targets = self._extract_strategy_targets([])  # Пустой список сигналов
                if len(features) > 0 and len(targets) > 0:
                    features_tensor = torch.FloatTensor(features).unsqueeze(0)
                    targets_tensor = torch.FloatTensor(targets).unsqueeze(0)
                    # Обучение
                    self.optimizer.zero_grad()
                    predictions = self.model(features_tensor)
                    loss = torch.nn.functional.mse_loss(predictions, targets_tensor)
                    loss.backward()
                    self.optimizer.step()
            return True
        except Exception as e:
            logger.error(f"Error learning strategy: {str(e)}")
            return False

    async def evolve(self, data: Any) -> bool:
        """Полная эволюция стратегии"""
        try:
            # Эволюция архитектуры модели
            await self._evolve_model_architecture()
            # Эволюция параметров стратегии
            await self._evolve_strategy_parameters()
            # Переобучение на истории
            await self._retrain_on_history()
            logger.info("Strategy evolved successfully")
            return True
        except Exception as e:
            logger.error(f"Strategy evolve error: {e}")
            return False

    def get_performance(self) -> float:
        """Получение производительности стратегии"""
        return self.performance

    def get_confidence(self) -> float:
        """Получение уверенности в стратегии"""
        return self.confidence

    def save_state(self, path: str = "") -> bool:
        """Сохранение состояния стратегии."""
        try:
            # Исправление: изменяем сигнатуру для совместимости
            state = {
                "model_state": self.model.state_dict(),
                "config": self.config,
                "generation": getattr(self, 'generation', 0),
                "fitness_score": getattr(self, 'fitness_score', 0.0),
                "total_trades": getattr(self, 'total_trades', 0),
                "win_rate": getattr(self, 'win_rate', 0.0),
                "profit_factor": getattr(self, 'profit_factor', 0.0),
                "max_drawdown": getattr(self, 'max_drawdown', 0.0),
                "improvement_rate": getattr(self, 'improvement_rate', 0.0),
                "adaptation_success_rate": getattr(self, 'adaptation_success_rate', 0.0),
                "last_evolution": getattr(self, 'last_evolution', None),
                "evolution_history": getattr(self, 'evolution_history', []),
                "performance_history": getattr(self, 'performance_history', []),
                "metadata": {
                    "strategy_name": self.name,
                    "symbol": self.symbol,
                    "created_at": getattr(self, 'created_at', datetime.now()).isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "version": "1.0.0",
                    "learning_rate": float(self._config_obj.learning_rate) if hasattr(self._config_obj, 'learning_rate') else 1e-3,
                }
            }
            
            with open(path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            self.logger.info(f"Strategy state saved to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving strategy state: {e}")
            return False

    def load_state(self, path: str) -> bool:
        """Загрузка состояния стратегии"""
        try:
            if os.path.exists(path):
                state = torch.load(path)
                self.model.load_state_dict(state["model_state"])
                self.optimizer.load_state_dict(state["optimizer_state"])
                self.config = state["config"]
                self.performance = state["performance"]
                self.confidence = state["confidence"]
                self.training_data = state["training_data"]
                self.signal_history = state["signal_history"]
                logger.info(f"Strategy state loaded from {path}")
                return True
        except Exception as e:
            logger.error(f"Error loading strategy state: {e}")
        return False

    def _extract_strategy_targets(self, signals: List[DomainSignal]) -> List[float]:
        """Извлечение целевых значений для обучения"""
        try:
            targets = []
            for signal in signals:
                if signal.direction == StrategyDirection.LONG:
                    targets.extend([1.0, 0.0, 0.0])  # buy, sell, hold
                elif signal.direction == StrategyDirection.SHORT:
                    targets.extend([0.0, 1.0, 0.0])
                else:
                    targets.extend([0.0, 0.0, 1.0])
            return targets
        except Exception as e:
            logger.error(f"Error extracting targets: {str(e)}")
            return [0.0, 0.0, 1.0]  # Default to hold

    async def _evolve_model_architecture(self) -> None:
        """Эволюция архитектуры модели"""
        try:
            # Увеличение размера скрытого слоя
            current_hidden = self.model.net[0].out_features
            new_hidden = min(256, current_hidden + 32)
            if new_hidden != current_hidden:
                self.model = StrategyMLModel(input_dim=20, hidden_dim=new_hidden)
                # Исправление: добавляем проверку на существование learning_rate
                learning_rate = getattr(self.config, 'learning_rate', 1e-3)
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
                logger.info(f"Model architecture evolved: hidden_dim={new_hidden}")
        except Exception as e:
            logger.error(f"Error evolving model architecture: {e}")

    async def _evolve_strategy_parameters(self) -> None:
        """Эволюция параметров стратегии"""
        try:
            # Адаптация порогов на основе производительности
            if self.performance > 0.8:
                self.config.confidence_threshold = min(
                    0.9, self.config.confidence_threshold + 0.05
                )
            elif self.performance < 0.3:
                self.config.confidence_threshold = max(
                    0.3, self.config.confidence_threshold - 0.05
                )
            # Адаптация скорости обучения
            if self.performance > 0.7:
                self.config.learning_rate = max(1e-4, self.config.learning_rate * 0.9)
            else:
                self.config.learning_rate = min(1e-2, self.config.learning_rate * 1.1)
            logger.info(
                f"Strategy parameters evolved: confidence_threshold={self.config.confidence_threshold:.3f}, learning_rate={self.config.learning_rate:.6f}"
            )
        except Exception as e:
            logger.error(f"Error evolving strategy parameters: {e}")

    async def _retrain_on_history(self) -> None:
        """Переобучение на исторических данных"""
        try:
            if len(self.training_data) < 100:
                return
            # Выборка последних данных
            recent_data = self.training_data[-100:]
            for data_point in recent_data:
                features = data_point["features"]
                targets = data_point["targets"]
                if len(features) > 0 and len(targets) > 0:
                    features_tensor = torch.FloatTensor(features).unsqueeze(0)
                    targets_tensor = torch.FloatTensor(targets)
                    self.optimizer.zero_grad()
                    predictions = self.model(features_tensor)
                    loss = nn.MSELoss()(predictions, targets_tensor)
                    loss.backward()
                    self.optimizer.step()
            logger.info("Model retrained on historical data")
        except Exception as e:
            logger.error(f"Error retraining on history: {e}")

    def _init_rl_agent(self, state_dim: int, action_dim: int) -> RLTradingAgent:
        """Инициализация RL агента"""
        return RLTradingAgent(state_dim, action_dim)

    async def train_rl_agent(self, historical_data: np.ndarray, reward_fn):
        """Обучение RL агента"""
        try:
            agent = self._init_rl_agent(historical_data.shape[1], 3)
            optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
            for episode in range(100):
                state = historical_data[0]
                total_reward = 0
                for step in range(len(historical_data) - 1):
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action_probs = torch.softmax(agent(state_tensor), dim=1)
                    action = torch.multinomial(action_probs, 1).item()
                    next_state = historical_data[int(step) + 1]
                    reward = reward_fn(state, action, next_state)
                    total_reward += reward
                    # Обучение
                    optimizer.zero_grad()
                    loss = -torch.log(action_probs[0][action]) * reward  # type: ignore[index]
                    loss.backward()
                    optimizer.step()
                    state = next_state
                if episode % 10 == 0:
                    logger.info(
                        f"RL Episode {episode}, Total Reward: {total_reward:.2f}"
                    )
        except Exception as e:
            logger.error(f"Error training RL agent: {e}")

    async def _validate_rl_agent(self, validation_data: np.ndarray, reward_fn):
        """Валидация RL агента"""
        try:
            agent = self._init_rl_agent(validation_data.shape[1], 3)
            total_reward = 0
            for step in range(len(validation_data) - 1):
                state = validation_data[step]
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs = torch.softmax(agent(state_tensor), dim=1)
                action = torch.argmax(action_probs).item()
                next_state = validation_data[step + 1]
                reward = reward_fn(state, action, next_state)
                total_reward += reward
            logger.info(f"RL Agent validation reward: {total_reward:.2f}")
            return total_reward
        except Exception as e:
            logger.error(f"Error validating RL agent: {e}")
            return 0.0

    async def generate_signals(self, market_data: pd.DataFrame) -> List[Signal]:
        """Генерация сигналов с использованием RL агента"""
        try:
            features = self._extract_strategy_features(market_data)
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            agent = self._init_rl_agent(len(features), 3)
            action_probs = torch.softmax(agent(features_tensor), dim=1)
            action = torch.argmax(action_probs).item()
            signal = self._action_to_signal(action, market_data)
            return [signal] if signal else []
        except Exception as e:
            logger.error(f"Error generating RL signals: {e}")
            return []

    def _action_to_signal(self, action, market_data):
        """Преобразование действия RL-агента в торговый сигнал"""
        try:
            if action == 0:  # Buy
                signal = Signal(
                    id="evolvable_signal",
                    symbol="BTC/USDT",
                    signal_type="buy",
                    confidence=Decimal("0.7"),
                    price=Decimal(str(market_data["close"].iloc[-1])),
                    amount=Decimal("0.1"),
                    created_at=datetime.now()
                )
                return signal
            elif action == 1:  # Sell
                signal = Signal(
                    id="evolvable_signal",
                    symbol="BTC/USDT",
                    signal_type="sell",
                    confidence=Decimal("0.7"),
                    price=Decimal(str(market_data["close"].iloc[-1])),
                    amount=Decimal("0.1"),
                    created_at=datetime.now()
                )
                return signal
            else:  # Hold
                return None
        except Exception as e:
            logger.error(f"Error converting action to signal: {e}")
            return None

    def get_evolution_stats(self) -> Dict[str, Any]:
        """Получение статистики эволюции"""
        try:
            # Исправление: исправляем неопределенные переменные
            evolution_stats = {
                "generation": getattr(self, 'generation', 0),
                "fitness_score": getattr(self, 'fitness_score', 0.0),
                "improvement_rate": getattr(self, 'improvement_rate', 0.0),
                "adaptation_success": getattr(self, 'adaptation_success_rate', 0.0)
            }
            performance_metrics = {
                "total_trades": getattr(self, 'total_trades', 0),
                "win_rate": getattr(self, 'win_rate', 0.0),
                "profit_factor": getattr(self, 'profit_factor', 0.0),
                "max_drawdown": getattr(self, 'max_drawdown', 0.0)
            }
            return {
                "evolution_score": self.performance,
                "adaptation_rate": getattr(self.config, 'adaptation_rate', 0.01),
                "learning_rate": getattr(self.config, 'learning_rate', 1e-3),
                "performance": self.performance,
                "confidence": self.confidence,
            }
        except Exception as e:
            logger.error(f"Error getting evolution stats: {e}")
            return {}
