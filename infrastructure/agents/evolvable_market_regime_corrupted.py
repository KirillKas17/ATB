"""
Эволюционный агент рыночного режима (исправленная версия).
Интегрируется с модульной архитектурой и поддерживает эволюцию.
"""

import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger

from infrastructure.core.evolution_manager import (
    EvolvableComponent,
    register_for_evolution,
)


class MarketRegimeML(nn.Module):
    """ML модель для классификации рыночного режима"""

    def __init__(
        self, input_dim: int = 15, hidden_dim: int = 64, num_regimes: int = 4
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_regimes),  # 4 режима: trending, ranging, volatile, stable
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)


class EvolvableMarketRegimeAgent(EvolvableComponent):
    """Эволюционный агент рыночного режима"""

    def __init__(self, name: str = "evolvable_market_regime"):
        super().__init__(name)
        # ML модель для эволюции
        self.ml_model = MarketRegimeML()
        self.optimizer = torch.optim.Adam(self.ml_model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        # Метрики производительности
        self.performance_metric = 0.5
        self.confidence_metric = 0.5
        self.successful_predictions = 0
        self.total_predictions = 0
        # История данных для обучения
        self.training_data: List[Dict[str, Any]] = []
        self.max_training_samples: int = 10000
        # Рыночные режимы
        self.regimes: Dict[int, str] = {
            0: "trending",
            1: "ranging",
            2: "volatile",
            3: "stable",
        }
        # Конфигурация
        self.config: Dict[str, float] = {
            "volatility_threshold": 0.02,
            "trend_threshold": 0.01,
            "regime_confidence_threshold": 0.7,
        }
        # Регистрация в эволюционном менеджере
        register_for_evolution(self)
        logger.info(f"EvolvableMarketRegimeAgent initialized: {name}")

    async def adapt(self, data: Any) -> bool:
        """Быстрая адаптация к новым данным"""
        try:
            if isinstance(data, dict) and "market_data" in data:
                market_data = data["market_data"]
                # Адаптация на основе рыночных условий
                if not market_data.empty:
                    volatility = (
                        market_data["close"].pct_change().rolling(20).std().iloc[-1]
                    )
                    trend_strength = abs(
                        market_data["close"].pct_change().rolling(20).mean().iloc[-1]
                    )
                    # Адаптация порогов
                    if volatility > 0.03:  # Высокая волатильность
                        self.config["volatility_threshold"] *= 1.1
                    elif volatility < 0.01:  # Низкая волатильность
                        self.config["volatility_threshold"] *= 0.9
                    if trend_strength > 0.015:  # Сильный тренд
                        self.config["trend_threshold"] *= 1.1
                    elif trend_strength < 0.005:  # Слабый тренд
                        self.config["trend_threshold"] *= 0.9
                    # Ограничение параметров
                    self.config["volatility_threshold"] = max(0.01, min(0.05, self.config["volatility_threshold"]))
                    self.config["trend_threshold"] = max(0.005, min(0.02, self.config["trend_threshold"]))
                    logger.debug(
                        f"MarketRegimeAgent adapted: volatility={volatility:.6f}, trend_strength={trend_strength:.6f}"
                    )
                    return True
        except Exception as e:
            logger.error(f"Error in MarketRegimeAgent adaptation: {e}")
        return False

    async def learn(self, data: Any) -> bool:
        """Обучение на новых данных"""
        try:
            if isinstance(data, dict) and "market_data" in data:
                market_data = data["market_data"]
                regime_label = data.get("regime_label", "trending")
                # Подготовка данных для обучения
                features = self._extract_features(market_data)
                target_regime = self._regime_to_index(regime_label)
                if len(features) > 0 and target_regime is not None:
                    # Обучение модели
                    features_tensor = torch.FloatTensor(features).unsqueeze(0)
                    target_tensor = torch.LongTensor([target_regime])
                    self.optimizer.zero_grad()
                    predictions = self.ml_model(features_tensor)
                    loss = self.criterion(predictions, target_tensor)
                    loss.backward()
                    self.optimizer.step()
                    # Обновление метрик
                    self._update_metrics(
                        loss.item(), predictions.detach().numpy()[0], target_regime
                    )
                    # Сохранение данных для обучения
                    self.training_data.append(
                        {
                            "features": features,
                            "target_regime": target_regime,
                            "timestamp": datetime.now(),
                        }
                    )
                    # Ограничение размера истории
                    if len(self.training_data) > self.max_training_samples:
                        self.training_data = self.training_data[
                            -self.max_training_samples :
                        ]
                    logger.debug(f"MarketRegimeAgent learned: loss={loss.item():.6f}")
                    return True
        except Exception as e:
            logger.error(f"Error in MarketRegimeAgent learning: {e}")
        return False

    async def evolve(self, data: Any) -> bool:
        """Полная эволюция компонента"""
        try:
            # Эволюция архитектуры модели
            await self._evolve_model_architecture()
            # Эволюция конфигурации
            await self._evolve_configuration()
            # Переобучение на всей истории
            await self._retrain_on_history()
            logger.info("MarketRegimeAgent evolved successfully")
            return True
        except Exception as e:
            logger.error(f"Error in MarketRegimeAgent evolution: {e}")
            return False

    def get_performance(self) -> float:
        """Получение текущей производительности"""
        if self.total_predictions == 0:
            return 0.0
        accuracy = self.successful_predictions / self.total_predictions
        confidence = self.confidence_metric
        # Комбинированная метрика производительности
        performance = 0.6 * accuracy + 0.4 * confidence
        return min(1.0, max(0.0, performance))

    def get_confidence(self) -> float:
        """Получение уверенности в решениях"""
        return self.confidence_metric

    def save_state(self, path: str) -> bool:
        """Сохранение состояния"""
        try:
            state = {
                "ml_model_state": self.ml_model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "performance_metric": self.performance_metric,
                "confidence_metric": self.confidence_metric,
                "successful_predictions": self.successful_predictions,
                "total_predictions": self.total_predictions,
                "training_data": self.training_data[-1000:],
                "evolution_count": self.evolution_count,
                "last_evolution": self.last_evolution,
                "config": self.config,
            }
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(state, f)
            logger.info(f"MarketRegimeAgent state saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving MarketRegimeAgent state: {e}")
            return False

    def load_state(self, path: str) -> bool:
        """Загрузка состояния"""
        try:
            if not os.path.exists(path):
                logger.warning(f"State file not found: {path}")
                return False
            with open(path, "rb") as f:
                state = pickle.load(f)
            self.ml_model.load_state_dict(state["ml_model_state"])
            self.optimizer.load_state_dict(state["optimizer_state"])
            self.performance_metric = state["performance_metric"]
            self.confidence_metric = state["confidence_metric"]
            self.successful_predictions = state["successful_predictions"]
            self.total_predictions = state["total_predictions"]
            self.training_data = state["training_data"]
            self.evolution_count = state["evolution_count"]
            self.last_evolution = state["last_evolution"]
            self.config = state.get("config", self.config)
            logger.info(f"MarketRegimeAgent state loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading MarketRegimeAgent state: {e}")
            return False

    def _extract_features(self, market_data: pd.DataFrame) -> List[float]:
        """Извлечение признаков из рыночных данных"""
        try:
            if market_data.empty:
                return []
            # Технические индикаторы
            close_prices = market_data["close"]
            volume = market_data["volume"]
            # Волатильность
            volatility_5 = close_prices.pct_change().rolling(5).std().iloc[-1]
            volatility_20 = close_prices.pct_change().rolling(20).std().iloc[-1]
            volatility_50 = close_prices.pct_change().rolling(50).std().iloc[-1]
            # Тренд
            sma_5 = close_prices.rolling(5).mean().iloc[-1]
            sma_20 = close_prices.rolling(20).mean().iloc[-1]
            sma_50 = close_prices.rolling(50).mean().iloc[-1]
            trend_strength_5 = (close_prices.iloc[-1] - sma_5) / sma_5
            trend_strength_20 = (close_prices.iloc[-1] - sma_20) / sma_20
            trend_strength_50 = (close_prices.iloc[-1] - sma_50) / sma_50
            # Моментум
            momentum_5 = (close_prices.iloc[-1] / close_prices.iloc[-6]) - 1
            momentum_20 = (close_prices.iloc[-1] / close_prices.iloc[-21]) - 1
            # Объем
            volume_sma = volume.rolling(20).mean().iloc[-1]
            volume_ratio = volume.iloc[-1] / volume_sma if volume_sma > 0 else 1.0
            # Комбинирование признаков
            features = [
                volatility_5, volatility_20, volatility_50,
                trend_strength_5, trend_strength_20, trend_strength_50,
                momentum_5, momentum_20,
                volume_ratio,
            ]
            # Нормализация
            features = [float(f) if not np.isnan(f) else 0.0 for f in features]
            return features
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return []

    def _regime_to_index(self, regime: str) -> Optional[int]:
        """Преобразование названия режима в индекс"""
        regime_map = {
            "trending": 0,
            "ranging": 1,
            "volatile": 2,
            "stable": 3,
        }
        return regime_map.get(regime)

    def _index_to_regime(self, index: Union[int, float]) -> str:
        """Преобразование индекса в название режима"""
        index = int(index)
        return self.regimes.get(index, "trending")

    def _update_metrics(
        self, loss: float, predictions: np.ndarray, target_regime: int
    ):
        """Обновление метрик производительности"""
        try:
            predicted_regime = np.argmax(predictions)
            self.total_predictions += 1
            if predicted_regime == target_regime:
                self.successful_predictions += 1
            # Обновление уверенности
            confidence = predictions[predicted_regime]
            self.confidence_metric = 0.9 * self.confidence_metric + 0.1 * confidence
            # Обновление производительности
            accuracy = self.successful_predictions / self.total_predictions
            self.performance_metric = 0.9 * self.performance_metric + 0.1 * accuracy
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

    async def _evolve_model_architecture(self):
        """Эволюция архитектуры модели"""
        try:
            # Простая эволюция - изменение размеров слоев
            current_hidden_dim = self.ml_model.net[0].out_features
            new_hidden_dim = current_hidden_dim + np.random.randint(-10, 11)
            new_hidden_dim = max(32, min(128, new_hidden_dim))
            # Создание новой модели
            new_model = MarketRegimeML(
                input_dim=15,
                hidden_dim=new_hidden_dim,
                num_regimes=4
            )
            # Копирование весов где возможно
            try:
                new_model.net[0].weight.data[:current_hidden_dim, :] = self.ml_model.net[0].weight.data
                new_model.net[0].bias.data[:current_hidden_dim] = self.ml_model.net[0].bias.data
            except (RuntimeError, IndexError, AttributeError) as e:
                # Логируем ошибку копирования весов и продолжаем с новой моделью
                logger.warning(f"Could not copy weights during market regime model evolution: {e}")
            self.ml_model = new_model
            self.optimizer = torch.optim.Adam(self.ml_model.parameters(), lr=1e-3)
            logger.info(f"Model architecture evolved: hidden_dim={new_hidden_dim}")
        except Exception as e:
            logger.error(f"Error evolving model architecture: {e}")

    async def _evolve_configuration(self):
        """Эволюция конфигурации"""
        try:
            # Эволюция параметров конфигурации
            self.config["volatility_threshold"] *= np.random.uniform(0.9, 1.1)
            self.config["trend_threshold"] *= np.random.uniform(0.9, 1.1)
            self.config["regime_confidence_threshold"] *= np.random.uniform(0.9, 1.1)
            # Ограничение параметров
            self.config["volatility_threshold"] = max(0.01, min(0.05, self.config["volatility_threshold"]))
            self.config["trend_threshold"] = max(0.005, min(0.02, self.config["trend_threshold"]))
            self.config["regime_confidence_threshold"] = max(0.5, min(0.9, self.config["regime_confidence_threshold"]))
            logger.info("Market regime configuration evolved")
        except Exception as e:
            logger.error(f"Error evolving configuration: {e}")

    async def _retrain_on_history(self):
        """Переобучение на всей истории"""
        try:
            if len(self.training_data) < 10:
                return
            # Подготовка данных
            features_list = [item["features"] for item in self.training_data]
            targets_list = [item["target_regime"] for item in self.training_data]
            if len(features_list) == 0:
                return
            # Преобразование в тензоры
            features_tensor = torch.FloatTensor(features_list)
            targets_tensor = torch.LongTensor(targets_list)
            # Обучение
            self.ml_model.train()
            for epoch in range(10):
                self.optimizer.zero_grad()
                predictions = self.ml_model(features_tensor)
                loss = self.criterion(predictions, targets_tensor)
                loss.backward()
                self.optimizer.step()
            logger.info("Model retrained on history")
        except Exception as e:
            logger.error(f"Error retraining on history: {e}")

    async def classify_market_regime(
        self,
        symbol: str,
        market_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Классификация рыночного режима"""
        try:
            if market_data.empty:
                return {
                    "regime": "trending",
                    "confidence": 0.5,
                    "features": {},
                }
            # Извлечение признаков
            features = self._extract_features(market_data)
            if len(features) == 0:
                return {
                    "regime": "trending",
                    "confidence": 0.5,
                    "features": {},
                }
            # Предсказание
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            self.ml_model.eval()
            with torch.no_grad():
                predictions = self.ml_model(features_tensor)
                predictions_np = predictions.numpy()[0]
            # Интерпретация результатов
            predicted_index = np.argmax(predictions_np)
            confidence = float(predictions_np[predicted_index])
            regime = self._index_to_regime(int(predicted_index))
            # Обновление метрик
            self.total_predictions += 1
            self.confidence_metric = 0.9 * self.confidence_metric + 0.1 * confidence
            # Дополнительные признаки
            feature_names = [
                "volatility_5", "volatility_20", "volatility_50",
                "trend_strength_5", "trend_strength_20", "trend_strength_50",
                "momentum_5", "momentum_20", "volume_ratio",
            ]
            features_dict = dict(zip(feature_names, features))
            return {
                "regime": regime,
                "confidence": confidence,
                "features": features_dict,
                "predictions": predictions_np.tolist(),
            }
        except Exception as e:
            logger.error(f"Error classifying market regime: {e}")
            return {
                "regime": "trending",
                "confidence": 0.5,
                "features": {},
            }

    async def get_regime_config(self) -> Dict[str, Any]:
        """Получение текущей конфигурации"""
        return self.config.copy()

    async def update_regime_config(self, config: Dict[str, Any]):
        """Обновление конфигурации"""
        try:
            self.config.update(config)
        except Exception as e:
            logger.error(f"Error updating regime config: {e}")

    async def process(
        self,
        symbol: str,
        market_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Обработка рыночных данных для классификации режима"""
        try:
            # Классификация режима
            regime_analysis = await self.classify_market_regime(symbol, market_data)
            # Дополнительная обработка
            processed_result = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "regime_analysis": regime_analysis,
                "processed": True,
            }
            return processed_result
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "regime_analysis": {
                    "regime": "trending",
                    "confidence": 0.5,
                    "features": {},
                },
                "processed": False,
                "error": str(e),
            }
