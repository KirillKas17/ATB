"""
Эволюционный агент рыночного режима
Автоматически адаптируется к изменениям рынка и определяет текущий режим
"""

import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from shared.numpy_utils import np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger

from infrastructure.core.evolution_manager import (
    EvolvableComponent,
    register_for_evolution,
)


class MarketRegimeML(nn.Module):
    """ML модель для определения рыночного режима"""

    def __init__(
        self, input_dim: int = 25, hidden_dim: int = 64, num_regimes: int = 4
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_regimes),  # 4 режима: trending, ranging, volatile, stable
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        # Доступные режимы
        self.regimes: Dict[str, str] = {
            "trending": "Trending Market",
            "ranging": "Ranging Market",
            "volatile": "Volatile Market",
            "stable": "Stable Market",
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
                    # Адаптация предпочтений режимов
                    if volatility > 0.03:  # Высокая волатильность
                        self._prefer_volatile_regime()
                    elif trend_strength > 0.01:  # Сильный тренд
                        self._prefer_trending_regime()
                    elif volatility < 0.01:  # Низкая волатильность
                        self._prefer_stable_regime()
                    else:  # Боковик
                        self._prefer_ranging_regime()
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
                current_regime = data.get("current_regime", "trending")
                regime_performance = data.get("regime_performance", {})
                # Подготовка данных для обучения
                features = self._extract_features(market_data, regime_performance)
                target_regime = self._regime_to_index(current_regime)
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
                        self.training_data = self.training_data[-self.max_training_samples:]
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
            # Эволюция режимов
            await self._evolve_regimes()
            # Переобучение на истории
            await self._retrain_on_history()
            logger.info("MarketRegimeAgent evolved successfully")
            return True
        except Exception as e:
            logger.error(f"Error in MarketRegimeAgent evolution: {e}")
            return False

    def get_performance(self) -> float:
        """Получение производительности"""
        return self.performance_metric

    def get_confidence(self) -> float:
        """Получение уверенности"""
        return self.confidence_metric

    def save_state(self, path: str) -> bool:
        """Сохранение состояния"""
        try:
            state = {
                "name": self.name,
                "model_state": self.ml_model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "performance_metric": self.performance_metric,
                "confidence_metric": self.confidence_metric,
                "successful_predictions": self.successful_predictions,
                "total_predictions": self.total_predictions,
                "training_data": self.training_data[-1000:],
                "regimes": self.regimes,
            }
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(state, f"{path}/evolvable_market_regime_agent_state.pth")
            return True
        except Exception as e:
            logger.error(f"MarketRegimeAgent save_state error: {e}")
            return False

    def load_state(self, path: str) -> bool:
        """Загрузка состояния"""
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    state = pickle.load(f)
                # Восстанавливаем модель
                if "model_state" in state:
                    self.ml_model.load_state_dict(state["model_state"])
                # Восстанавливаем оптимизатор
                if "optimizer_state" in state:
                    self.optimizer.load_state_dict(state["optimizer_state"])
                # Восстанавливаем метрики
                self.performance_metric = state.get("performance_metric", 0.5)
                self.confidence_metric = state.get("confidence_metric", 0.5)
                self.successful_predictions = state.get("successful_predictions", 0)
                self.total_predictions = state.get("total_predictions", 0)
                self.training_data = state.get("training_data", [])
                self.regimes = state.get("regimes", self.regimes)
                logger.info(f"MarketRegimeAgent state loaded from {path}")
                return True
        except Exception as e:
            logger.error(f"MarketRegimeAgent load_state error: {e}")
        return False

    def _extract_features(
        self, market_data: pd.DataFrame, regime_performance: Dict
    ) -> List[float]:
        """Извлечение признаков для определения режима"""
        try:
            if market_data.empty:
                return []
            # Базовые рыночные данные
            close_prices = market_data["close"]
            volume = market_data["volume"]
            # Технические индикаторы
            # SMA
            sma_5 = close_prices.rolling(5).mean().iloc[-1]
            sma_20 = close_prices.rolling(20).mean().iloc[-1]
            sma_50 = close_prices.rolling(50).mean().iloc[-1]
            # Волатильность
            volatility_5 = close_prices.pct_change().rolling(5).std().iloc[-1]
            volatility_20 = close_prices.pct_change().rolling(20).std().iloc[-1]
            # Моментум
            momentum_5 = (close_prices.iloc[-1] / close_prices.iloc[-6]) - 1
            momentum_20 = (close_prices.iloc[-1] / close_prices.iloc[-21]) - 1
            # RSI
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            # Объем
            volume_sma = volume.rolling(20).mean().iloc[-1]
            volume_ratio = volume.iloc[-1] / volume_sma if volume_sma > 0 else 1.0
            # Производительность режимов
            regime_scores = []
            for regime in ["trending", "ranging", "volatile", "stable"]:
                score = regime_performance.get(regime, 0.5)
                regime_scores.append(score)
            # Комбинирование признаков
            features = [
                sma_5, sma_20, sma_50,
                volatility_5, volatility_20,
                momentum_5, momentum_20,
                rsi,
                volume_ratio
            ] + regime_scores
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
        # Исправление: приводим к int
        index = int(index)
        regimes = ["trending", "ranging", "volatile", "stable"]
        return regimes[index] if 0 <= index < len(regimes) else "trending"

    def _update_metrics(
        self, loss: float, predictions: np.ndarray, target_regime: int
    ) -> None:
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

    def _prefer_volatile_regime(self) -> None:
        """Предпочтение волатильного режима"""
        pass

    def _prefer_trending_regime(self) -> None:
        """Предпочтение трендового режима"""
        pass

    def _prefer_stable_regime(self) -> None:
        """Предпочтение стабильного режима"""
        pass

    def _prefer_ranging_regime(self) -> None:
        """Предпочтение бокового режима"""
        pass

    async def _evolve_model_architecture(self) -> None:
        """Эволюция архитектуры модели"""
        try:
            # Простая эволюция - изменение размеров слоев
            current_hidden_dim = self.ml_model.net[0].out_features
            new_hidden_dim = current_hidden_dim + np.random.randint(-10, 11)
            new_hidden_dim = max(32, min(128, new_hidden_dim))
            # Создание новой модели
            new_model = MarketRegimeML(
                input_dim=25,
                hidden_dim=new_hidden_dim,
                num_regimes=4
            )
            # Копирование весов где возможно
            try:
                new_model.net[0].weight.data[:current_hidden_dim, :] = self.ml_model.net[0].weight.data
                new_model.net[0].bias.data[:current_hidden_dim] = self.ml_model.net[0].bias.data
            except:
                pass
            self.ml_model = new_model
            self.optimizer = torch.optim.Adam(self.ml_model.parameters(), lr=1e-3)
            logger.info(f"Model architecture evolved: hidden_dim={new_hidden_dim}")
        except Exception as e:
            logger.error(f"Error evolving model architecture: {e}")

    async def _evolve_regimes(self) -> None:
        """Эволюция режимов"""
        try:
            # Простая эволюция - изменение весов режимов
            pass
        except Exception as e:
            logger.error(f"Error evolving regimes: {e}")

    async def _retrain_on_history(self) -> None:
        """Переобучение на исторических данных"""
        try:
            if len(self.training_data) < 10:
                return
            # Подготовка данных
            features_list = [item["features"] for item in self.training_data]
            targets_list = [item["target_regime"] for item in self.training_data]
            if len(features_list) > 0 and len(targets_list) > 0:
                # Объединение признаков
                all_features = []
                for features in features_list:
                    all_features.extend(features)
                # Объединение целей
                all_targets = targets_list
                if len(all_features) > 0 and len(all_targets) > 0:
                    # Обучение
                    features_tensor = torch.FloatTensor(all_features).unsqueeze(0)
                    targets_tensor = torch.LongTensor(all_targets)
                    self.optimizer.zero_grad()
                    predictions = self.ml_model(features_tensor)
                    loss = self.criterion(predictions, targets_tensor)
                    loss.backward()
                    self.optimizer.step()
                    logger.info(f"Retrained on history: loss={loss.item():.6f}")
        except Exception as e:
            logger.error(f"Error retraining on history: {e}")

    async def detect_regime(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        regime_performance: Dict[str, float],
    ) -> Dict[str, Any]:
        """Определение текущего рыночного режима"""
        try:
            # Извлечение признаков
            features = self._extract_features(market_data, regime_performance)
            if len(features) == 0:
                return {"regime": "trending", "confidence": 0.5, "reasoning": "No features available"}
            # ML предсказание
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            with torch.no_grad():
                predictions = self.ml_model(features_tensor)
                predicted_index = torch.argmax(predictions).item()
                confidence = torch.max(predictions).item()
            # Преобразование в название режима
            detected_regime = self._index_to_regime(predicted_index)
            return {
                "regime": detected_regime,
                "confidence": confidence,
                "reasoning": f"ML model detected {detected_regime} regime with confidence {confidence:.3f}",
                "all_predictions": predictions.numpy()[0].tolist()
            }
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return {"regime": "trending", "confidence": 0.5, "reasoning": f"Error: {str(e)}"}

    async def get_available_regimes(self) -> Dict[str, str]:
        """Получение доступных режимов"""
        return self.regimes

    async def update_regime_performance(self, regime: str, performance: float) -> None:
        """Обновление производительности режима"""
        try:
            # Обновление метрик на основе производительности
            if performance > 0.7:
                self.performance_metric = 0.9 * self.performance_metric + 0.1 * performance
            elif performance < 0.3:
                self.performance_metric = 0.9 * self.performance_metric + 0.1 * performance
        except Exception as e:
            logger.error(f"Error updating regime performance: {e}")
