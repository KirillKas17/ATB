"""
Эволюционный агент стратегий
Автоматически адаптируется к изменениям рынка и оптимизирует выбор стратегий
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


class StrategyML(nn.Module):
    """ML модель для выбора стратегий"""

    def __init__(
        self, input_dim: int = 25, hidden_dim: int = 64, num_strategies: int = 5
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(
                hidden_dim // 2, num_strategies
            ),  # 5 стратегий: trend, mean_reversion, momentum, volatility, arbitrage
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=1)


class EvolvableStrategyAgent(EvolvableComponent):
    """Эволюционный агент стратегий"""

    def __init__(self, name: str = "evolvable_strategy") -> None:
        super().__init__(name)
        # ML модель для эволюции
        self.ml_model = StrategyML()
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
        # Доступные стратегии
        self.strategies: Dict[str, str] = {
            "trend": "Trend Following",
            "mean_reversion": "Mean Reversion",
            "momentum": "Momentum",
            "volatility": "Volatility Breakout",
            "arbitrage": "Arbitrage",
        }
        # Регистрация в эволюционном менеджере
        register_for_evolution(self)
        logger.info(f"EvolvableStrategyAgent initialized: {name}")

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
                    # Адаптация предпочтений стратегий
                    if volatility > 0.03:  # Высокая волатильность
                        self._prefer_volatility_strategies()
                    elif trend_strength > 0.01:  # Сильный тренд
                        self._prefer_trend_strategies()
                    else:  # Боковик
                        self._prefer_mean_reversion_strategies()
                    logger.debug(
                        f"StrategyAgent adapted: volatility={volatility:.6f}, trend_strength={trend_strength:.6f}"
                    )
                    return True
        except Exception as e:
            logger.error(f"Error in StrategyAgent adaptation: {e}")
        return False

    async def learn(self, data: Any) -> bool:
        """Обучение на новых данных"""
        try:
            if isinstance(data, dict) and "market_data" in data:
                market_data = data["market_data"]
                best_strategy = data.get("best_strategy", "trend")
                strategy_performance = data.get("strategy_performance", {})
                # Подготовка данных для обучения
                features = self._extract_features(market_data, strategy_performance)
                target_strategy = self._strategy_to_index(best_strategy)
                if len(features) > 0 and target_strategy is not None:
                    # Обучение модели
                    features_tensor = torch.FloatTensor(features).unsqueeze(0)
                    target_tensor = torch.LongTensor([target_strategy])
                    self.optimizer.zero_grad()
                    predictions = self.ml_model(features_tensor)
                    loss = self.criterion(predictions, target_tensor)
                    loss.backward()
                    self.optimizer.step()
                    # Обновление метрик
                    self._update_metrics(
                        loss.item(), predictions.detach().numpy()[0], target_strategy
                    )
                    # Сохранение данных для обучения
                    self.training_data.append(
                        {
                            "features": features,
                            "target_strategy": target_strategy,
                            "timestamp": datetime.now(),
                        }
                    )
                    # Ограничение размера истории
                    if len(self.training_data) > self.max_training_samples:
                        self.training_data = self.training_data[-self.max_training_samples:]
                    logger.debug(f"StrategyAgent learned: loss={loss.item():.6f}")
                    return True
        except Exception as e:
            logger.error(f"Error in StrategyAgent learning: {e}")
        return False

    async def evolve(self, data: Any) -> bool:
        """Полная эволюция компонента"""
        try:
            # Эволюция архитектуры модели
            await self._evolve_model_architecture()
            # Эволюция стратегий
            await self._evolve_strategies()
            # Переобучение на истории
            await self._retrain_on_history()
            logger.info("StrategyAgent evolved successfully")
            return True
        except Exception as e:
            logger.error(f"Error in StrategyAgent evolution: {e}")
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
                "strategies": self.strategies,
            }
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(state, f"{path}/evolvable_strategy_agent_state.pth")
            return True
        except Exception as e:
            logger.error(f"StrategyAgent save_state error: {e}")
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
                self.strategies = state.get("strategies", self.strategies)
                logger.info(f"StrategyAgent state loaded from {path}")
                return True
        except Exception as e:
            logger.error(f"StrategyAgent load_state error: {e}")
        return False

    def _extract_features(
        self, market_data: pd.DataFrame, strategy_performance: Dict
    ) -> List[float]:
        """Извлечение признаков для выбора стратегии"""
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
            # Производительность стратегий
            strategy_scores = []
            for strategy in ["trend", "mean_reversion", "momentum", "volatility", "arbitrage"]:
                score = strategy_performance.get(strategy, 0.5)
                strategy_scores.append(score)
            # Комбинирование признаков
            features = [
                sma_5, sma_20, sma_50,
                volatility_5, volatility_20,
                momentum_5, momentum_20,
                rsi,
                volume_ratio
            ] + strategy_scores
            # Нормализация
            features = [float(f) if not np.isnan(f) else 0.0 for f in features]
            return features
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return []

    def _strategy_to_index(self, strategy: str) -> Optional[int]:
        """Преобразование названия стратегии в индекс"""
        strategy_map = {
            "trend": 0,
            "mean_reversion": 1,
            "momentum": 2,
            "volatility": 3,
            "arbitrage": 4,
        }
        return strategy_map.get(strategy)

    def _index_to_strategy(self, index: Union[int, float]) -> str:
        """Преобразование индекса в название стратегии"""
        # Исправление: приводим к int
        index = int(index)
        strategies = ["trend", "mean_reversion", "momentum", "volatility", "arbitrage"]
        return strategies[index] if 0 <= index < len(strategies) else "trend"

    def _update_metrics(
        self, loss: float, predictions: np.ndarray, target_strategy: int
    ) -> None:
        """Обновление метрик производительности"""
        try:
            predicted_strategy = np.argmax(predictions)
            self.total_predictions += 1
            if predicted_strategy == target_strategy:
                self.successful_predictions += 1
            # Обновление уверенности
            confidence = predictions[predicted_strategy]
            self.confidence_metric = 0.9 * self.confidence_metric + 0.1 * confidence
            # Обновление производительности
            accuracy = self.successful_predictions / self.total_predictions
            self.performance_metric = 0.9 * self.performance_metric + 0.1 * accuracy
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

    def _prefer_volatility_strategies(self) -> None:
        """Предпочтение стратегий волатильности"""
        pass

    def _prefer_trend_strategies(self) -> None:
        """Предпочтение трендовых стратегий"""
        pass

    def _prefer_mean_reversion_strategies(self) -> None:
        """Предпочтение стратегий возврата к среднему"""
        pass

    async def _evolve_model_architecture(self) -> None:
        """Эволюция архитектуры модели"""
        try:
            # Простая эволюция - изменение размеров слоев
            current_hidden_dim = self.ml_model.net[0].out_features
            new_hidden_dim = current_hidden_dim + np.random.randint(-10, 11)
            new_hidden_dim = max(32, min(128, new_hidden_dim))
            # Создание новой модели
            new_model = StrategyML(
                input_dim=25,
                hidden_dim=new_hidden_dim,
                num_strategies=5
            )
            # Копирование весов где возможно
            try:
                new_model.net[0].weight.data[:current_hidden_dim, :] = self.ml_model.net[0].weight.data
                new_model.net[0].bias.data[:current_hidden_dim] = self.ml_model.net[0].bias.data
            except Exception as e:
                # ИСПРАВЛЕНО: Логирование вместо поглощения исключения
                logger.warning(f"Failed to copy model weights during evolution: {e}")
                logger.debug(f"Model shapes - current: {self.ml_model.net[0].weight.shape}, new: {new_model.net[0].weight.shape}")
                # Продолжаем с новой моделью без копирования весов
            self.ml_model = new_model
            self.optimizer = torch.optim.Adam(self.ml_model.parameters(), lr=1e-3)
            logger.info(f"Model architecture evolved: hidden_dim={new_hidden_dim}")
        except Exception as e:
            logger.error(f"Error evolving model architecture: {e}")

    async def _evolve_strategies(self) -> None:
        """Эволюция стратегий"""
        try:
            # Простая эволюция - изменение весов стратегий
            pass
        except Exception as e:
            logger.error(f"Error evolving strategies: {e}")

    async def _retrain_on_history(self) -> None:
        """Переобучение на исторических данных"""
        try:
            if len(self.training_data) < 10:
                return
            # Подготовка данных
            features_list = [item["features"] for item in self.training_data]
            targets_list = [item["target_strategy"] for item in self.training_data]
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

    async def select_strategy(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        strategy_performance: Dict[str, float],
    ) -> Dict[str, Any]:
        """Выбор оптимальной стратегии"""
        try:
            # Извлечение признаков
            features = self._extract_features(market_data, strategy_performance)
            if len(features) == 0:
                return {"strategy": "trend", "confidence": 0.5, "reasoning": "No features available"}
            # ML предсказание
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            with torch.no_grad():
                predictions = self.ml_model(features_tensor)
                predicted_index = torch.argmax(predictions).item()
                confidence = torch.max(predictions).item()
            # Преобразование в название стратегии
            selected_strategy = self._index_to_strategy(predicted_index)
            return {
                "strategy": selected_strategy,
                "confidence": confidence,
                "reasoning": f"ML model selected {selected_strategy} with confidence {confidence:.3f}",
                "all_predictions": predictions.numpy()[0].tolist()
            }
        except Exception as e:
            logger.error(f"Error selecting strategy: {e}")
            return {"strategy": "trend", "confidence": 0.5, "reasoning": f"Error: {str(e)}"}

    async def get_available_strategies(self) -> Dict[str, str]:
        """Получение доступных стратегий"""
        return self.strategies

    async def update_strategy_performance(self, strategy: str, performance: float) -> None:
        """Обновление производительности стратегии"""
        try:
            # Обновление метрик на основе производительности
            if performance > 0.7:
                self.performance_metric = 0.9 * self.performance_metric + 0.1 * performance
            elif performance < 0.3:
                self.performance_metric = 0.9 * self.performance_metric + 0.1 * performance
        except Exception as e:
            logger.error(f"Error updating strategy performance: {e}")
