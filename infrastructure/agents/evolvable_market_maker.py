"""
Эволюционный агент маркет-мейкинга
Автоматически адаптируется к изменениям рынка и оптимизирует маркет-мейкинг
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


class MarketMakerML(nn.Module):
    """ML модель для маркет-мейкинга"""

    def __init__(
        self, input_dim: int = 30, hidden_dim: int = 64, output_dim: int = 4
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),  # bid_price, ask_price, bid_size, ask_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class EvolvableMarketMakerAgent(EvolvableComponent):
    """Эволюционный агент маркет-мейкинга"""

    def __init__(self, name: str = "evolvable_market_maker"):
        super().__init__(name)
        # ML модель для эволюции
        self.ml_model = MarketMakerML()
        self.optimizer = torch.optim.Adam(self.ml_model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        # Метрики производительности
        self.performance_metric = 0.5
        self.confidence_metric = 0.5
        self.successful_predictions = 0
        self.total_predictions = 0
        # История данных для обучения
        self.training_data: List[Dict[str, Any]] = []
        self.max_training_samples: int = 10000
        # Параметры маркет-мейкинга
        self.spread_multiplier = 1.5
        self.size_multiplier = 1.0
        self.inventory_target = 0.0
        # Регистрация в эволюционном менеджере
        register_for_evolution(self)
        logger.info(f"EvolvableMarketMakerAgent initialized: {name}")

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
                    volume = market_data["volume"].rolling(20).mean().iloc[-1]
                    # Адаптация параметров маркет-мейкинга
                    if volatility > 0.03:  # Высокая волатильность
                        self.spread_multiplier *= 1.2
                        self.size_multiplier *= 0.8
                    elif volatility < 0.01:  # Низкая волатильность
                        self.spread_multiplier *= 0.9
                        self.size_multiplier *= 1.1
                    # Ограничение параметров
                    self.spread_multiplier = max(1.0, min(3.0, self.spread_multiplier))
                    self.size_multiplier = max(0.5, min(2.0, self.size_multiplier))
                    logger.debug(
                        f"MarketMakerAgent adapted: volatility={volatility:.6f}, spread_multiplier={self.spread_multiplier:.3f}"
                    )
                    return True
        except Exception as e:
            logger.error(f"Error in MarketMakerAgent adaptation: {e}")
        return False

    async def learn(self, data: Any) -> bool:
        """Обучение на новых данных"""
        try:
            if isinstance(data, dict) and "market_data" in data:
                market_data = data["market_data"]
                order_book = data.get("order_book", {})
                # Подготовка данных для обучения
                features = self._extract_features(market_data, order_book)
                targets = self._extract_targets(order_book)
                if len(features) > 0 and len(targets) > 0:
                    # Обучение модели
                    features_tensor = torch.FloatTensor(features).unsqueeze(0)
                    targets_tensor = torch.FloatTensor(targets).unsqueeze(0)
                    self.optimizer.zero_grad()
                    predictions = self.ml_model(features_tensor)
                    loss = self.criterion(predictions, targets_tensor)
                    loss.backward()
                    self.optimizer.step()
                    # Обновление метрик
                    self._update_metrics(
                        loss.item(), predictions.detach().numpy()[0], targets
                    )
                    # Сохранение данных для обучения
                    self.training_data.append(
                        {
                            "features": features,
                            "targets": targets,
                            "timestamp": datetime.now(),
                        }
                    )
                    # Ограничение размера истории
                    if len(self.training_data) > self.max_training_samples:
                        self.training_data = self.training_data[-self.max_training_samples:]
                    logger.debug(f"MarketMakerAgent learned: loss={loss.item():.6f}")
                    return True
        except Exception as e:
            logger.error(f"Error in MarketMakerAgent learning: {e}")
        return False

    async def evolve(self, data: Any) -> bool:
        """Полная эволюция компонента"""
        try:
            # Эволюция архитектуры модели
            await self._evolve_model_architecture()
            # Эволюция параметров маркет-мейкинга
            await self._evolve_market_making_parameters()
            # Переобучение на истории
            await self._retrain_on_history()
            logger.info("MarketMakerAgent evolved successfully")
            return True
        except Exception as e:
            logger.error(f"Error in MarketMakerAgent evolution: {e}")
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
                "spread_multiplier": self.spread_multiplier,
                "size_multiplier": self.size_multiplier,
                "inventory_target": self.inventory_target,
            }
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(state, f"{path}/evolvable_market_maker_agent_state.pth")
            return True
        except Exception as e:
            logger.error(f"MarketMakerAgent save_state error: {e}")
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
                self.spread_multiplier = state.get("spread_multiplier", 1.5)
                self.size_multiplier = state.get("size_multiplier", 1.0)
                self.inventory_target = state.get("inventory_target", 0.0)
                logger.info(f"MarketMakerAgent state loaded from {path}")
                return True
        except Exception as e:
            logger.error(f"MarketMakerAgent load_state error: {e}")
        return False

    def _extract_features(
        self, market_data: pd.DataFrame, order_book: Dict
    ) -> List[float]:
        """Извлечение признаков для маркет-мейкинга"""
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
            # Данные ордербука
            bid_price = order_book.get("best_bid", close_prices.iloc[-1])
            ask_price = order_book.get("best_ask", close_prices.iloc[-1])
            bid_size = order_book.get("best_bid_size", 1.0)
            ask_size = order_book.get("best_ask_size", 1.0)
            spread = (ask_price - bid_price) / bid_price if bid_price > 0 else 0.0
            # Комбинирование признаков
            features = [
                sma_5, sma_20, sma_50,
                volatility_5, volatility_20,
                momentum_5, momentum_20,
                rsi, volume_ratio,
                bid_price, ask_price, bid_size, ask_size, spread
            ]
            # Нормализация
            features = [float(f) if not np.isnan(f) else 0.0 for f in features]
            return features
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return []

    def _extract_targets(self, order_book: Dict) -> List[float]:
        """Извлечение целевых значений для обучения"""
        try:
            bid_price = order_book.get("best_bid", 1.0)
            ask_price = order_book.get("best_ask", 1.0)
            bid_size = order_book.get("best_bid_size", 1.0)
            ask_size = order_book.get("best_ask_size", 1.0)
            return [bid_price, ask_price, bid_size, ask_size]
        except Exception as e:
            logger.error(f"Error extracting targets: {e}")
            return [1.0, 1.0, 1.0, 1.0]

    def _update_metrics(
        self, loss: float, predictions: np.ndarray, targets: List[float]
    ) -> None:
        """Обновление метрик производительности"""
        try:
            self.total_predictions += 1
            # Оценка точности предсказаний
            mse = np.mean((predictions - targets) ** 2)
            if mse < 0.1:  # Хорошее предсказание
                self.successful_predictions += 1
            # Обновление уверенности
            confidence = 1.0 - mse
            self.confidence_metric = 0.9 * self.confidence_metric + 0.1 * confidence
            # Обновление производительности
            accuracy = self.successful_predictions / self.total_predictions
            self.performance_metric = 0.9 * self.performance_metric + 0.1 * accuracy
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

    async def _evolve_model_architecture(self) -> None:
        """Эволюция архитектуры модели"""
        try:
            # Простая эволюция - изменение размеров слоев
            current_hidden_dim = self.ml_model.net[0].out_features
            new_hidden_dim = current_hidden_dim + np.random.randint(-10, 11)
            new_hidden_dim = max(32, min(128, new_hidden_dim))
            # Создание новой модели
            new_model = MarketMakerML(
                input_dim=30,
                hidden_dim=new_hidden_dim,
                output_dim=4
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

    async def _evolve_market_making_parameters(self) -> None:
        """Эволюция параметров маркет-мейкинга"""
        try:
            # Адаптация параметров на основе производительности
            if self.performance_metric < 0.5:
                self.spread_multiplier *= 1.1  # Увеличиваем спред
                self.size_multiplier *= 0.9  # Уменьшаем размер
            elif self.performance_metric > 0.8:
                self.spread_multiplier *= 0.9  # Уменьшаем спред
                self.size_multiplier *= 1.1  # Увеличиваем размер
            # Ограничения
            self.spread_multiplier = max(1.0, min(3.0, self.spread_multiplier))
            self.size_multiplier = max(0.5, min(2.0, self.size_multiplier))
            logger.info(f"Market making parameters evolved: spread_multiplier={self.spread_multiplier:.3f}, size_multiplier={self.size_multiplier:.3f}")
        except Exception as e:
            logger.error(f"Error evolving market making parameters: {e}")

    async def _retrain_on_history(self) -> None:
        """Переобучение на исторических данных"""
        try:
            if len(self.training_data) < 10:
                return
            # Подготовка данных
            features_list = [item["features"] for item in self.training_data]
            targets_list = [item["targets"] for item in self.training_data]
            if len(features_list) > 0 and len(targets_list) > 0:
                # Объединение признаков
                all_features = []
                for features in features_list:
                    all_features.extend(features)
                # Объединение целей
                all_targets = []
                for targets in targets_list:
                    all_targets.extend(targets)
                if len(all_features) > 0 and len(all_targets) > 0:
                    # Обучение
                    features_tensor = torch.FloatTensor(all_features).unsqueeze(0)
                    targets_tensor = torch.FloatTensor(all_targets).unsqueeze(0)
                    self.optimizer.zero_grad()
                    predictions = self.ml_model(features_tensor)
                    loss = self.criterion(predictions, targets_tensor)
                    loss.backward()
                    self.optimizer.step()
                    logger.info(f"Retrained on history: loss={loss.item():.6f}")
        except Exception as e:
            logger.error(f"Error retraining on history: {e}")

    async def generate_quotes(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        order_book: Dict[str, Any],
        inventory: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Генерация котировок для маркет-мейкинга"""
        try:
            # Извлечение признаков
            features = self._extract_features(market_data, order_book)
            if len(features) == 0:
                return {
                    "bid_price": 1.0,
                    "ask_price": 1.0,
                    "bid_size": 1.0,
                    "ask_size": 1.0,
                    "confidence": 0.5,
                    "reasoning": "No features available"
                }
            # ML предсказание
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            with torch.no_grad():
                predictions = self.ml_model(features_tensor)
                bid_price = predictions[0][0].item()
                ask_price = predictions[0][1].item()
                bid_size = predictions[0][2].item()
                ask_size = predictions[0][3].item()
                confidence = torch.max(predictions).item()
            # Применение параметров маркет-мейкинга
            mid_price = (bid_price + ask_price) / 2
            spread = (ask_price - bid_price) * self.spread_multiplier
            bid_price = mid_price - spread / 2
            ask_price = mid_price + spread / 2
            bid_size *= self.size_multiplier
            ask_size *= self.size_multiplier
            # Учет инвентаря
            if inventory is not None:
                inventory_skew = inventory / 1000.0  # Нормализация
                bid_price *= (1 + inventory_skew * 0.001)
                ask_price *= (1 - inventory_skew * 0.001)
            return {
                "bid_price": bid_price,
                "ask_price": ask_price,
                "bid_size": bid_size,
                "ask_size": ask_size,
                "confidence": confidence,
                "reasoning": f"ML model generated quotes with confidence {confidence:.3f}",
                "features": features
            }
        except Exception as e:
            logger.error(f"Error generating quotes: {e}")
            return {
                "bid_price": 1.0,
                "ask_price": 1.0,
                "bid_size": 1.0,
                "ask_size": 1.0,
                "confidence": 0.5,
                "reasoning": f"Error: {str(e)}"
            }

    async def get_market_making_parameters(self) -> Dict[str, float]:
        """Получение текущих параметров маркет-мейкинга"""
        return {
            "spread_multiplier": self.spread_multiplier,
            "size_multiplier": self.size_multiplier,
            "inventory_target": self.inventory_target,
            "performance": self.performance_metric,
            "confidence": self.confidence_metric,
        }

    async def update_market_making_metrics(self, metrics: Dict[str, float]) -> None:
        """Обновление метрик маркет-мейкинга"""
        try:
            # Обновление метрик на основе производительности
            if "performance" in metrics:
                performance = metrics["performance"]
                if performance > 0.7:
                    self.performance_metric = 0.9 * self.performance_metric + 0.1 * performance
                elif performance < 0.3:
                    self.performance_metric = 0.9 * self.performance_metric + 0.1 * performance
        except Exception as e:
            logger.error(f"Error updating market making metrics: {e}")
