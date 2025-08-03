"""
Эволюционный агент управления рисками
Автоматически адаптируется к изменениям рынка и оптимизирует управление рисками
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


class RiskML(nn.Module):
    """ML модель для управления рисками"""

    def __init__(
        self, input_dim: int = 20, hidden_dim: int = 64, output_dim: int = 3
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),  # risk_score, position_size, stop_loss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class EvolvableRiskAgent(EvolvableComponent):
    """Эволюционный агент управления рисками"""

    def __init__(self, name: str = "evolvable_risk"):
        super().__init__(name)
        # ML модель для эволюции
        self.ml_model = RiskML()
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
        # Параметры риска
        self.max_position_size = 1.0
        self.max_drawdown = 0.1
        self.risk_per_trade = 0.02
        # Регистрация в эволюционном менеджере
        register_for_evolution(self)
        logger.info(f"EvolvableRiskAgent initialized: {name}")

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
                    # Адаптация параметров риска
                    if volatility > 0.03:  # Высокая волатильность
                        self.risk_per_trade *= 0.8
                        self.max_position_size *= 0.8
                    elif volatility < 0.01:  # Низкая волатильность
                        self.risk_per_trade *= 1.2
                        self.max_position_size *= 1.2
                    # Ограничение параметров
                    self.risk_per_trade = max(0.005, min(0.05, self.risk_per_trade))
                    self.max_position_size = max(0.1, min(2.0, self.max_position_size))
                    logger.debug(
                        f"RiskAgent adapted: volatility={volatility:.6f}, risk_per_trade={self.risk_per_trade:.4f}"
                    )
                    return True
        except Exception as e:
            logger.error(f"Error in RiskAgent adaptation: {e}")
        return False

    async def learn(self, data: Any) -> bool:
        """Обучение на новых данных"""
        try:
            if isinstance(data, dict) and "market_data" in data:
                market_data = data["market_data"]
                risk_metrics = data.get("risk_metrics", {})
                # Подготовка данных для обучения
                features = self._extract_features(market_data, risk_metrics)
                targets = self._extract_targets(risk_metrics)
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
                    logger.debug(f"RiskAgent learned: loss={loss.item():.6f}")
                    return True
        except Exception as e:
            logger.error(f"Error in RiskAgent learning: {e}")
        return False

    async def evolve(self, data: Any) -> bool:
        """Полная эволюция компонента"""
        try:
            # Эволюция архитектуры модели
            await self._evolve_model_architecture()
            # Эволюция параметров риска
            await self._evolve_risk_parameters()
            # Переобучение на истории
            await self._retrain_on_history()
            logger.info("RiskAgent evolved successfully")
            return True
        except Exception as e:
            logger.error(f"Error in RiskAgent evolution: {e}")
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
                "max_position_size": self.max_position_size,
                "max_drawdown": self.max_drawdown,
                "risk_per_trade": self.risk_per_trade,
            }
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(state, f"{path}/evolvable_risk_agent_state.pth")
            return True
        except Exception as e:
            logger.error(f"RiskAgent save_state error: {e}")
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
                self.max_position_size = state.get("max_position_size", 1.0)
                self.max_drawdown = state.get("max_drawdown", 0.1)
                self.risk_per_trade = state.get("risk_per_trade", 0.02)
                logger.info(f"RiskAgent state loaded from {path}")
                return True
        except Exception as e:
            logger.error(f"RiskAgent load_state error: {e}")
        return False

    def _extract_features(
        self, market_data: pd.DataFrame, risk_metrics: Dict
    ) -> List[float]:
        """Извлечение признаков для оценки риска"""
        try:
            if market_data.empty:
                return []
            # Базовые рыночные данные
            close_prices = market_data["close"]
            volume = market_data["volume"]
            # Технические индикаторы риска
            # Волатильность
            volatility_5 = close_prices.pct_change().rolling(5).std().iloc[-1]
            volatility_20 = close_prices.pct_change().rolling(20).std().iloc[-1]
            # Максимальная просадка
            rolling_max = close_prices.expanding().max()
            drawdown = (close_prices - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            # VaR (Value at Risk)
            returns = close_prices.pct_change().dropna()
            var_95 = np.percentile(returns, 5)
            # Корреляция с объемом
            volume_correlation = close_prices.corr(volume)
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
            # Комбинирование признаков
            features = [
                volatility_5, volatility_20,
                max_drawdown, var_95,
                volume_correlation,
                momentum_5, momentum_20,
                rsi, volume_ratio
            ]
            # Нормализация
            features = [float(f) if not np.isnan(f) else 0.0 for f in features]
            return features
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return []

    def _extract_targets(self, risk_metrics: Dict) -> List[float]:
        """Извлечение целевых значений для обучения"""
        try:
            risk_score = risk_metrics.get("risk_score", 0.5)
            position_size = risk_metrics.get("position_size", 1.0)
            stop_loss = risk_metrics.get("stop_loss", 0.02)
            return [risk_score, position_size, stop_loss]
        except Exception as e:
            logger.error(f"Error extracting targets: {e}")
            return [0.5, 1.0, 0.02]

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
            new_model = RiskML(
                input_dim=20,
                hidden_dim=new_hidden_dim,
                output_dim=3
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

    async def _evolve_risk_parameters(self) -> None:
        """Эволюция параметров риска"""
        try:
            # Адаптация параметров на основе производительности
            if self.performance_metric < 0.5:
                self.risk_per_trade *= 0.9  # Более консервативно
                self.max_position_size *= 0.9
            elif self.performance_metric > 0.8:
                self.risk_per_trade *= 1.1  # Более агрессивно
                self.max_position_size *= 1.1
            # Ограничения
            self.risk_per_trade = max(0.005, min(0.05, self.risk_per_trade))
            self.max_position_size = max(0.1, min(2.0, self.max_position_size))
            logger.info(f"Risk parameters evolved: risk_per_trade={self.risk_per_trade:.4f}, max_position_size={self.max_position_size:.2f}")
        except Exception as e:
            logger.error(f"Error evolving risk parameters: {e}")

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

    async def assess_risk(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        positions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Оценка риска для символа"""
        try:
            # Извлечение признаков
            risk_metrics = {}
            if positions:
                risk_metrics = {
                    "risk_score": positions.get("risk_score", 0.5),
                    "position_size": positions.get("position_size", 1.0),
                    "stop_loss": positions.get("stop_loss", 0.02),
                }
            features = self._extract_features(market_data, risk_metrics)
            if len(features) == 0:
                return {
                    "risk_score": 0.5,
                    "position_size": 1.0,
                    "stop_loss": 0.02,
                    "confidence": 0.5,
                    "reasoning": "No features available"
                }
            # ML предсказание
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            with torch.no_grad():
                predictions = self.ml_model(features_tensor)
                risk_score = predictions[0][0].item()
                position_size = predictions[0][1].item()
                stop_loss = predictions[0][2].item()
                confidence = torch.max(predictions).item()
            # Применение ограничений
            position_size = min(position_size, self.max_position_size)
            stop_loss = max(0.005, min(0.1, stop_loss))
            risk_score = max(0.0, min(1.0, risk_score))
            return {
                "risk_score": risk_score,
                "position_size": position_size,
                "stop_loss": stop_loss,
                "confidence": confidence,
                "reasoning": f"ML model assessed risk with confidence {confidence:.3f}",
                "features": features
            }
        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
            return {
                "risk_score": 0.5,
                "position_size": 1.0,
                "stop_loss": 0.02,
                "confidence": 0.5,
                "reasoning": f"Error: {str(e)}"
            }

    async def get_risk_parameters(self) -> Dict[str, float]:
        """Получение текущих параметров риска"""
        return {
            "max_position_size": self.max_position_size,
            "max_drawdown": self.max_drawdown,
            "risk_per_trade": self.risk_per_trade,
            "performance": self.performance_metric,
            "confidence": self.confidence_metric,
        }

    async def update_risk_metrics(self, metrics: Dict[str, float]) -> None:
        """Обновление метрик риска"""
        try:
            # Обновление метрик на основе производительности
            if "performance" in metrics:
                performance = metrics["performance"]
                if performance > 0.7:
                    self.performance_metric = 0.9 * self.performance_metric + 0.1 * performance
                elif performance < 0.3:
                    self.performance_metric = 0.9 * self.performance_metric + 0.1 * performance
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
